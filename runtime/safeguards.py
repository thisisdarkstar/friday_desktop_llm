"""Production Safeguard & Normalization Engine for FRIDAY AI.

Provides deterministic, sub-millisecond (<1ms) post-processing, alias resolution,
action-precedence arbitration, and graceful model cascading:
1. Tool Name Aliases & Synonyms
2. Application Name Normalization
3. Action-Over-Chitchat Precedence Guardrail
4. Argument Coercion & Clamping
5. Automatic 1.5B Cascade for complex compound clauses
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"

# =============================================================================
# 1. CANONICAL MAPS & SYNONYMS
# =============================================================================

TOOL_CANONICAL_MAP = {
    # Synonyms / Hallucinated variants -> Canonical FRIDAY MCP tool
    "check_system_info": "get_system_info",
    "system_info": "get_system_info",
    "get_version": "chat_response",
    "check_system": "get_system_info",
    "system_status": "get_system_info",
    "file_delete": "delete_file",
    "remove_file": "delete_file",
    "organize_directory": "organize_files",
    "change_volume": "set_volume",
    "volume_set": "set_volume",
    "adjust_volume": "set_volume",
    "brightness_set": "set_brightness",
    "adjust_brightness": "set_brightness",
    "speak": "chat_response",
    "assistant_reply": "chat_response",
    "conversation": "chat_response",
    "chat_message_send": "send_whatsapp_message",
    "whatsapp_send": "send_whatsapp_message",
    "telegram_send": "send_telegram_message",
}

APP_CANONICAL_MAP = {
    "google": "chrome",
    "google chrome": "chrome",
    "browser": "chrome",
    "internet": "chrome",
    "ms word": "winword",
    "word": "winword",
    "word document": "winword",
    "excel": "excel",
    "ms excel": "excel",
    "powerpoint": "powerpnt",
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "spotify music": "spotify",
    "whatsapp web": "whatsapp",
    "slack desktop": "slack",
    "discord app": "discord",
}

# Strong action trigger patterns that should NEVER be swallowed by a greeting
ACTION_PATTERNS = [
    (r"\b(?:set|turn|change|put)\s+volume\s+(?:to\s+)?(\d+)", "set_volume", lambda m: {"level": int(m.group(1))}),
    (r"\b(?:mute|unmute)\s+(?:audio|sound|volume)", "mute_audio", lambda m: {"mute": "unmute" not in m.group(0)}),
    (r"\b(?:set|turn|change)\s+brightness\s+(?:to\s+)?(\d+)", "set_brightness", lambda m: {"level": int(m.group(1))}),
    (r"\b(?:take|grab|capture)\s+(?:a\s+)?screenshot", "take_screenshot", lambda m: {}),
    (r"\b(?:open|launch|start)\s+([a-zA-Z0-9_\s]+)", "launch_app", lambda m: {"name": m.group(1).strip()}),
    (r"\b(?:close|quit|kill|exit)\s+([a-zA-Z0-9_\s]+)", "close_app", lambda m: {"name": m.group(1).strip()}),
]


# =============================================================================
# 2. DETERMINISTIC SAFEGUARD PIPELINE
# =============================================================================

def apply_safeguards(raw_calls: Optional[List[Dict[str, Any]]], user_query: str) -> List[Dict[str, Any]]:
    """Applies the deterministic multi-stage safeguard pipeline to raw tool calls."""
    if not raw_calls:
        raw_calls = []

    safeguarded_calls = []

    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        tool = call.get("tool", "")
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}

        # 1. Normalize tool name
        clean_tool = TOOL_CANONICAL_MAP.get(tool.lower(), tool)

        # 2. Normalize app name if applicable
        if "name" in args and isinstance(args["name"], str):
            app_norm = args["name"].lower().strip()
            args["name"] = APP_CANONICAL_MAP.get(app_norm, args["name"])

        # 3. Argument Type Coercion & Bounds Clamping
        if clean_tool in ("set_volume", "set_brightness") and "level" in args:
            try:
                lvl = int(re.sub(r"[^\d]", "", str(args["level"])))
                args["level"] = max(0, min(100, lvl))
            except Exception:
                pass

        if clean_tool == "manage_window" and "action" in args:
            args["action"] = str(args["action"]).lower()
            if args["action"] == "close" and "app" in args:
                clean_tool = "close_app"
                args = {"name": args.get("app")}

        safeguarded_calls.append({"tool": clean_tool, "args": args})

    # Safeguard 3: Action-over-Chitchat Precedence Check
    # If the model ONLY output chat_response, but the prompt clearly contains an imperative command:
    if len(safeguarded_calls) == 1 and safeguarded_calls[0]["tool"] == "chat_response":
        q_lower = user_query.lower()
        extracted_actions = []
        for pat, tool_name, arg_fn in ACTION_PATTERNS:
            match = re.search(pat, q_lower)
            if match:
                try:
                    args = arg_fn(match)
                    if "name" in args and isinstance(args["name"], str):
                        args["name"] = APP_CANONICAL_MAP.get(args["name"], args["name"])
                    extracted_actions.append({"tool": tool_name, "args": args})
                except Exception:
                    pass
        if extracted_actions:
            # Replace pure greeting with the actual requested action!
            safeguarded_calls = extracted_actions

    return safeguarded_calls


def query_with_fallback(
    prompt: str,
    primary_model: str = "friday:0.5b-ultimate",
    fallback_model: str = "friday:1.5b",
    endpoint: str = OLLAMA_ENDPOINT,
    session: Optional[requests.Session] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Queries primary 0.5B model, applies safeguards, and seamlessly cascades to 1.5B if required."""
    s = session or requests.Session()

    def _call_ollama(model: str) -> Optional[List[Dict[str, Any]]]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "top_k": 1, "top_p": 0.95, "num_predict": 128},
        }
        try:
            resp = s.post(endpoint, json=payload, timeout=2.5)
            raw = resp.json().get("response", "").strip()
            if raw.startswith("[") and not raw.endswith("]"):
                raw += "]"
            return json.loads(raw)
        except Exception:
            return None

    # Step 1: Query Primary 0.5B Model (~250-320 ms)
    primary_calls = _call_ollama(primary_model)
    safeguarded = apply_safeguards(primary_calls, prompt)

    # Step 2: Check if Primary Output is Missing or Dropped on Complex Conjunction Chains
    # A chain is suspected incomplete if prompt has 2+ conjunctions ("after that", "and then")
    # but model output fewer tools than clauses.
    conjunction_count = len(re.findall(r"\b(?:after that|and then|then|followed by)\b", prompt.lower()))
    is_suspected_dropout = conjunction_count >= 1 and len(safeguarded) <= 1

    if not safeguarded or is_suspected_dropout:
        # Step 3: Cascade to 1.5B Fallback
        fallback_calls = _call_ollama(fallback_model)
        if fallback_calls:
            fb_safeguarded = apply_safeguards(fallback_calls, prompt)
            if fb_safeguarded and len(fb_safeguarded) >= len(safeguarded):
                return fb_safeguarded, fallback_model

    return safeguarded, primary_model
