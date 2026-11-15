"""DesktopIntent-LLM to FRIDAY AI Integration Bridge.

Translates the fine-tuned DesktopIntent neural JSON tool calls directly into
FRIDAY's native `SimpleIntentRoute` dataclass objects.
Provides sub-50ms deterministic neural routing with zero regex brittleness.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests

OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "friday"


@dataclass
class SimpleIntentRoute:
    category: str
    canonical_action: str
    target: Optional[str]
    normalized_text: str
    matched_rule: str


class FridayNeuralRouter:
    """Drop-in neural router for FRIDAY AI Assistant."""

    def __init__(self, model_name: str = DEFAULT_MODEL, endpoint: str = OLLAMA_ENDPOINT):
        self.model_name = model_name
        self.endpoint = endpoint
        self.session = requests.Session()

    def route_intents(
        self,
        prompt: str,
        timeout: float = 3.0,
    ) -> List[SimpleIntentRoute]:
        """Sends the user prompt to DesktopIntent-LLM and converts the output to FRIDAY routes.
        
        Returns an empty list [] if the model delegates to a high-level reasoning model,
        signaling FRIDAY to proceed with its cloud LLM pipeline.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 0.95,
                "num_predict": 128,
                "stop": ["<|im_end|>", "]"],
            },
        }

        try:
            start_time = time.perf_counter()
            resp = self.session.post(self.endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            raw_output = resp.json().get("response", "").strip()
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Ensure proper closing bracket if truncated by stop token
            if raw_output.startswith("[") and not raw_output.endswith("]"):
                raw_output += "]"

            tool_calls = json.loads(raw_output)
            routes = []

            for call in tool_calls:
                tool = call.get("tool", "")
                args = call.get("args", {})
                route = self._map_tool_to_route(tool, args, prompt)
                if route:
                    routes.append(route)

            return routes

        except Exception as e:
            # On connection failure or invalid JSON, return empty list for fallback
            return []

    def _map_tool_to_route(
        self, tool: str, args: Dict[str, Any], raw_text: str
    ) -> Optional[SimpleIntentRoute]:
        """Maps an individual tool call to FRIDAY's SimpleIntentRoute."""
        # 1. Audio
        if tool == "set_volume":
            return SimpleIntentRoute(
                category="system",
                canonical_action="VOLUME_SET",
                target=str(args.get("level", 50)),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool in ("mute_audio", "set_volume_mute"):
            action = "VOLUME_MUTE" if args.get("mute", True) else "VOLUME_UNMUTE"
            return SimpleIntentRoute(
                category="system",
                canonical_action=action,
                target=None,
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "change_volume":
            delta = args.get("delta", 10)
            action = "VOLUME_UP" if delta > 0 else "VOLUME_DOWN"
            return SimpleIntentRoute(
                category="system",
                canonical_action=action,
                target=str(abs(delta)),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 2. Display & Toggles
        elif tool == "set_brightness":
            return SimpleIntentRoute(
                category="system",
                canonical_action="BRIGHTNESS_SET",
                target=str(args.get("level", 50)),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "change_brightness":
            delta = args.get("delta", 10)
            action = "BRIGHTNESS_UP" if delta > 0 else "BRIGHTNESS_DOWN"
            return SimpleIntentRoute(
                category="system",
                canonical_action=action,
                target=str(abs(delta)),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "toggle_feature":
            feature = args.get("feature", "").upper()
            state = args.get("state", "toggle")
            return SimpleIntentRoute(
                category="system",
                canonical_action=f"TOGGLE_{feature}",
                target=state,
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 3. System Power & Hardware Info
        elif tool == "set_power_state":
            action = args.get("action", "lock").upper()
            return SimpleIntentRoute(
                category="system",
                canonical_action=action,
                target=None,
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "get_system_stats":
            return SimpleIntentRoute(
                category="system",
                canonical_action="SYSTEM_INFO",
                target=None,
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "get_time_or_date":
            qtype = args.get("query_type", "time").upper()
            return SimpleIntentRoute(
                category="system",
                canonical_action=qtype,
                target=None,
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 4. Apps & Process Lifecycle
        elif tool == "launch_app":
            return SimpleIntentRoute(
                category="apps",
                canonical_action="APP_LAUNCH",
                target=args.get("app", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "kill_app":
            return SimpleIntentRoute(
                category="apps",
                canonical_action="APP_KILL",
                target=args.get("app", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 5. Windows, Snapping & Multi-Monitor Presets
        elif tool == "manage_window":
            return SimpleIntentRoute(
                category="window",
                canonical_action=args.get("action", "focus").upper(),
                target=args.get("app", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "snap_window":
            pos = args.get("position", "left").upper()
            return SimpleIntentRoute(
                category="window",
                canonical_action=f"SNAP_{pos}",
                target=args.get("app", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "set_workspace_preset":
            return SimpleIntentRoute(
                category="window",
                canonical_action="LAYOUT_PRESET",
                target=args.get("preset", "default"),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 6. Browser & Web Search
        elif tool == "search_in_browser":
            return SimpleIntentRoute(
                category="search",
                canonical_action="SEARCH_WEB",
                target=args.get("query", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "browser_tab_action":
            action = args.get("action", "new").upper()
            return SimpleIntentRoute(
                category="tabs",
                canonical_action=f"TAB_{action}",
                target=None,
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "browser_navigate":
            return SimpleIntentRoute(
                category="tabs",
                canonical_action="NAVIGATE",
                target=args.get("target", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 7. GUI Shortcuts & Input
        elif tool == "press_shortcut":
            return SimpleIntentRoute(
                category="gui",
                canonical_action="HOTKEY",
                target=args.get("action", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "take_screenshot":
            return SimpleIntentRoute(
                category="system",
                canonical_action="SCREENSHOT",
                target=args.get("target", "screen"),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 8. Productivity & Timers
        elif tool == "create_timer":
            return SimpleIntentRoute(
                category="productivity",
                canonical_action="TIMER_SET",
                target=str(args.get("duration_seconds", 300)),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "create_reminder":
            return SimpleIntentRoute(
                category="productivity",
                canonical_action="REMINDER_SET",
                target=args.get("message", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "calculate":
            return SimpleIntentRoute(
                category="productivity",
                canonical_action="CALCULATE",
                target=args.get("expression", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 9. Filesystem Operations
        elif tool == "create_directory":
            return SimpleIntentRoute(
                category="files",
                canonical_action="CREATE_DIRECTORY",
                target=args.get("path", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "delete_file":
            return SimpleIntentRoute(
                category="files",
                canonical_action="DELETE_FILE",
                target=args.get("path", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "copy_file":
            return SimpleIntentRoute(
                category="files",
                canonical_action="COPY_FILE",
                target=f"{args.get('source', '')} -> {args.get('destination', '')}",
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "move_file":
            return SimpleIntentRoute(
                category="files",
                canonical_action="MOVE_FILE",
                target=f"{args.get('source', '')} -> {args.get('destination', '')}",
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "organize_directory_by_type":
            return SimpleIntentRoute(
                category="files",
                canonical_action="ORGANIZE_DIRECTORY",
                target=args.get("directory_path", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 10. Calendar & Scheduling
        elif tool == "create_calendar_event":
            return SimpleIntentRoute(
                category="calendar",
                canonical_action="CREATE_EVENT",
                target=f"{args.get('title', '')} | {args.get('start_time', '')}",
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool in ("list_calendar_events", "get_calendar_events"):
            return SimpleIntentRoute(
                category="calendar",
                canonical_action="LIST_EVENTS",
                target=None,
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "delete_calendar_event":
            return SimpleIntentRoute(
                category="calendar",
                canonical_action="DELETE_EVENT",
                target=args.get("title", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 11. Communication, Email & Office
        elif tool == "send_whatsapp_message":
            return SimpleIntentRoute(
                category="communication",
                canonical_action="WHATSAPP_SEND",
                target=f"{args.get('contact', '')}: {args.get('message', '')}",
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "send_telegram_message":
            return SimpleIntentRoute(
                category="communication",
                canonical_action="TELEGRAM_SEND",
                target=f"{args.get('contact', '')}: {args.get('message', '')}",
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "compose_email":
            return SimpleIntentRoute(
                category="communication",
                canonical_action="EMAIL_COMPOSE",
                target=args.get("recipient", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "read_emails":
            return SimpleIntentRoute(
                category="communication",
                canonical_action="EMAIL_READ",
                target=None,
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "create_word_document":
            return SimpleIntentRoute(
                category="communication",
                canonical_action="WORD_DOC_CREATE",
                target=args.get("title", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 12. Voice & Speech Control
        elif tool in ("switch_tts_voice", "set_tts_voice", "change_tts_voice"):
            return SimpleIntentRoute(
                category="voice",
                canonical_action="SWITCH_VOICE",
                target=args.get("voice_name_or_id", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "list_tts_voices":
            return SimpleIntentRoute(
                category="voice",
                canonical_action="LIST_VOICES",
                target=None,
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 13. Media & Terminal
        elif tool == "play_music":
            return SimpleIntentRoute(
                category="media",
                canonical_action="PLAY_MUSIC",
                target=args.get("query", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )
        elif tool == "run_terminal_command":
            return SimpleIntentRoute(
                category="system",
                canonical_action="RUN_COMMAND",
                target=args.get("command", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        # 14. Fallback & Delegation
        elif tool == "delegate_to_reasoning_model":
            # Return None to trigger cloud reasoning escalation in FRIDAY
            return None
        elif tool == "chat_response":
            return SimpleIntentRoute(
                category="chat",
                canonical_action="CHAT",
                target=args.get("message", ""),
                normalized_text=raw_text,
                matched_rule="neural_desktop_intent",
            )

        return None
