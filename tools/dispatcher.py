"""Central Tool Dispatcher for DesktopIntent.
Validates LLM-generated JSON tool calls with Pydantic and executes native tools.
"""

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from tools.audio import set_volume, mute_audio, change_volume, get_volume
from tools.display import set_brightness, change_brightness, get_brightness
from tools.apps import launch_app, kill_app, list_running_apps
from tools.windows import manage_window
from tools.productivity import create_timer, create_reminder, take_note, calculate
from tools.system import lock_pc, get_system_stats


class ToolCall(BaseModel):
    tool: str = Field(..., description="The name of the tool to invoke")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


TOOL_REGISTRY = {
    "set_volume": lambda args: set_volume(args.get("level", 50)),
    "mute_audio": lambda args: mute_audio(args.get("mute", True)),
    "set_volume_mute": lambda args: mute_audio(args.get("mute", True)),
    "change_volume": lambda args: change_volume(args.get("delta", 0)),
    "get_volume": lambda args: get_volume(),
    "set_brightness": lambda args: set_brightness(args.get("level", 50)),
    "change_brightness": lambda args: change_brightness(args.get("delta", 0)),
    "get_brightness": lambda args: get_brightness(),
    "launch_app": lambda args: launch_app(args.get("name", "")),
    "kill_app": lambda args: kill_app(args.get("name", "")),
    "list_running_apps": lambda args: list_running_apps(),
    "manage_window": lambda args: manage_window(args.get("app", ""), args.get("action", "focus")),
    "create_timer": lambda args: create_timer(
        args.get("duration_seconds", 60), args.get("label", "Timer")
    ),
    "create_reminder": lambda args: create_reminder(
        args.get("message", "Reminder"), args.get("minutes_from_now", 10)
    ),
    "take_note": lambda args: take_note(args.get("text", ""), args.get("filename", "notes.txt")),
    "calculate": lambda args: calculate(args.get("expression", "")),
    "lock_pc": lambda args: lock_pc(),
    "get_system_stats": lambda args: get_system_stats(),
    "get_system_info": lambda args: get_system_stats(),
    "get_time": lambda args: {"status": "success", "query": "time"},
    "press_shortcut": lambda args: {"status": "success", "action": args.get("action", "")},
    "search_in_browser": lambda args: {
        "status": "success",
        "query": args.get("query", ""),
        "engine": args.get("engine", "google"),
    },
    "delegate_to_reasoning_model": lambda args: {
        "status": "delegated",
        "target": "cloud_reasoning_model",
        "prompt": args.get("prompt", ""),
    },
    "chat_response": lambda args: {
        "status": "chat",
        "message": args.get("message", "How can I help you with your desktop?"),
    },
}


def parse_llm_output(raw_output: str) -> List[ToolCall]:
    """Extract and parse tool call array from raw LLM completion."""
    cleaned = raw_output.strip()

    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            [l for l in lines if not l.strip().startswith("```") and not l.strip() == "```"]
        ).strip()

    # Find the outermost array [ ... ]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    data = json.loads(cleaned)

    # If output was a single dict, wrap in a list
    if isinstance(data, dict):
        if "actions" in data and isinstance(data["actions"], list):
            data = data["actions"]
        elif "tool" in data:
            data = [data]
        elif "name" in data:
            data = [{"tool": data["name"], "args": data.get("parameters", {})}]

    tool_calls = []
    for item in data:
        # Standardize OpenAI format if present
        if "name" in item and "tool" not in item:
            item["tool"] = item["name"]
            item["args"] = item.get("parameters", {})
        tool_calls.append(ToolCall(**item))

    return tool_calls


def execute_tool_call(tool_call: ToolCall) -> Dict[str, Any]:
    """Execute a single validated tool call."""
    tool_name = tool_call.tool
    handler = TOOL_REGISTRY.get(tool_name)

    if not handler:
        return {
            "tool": tool_name,
            "status": "error",
            "message": f"Tool '{tool_name}' not recognized",
        }

    try:
        res = handler(tool_call.args)
        return {"tool": tool_name, "args": tool_call.args, "result": res}
    except Exception as e:
        return {"tool": tool_name, "status": "error", "message": str(e)}


def dispatch(raw_llm_output: str) -> List[Dict[str, Any]]:
    """Parse and sequentially execute an ordered list of tool calls."""
    try:
        calls = parse_llm_output(raw_llm_output)
    except Exception as e:
        return [{"status": "parse_error", "message": f"Could not parse tool calls: {str(e)}"}]

    results = []
    for call in calls:
        res = execute_tool_call(call)
        results.append(res)
    return results
