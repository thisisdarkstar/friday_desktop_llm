"""Productivity tools: timers, reminders, notes, and calculations."""

import ast
import operator
import threading
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

# Safe mathematical calculation evaluator
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):  # numbers
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](_eval_node(node.operand))
    raise ValueError(f"Unsupported math syntax: {ast.dump(node)}")


def calculate(expression: str) -> dict:
    """Safely calculate a mathematical expression (e.g. '15% of 850' or '24 * 3.5')."""
    expr = expression.lower().replace("% of", "* 0.01 *").replace("x", "*").replace("^", "**")
    try:
        parsed = ast.parse(expr, mode="eval")
        result = _eval_node(parsed.body)
        return {"status": "success", "expression": expression, "result": result}
    except Exception as e:
        return {"status": "error", "message": f"Could not calculate '{expression}': {str(e)}"}


def take_note(text: str, filename: str = "notes.txt") -> dict:
    """Append a timestamped note to a desktop file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {text}\n"

    desktop = Path(os.environ.get("USERPROFILE", ".")) / "Desktop"
    target_file = desktop / filename

    try:
        with open(target_file, "a", encoding="utf-8") as f:
            f.write(entry)
        return {"status": "success", "message": f"Saved note to {target_file}", "note": text}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _timer_worker(seconds: int, label: str):
    time.sleep(seconds)
    # Beep or send notification
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception:
        pass


def create_timer(duration_seconds: int, label: str = "Timer") -> dict:
    """Set a countdown timer in seconds."""
    duration_seconds = max(1, int(duration_seconds))
    thread = threading.Thread(target=_timer_worker, args=(duration_seconds, label), daemon=True)
    thread.start()
    return {
        "status": "success",
        "message": f"Timer '{label}' set for {duration_seconds} seconds",
        "duration_seconds": duration_seconds,
    }


def create_reminder(message: str, minutes_from_now: int = 10) -> dict:
    """Create a reminder for N minutes from now."""
    seconds = int(minutes_from_now) * 60
    return create_timer(seconds, label=f"Reminder: {message}")
