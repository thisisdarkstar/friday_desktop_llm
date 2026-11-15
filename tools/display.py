"""Display brightness tools using screen_brightness_control."""

import screen_brightness_control as sbc


def set_brightness(level: int) -> dict:
    """Set display brightness level from 0 to 100."""
    level = max(0, min(100, int(level)))
    try:
        sbc.set_brightness(level)
        return {"status": "success", "brightness": level}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_brightness() -> dict:
    """Get current display brightness."""
    try:
        vals = sbc.get_brightness()
        # Filter out None values from inactive displays
        valid_vals = [v for v in vals if v is not None]
        avg = int(sum(valid_vals) / len(valid_vals)) if valid_vals else 50
        return {"status": "success", "brightness": avg, "displays": vals}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def change_brightness(delta: int) -> dict:
    """Increase or decrease brightness by delta percentage points."""
    curr = get_brightness().get("brightness", 50)
    new_level = max(0, min(100, curr + delta))
    return set_brightness(new_level)
