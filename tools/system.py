"""System status and control tools."""

import ctypes
import psutil


def lock_pc() -> dict:
    """Lock the Windows workstation."""
    try:
        user32 = ctypes.windll.User32
        success = user32.LockWorkStation()
        return {"status": "success" if success else "error", "message": "Workstation locked"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_system_stats() -> dict:
    """Get CPU, Memory, and Battery status."""
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
        battery = psutil.sensors_battery()
        battery_pct = battery.percent if battery else None
        plugged = battery.power_plugged if battery else None

        return {
            "status": "success",
            "cpu_percent": cpu,
            "ram_percent": ram,
            "battery_percent": battery_pct,
            "power_plugged": plugged,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
