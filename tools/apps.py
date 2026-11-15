"""Application management tools: launch and terminate desktop applications."""

import os
import subprocess
import psutil

# Common Windows app shortcuts and URI schemes
APP_SHORTCUTS = {
    "notepad": "notepad.exe",
    "calc": "calc.exe",
    "calculator": "calc.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "files": "explorer.exe",
    "chrome": "chrome",
    "google chrome": "chrome",
    "brave": "brave",
    "brave browser": "brave",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "spotify": "spotify",
    "code": "code",
    "vs code": "code",
    "vscode": "code",
    "terminal": "wt",
    "windows terminal": "wt",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "task manager": "taskmgr.exe",
    "taskmgr": "taskmgr.exe",
    "settings": "ms-settings:",
    "whatsapp": "whatsapp:",
    "discord": "discord",
}


def launch_app(name: str) -> dict:
    """Launch an application by common name, executable, or protocol."""
    clean_name = name.lower().strip()
    target = APP_SHORTCUTS.get(clean_name, clean_name)

    try:
        if target.endswith(":") or target.startswith("http"):
            # URI scheme (e.g. ms-settings:, whatsapp:, https://...)
            os.startfile(target)
        else:
            # Command / executable
            subprocess.Popen(f"start {target}", shell=True)
        return {"status": "success", "message": f"Launched {name}", "target": target}
    except Exception as e:
        return {"status": "error", "message": f"Failed to launch {name}: {str(e)}"}


def kill_app(name: str) -> dict:
    """Terminate all running processes matching the application name."""
    clean_name = name.lower().strip()
    # Remove .exe extension if user passed it
    if clean_name.endswith(".exe"):
        clean_name = clean_name[:-4]

    killed_pids = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            p_name = proc.info["name"].lower()
            if clean_name in p_name:
                proc.terminate()
                killed_pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if killed_pids:
        return {"status": "success", "message": f"Terminated {name}", "pids": killed_pids}
    return {"status": "not_found", "message": f"No running processes found for {name}"}


def list_running_apps() -> dict:
    """List running graphical desktop applications."""
    apps = set()
    for proc in psutil.process_iter(["name"]):
        try:
            name = proc.info["name"]
            if name.endswith(".exe"):
                apps.add(name[:-4])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"status": "success", "apps": sorted(list(apps))[:50]}
