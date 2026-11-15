"""Window management tools using Windows Win32 API."""

import win32gui
import win32con
import win32service


def _get_input_desktop():
    try:
        return win32service.OpenInputDesktop(
            0, False, win32con.DESKTOP_SWITCHDESKTOP | win32con.GENERIC_READ | win32con.GENERIC_ALL
        )
    except Exception:
        return None


def find_windows_by_title_or_app(query: str):
    """Find visible top-level windows matching query in title."""
    q = query.lower().strip()
    matches = []

    def enum_cb(hwnd, results):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd).strip()
            if title and q in title.lower():
                results.append((hwnd, title))

    desk = _get_input_desktop()
    if desk:
        win32gui.EnumDesktopWindows(desk, enum_cb, matches)
    else:
        win32gui.EnumWindows(enum_cb, matches)

    return matches


def manage_window(app: str, action: str) -> dict:
    """
    Manage application window state.
    Actions supported: 'minimize', 'maximize', 'restore', 'focus', 'close'.
    """
    action = action.lower().strip()
    windows = find_windows_by_title_or_app(app)

    if not windows:
        return {"status": "not_found", "message": f"No active window found for '{app}'"}

    hwnd, title = windows[0]

    try:
        if action == "minimize":
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        elif action == "maximize":
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        elif action == "restore":
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        elif action in ("focus", "bring_to_front"):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
        elif action == "close":
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        else:
            return {"status": "error", "message": f"Unknown window action '{action}'"}

        return {"status": "success", "action": action, "window": title, "hwnd": hwnd}
    except Exception as e:
        return {"status": "error", "message": f"Failed to {action} window: {str(e)}"}
