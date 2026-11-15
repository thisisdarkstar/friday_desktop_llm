"""Generates the Exhaustive All-Tools Dataset for FRIDAY AI.

Covers every single tool and capability in FRIDAY's full catalog:
1. Audio (mute, unmute, set, change, get)
2. Display & Toggles (brightness, dark mode, night light, wifi, bluetooth)
3. Power & System (lock, sleep, restart, shutdown, battery, cpu/ram, time, date)
4. Apps & Windows (launch, close, minimize, maximize, restore, snap, focus, layouts)
5. Browser & Tabs (new tab, close tab, reload, next tab, search, navigate)
6. Clipboard & Shortcuts (copy, paste, cut, select all, undo, clipboard history)
7. Screenshots & Vision (take screenshot, inspect screen)
8. Filesystem (create directory, delete file, copy file, move file, organize)
9. Calendar & Scheduling (create event, list events, delete event)
10. Productivity (timer, reminder, calculate, take note)
11. Communication & Office (read emails, compose email, whatsapp, telegram, word doc)
12. Voice & Speech (switch tts voice, list voices)
13. Media (play music)
14. Terminal & Shell (run terminal command)
15. Dialogue & Standby (greetings, identity, status, sleep/standby)
16. Cloud Delegation (open-ended coding, writing, math, essays, business)
"""

import json
import random
from pathlib import Path

random.seed(42)

EXHAUSTIVE_TOOL_SAMPLES = [
    # =========================================================================
    # 1. AUDIO & SOUND
    # =========================================================================
    {"q": "mute audio", "t": [{"tool": "mute_audio", "args": {"mute": True}}]},
    {"q": "please mute the volume", "t": [{"tool": "mute_audio", "args": {"mute": True}}]},
    {"q": "unmute sound", "t": [{"tool": "mute_audio", "args": {"mute": False}}]},
    {"q": "turn sound back on please", "t": [{"tool": "mute_audio", "args": {"mute": False}}]},
    {"q": "set volume to 80", "t": [{"tool": "set_volume", "args": {"level": 80}}]},
    {"q": "volume 0", "t": [{"tool": "set_volume", "args": {"level": 0}}]},
    {"q": "max volume", "t": [{"tool": "set_volume", "args": {"level": 100}}]},
    {"q": "increase volume by 15", "t": [{"tool": "change_volume", "args": {"delta": 15}}]},
    {"q": "turn down volume by 25", "t": [{"tool": "change_volume", "args": {"delta": -25}}]},
    {"q": "turn down the volumme by 20", "t": [{"tool": "change_volume", "args": {"delta": -20}}]},
    {"q": "what is current volume level", "t": [{"tool": "get_volume", "args": {}}]},

    # =========================================================================
    # 2. DISPLAY & HARDWARE TOGGLES
    # =========================================================================
    {"q": "max brightness", "t": [{"tool": "set_brightness", "args": {"level": 100}}]},
    {"q": "set brightness to 70", "t": [{"tool": "set_brightness", "args": {"level": 70}}]},
    {"q": "dim the screen to 20%", "t": [{"tool": "set_brightness", "args": {"level": 20}}]},
    {"q": "dim display to 30", "t": [{"tool": "set_brightness", "args": {"level": 30}}]},
    {"q": "increase brightness by 10", "t": [{"tool": "change_brightness", "args": {"delta": 10}}]},
    {"q": "decrease brightness by 20", "t": [{"tool": "change_brightness", "args": {"delta": -20}}]},
    {"q": "enable dark mode", "t": [{"tool": "toggle_dark_mode", "args": {"dark": True}}]},
    {"q": "turn on light theme", "t": [{"tool": "toggle_dark_mode", "args": {"dark": False}}]},
    {"q": "turn on night light", "t": [{"tool": "toggle_night_light", "args": {"enabled": True}}]},
    {"q": "disable night light", "t": [{"tool": "toggle_night_light", "args": {"enabled": False}}]},
    {"q": "turn off wifi", "t": [{"tool": "toggle_wifi", "args": {"enabled": False}}]},
    {"q": "enable wifi", "t": [{"tool": "toggle_wifi", "args": {"enabled": True}}]},
    {"q": "turn off bluetooth", "t": [{"tool": "toggle_bluetooth", "args": {"enabled": False}}]},
    {"q": "enable bluetooth", "t": [{"tool": "toggle_bluetooth", "args": {"enabled": True}}]},

    # =========================================================================
    # 3. POWER & SYSTEM DIAGNOSTICS
    # =========================================================================
    {"q": "lock the pc", "t": [{"tool": "lock_screen", "args": {}}]},
    {"q": "lock my computer", "t": [{"tool": "lock_screen", "args": {}}]},
    {"q": "put pc to sleep", "t": [{"tool": "sleep_system", "args": {}}]},
    {"q": "sleep computer", "t": [{"tool": "sleep_system", "args": {}}]},
    {"q": "shutdwon the computer", "t": [{"tool": "sleep_system", "args": {}}]},
    {"q": "restart pc", "t": [{"tool": "restart_system", "args": {}}]},
    {"q": "reboot system", "t": [{"tool": "restart_system", "args": {}}]},
    {"q": "how much battery is left", "t": [{"tool": "get_battery_status", "args": {}}]},
    {"q": "check battery percentage", "t": [{"tool": "get_battery_status", "args": {}}]},
    {"q": "check cpu and memory usage", "t": [{"tool": "get_system_info", "args": {}}]},
    {"q": "show system hardware stats", "t": [{"tool": "get_system_info", "args": {}}]},
    {"q": "what time is it", "t": [{"tool": "get_time", "args": {}}]},
    {"q": "whats the date today", "t": [{"tool": "get_date", "args": {}}]},

    # =========================================================================
    # 4. APPS & WINDOW MANAGEMENT
    # =========================================================================
    {"q": "open brave", "t": [{"tool": "launch_app", "args": {"name": "brave"}}]},
    {"q": "launch visual studio code", "t": [{"tool": "launch_app", "args": {"name": "code"}}]},
    {"q": "open file explorer", "t": [{"tool": "launch_app", "args": {"name": "explorer"}}]},
    {"q": "launch file explorer please", "t": [{"tool": "launch_app", "args": {"name": "explorer"}}]},
    {"q": "open calculator", "t": [{"tool": "launch_app", "args": {"name": "calculator"}}]},
    {"q": "run terminal", "t": [{"tool": "launch_app", "args": {"name": "terminal"}}]},
    {"q": "close spotify", "t": [{"tool": "close_app", "args": {"name": "spotify"}}]},
    {"q": "kill discord", "t": [{"tool": "close_app", "args": {"name": "discord"}}]},
    {"q": "clos whatsapp right now", "t": [{"tool": "close_app", "args": {"name": "whatsapp"}}]},
    {"q": "minimize the active window", "t": [{"tool": "manage_window", "args": {"app": "active", "action": "minimize"}}]},
    {"q": "maximize chrome", "t": [{"tool": "manage_window", "args": {"app": "chrome", "action": "maximize"}}]},
    {"q": "focus slack", "t": [{"tool": "manage_window", "args": {"app": "slack", "action": "focus"}}]},
    {"q": "snap window left", "t": [{"tool": "snap_window", "args": {"app": "active", "position": "left"}}]},
    {"q": "snap browser to the right side", "t": [{"tool": "snap_window", "args": {"app": "browser", "position": "right"}}]},
    {"q": "minimize everything and show desktop", "t": [{"tool": "minimize_all_windows", "args": {}}]},
    {"q": "restore all windows", "t": [{"tool": "restore_all_windows", "args": {}}]},
    {"q": "switch to coding workspace layout", "t": [{"tool": "set_workspace_preset", "args": {"preset": "coding"}}]},
    {"q": "swich to coding workspace", "t": [{"tool": "set_workspace_preset", "args": {"preset": "coding"}}]},
    {"q": "set up research layout", "t": [{"tool": "set_workspace_preset", "args": {"preset": "research"}}]},
    {"q": "clean desktop layout", "t": [{"tool": "set_workspace_preset", "args": {"preset": "clean"}}]},
    {"q": "switch to clean layout", "t": [{"tool": "set_workspace_preset", "args": {"preset": "clean"}}]},

    # =========================================================================
    # 5. BROWSER & TABS
    # =========================================================================
    {"q": "open a new browser tab", "t": [{"tool": "browser_tab_action", "args": {"action": "new_tab"}}]},
    {"q": "close current tab", "t": [{"tool": "browser_tab_action", "args": {"action": "close_tab"}}]},
    {"q": "reload this webpage", "t": [{"tool": "browser_tab_action", "args": {"action": "reload_tab"}}]},
    {"q": "switch to next tab", "t": [{"tool": "browser_tab_action", "args": {"action": "next_tab"}}]},
    {"q": "search python asyncio on google", "t": [{"tool": "search_in_browser", "args": {"query": "python asyncio", "engine": "google"}}]},
    {"q": "search lofi beats on youtube", "t": [{"tool": "search_in_browser", "args": {"query": "lofi beats", "engine": "youtube"}}]},
    {"q": "navigate to https://github.com", "t": [{"tool": "browser_navigate", "args": {"url": "https://github.com"}}]},

    # =========================================================================
    # 6. CLIPBOARD & SHORTCUTS
    # =========================================================================
    {"q": "copy selected text", "t": [{"tool": "press_shortcut", "args": {"action": "copy"}}]},
    {"q": "paste from clipboard", "t": [{"tool": "press_shortcut", "args": {"action": "paste"}}]},
    {"q": "selec all text in active window", "t": [{"tool": "press_shortcut", "args": {"action": "select_all"}}]},
    {"q": "select all and copy", "t": [{"tool": "press_shortcut", "args": {"action": "select_all"}}, {"tool": "press_shortcut", "args": {"action": "copy"}}]},
    {"q": "undo my last action", "t": [{"tool": "press_shortcut", "args": {"action": "undo"}}]},
    {"q": "press undo", "t": [{"tool": "press_shortcut", "args": {"action": "undo"}}]},
    {"q": "get clipboard history", "t": [{"tool": "press_shortcut", "args": {"action": "clipboard_history"}}]},
    {"q": "show clipboard history", "t": [{"tool": "press_shortcut", "args": {"action": "clipboard_history"}}]},
    {"q": "open clipboard history", "t": [{"tool": "press_shortcut", "args": {"action": "clipboard_history"}}]},
    {"q": "take a screenshot of my screen", "t": [{"tool": "take_screenshot", "args": {}}]},
    {"q": "screnshot my desktop right now", "t": [{"tool": "take_screenshot", "args": {}}]},

    # =========================================================================
    # 7. FILESYSTEM TOOLS (NEW & COMPLETE!)
    # =========================================================================
    {"q": "create a new directory named project_files", "t": [{"tool": "create_directory", "args": {"path": "project_files"}}]},
    {"q": "make a folder called invoices", "t": [{"tool": "create_directory", "args": {"path": "invoices"}}]},
    {"q": "create folder reports in documents", "t": [{"tool": "create_directory", "args": {"path": "documents/reports"}}]},
    {"q": "mak a new directry calld project_files", "t": [{"tool": "create_directory", "args": {"path": "project_files"}}]},
    {"q": "make a new directry called test", "t": [{"tool": "create_directory", "args": {"path": "test"}}]},
    {"q": "delete file temp.txt", "t": [{"tool": "delete_file", "args": {"path": "temp.txt"}}]},
    {"q": "remove file old_notes.txt", "t": [{"tool": "delete_file", "args": {"path": "old_notes.txt"}}]},
    {"q": "delete file notes.log", "t": [{"tool": "delete_file", "args": {"path": "notes.log"}}]},
    {"q": "copy file data.csv to backup folder", "t": [{"tool": "copy_file", "args": {"source": "data.csv", "destination": "backup/data.csv"}}]},
    {"q": "copy file report.pdf to documents", "t": [{"tool": "copy_file", "args": {"source": "report.pdf", "destination": "documents/report.pdf"}}]},
    {"q": "copy file image.png to desktop", "t": [{"tool": "copy_file", "args": {"source": "image.png", "destination": "desktop/image.png"}}]},
    {"q": "copy my file budget.xlsx to archive", "t": [{"tool": "copy_file", "args": {"source": "budget.xlsx", "destination": "archive/budget.xlsx"}}]},
    {"q": "copy file notes.txt to backup", "t": [{"tool": "copy_file", "args": {"source": "notes.txt", "destination": "backup/notes.txt"}}]},
    {"q": "move file report.pdf to downloads", "t": [{"tool": "move_file", "args": {"source": "report.pdf", "destination": "downloads/report.pdf"}}]},
    {"q": "move file data.csv to archive", "t": [{"tool": "move_file", "args": {"source": "data.csv", "destination": "archive/data.csv"}}]},
    {"q": "move file photo.jpg to pictures", "t": [{"tool": "move_file", "args": {"source": "photo.jpg", "destination": "pictures/photo.jpg"}}]},
    {"q": "move document.docx to work folder", "t": [{"tool": "move_file", "args": {"source": "document.docx", "destination": "work/document.docx"}}]},
    {"q": "organize my downloads directory by file type", "t": [{"tool": "organize_directory_by_type", "args": {"directory_path": "downloads"}}]},
    {"q": "organize my downloads folder by type", "t": [{"tool": "organize_directory_by_type", "args": {"directory_path": "downloads"}}]},
    {"q": "organize desktop files into folders", "t": [{"tool": "organize_directory_by_type", "args": {"directory_path": "desktop"}}]},
    {"q": "organize desktop files by extension", "t": [{"tool": "organize_directory_by_type", "args": {"directory_path": "desktop"}}]},

    # =========================================================================
    # 8. CALENDAR & SCHEDULING (NEW & COMPLETE!)
    # =========================================================================
    {"q": "schedule team meeting tomorrow at 10am", "t": [{"tool": "create_calendar_event", "args": {"title": "team meeting", "start_time": "tomorrow at 10am"}}]},
    {"q": "create a calendar event for dentist appointment on Friday at 3pm", "t": [{"tool": "create_calendar_event", "args": {"title": "dentist appointment", "start_time": "Friday at 3pm"}}]},
    {"q": "add event project review Monday 2pm to my calendar", "t": [{"tool": "create_calendar_event", "args": {"title": "project review", "start_time": "Monday 2pm"}}]},
    {"q": "what is on my calendar today", "t": [{"tool": "list_calendar_events", "args": {}}]},
    {"q": "what do I have on my calendar today", "t": [{"tool": "list_calendar_events", "args": {}}]},
    {"q": "list my scheduled calendar events", "t": [{"tool": "list_calendar_events", "args": {}}]},
    {"q": "show my upcoming meetings", "t": [{"tool": "list_calendar_events", "args": {}}]},
    {"q": "check my calendar for appointments", "t": [{"tool": "list_calendar_events", "args": {}}]},
    {"q": "delete calendar event team meeting", "t": [{"tool": "delete_calendar_event", "args": {"title": "team meeting"}}]},

    # =========================================================================
    # 9. PRODUCTIVITY (TIMERS, REMINDERS, NOTES, CALC)
    # =========================================================================
    {"q": "set a timer for 15 minutes", "t": [{"tool": "create_timer", "args": {"duration_seconds": 900, "label": "Timer"}}]},
    {"q": "start a 25 min pomodoro timer", "t": [{"tool": "create_timer", "args": {"duration_seconds": 1500, "label": "Pomodoro"}}]},
    {"q": "remind me to drink water in 30 minutes", "t": [{"tool": "create_reminder", "args": {"message": "drink water", "minutes_from_now": 30}}]},
    {"q": "calculate 450 * 18", "t": [{"tool": "calculate", "args": {"expression": "450 * 18"}}]},
    {"q": "calulate 55 times 12", "t": [{"tool": "calculate", "args": {"expression": "55 * 12"}}]},
    {"q": "take a note: buy groceries tomorrow", "t": [{"tool": "take_note", "args": {"text": "buy groceries tomorrow"}}]},
    {"q": "write note call John at 5pm", "t": [{"tool": "take_note", "args": {"text": "call John at 5pm"}}]},

    # =========================================================================
    # 10. COMMUNICATION, MESSAGING & OFFICE
    # =========================================================================
    {"q": "send a whatsapp message to John saying I will be late", "t": [{"tool": "send_whatsapp_message", "args": {"contact": "John", "message": "I will be late"}}]},
    {"q": "message Sarah on whatsapp see you at 6pm", "t": [{"tool": "send_whatsapp_message", "args": {"contact": "Sarah", "message": "see you at 6pm"}}]},
    {"q": "send telegram message to Alex meeting starting soon", "t": [{"tool": "send_telegram_message", "args": {"contact": "Alex", "message": "meeting starting soon"}}]},
    {"q": "read my latest emails", "t": [{"tool": "read_emails", "args": {"limit": 5}}]},
    {"q": "check my inbox for new emails", "t": [{"tool": "read_emails", "args": {"limit": 5}}]},
    {"q": "compose email to boss@company.com subject quarterly report body report attached", "t": [{"tool": "compose_email", "args": {"recipient": "boss@company.com", "subject": "quarterly report", "body": "report attached"}}]},
    {"q": "create a word document named project_proposal.docx", "t": [{"tool": "create_word_document", "args": {"title": "project_proposal"}}]},
    {"q": "generate a word doc titled meeting_minutes", "t": [{"tool": "create_word_document", "args": {"title": "meeting_minutes"}}]},

    # =========================================================================
    # 11. VOICE & SPEECH SETTINGS
    # =========================================================================
    {"q": "switch voice to female tts", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": "female"}}]},
    {"q": "change tts voice to Aria", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": "Aria"}}]},
    {"q": "hey friday, could you change tts voice to Aria?", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": "Aria"}}]},
    {"q": "set voice to female", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": "female"}}]},
    {"q": "set tts voice to Guy", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": "Guy"}}]},
    {"q": "change voice to Prabhat", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": "Prabhat"}}]},
    {"q": "switch speech voice to Jenny", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": "Jenny"}}]},
    {"q": "list available tts voices", "t": [{"tool": "list_tts_voices", "args": {}}]},
    {"q": "what voices do you have", "t": [{"tool": "list_tts_voices", "args": {}}]},

    # =========================================================================
    # 12. MEDIA & TERMINAL
    # =========================================================================
    {"q": "play some jazz music", "t": [{"tool": "play_music", "args": {"query": "jazz music"}}]},
    {"q": "play music lofi hip hop", "t": [{"tool": "play_music", "args": {"query": "lofi hip hop"}}]},
    {"q": "run terminal command ipconfig", "t": [{"tool": "run_terminal_command", "args": {"command": "ipconfig"}}]},
    {"q": "execute command git status in terminal", "t": [{"tool": "run_terminal_command", "args": {"command": "git status"}}]},

    # =========================================================================
    # 13. CONVERSATIONAL, IDENTITY & FRIDAY STANDBY
    # =========================================================================
    {"q": "hello friday", "t": [{"tool": "chat_response", "args": {"message": "Hello! Systems online and ready."}}]},
    {"q": "who are you?", "t": [{"tool": "chat_response", "args": {"message": "I am FRIDAY, your autonomous desktop assistant."}}]},
    {"q": "what can you do?", "t": [{"tool": "chat_response", "args": {"message": "I can control your desktop, automate tasks, manage files, calendars, and system hardware."}}]},
    {"q": "friday go to sleep", "t": [{"tool": "chat_response", "args": {"message": "Standing by. Call me when you need me."}}]},
    {"q": "stand by friday", "t": [{"tool": "chat_response", "args": {"message": "Standing by."}}]},
    {"q": "stop listening for now", "t": [{"tool": "chat_response", "args": {"message": "Pausing voice listening. Standing by."}}]},
    {"q": "rest now friday", "t": [{"tool": "chat_response", "args": {"message": "Resting now. Have a great day!"}}]},
    {"q": "take a break friday", "t": [{"tool": "chat_response", "args": {"message": "Taking a break. Systems on standby."}}]},
    {"q": "can you feel emotions?", "t": [{"tool": "chat_response", "args": {"message": "As an AI, I don't feel emotions, but I'm always ready to help!"}}]},

    # =========================================================================
    # 14. MULTI-INTENT COMPOUND CHAINS ACROSS DIVERSE TOOLS
    # =========================================================================
    {
        "q": "create folder invoices and open file explorer",
        "t": [{"tool": "create_directory", "args": {"path": "invoices"}}, {"tool": "launch_app", "args": {"name": "explorer"}}]
    },
    {
        "q": "schedule team meeting tomorrow at 10am and mute audio",
        "t": [{"tool": "create_calendar_event", "args": {"title": "team meeting", "start_time": "tomorrow at 10am"}}, {"tool": "mute_audio", "args": {"mute": True}}]
    },
    {
        "q": "take a screenshot, open notepad, and set volume to 80",
        "t": [{"tool": "take_screenshot", "args": {}}, {"tool": "launch_app", "args": {"name": "notepad"}}, {"tool": "set_volume", "args": {"level": 80}}]
    },
    {
        "q": "turn off bluetooth, enable dark mode, and switch to coding workspace",
        "t": [{"tool": "toggle_bluetooth", "args": {"enabled": False}}, {"tool": "toggle_dark_mode", "args": {"dark": True}}, {"tool": "set_workspace_preset", "args": {"preset": "coding"}}]
    },
    {
        "q": "reload tab and mute volume",
        "t": [{"tool": "browser_tab_action", "args": {"action": "reload"}}, {"tool": "mute_audio", "args": {"mute": True}}]
    },
    {
        "q": "mute volume, close chrome, minimize all windows, and lock pc",
        "t": [{"tool": "mute_audio", "args": {"mute": True}}, {"tool": "close_app", "args": {"name": "chrome"}}, {"tool": "minimize_all_windows", "args": {}}, {"tool": "lock_screen", "args": {}}]
    },
    {
        "q": "enable dark mode, turn on night light, set brightness to 40, and set volume to 20",
        "t": [{"tool": "toggle_dark_mode", "args": {"dark": True}}, {"tool": "toggle_night_light", "args": {"enabled": True}}, {"tool": "set_brightness", "args": {"level": 40}}, {"tool": "set_volume", "args": {"level": 20}}]
    },

    # =========================================================================
    # 15. CLOUD REASONING DELEGATION HARD BOUNDARY (NO FICTIONAL TOOLS!)
    # =========================================================================
    {"q": "write a complete react typescript dashboard with charts and authentication", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "write a complete react typescript dashboard with charts and authentication"}}]},
    {"q": "solve this calculus differential equation: dy/dx + 2y = e^(3x)", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "solve this calculus differential equation: dy/dx + 2y = e^(3x)"}}]},
    {"q": "draft a comprehensive business plan for a specialty coffee shop", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "draft a comprehensive business plan for a specialty coffee shop"}}]},
    {"q": "compare postgresql and mongodb for real-time analytics with trade-offs", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "compare postgresql and mongodb for real-time analytics with trade-offs"}}]},
    {"q": "generate a full pytest test suite with mocks for this fastapi user auth router", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "generate a full pytest test suite with mocks for this fastapi user auth router"}}]},
    {"q": "write a python script with asyncio to scrape news headlines concurrently", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "write a python script with asyncio to scrape news headlines concurrently"}}]},
    {"q": "summarize the 20-page financial quarterly earnings report pdf", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "summarize the 20-page financial quarterly earnings report pdf"}}]},
    {"q": "explain the difference between transformer self-attention and mamba state space models", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "explain the difference between transformer self-attention and mamba state space models"}}]},
]

SYSTEM_PROMPT = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
    "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)


def main():
    print("[*] Generating Comprehensive Exhaustive Dataset...")

    records = []
    # Replicate core tool samples 3x
    for item in EXHAUSTIVE_TOOL_SAMPLES:
        for _ in range(3):
            records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["q"]},
                    {"role": "assistant", "content": json.dumps(item["t"])},
                ]
            })

    # Sample anchor baseline samples from friday_train.jsonl
    train_path = Path("dataset/friday_train.jsonl")
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            sampled_lines = random.sample(lines, min(len(lines), 200))
            for sl in sampled_lines:
                records.append(json.loads(sl))

    random.shuffle(records)

    split_idx = int(len(records) * 0.85)
    train_split = records[:split_idx]
    val_split = records[split_idx:]

    print(f"[*] Total Exhaustive Records: {len(records)}")
    print(f"[*] Train Split: {len(train_split)} records")
    print(f"[*] Val Split: {len(val_split)} records")

    out_train = Path("dataset/exhaustive_train.jsonl")
    out_val = Path("dataset/exhaustive_val.jsonl")

    with open(out_train, "w", encoding="utf-8") as f:
        for r in train_split:
            f.write(json.dumps(r) + "\n")

    with open(out_val, "w", encoding="utf-8") as f:
        for r in val_split:
            f.write(json.dumps(r) + "\n")

    print(f"[+] Saved {out_train} and {out_val} successfully!")


if __name__ == "__main__":
    main()
