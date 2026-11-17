"""Generates the DesktopIntent-v6-DailyEssentials Dataset for FRIDAY AI Assistant.

High-density combinatorial expansion covering all 8 remaining capability gaps:
1. Email Reading & Calendar Management (read_emails, list_calendar_events, update/delete/create event)
2. Screen Recording & Video Media (start_screen_recording, stop_screen_recording, download_media_video)
3. System Power & Hardware Controls (shutdown_system, restart_system, cancel_shutdown, turn_off_monitors)
4. Documents, PDF, Archive & Filesystem (create_archive, extract_archive, read_pdf_document, create_pdf, analyze_spreadsheet, read_file, write_file, copy_file, move_file)
5. Clipboard Utilities (get_clipboard, set_clipboard, clear_clipboard_history, get_clipboard_history)
6. Window Focus, Switching & Detailed Windowing (switch_app, focus_window, minimize_window, maximize_window, close_window)
7. Browser Tab Control & In-Page Actions (browser_new_tab, browser_close_tab, browser_switch_tab, browser_scroll)
8. Contact Creation & Address Book (add_contact, list_contacts, delete_contact)
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

random.seed(42)

BASE_SYSTEM_PROMPT = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
    "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

DATASET_DIR = Path("D:/Projects/desktop-llm/dataset")
HF_SUBSET_DIR = Path("D:/Projects/desktop-llm/hf_export/friday-desktop-intent/subsets/v6_daily_essentials")


# =============================================================================
# DOMAIN 1: EMAIL READING & CALENDAR MANAGEMENT (~700 samples)
# =============================================================================
def generate_domain_email_calendar() -> List[Dict[str, Any]]:
    samples = []

    # 1.1 Read Emails (General & Unread)
    counts = [3, 5, 10, 15, 20]
    for c in counts:
        phrases = [
            (f"read my {c} unread emails", {"unread_only": True, "limit": c}),
            (f"fetch latest {c} unread emails", {"unread_only": True, "limit": c}),
            (f"check my last {c} emails", {"unread_only": False, "limit": c}),
            (f"show me {c} recent emails", {"unread_only": False, "limit": c}),
            (f"read the top {c} emails in my inbox", {"unread_only": False, "limit": c}),
            (f"do I have {c} new emails", {"unread_only": True, "limit": c}),
        ]
        for q, args in phrases:
            samples.append({"q": q, "t": [{"tool": "read_emails", "args": args}]})
            samples.append({"q": f"friday {q}", "t": [{"tool": "read_emails", "args": args}]})

    # Read Emails by Query / Sender
    senders = ["boss", "manager", "HR", "Google", "Amazon", "GitHub", "Microsoft", "Alice", "Bob", "John", "David", "Stripe", "PayPal"]
    for s in senders:
        phrases = [
            f"check emails from {s}", f"read emails from {s}",
            f"do I have any emails from {s}", f"find emails from {s} in inbox",
            f"search emails sent by {s}", f"show messages from {s}"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "read_emails", "args": {"unread_only": False, "query": s}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "read_emails", "args": {"unread_only": False, "query": s}}]})

    topics = ["invoice", "receipt", "salary", "interview", "flight", "hotel booking", "project update", "verification code", "security alert"]
    for top in topics:
        samples.append({"q": f"check emails about {top}", "t": [{"tool": "read_emails", "args": {"unread_only": False, "query": top}}]})
        samples.append({"q": f"search inbox for {top}", "t": [{"tool": "read_emails", "args": {"unread_only": False, "query": top}}]})

    # 1.2 List Calendar Events
    time_scopes = [
        ("today", "today"), ("tomorrow", "tomorrow"), ("this week", "this week"),
        ("this weekend", "weekend"), ("next week", "next week"), ("next Monday", "Monday"),
        ("Friday", "Friday"), ("this afternoon", "afternoon"), ("this evening", "evening")
    ]
    for scope_str, scope_key in time_scopes:
        cal_phrases = [
            f"what's on my calendar {scope_str}", f"show my schedule for {scope_str}",
            f"list calendar events for {scope_str}", f"do I have any meetings {scope_str}",
            f"what meetings do I have {scope_str}", f"check my calendar {scope_str}",
            f"view schedule for {scope_str}", f"what appointments do I have {scope_str}"
        ]
        for p in cal_phrases:
            samples.append({"q": p, "t": [{"tool": "list_calendar_events", "args": {"start_date": scope_key, "limit": 10}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "list_calendar_events", "args": {"start_date": scope_key, "limit": 10}}]})

    # 1.3 Create Calendar Event
    meeting_titles = [
        "Team Sync", "Dentist Appointment", "Lunch with Sarah", "Project Demo",
        "Q3 Roadmap Review", "Doctor Checkup", "Client Call with Acme Corp", "Design Review",
        "1-on-1 with Alex", "Weekly Standup", "Budget Planning", "Product Launch Sync",
        "Code Architecture Meeting", "Investor Pitch", "Gym Session", "Coffee with Mike"
    ]
    meeting_times = [
        "tomorrow at 10 AM", "today at 3 PM", "Friday at 2 PM", "Monday at 11 AM",
        "tomorrow at 4:30 PM", "Wednesday at 9 AM", "tomorrow at noon", "Thursday at 5 PM"
    ]
    for title in meeting_titles:
        for t_val in meeting_times[:3]:
            dur = random.choice([30, 45, 60])
            samples.append({"q": f"schedule a meeting '{title}' for {t_val}", "t": [{"tool": "create_calendar_event", "args": {"title": title, "start_time": t_val}}]})
            samples.append({"q": f"add '{title}' to my calendar at {t_val} for {dur} minutes", "t": [{"tool": "create_calendar_event", "args": {"title": title, "start_time": t_val, "duration_minutes": dur}}]})
            samples.append({"q": f"create calendar appointment '{title}' {t_val}", "t": [{"tool": "create_calendar_event", "args": {"title": title, "start_time": t_val}}]})

    # 1.4 Update & Delete Calendar Events
    for title in meeting_titles[:8]:
        new_t = "tomorrow at 3 PM"
        samples.append({"q": f"reschedule '{title}' to {new_t}", "t": [{"tool": "update_calendar_event", "args": {"title": title, "start_time": new_t}}]})
        samples.append({"q": f"move '{title}' meeting to {new_t}", "t": [{"tool": "update_calendar_event", "args": {"title": title, "start_time": new_t}}]})
        samples.append({"q": f"cancel my '{title}' meeting", "t": [{"tool": "delete_calendar_event", "args": {"title": title}}]})
        samples.append({"q": f"delete '{title}' from calendar", "t": [{"tool": "delete_calendar_event", "args": {"title": title}}]})

    return samples


# =============================================================================
# DOMAIN 2: SCREEN RECORDING & VIDEO MEDIA (~450 samples)
# =============================================================================
def generate_domain_screen_record_media() -> List[Dict[str, Any]]:
    samples = []

    # 2.1 Start Screen Recording
    fps_options = [15, 30, 60]
    for fps in fps_options:
        phrases = [
            f"start screen recording at {fps} fps", f"record screen at {fps} fps",
            f"capture screen video at {fps} fps", f"begin {fps} fps desktop recording"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "start_screen_recording", "args": {"fps": fps}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "start_screen_recording", "args": {"fps": fps}}]})

    generic_record = [
        "start screen recording", "start recording my screen", "record screen",
        "begin screen recording", "start recording video of my desktop",
        "start capturing screen video", "start desktop recording",
        "record my display", "start screen record now", "record current screen"
    ]
    for p in generic_record:
        samples.append({"q": p, "t": [{"tool": "start_screen_recording", "args": {"fps": 15}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "start_screen_recording", "args": {"fps": 15}}]})
        samples.append({"q": f"can you please {p}", "t": [{"tool": "start_screen_recording", "args": {"fps": 15}}]})

    # 2.2 Stop Screen Recording
    stop_phrases = [
        "stop screen recording", "stop recording", "finish screen recording",
        "end screen record", "stop recording my screen", "halt screen recording",
        "save and stop screen recording", "done recording screen", "terminate screen record",
        "finish recording desktop", "stop video capture"
    ]
    for p in stop_phrases:
        samples.append({"q": p, "t": [{"tool": "stop_screen_recording", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "stop_screen_recording", "args": {}}]})
        samples.append({"q": f"please {p}", "t": [{"tool": "stop_screen_recording", "args": {}}]})

    # 2.3 Download Media Video
    sample_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://youtu.be/kJQP7kiw5Fk",
        "https://vimeo.com/12345678", "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
        "https://youtube.com/watch?v=9bZkp7q19f0", "https://youtu.be/fJ9rUzIMcZQ",
        "https://www.dailymotion.com/video/x7tgad0", "https://vimeo.com/98765432"
    ]
    for u in sample_urls:
        q_video = [
            f"download video from {u}", f"download this youtube video {u}",
            f"save video from {u}", f"download media at {u}", f"grab video {u}"
        ]
        for q in q_video:
            samples.append({"q": q, "t": [{"tool": "download_media_video", "args": {"url": u, "extract_audio_only": False}}]})
            samples.append({"q": f"friday {q}", "t": [{"tool": "download_media_video", "args": {"url": u, "extract_audio_only": False}}]})

        q_audio = [
            f"extract audio from {u}", f"download mp3 from {u}",
            f"download audio only from {u}", f"save audio from {u} as mp3"
        ]
        for q in q_audio:
            samples.append({"q": q, "t": [{"tool": "download_media_video", "args": {"url": u, "extract_audio_only": True}}]})
            samples.append({"q": f"friday {q}", "t": [{"tool": "download_media_video", "args": {"url": u, "extract_audio_only": True}}]})

    return samples


# =============================================================================
# DOMAIN 3: SYSTEM POWER & HARDWARE CONTROLS (~450 samples)
# =============================================================================
def generate_domain_system_power() -> List[Dict[str, Any]]:
    samples = []

    # 3.1 Shutdown System with various delays
    delays = [
        ("0", 0, ["now", "immediately", "right now"]),
        ("5", 300, ["in 5 minutes", "after 5 mins"]),
        ("10", 600, ["in 10 minutes", "in ten minutes"]),
        ("15", 900, ["in 15 minutes"]),
        ("20", 1200, ["in 20 minutes"]),
        ("30", 1800, ["in 30 minutes", "in half an hour"]),
        ("45", 2700, ["in 45 minutes"]),
        ("60", 3600, ["in 1 hour", "in 60 minutes", "in an hour"]),
        ("120", 7200, ["in 2 hours"])
    ]
    for _, sec, phrases in delays:
        for ph in phrases:
            q_list = [
                f"shut down the computer {ph}", f"shut down pc {ph}",
                f"turn off the computer {ph}", f"power off pc {ph}",
                f"schedule shutdown {ph}"
            ]
            for q in q_list:
                samples.append({"q": q, "t": [{"tool": "shutdown_system", "args": {"delay_seconds": sec, "force": False}}]})
                samples.append({"q": f"friday {q}", "t": [{"tool": "shutdown_system", "args": {"delay_seconds": sec, "force": False}}]})

    # Immediate shutdown
    samples.append({"q": "shut down my computer", "t": [{"tool": "shutdown_system", "args": {"delay_seconds": 0, "force": False}}]})
    samples.append({"q": "turn off the pc", "t": [{"tool": "shutdown_system", "args": {"delay_seconds": 0, "force": False}}]})
    samples.append({"q": "power off the computer", "t": [{"tool": "shutdown_system", "args": {"delay_seconds": 0, "force": False}}]})

    # 3.2 Cancel Shutdown
    cancel_phrases = [
        "cancel the shutdown", "cancel shutdown", "abort shutdown",
        "stop scheduled shutdown", "don't shut down the pc", "abort computer shutdown",
        "cancel system shutdown", "halt scheduled shutdown", "dismiss shutdown"
    ]
    for p in cancel_phrases:
        samples.append({"q": p, "t": [{"tool": "cancel_shutdown", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "cancel_shutdown", "args": {}}]})
        samples.append({"q": f"please {p}", "t": [{"tool": "cancel_shutdown", "args": {}}]})

    # 3.3 Restart System
    restart_delays = [
        (0, ["restart my computer", "reboot the pc", "restart system", "reboot now", "restart pc immediately"]),
        (300, ["restart computer in 5 minutes", "reboot pc in 5 mins"]),
        (600, ["restart pc in 10 minutes", "reboot system in 10 minutes"]),
        (900, ["restart computer in 15 minutes"])
    ]
    for sec, ph_list in restart_delays:
        for p in ph_list:
            samples.append({"q": p, "t": [{"tool": "restart_system", "args": {"delay_seconds": sec, "force": False}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "restart_system", "args": {"delay_seconds": sec, "force": False}}]})

    # 3.4 Turn Off Monitors (Display Sleep)
    mon_sleep = [
        "turn off the monitors", "turn off my screens", "turn off display",
        "sleep monitors", "put screens to sleep", "turn off monitors to save battery",
        "blank the screen", "turn off the display now", "sleep the screens",
        "power down monitors", "turn off desktop displays"
    ]
    for p in mon_sleep:
        samples.append({"q": p, "t": [{"tool": "turn_off_monitors", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "turn_off_monitors", "args": {}}]})

    # 3.5 Hibernate System
    hib_phrases = [
        "hibernate the computer", "hibernate pc", "put computer into hibernation",
        "hibernate system now", "hibernate desktop", "enter hibernation mode"
    ]
    for p in hib_phrases:
        samples.append({"q": p, "t": [{"tool": "hibernate_system", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "hibernate_system", "args": {}}]})

    return samples


# =============================================================================
# DOMAIN 4: DOCUMENTS, PDF, ARCHIVE & FILESYSTEM (~750 samples)
# =============================================================================
def generate_domain_documents_files() -> List[Dict[str, Any]]:
    samples = []

    # 4.1 Archives (Zip / Unzip)
    folders = [
        ("downloads/project", "downloads/project.zip"),
        ("documents/reports", "documents/reports.zip"),
        ("desktop/photos", "desktop/photos.zip"),
        ("code/backend", "code/backend.zip"),
        ("documents/invoices", "documents/invoices.zip"),
        ("downloads/client_files", "downloads/client_files.zip"),
        ("desktop/backup", "desktop/backup.zip"),
        ("projects/model", "projects/model.zip")
    ]
    for src, out_zip in folders:
        # Zip
        samples.append({"q": f"create a zip archive of {src}", "t": [{"tool": "create_archive", "args": {"source_path": src, "output_zip_path": out_zip}}]})
        samples.append({"q": f"zip the folder {src}", "t": [{"tool": "create_archive", "args": {"source_path": src, "output_zip_path": out_zip}}]})
        samples.append({"q": f"compress {src} into a zip", "t": [{"tool": "create_archive", "args": {"source_path": src, "output_zip_path": out_zip}}]})
        samples.append({"q": f"archive {src} to {out_zip}", "t": [{"tool": "create_archive", "args": {"source_path": src, "output_zip_path": out_zip}}]})
        samples.append({"q": f"friday zip {src}", "t": [{"tool": "create_archive", "args": {"source_path": src, "output_zip_path": out_zip}}]})

        # Unzip
        samples.append({"q": f"extract archive {out_zip}", "t": [{"tool": "extract_archive", "args": {"zip_path": out_zip, "extract_to_dir": src}}]})
        samples.append({"q": f"unzip {out_zip}", "t": [{"tool": "extract_archive", "args": {"zip_path": out_zip, "extract_to_dir": src}}]})
        samples.append({"q": f"extract files from {out_zip}", "t": [{"tool": "extract_archive", "args": {"zip_path": out_zip, "extract_to_dir": src}}]})
        samples.append({"q": f"uncompress {out_zip} to {src}", "t": [{"tool": "extract_archive", "args": {"zip_path": out_zip, "extract_to_dir": src}}]})
        samples.append({"q": f"friday unzip {out_zip} here", "t": [{"tool": "extract_archive", "args": {"zip_path": out_zip, "extract_to_dir": src}}]})

    # 4.2 PDF Documents (Read / Create)
    pdf_docs = [
        "report.pdf", "invoice_october.pdf", "contract.pdf", "research_paper.pdf",
        "user_guide.pdf", "financial_statement.pdf", "meeting_notes.pdf", "resume.pdf",
        "agreement.pdf", "whitepaper.pdf"
    ]
    for pdf in pdf_docs:
        pages = random.choice([5, 10, 20])
        samples.append({"q": f"read {pdf}", "t": [{"tool": "read_pdf_document", "args": {"file_path": pdf, "max_pages": 10}}]})
        samples.append({"q": f"read pdf document {pdf}", "t": [{"tool": "read_pdf_document", "args": {"file_path": pdf, "max_pages": 10}}]})
        samples.append({"q": f"extract text from {pdf}", "t": [{"tool": "read_pdf_document", "args": {"file_path": pdf, "max_pages": 10}}]})
        samples.append({"q": f"view first {pages} pages of {pdf}", "t": [{"tool": "read_pdf_document", "args": {"file_path": pdf, "max_pages": pages}}]})
        samples.append({"q": f"friday read {pdf}", "t": [{"tool": "read_pdf_document", "args": {"file_path": pdf, "max_pages": 10}}]})

    # 4.3 Spreadsheet Analysis (Excel / CSV)
    spreadsheets = [
        ("sales_q3.csv", "summary"), ("financials.xlsx", "summary"),
        ("users.csv", "summary"), ("inventory.xlsx", "summary"),
        ("analytics_data.csv", "summary"), ("budget_2026.xlsx", "summary")
    ]
    for sheet, q_type in spreadsheets:
        samples.append({"q": f"analyze spreadsheet {sheet}", "t": [{"tool": "analyze_spreadsheet", "args": {"file_path": sheet, "query_type": q_type}}]})
        samples.append({"q": f"summarize data in {sheet}", "t": [{"tool": "analyze_spreadsheet", "args": {"file_path": sheet, "query_type": q_type}}]})
        samples.append({"q": f"inspect data in {sheet}", "t": [{"tool": "analyze_spreadsheet", "args": {"file_path": sheet, "query_type": q_type}}]})
        samples.append({"q": f"analyze table in {sheet}", "t": [{"tool": "analyze_spreadsheet", "args": {"file_path": sheet, "query_type": q_type}}]})
        samples.append({"q": f"friday examine {sheet}", "t": [{"tool": "analyze_spreadsheet", "args": {"file_path": sheet, "query_type": q_type}}]})

    # 4.4 Read & Write Files
    plain_files = [
        ("notes.txt", "notes.txt"), ("todo.md", "todo.md"), ("config.json", "config.json"),
        ("script.py", "script.py"), ("log.txt", "log.txt"), ("readme.txt", "readme.txt"),
        ("environment.env", "environment.env"), ("server.js", "server.js")
    ]
    for _, fname in plain_files:
        samples.append({"q": f"read {fname}", "t": [{"tool": "read_file", "args": {"path": fname, "start_line": 1}}]})
        samples.append({"q": f"show contents of {fname}", "t": [{"tool": "read_file", "args": {"path": fname, "start_line": 1}}]})
        samples.append({"q": f"read the first 30 lines of {fname}", "t": [{"tool": "read_file", "args": {"path": fname, "start_line": 1, "end_line": 30}}]})
        samples.append({"q": f"display file {fname}", "t": [{"tool": "read_file", "args": {"path": fname, "start_line": 1}}]})
        samples.append({"q": f"friday read {fname}", "t": [{"tool": "read_file", "args": {"path": fname, "start_line": 1}}]})

    # 4.5 Copy, Move & Delete Files
    file_ops = [
        ("downloads/report.pdf", "documents/report.pdf"),
        ("desktop/data.csv", "documents/data.csv"),
        ("downloads/setup.exe", "desktop/setup.exe"),
        ("downloads/archive.zip", "documents/archive.zip"),
        ("documents/contract_v1.docx", "documents/archive/contract_v1.docx"),
        ("desktop/temp.png", "pictures/temp.png")
    ]
    for src, dst in file_ops:
        samples.append({"q": f"copy {src} to {dst}", "t": [{"tool": "copy_file", "args": {"source_path": src, "destination_path": dst}}]})
        samples.append({"q": f"move {src} to {dst}", "t": [{"tool": "move_file", "args": {"source_path": src, "destination_path": dst}}]})
        samples.append({"q": f"friday move {src} into {dst}", "t": [{"tool": "move_file", "args": {"source_path": src, "destination_path": dst}}]})
        samples.append({"q": f"delete file {src}", "t": [{"tool": "delete_file", "args": {"path": src, "confirmed": True}}]})
        samples.append({"q": f"remove {src}", "t": [{"tool": "delete_file", "args": {"path": src, "confirmed": True}}]})

    return samples


# =============================================================================
# DOMAIN 5: CLIPBOARD UTILITIES (~350 samples)
# =============================================================================
def generate_domain_clipboard() -> List[Dict[str, Any]]:
    samples = []

    # 5.1 Get Clipboard
    get_clip = [
        "what is on my clipboard", "read clipboard", "get clipboard content",
        "show clipboard", "what did I just copy", "paste clipboard text",
        "check clipboard", "read my current clipboard", "inspect clipboard content",
        "what text is currently copied", "view clipboard contents"
    ]
    for p in get_clip:
        samples.append({"q": p, "t": [{"tool": "get_clipboard", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "get_clipboard", "args": {}}]})
        samples.append({"q": f"can you {p}", "t": [{"tool": "get_clipboard", "args": {}}]})

    # 5.2 Set Clipboard
    snippets = [
        "https://github.com/ggerganov/llama.cpp", "192.168.1.1", "john.doe@example.com",
        "sudo apt update && sudo apt upgrade", "AKIAIOSFODNN7EXAMPLE", "Meeting ID: 849 203 192",
        "python main.py --port 8000", "docker-compose up -d", "sk-proj-1234567890",
        "Password123!", "git commit -m 'Initial commit'", "Hello world"
    ]
    for s in snippets:
        samples.append({"q": f"copy '{s}' to clipboard", "t": [{"tool": "set_clipboard", "args": {"content": s}}]})
        samples.append({"q": f"set clipboard to '{s}'", "t": [{"tool": "set_clipboard", "args": {"content": s}}]})
        samples.append({"q": f"put '{s}' on my clipboard", "t": [{"tool": "set_clipboard", "args": {"content": s}}]})
        samples.append({"q": f"copy text '{s}'", "t": [{"tool": "set_clipboard", "args": {"content": s}}]})
        samples.append({"q": f"friday copy '{s}' to clipboard", "t": [{"tool": "set_clipboard", "args": {"content": s}}]})

    # 5.3 Clear Clipboard History
    clear_phrases = [
        "clear my clipboard", "clear clipboard history", "wipe clipboard",
        "empty clipboard", "clear copied text history", "wipe clipboard data",
        "reset clipboard", "erase clipboard history", "purge clipboard"
    ]
    for p in clear_phrases:
        samples.append({"q": p, "t": [{"tool": "clear_clipboard_history", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "clear_clipboard_history", "args": {}}]})

    return samples


# =============================================================================
# DOMAIN 6: WINDOW FOCUS, SWITCHING & DETAILED WINDOWING (~500 samples)
# =============================================================================
def generate_domain_windowing() -> List[Dict[str, Any]]:
    samples = []

    apps = [
        "chrome", "spotify", "discord", "vscode", "notepad", "terminal", "slack",
        "calculator", "excel", "word", "powerpoint", "brave", "edge", "vlc",
        "telegram", "obsidian", "steam", "postman"
    ]
    for app in apps:
        # Switch / Focus
        samples.append({"q": f"switch to {app}", "t": [{"tool": "switch_app", "args": {"app_name": app}}]})
        samples.append({"q": f"focus {app}", "t": [{"tool": "focus_window", "args": {"window_title": app}}]})
        samples.append({"q": f"bring {app} to front", "t": [{"tool": "focus_window", "args": {"window_title": app}}]})
        samples.append({"q": f"alt tab to {app}", "t": [{"tool": "switch_app", "args": {"app_name": app}}]})
        samples.append({"q": f"switch window to {app}", "t": [{"tool": "switch_app", "args": {"app_name": app}}]})
        samples.append({"q": f"friday switch to {app}", "t": [{"tool": "switch_app", "args": {"app_name": app}}]})

        # Explicit Window Minimization / Maximization
        samples.append({"q": f"minimize {app}", "t": [{"tool": "minimize_window", "args": {"window_title": app}}]})
        samples.append({"q": f"maximize {app}", "t": [{"tool": "maximize_window", "args": {"window_title": app}}]})
        samples.append({"q": f"restore {app} window", "t": [{"tool": "restore_window", "args": {"window_title": app}}]})
        samples.append({"q": f"close {app} window", "t": [{"tool": "close_window", "args": {"window_title": app}}]})
        samples.append({"q": f"friday minimize {app}", "t": [{"tool": "minimize_window", "args": {"window_title": app}}]})

    # Show Desktop
    desktop_phrases = [
        "show desktop", "minimize everything and show desktop", "peek desktop",
        "hide all windows and show desktop", "show my desktop", "view desktop",
        "minimize all windows to desktop"
    ]
    for p in desktop_phrases:
        samples.append({"q": p, "t": [{"tool": "minimize_window", "args": {"window_title": "all"}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "minimize_window", "args": {"window_title": "all"}}]})

    return samples


# =============================================================================
# DOMAIN 7: BROWSER TAB CONTROL & IN-PAGE ACTIONS (~450 samples)
# =============================================================================
def generate_domain_browser_tabs() -> List[Dict[str, Any]]:
    samples = []

    # 7.1 New Tab
    sites = [
        ("google.com", "https://google.com"), ("youtube.com", "https://youtube.com"),
        ("github.com", "https://github.com"), ("reddit.com", "https://reddit.com"),
        ("twitter.com", "https://twitter.com"), ("wikipedia.org", "https://wikipedia.org"),
        ("stackoverflow.com", "https://stackoverflow.com"), ("chatgpt.com", "https://chatgpt.com")
    ]
    for s_name, s_url in sites:
        samples.append({"q": f"open a new tab with {s_name}", "t": [{"tool": "browser_new_tab", "args": {"url": s_url}}]})
        samples.append({"q": f"new tab with {s_name}", "t": [{"tool": "browser_new_tab", "args": {"url": s_url}}]})
        samples.append({"q": f"open {s_name} in a new tab", "t": [{"tool": "browser_new_tab", "args": {"url": s_url}}]})
        samples.append({"q": f"friday open new tab {s_name}", "t": [{"tool": "browser_new_tab", "args": {"url": s_url}}]})

    blank_tabs = ["open a new tab", "new tab", "create a new browser tab", "new tab in chrome", "open blank tab"]
    for b in blank_tabs:
        samples.append({"q": b, "t": [{"tool": "browser_new_tab", "args": {"url": "about:blank"}}]})
        samples.append({"q": f"friday {b}", "t": [{"tool": "browser_new_tab", "args": {"url": "about:blank"}}]})

    # 7.2 Close Tab
    close_tabs = [
        "close current tab", "close this tab", "close active browser tab",
        "close the current browser tab", "close tab in chrome", "close web tab",
        "shut this tab", "close active tab now"
    ]
    for q in close_tabs:
        samples.append({"q": q, "t": [{"tool": "browser_close_tab", "args": {}}]})
        samples.append({"q": f"friday {q}", "t": [{"tool": "browser_close_tab", "args": {}}]})

    # 7.3 Switch Tab
    targets = ["youtube", "github", "gmail", "reddit", "twitter", "calendar", "docs", "amazon", "next", "prev"]
    for t in targets:
        samples.append({"q": f"switch to {t} tab", "t": [{"tool": "browser_switch_tab", "args": {"tab_id": t}}]})
        samples.append({"q": f"go to the {t} tab", "t": [{"tool": "browser_switch_tab", "args": {"tab_id": t}}]})
        samples.append({"q": f"friday switch to {t} tab", "t": [{"tool": "browser_switch_tab", "args": {"tab_id": t}}]})

    # 7.4 Browser Scroll
    scroll_configs = [
        ("scroll down", "down", 500),
        ("scroll down the page", "down", 500),
        ("scroll down a bit", "down", 300),
        ("scroll down a lot", "down", 1000),
        ("scroll up", "up", 500),
        ("scroll up the page", "up", 500),
        ("scroll up a bit", "up", 300),
        ("scroll to the top of page", "up", 1500),
        ("scroll to the top", "up", 1500),
        ("scroll to the bottom", "down", 1500),
        ("scroll to bottom of page", "down", 1500),
    ]
    for q, direction, amount in scroll_configs:
        samples.append({"q": q, "t": [{"tool": "browser_scroll", "args": {"direction": direction, "amount": amount}}]})
        samples.append({"q": f"friday {q}", "t": [{"tool": "browser_scroll", "args": {"direction": direction, "amount": amount}}]})

    return samples


# =============================================================================
# DOMAIN 8: CONTACT CREATION & ADDRESS BOOK (~350 samples)
# =============================================================================
def generate_domain_contacts() -> List[Dict[str, Any]]:
    samples = []

    names_and_contacts = [
        ("John Smith", "555-0192", "john@example.com"),
        ("Alice Johnson", "555-0144", "alice@example.com"),
        ("David Miller", "555-0188", "david@example.com"),
        ("Sarah Williams", "555-0123", "sarah@example.com"),
        ("Michael Brown", "555-0177", "michael@example.com"),
        ("Emily Davis", "555-0155", "emily@example.com"),
        ("Robert Wilson", "555-0166", "robert@example.com"),
        ("Jessica Taylor", "555-0111", "jessica@example.com"),
        ("Alex Martinez", "555-0133", "alex@example.com"),
        ("Dr. Peterson", "555-0199", "peterson@clinic.com")
    ]
    for name, phone, email in names_and_contacts:
        samples.append({"q": f"add contact {name} with phone {phone}", "t": [{"tool": "add_contact", "args": {"name": name, "phone": phone, "email": email}}]})
        samples.append({"q": f"save {name}'s number as {phone}", "t": [{"tool": "add_contact", "args": {"name": name, "phone": phone, "email": email}}]})
        samples.append({"q": f"add {name} to my contacts with email {email}", "t": [{"tool": "add_contact", "args": {"name": name, "phone": phone, "email": email}}]})
        samples.append({"q": f"create contact for {name} phone {phone} email {email}", "t": [{"tool": "add_contact", "args": {"name": name, "phone": phone, "email": email}}]})
        samples.append({"q": f"save contact {name} {phone}", "t": [{"tool": "add_contact", "args": {"name": name, "phone": phone, "email": email}}]})
        samples.append({"q": f"friday add contact {name}", "t": [{"tool": "add_contact", "args": {"name": name, "phone": phone, "email": email}}]})

    # List Contacts
    list_phrases = [
        "list all my contacts", "show my address book", "list contacts",
        "view all saved contacts", "show contact list", "get my contacts",
        "who is in my address book", "show contacts"
    ]
    for p in list_phrases:
        samples.append({"q": p, "t": [{"tool": "list_contacts", "args": {"limit": 50}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "list_contacts", "args": {"limit": 50}}]})

    # Delete Contacts
    for name, _, _ in names_and_contacts:
        samples.append({"q": f"delete contact {name}", "t": [{"tool": "delete_contact", "args": {"name_or_id": name}}]})
        samples.append({"q": f"remove {name} from contacts", "t": [{"tool": "delete_contact", "args": {"name_or_id": name}}]})
        samples.append({"q": f"delete {name} from address book", "t": [{"tool": "delete_contact", "args": {"name_or_id": name}}]})

    return samples


def augment_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Augment with voice fillers and acoustic disfluencies."""
    augmented = []
    fillers = [
        ("can you please ", ""), ("hey friday ", ""),
        ("um ", ""), ("uh ", ""), ("could you ", ""),
        ("please ", ""), ("yo friday ", "")
    ]
    for s in samples:
        augmented.append(s)
        if random.random() < 0.35:
            prefix, _ = random.choice(fillers)
            if not s["q"].lower().startswith(("friday", "hey friday", "could", "can", "please")):
                augmented.append({"q": f"{prefix}{s['q']}".strip(), "t": s["t"]})
    return augmented


def format_chatml(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Format single sample into Qwen2.5 ChatML format."""
    return {
        "messages": [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": sample["q"]},
            {"role": "assistant", "content": json.dumps(sample["t"], separators=(",", ":"))}
        ]
    }


def main():
    print("=" * 75)
    print("🚀 GENERATING DesktopIntent-v6-DailyEssentials DATASET")
    print("=" * 75)

    d1 = generate_domain_email_calendar()
    print(f"  [+] Domain 1 (Email Reading & Calendar Ops):   {len(d1):,} samples")

    d2 = generate_domain_screen_record_media()
    print(f"  [+] Domain 2 (Screen Recording & Video Media): {len(d2):,} samples")

    d3 = generate_domain_system_power()
    print(f"  [+] Domain 3 (System Power, Shutdown, Display):{len(d3):,} samples")

    d4 = generate_domain_documents_files()
    print(f"  [+] Domain 4 (Documents, PDF, Zip & Files):    {len(d4):,} samples")

    d5 = generate_domain_clipboard()
    print(f"  [+] Domain 5 (Clipboard Get/Set/History):      {len(d5):,} samples")

    d6 = generate_domain_windowing()
    print(f"  [+] Domain 6 (Window Focus, Switch, Min/Max):  {len(d6):,} samples")

    d7 = generate_domain_browser_tabs()
    print(f"  [+] Domain 7 (Browser Tab & Scroll Actions):   {len(d7):,} samples")

    d8 = generate_domain_contacts()
    print(f"  [+] Domain 8 (Contact Add/List/Delete):        {len(d8):,} samples")

    raw_samples = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8
    augmented_samples = augment_samples(raw_samples)
    print(f"  [+] Speech & Disfluency Augmentation:          {len(augmented_samples) - len(raw_samples):,} samples")

    random.shuffle(augmented_samples)

    split_idx = int(len(augmented_samples) * 0.9)
    train_raw = augmented_samples[:split_idx]
    val_raw = augmented_samples[split_idx:]

    print("-" * 75)
    print(f"  Total Master v6-DailyEssentials Samples:       {len(augmented_samples):,}")
    print(f"    Train Split:                                 {len(train_raw):,}")
    print(f"    Validation Split:                            {len(val_raw):,}")
    print("-" * 75)

    # Convert to ChatML
    train_chatml = [format_chatml(s) for s in train_raw]
    val_chatml = [format_chatml(s) for s in val_raw]

    # Save to local dataset/
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATASET_DIR / "v6_daily_essentials_train.jsonl"
    val_path = DATASET_DIR / "v6_daily_essentials_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_chatml:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_chatml:
            f.write(json.dumps(item) + "\n")

    print(f"[✓] Saved local dataset to: {train_path} & {val_path}")

    # Also save to HuggingFace export subset
    HF_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    hf_train_path = HF_SUBSET_DIR / "train.jsonl"
    hf_val_path = HF_SUBSET_DIR / "validation.jsonl"

    with open(hf_train_path, "w", encoding="utf-8") as f:
        for item in train_chatml:
            f.write(json.dumps(item) + "\n")

    with open(hf_val_path, "w", encoding="utf-8") as f:
        for item in val_chatml:
            f.write(json.dumps(item) + "\n")

    print(f"[✓] Saved HuggingFace subset to: {hf_train_path} & {hf_val_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
