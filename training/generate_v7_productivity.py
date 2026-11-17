"""Generates the DesktopIntent-v7-Productivity Dataset for FRIDAY AI Assistant.

Comprehensive high-density combinatorial expansion covering the 7 core missing everyday productivity domains:
1. System Health, Telemetry & Process Management (get_system_health, get_detailed_hardware_stats, kill_process, list_processes)
2. Window Snapping, Workspace Presets & Layouts (snap_window, arrange_workspace_preset, save_workspace_profile, load_workspace_profile, list_windows)
3. File Organization, Explorer Navigation & Grep (organize_directory_by_type, open_in_explorer, rename_file, search_files, grep_files)
4. Ambient Information, Weather, Wikipedia & News (get_weather, search_wikipedia, news_search)
5. Direct Communication & Messaging (send_whatsapp_message, send_telegram_message, compose_email, search_contacts, get_contact)
6. Multimedia Chores & Media Conversions (download_youtube, convert_media_ffmpeg, capture_monitor_screenshot)
7. Voice Customization & Desktop Notifications (switch_tts_voice, list_tts_voices, send_desktop_toast)
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
HF_SUBSET_DIR = Path("D:/Projects/desktop-llm/hf_export/friday-desktop-intent/subsets/v7_productivity")


# =============================================================================
# DOMAIN 1: SYSTEM HEALTH, TELEMETRY & PROCESS MANAGEMENT (~400 samples)
# =============================================================================
def generate_domain_health_processes() -> List[Dict[str, Any]]:
    samples = []

    # 1.1 get_system_health & health check
    health_phrases = [
        "run a system health check", "check system health", "run diagnostics",
        "perform pre-flight diagnostics", "is my system healthy", "check hardware and cuda status",
        "run health check on all subsystems", "audit system performance", "diagnose system status",
        "run assistant diagnostic check", "check system integrity", "do a full system diagnostics",
        "why is my computer lagging", "why is my pc so slow", "check if all subsystems are operational",
        "verify system components", "system self-test", "run full diagnostics scan",
        "check if microphone and GPU are working properly"
    ]
    for p in health_phrases:
        samples.append({"q": p, "t": [{"tool": "get_system_health", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "get_system_health", "args": {}}]})
        samples.append({"q": f"hey friday {p}", "t": [{"tool": "get_system_health", "args": {}}]})

    # 1.2 get_detailed_hardware_stats
    telemetry_phrases = [
        "get detailed hardware stats", "show detailed hardware statistics", "how much RAM is being used",
        "what is my current CPU usage", "check CPU cores and memory load", "show memory breakdown",
        "how much disk space is free on drive C", "show CPU utilization and thermals",
        "check hardware resource consumption", "my laptop is running hot check hardware specs",
        "how much memory is left", "show CPU and RAM usage breakdown", "detailed system specs and load",
        "what is my CPU utilization right now", "check disk space across drives",
        "how much vram is free", "show hardware telemetry", "what are my current system specs under load"
    ]
    for p in telemetry_phrases:
        samples.append({"q": p, "t": [{"tool": "get_detailed_hardware_stats", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "get_detailed_hardware_stats", "args": {}}]})

    # 1.3 list_processes
    proc_list_phrases = [
        ("list all running processes", {}),
        ("show active processes", {}),
        ("what processes are running", {}),
        ("list open tasks in task manager", {"limit": 25}),
        ("show top running applications and PIDs", {"limit": 25}),
        ("list processes eating my memory", {"limit": 25}),
        ("show running tasks", {}),
        ("what tasks are active right now", {}),
        ("filter processes by chrome", {"filter_name": "chrome"}),
        ("show python processes", {"filter_name": "python"}),
        ("list processes matching discord", {"filter_name": "discord"}),
        ("check if steam is running", {"filter_name": "steam"}),
        ("find process spotify", {"filter_name": "spotify"}),
        ("list top 50 processes", {"limit": 50}),
        ("search running tasks for node", {"filter_name": "node"})
    ]
    for q, args in proc_list_phrases:
        samples.append({"q": q, "t": [{"tool": "list_processes", "args": args}]})
        samples.append({"q": f"friday {q}", "t": [{"tool": "list_processes", "args": args}]})

    # 1.4 kill_process by name or PID
    target_procs = [
        ("chrome.exe", "chrome"), ("Discord.exe", "discord"), ("Spotify.exe", "spotify"),
        ("Code.exe", "vscode"), ("python.exe", "python"), ("excel.exe", "excel"),
        ("photoshop.exe", "photoshop"), ("steam.exe", "steam"), ("zoom.exe", "zoom"),
        ("slack.exe", "slack"), ("notepad.exe", "notepad"), ("vlc.exe", "vlc"),
        ("Taskmgr.exe", "task manager"), ("blender.exe", "blender"), ("obs64.exe", "obs"),
        ("firefox.exe", "firefox"), ("msedge.exe", "edge"), ("node.exe", "node"),
        ("git.exe", "git"), ("cmd.exe", "command prompt")
    ]
    for exe, nickname in target_procs:
        patterns = [
            (f"kill process {exe}", {"name_or_pid": exe}),
            (f"kill {nickname}", {"name_or_pid": exe}),
            (f"terminate {nickname} process", {"name_or_pid": exe}),
            (f"end task {nickname}", {"name_or_pid": exe}),
            (f"force close {nickname}", {"name_or_pid": exe, "force": True}),
            (f"nuke {nickname}", {"name_or_pid": exe, "force": True}),
            (f"shut down {nickname} it is frozen", {"name_or_pid": exe, "force": True}),
            (f"stop {nickname} process immediately", {"name_or_pid": exe, "force": True})
        ]
        for q, args in patterns:
            samples.append({"q": q, "t": [{"tool": "kill_process", "args": args}]})
            samples.append({"q": f"friday {q}", "t": [{"tool": "kill_process", "args": args}]})

    # Kill by PID
    pids = [1024, 4512, 8080, 9928, 3044, 12890, 7712, 2344, 5500, 6892]
    for pid in pids:
        samples.append({"q": f"kill process PID {pid}", "t": [{"tool": "kill_process", "args": {"name_or_pid": pid}}]})
        samples.append({"q": f"terminate PID {pid}", "t": [{"tool": "kill_process", "args": {"name_or_pid": pid}}]})
        samples.append({"q": f"end task with pid {pid}", "t": [{"tool": "kill_process", "args": {"name_or_pid": pid}}]})
        samples.append({"q": f"force kill pid {pid}", "t": [{"tool": "kill_process", "args": {"name_or_pid": pid, "force": True}}]})

    return samples


# =============================================================================
# DOMAIN 2: WINDOW SNAPPING, WORKSPACE PRESETS & LAYOUTS (~600 samples)
# =============================================================================
def generate_domain_windowing_layouts() -> List[Dict[str, Any]]:
    samples = []

    apps = ["Chrome", "VS Code", "Discord", "Notepad", "Terminal", "Spotify", "Slack", "Excel", "Word", "Edge"]
    positions = [
        ("left", ["left", "left half", "left side", "to the left"]),
        ("right", ["right", "right half", "right side", "to the right"]),
        ("top", ["top", "top half", "upper half"]),
        ("bottom", ["bottom", "bottom half", "lower half"]),
        ("center", ["center", "centered", "middle of the screen"]),
        ("top-left", ["top left", "top left corner", "upper left quadrant"]),
        ("top-right", ["top right", "top right corner", "upper right quadrant"]),
        ("bottom-left", ["bottom left", "bottom left corner"]),
        ("bottom-right", ["bottom right", "bottom right corner"])
    ]

    # 2.1 Single Window Snapping
    for app in apps:
        for pos_key, phrs in positions:
            for phr in phrs[:2]:
                q1 = f"snap {app} to {phr}"
                q2 = f"tile {app} {phr}"
                q3 = f"dock {app} on the {phr}"
                samples.append({"q": q1, "t": [{"tool": "snap_window", "args": {"title_keyword": app, "position": pos_key}}]})
                samples.append({"q": f"friday {q1}", "t": [{"tool": "snap_window", "args": {"title_keyword": app, "position": pos_key}}]})
                samples.append({"q": q2, "t": [{"tool": "snap_window", "args": {"title_keyword": app, "position": pos_key}}]})
                samples.append({"q": q3, "t": [{"tool": "snap_window", "args": {"title_keyword": app, "position": pos_key}}]})

    # 2.2 Dual Window Snapping (Split screen 50/50 combos)
    combos = [
        ("Chrome", "VS Code"),
        ("Terminal", "VS Code"),
        ("Discord", "Chrome"),
        ("Notepad", "Excel"),
        ("Word", "Edge"),
        ("Slack", "Chrome"),
        ("Spotify", "Terminal"),
        ("Browser", "Editor")
    ]
    for app_a, app_b in combos:
        dual_phrases = [
            f"split screen with {app_a} on the left and {app_b} on the right",
            f"snap {app_a} to left and {app_b} to right",
            f"put {app_a} on the left half and {app_b} on the right half",
            f"tile {app_a} left and {app_b} right side-by-side",
            f"arrange {app_a} on left side and {app_b} on right side"
        ]
        for dp in dual_phrases:
            dual_tools = [
                {"tool": "snap_window", "args": {"title_keyword": app_a, "position": "left"}},
                {"tool": "snap_window", "args": {"title_keyword": app_b, "position": "right"}}
            ]
            samples.append({"q": dp, "t": dual_tools})
            samples.append({"q": f"friday {dp}", "t": dual_tools})

    # 2.3 Presets
    presets = ["coding", "writing", "multitask", "research", "gaming", "streaming"]
    for pr in presets:
        preset_phrases = [
            f"arrange workspace for {pr}", f"set workspace preset to {pr}",
            f"apply {pr} workspace preset", f"switch layout to {pr} mode",
            f"arrange windows for {pr}", f"organize open windows for {pr}",
            f"set window preset to {pr}", f"activate {pr} window arrangement"
        ]
        for p in preset_phrases:
            samples.append({"q": p, "t": [{"tool": "arrange_workspace_preset", "args": {"preset_name": pr}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "arrange_workspace_preset", "args": {"preset_name": pr}}]})

    # 2.4 Save / Load Workspace Profile
    profiles = ["Dev Setup", "Morning Routine", "Trading View", "Study Session", "Dual Monitor Work", "Video Editing", "Clean Desktop"]
    for prof in profiles:
        samples.append({"q": f"save this workspace layout as {prof}", "t": [{"tool": "save_workspace_profile", "args": {"profile_name": prof}}]})
        samples.append({"q": f"save current window positions as {prof}", "t": [{"tool": "save_workspace_profile", "args": {"profile_name": prof}}]})
        samples.append({"q": f"load my {prof} workspace", "t": [{"tool": "load_workspace_profile", "args": {"profile_name": prof}}]})
        samples.append({"q": f"restore {prof} layout", "t": [{"tool": "load_workspace_profile", "args": {"profile_name": prof}}]})
        samples.append({"q": f"switch to {prof} profile", "t": [{"tool": "load_workspace_profile", "args": {"profile_name": prof}}]})

    # 2.5 List Open Windows
    list_win_phrases = [
        "list all open windows", "what windows are open", "show open window titles and positions",
        "show list of active windows", "which windows are currently open on desktop",
        "enumerate visible windows", "give me a list of running window titles"
    ]
    for p in list_win_phrases:
        samples.append({"q": p, "t": [{"tool": "list_windows", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "list_windows", "args": {}}]})

    return samples


# =============================================================================
# DOMAIN 3: FILE CHORES, DIRECTORY ORGANIZATION & EXPLORER (~350 samples)
# =============================================================================
def generate_domain_file_organization() -> List[Dict[str, Any]]:
    samples = []

    # 3.1 organize_directory_by_type
    dirs = [
        ("~/Downloads", "Downloads folder"),
        ("~/Desktop", "Desktop"),
        ("~/Documents", "Documents folder"),
        ("~/Pictures", "Pictures folder"),
        ("C:/Users/User/Downloads", "Downloads directory"),
        ("D:/Downloads", "D drive downloads"),
        ("~/Desktop", "desktop icons")
    ]
    for dir_path, dir_label in dirs:
        org_phrases = [
            f"organize my {dir_label} by file type",
            f"clean up my {dir_label}",
            f"tidy up files in {dir_label}",
            f"group files in {dir_label} into categorized folders",
            f"sort {dir_label} by extension",
            f"auto-sort files in {dir_label}",
            f"declutter my {dir_label}",
            f"clean and arrange {dir_label} into folders"
        ]
        for p in org_phrases:
            samples.append({"q": p, "t": [{"tool": "organize_directory_by_type", "args": {"directory_path": dir_path}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "organize_directory_by_type", "args": {"directory_path": dir_path}}]})

    # 3.2 open_in_explorer
    paths = [
        ("~/Downloads", "Downloads folder in explorer"),
        ("~/Documents", "Documents in windows explorer"),
        ("C:/Projects", "C:/Projects folder in explorer"),
        ("D:/Work", "D:/Work directory in file explorer"),
        ("~/Desktop", "desktop in file explorer"),
        ("C:/Program Files", "Program Files in explorer"),
        ("~/Music", "Music folder in Windows Explorer"),
        ("~/Videos", "Videos in Explorer"),
        ("C:/Windows/System32", "System32 in file explorer"),
        ("D:/Backup", "Backup folder in explorer")
    ]
    for path_str, phrase_suffix in paths:
        explorer_phrases = [
            f"open {phrase_suffix}",
            f"reveal {path_str} in Windows Explorer",
            f"show {path_str} in folder",
            f"open folder {path_str}",
            f"browse {path_str} in explorer",
            f"open directory {path_str} in file manager"
        ]
        for p in explorer_phrases:
            samples.append({"q": p, "t": [{"tool": "open_in_explorer", "args": {"path": path_str}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "open_in_explorer", "args": {"path": path_str}}]})

    # 3.3 rename_file
    rename_pairs = [
        ("report.pdf", "report_final.pdf"),
        ("notes.txt", "meeting_notes_nov.txt"),
        ("presentation.pptx", "pitch_deck_v2.pptx"),
        ("image.png", "hero_banner.png"),
        ("data.csv", "sales_q3.csv"),
        ("draft.docx", "submitted_essay.docx"),
        ("video.mp4", "lecture_1.mp4"),
        ("archive.zip", "project_backup.zip"),
        ("config.json.bak", "config.json"),
        ("app.py.old", "app.py"),
        ("photo.jpg", "profile_picture.jpg")
    ]
    for src, dst in rename_pairs:
        rename_phrases = [
            f"rename {src} to {dst}",
            f"change filename of {src} to {dst}",
            f"rename file {src} as {dst}",
            f"change name of {src} to {dst}"
        ]
        for p in rename_phrases:
            samples.append({"q": p, "t": [{"tool": "rename_file", "args": {"source_path": src, "new_name": dst}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "rename_file", "args": {"source_path": src, "new_name": dst}}]})

    # 3.4 search_files
    file_searches = [
        ("*.pdf", "~/Documents", "find all PDF files in Documents"),
        ("*.png", "~/Desktop", "search for all PNG images on Desktop"),
        ("test_*.py", ".", "find all test python files"),
        ("config.yaml", "C:/Projects", "locate config.yaml in C:/Projects"),
        ("*.docx", "~/Downloads", "search for Word documents in Downloads"),
        ("*.csv", ".", "find all CSV spreadsheets"),
        ("*.log", "C:/Logs", "find all log files in C:/Logs"),
        ("*.zip", "~/Downloads", "search for zip archives in Downloads"),
        ("*.json", ".", "find json configuration files"),
        ("*.mp4", "~/Videos", "search for mp4 video files in Videos")
    ]
    for pat, d, phr in file_searches:
        samples.append({"q": phr, "t": [{"tool": "search_files", "args": {"pattern": pat, "directory": d}}]})
        samples.append({"q": f"friday {phr}", "t": [{"tool": "search_files", "args": {"pattern": pat, "directory": d}}]})

    # 3.5 grep_files
    grep_searches = [
        ("TODO", "*.py", "search for TODO in all python files"),
        ("DATABASE_URL", "*.env", "grep for DATABASE_URL in env files"),
        ("API_KEY", "*", "find files containing API_KEY"),
        ("import torch", "*.py", "grep for import torch across python files"),
        ("FIXME", "*.ts", "search for FIXME in TypeScript files"),
        ("class Model", "*.py", "grep for class Model in python source"),
        ("password", "*.conf", "find occurrences of password in config files"),
        ("localhost:8000", "*", "grep for localhost:8000 in codebase")
    ]
    for q_text, pat, phr in grep_searches:
        samples.append({"q": phr, "t": [{"tool": "grep_files", "args": {"query": q_text, "file_pattern": pat}}]})
        samples.append({"q": f"friday {phr}", "t": [{"tool": "grep_files", "args": {"query": q_text, "file_pattern": pat}}]})

    return samples


# =============================================================================
# DOMAIN 4: AMBIENT INFORMATION, WEATHER & NEWS (~350 samples)
# =============================================================================
def generate_domain_weather_info() -> List[Dict[str, Any]]:
    samples = []

    # 4.1 Weather (Auto / Current Location)
    auto_weather_phrases = [
        "what's the weather today", "is it going to rain today", "what is the temperature right now",
        "current weather forecast", "do I need an umbrella today", "how hot is it outside",
        "what's the weather like outside", "weather forecast for today", "check current temperature",
        "is it sunny outside", "will it rain this afternoon", "show today's weather forecast",
        "give me the local weather", "what's the forecast for tonight", "how is the weather looking"
    ]
    for p in auto_weather_phrases:
        samples.append({"q": p, "t": [{"tool": "get_weather", "args": {"location": "auto"}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "get_weather", "args": {"location": "auto"}}]})
        samples.append({"q": f"hey friday {p}", "t": [{"tool": "get_weather", "args": {"location": "auto"}}]})

    # 4.2 Weather (Specific Cities)
    cities = [
        "New York", "London", "Tokyo", "Paris", "Mumbai", "San Francisco",
        "Berlin", "Toronto", "Sydney", "Delhi", "Seattle", "Chicago", "Dubai", "Singapore",
        "Boston", "Austin", "Vancouver", "Amsterdam", "Seoul", "Bangalore"
    ]
    for city in cities:
        city_phrases = [
            f"what's the weather in {city}",
            f"weather forecast for {city}",
            f"how cold is it in {city}",
            f"temperature in {city} right now",
            f"is it raining in {city}",
            f"check weather for {city}",
            f"forecast in {city} tomorrow"
        ]
        for p in city_phrases:
            samples.append({"q": p, "t": [{"tool": "get_weather", "args": {"location": city}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "get_weather", "args": {"location": city}}]})

    # 4.3 search_wikipedia
    wiki_topics = [
        "Nikola Tesla", "Quantum Computing", "French Revolution", "James Webb Space Telescope",
        "Photosynthesis", "Theory of Relativity", "Alan Turing", "Black Hole",
        "Artificial Intelligence", "Renewable Energy", "Mars Rover", "CRISPR",
        "Large Language Models", "Great Wall of China", "Johannes Gutenberg", "Isaac Newton"
    ]
    for topic in wiki_topics:
        wiki_phrases = [
            f"search Wikipedia for {topic}",
            f"who was {topic} according to Wikipedia",
            f"look up {topic} on Wikipedia",
            f"summarize Wikipedia article for {topic}",
            f"give me a Wikipedia summary on {topic}",
            f"what does Wikipedia say about {topic}"
        ]
        for p in wiki_phrases:
            samples.append({"q": p, "t": [{"tool": "search_wikipedia", "args": {"query": topic, "sentences": 3}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "search_wikipedia", "args": {"query": topic, "sentences": 3}}]})

    # 4.4 news_search
    news_queries = [
        ("tech news", "tech", "show me latest tech news"),
        ("AI developments", "artificial intelligence", "what are today's headlines in AI"),
        ("space news", "space", "search breaking news about space exploration"),
        ("financial markets", "finance", "show top financial and market news"),
        ("cybersecurity alerts", "cybersecurity", "check latest cybersecurity news"),
        ("world news", "world news", "what is happening in world news today"),
        ("science discoveries", "science", "search recent science news"),
        ("gaming news", "gaming", "show top gaming industry news")
    ]
    for label, query_term, phrase in news_queries:
        samples.append({"q": phrase, "t": [{"tool": "news_search", "args": {"query": query_term, "max_results": 5}}]})
        samples.append({"q": f"friday {phrase}", "t": [{"tool": "news_search", "args": {"query": query_term, "max_results": 5}}]})

    return samples


# =============================================================================
# DOMAIN 5: DIRECT MESSAGING & EMAIL DRAFTING (~400 samples)
# =============================================================================
def generate_domain_messaging_contacts() -> List[Dict[str, Any]]:
    samples = []

    # 5.1 WhatsApp
    contacts = [
        ("Mom", "I'll be home in 20 minutes"),
        ("Dad", "Reached safely, don't worry"),
        ("Alex", "The meeting has been rescheduled to 4 PM"),
        ("Sarah", "Did you receive the documents I sent?"),
        ("David", "Let's catch up tomorrow afternoon"),
        ("Emily", "Happy Birthday! Have a great day"),
        ("John", "Can you send me the latest repo link?"),
        ("Bro", "Are you watching the match tonight?"),
        ("Sis", "Call me when you get free"),
        ("Wife", "Leaving office now, need anything from store?"),
        ("Manager", "Finished the weekly report, submitted on portal"),
        ("Jessica", "Thanks for your help with the presentation")
    ]
    for contact, msg in contacts:
        wa_phrases = [
            f"send a WhatsApp to {contact} saying {msg}",
            f"WhatsApp {contact}: {msg}",
            f"message {contact} on WhatsApp that {msg}",
            f"shoot a WhatsApp text to {contact} with message: {msg}",
            f"drop {contact} a WhatsApp message saying {msg}",
            f"text {contact} on WhatsApp: {msg}"
        ]
        for p in wa_phrases:
            samples.append({"q": p, "t": [{"tool": "send_whatsapp_message", "args": {"contact": contact, "message": msg}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "send_whatsapp_message", "args": {"contact": contact, "message": msg}}]})

    # WhatsApp with Phone Numbers
    phone_numbers = ["+15551234567", "+447911123456", "+919876543210", "+14155552671"]
    for num in phone_numbers:
        samples.append({
            "q": f"send a WhatsApp message to {num} saying hello there",
            "t": [{"tool": "send_whatsapp_message", "args": {"contact": num, "message": "hello there", "phone_number": num}}]
        })
        samples.append({
            "q": f"WhatsApp {num} that the package has arrived",
            "t": [{"tool": "send_whatsapp_message", "args": {"contact": num, "message": "the package has arrived", "phone_number": num}}]
        })

    # 5.2 Telegram
    tg_users = [
        ("@alex_dev", "The backend server is up and running"),
        ("@sarah_ops", "Deployment to production succeeded"),
        ("David", "Call me when you're free"),
        ("Team Channel", "Standup begins in 10 minutes"),
        ("@mike_design", "Here are the updated UI mockups"),
        ("Work Group", "Please review PR #42 when possible")
    ]
    for user, msg in tg_users:
        tg_phrases = [
            f"send a Telegram to {user} saying {msg}",
            f"message {user} on Telegram that {msg}",
            f"Telegram {user}: {msg}",
            f"drop a message on Telegram to {user} saying {msg}",
            f"ping {user} on Telegram with {msg}"
        ]
        for p in tg_phrases:
            samples.append({"q": p, "t": [{"tool": "send_telegram_message", "args": {"recipient": user, "message": msg}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "send_telegram_message", "args": {"recipient": user, "message": msg}}]})

    # 5.3 compose_email
    email_drafts = [
        ("boss@company.com", "Leave Request", "I would like to request leave for tomorrow due to personal matters."),
        ("prof_smith@university.edu", "Project Submission Extension", "Dear Professor, I am writing to politely request a 2-day extension on the final submission."),
        ("support@service.com", "Account Issue", "Hello, I am having trouble logging into my portal. Please assist."),
        ("client@agency.io", "Project Status Update", "Hi team, please find attached the progress update for Phase 1."),
        ("hr@company.com", "Benefits Inquiry", "Hello HR, could you please clarify the health insurance enrollment deadline?"),
        ("alex@startup.com", "Coffee Chat Followup", "Hey Alex, great meeting you yesterday. Let's stay in touch!")
    ]
    for to_addr, sub, body in email_drafts:
        email_phrases = [
            f"draft an email to {to_addr} with subject {sub} and body {body}",
            f"compose an email to {to_addr} regarding {sub}",
            f"prepare an email draft to {to_addr} about {sub}",
            f"write an email to {to_addr} with subject {sub}"
        ]
        for p in email_phrases:
            samples.append({"q": p, "t": [{"tool": "compose_email", "args": {"to": to_addr, "subject": sub, "body": body}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "compose_email", "args": {"to": to_addr, "subject": sub, "body": body}}]})

    # 5.4 search_contacts & get_contact
    contact_names = ["Sarah", "Alex", "David", "Dr. Wilson", "Alice Johnson", "Michael Brown", "Emma Watson", "Tech Lead", "Accountant"]
    for name in contact_names:
        c_phrases = [
            f"search contacts for {name}",
            f"find {name} in address book",
            f"lookup contact {name}",
            f"get contact details for {name}",
            f"check contact info for {name}"
        ]
        for p in c_phrases:
            samples.append({"q": p, "t": [{"tool": "search_contacts", "args": {"query": name}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "search_contacts", "args": {"query": name}}]})

    return samples


# =============================================================================
# DOMAIN 6: MULTIMEDIA CHORES & MEDIA CONVERSIONS (~350 samples)
# =============================================================================
def generate_domain_multimedia_conversions() -> List[Dict[str, Any]]:
    samples = []

    # 6.1 download_youtube
    yt_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/kJQP7kiw5Fk",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",
        "https://youtu.be/3JZ_D3ELwOQ",
        "https://www.youtube.com/watch?v=fJ9rUzIMcZQ",
        "https://youtu.be/L_LUpnjgPso",
        "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
        "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
        "https://youtu.be/kXYiU_JCYtU",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0"
    ]
    for url in yt_urls:
        yt_phrases = [
            (f"download this YouTube video {url}", {"url": url, "format": "mp4", "quality": "best"}),
            (f"download video from {url} in 1080p", {"url": url, "format": "mp4", "quality": "1080p"}),
            (f"download YouTube video {url} in 720p", {"url": url, "format": "mp4", "quality": "720p"}),
            (f"download YouTube link {url} in 4k", {"url": url, "format": "mp4", "quality": "4k"}),
            (f"rip audio from YouTube {url} as mp3", {"url": url, "format": "mp3", "quality": "best"}),
            (f"save youtube video {url}", {"url": url, "format": "mp4", "quality": "best"}),
            (f"download youtube audio {url}", {"url": url, "format": "mp3", "quality": "best"}),
            (f"fetch youtube audio track from {url}", {"url": url, "format": "mp3", "quality": "best"}),
            (f"download YouTube link {url} in best quality", {"url": url, "format": "mp4", "quality": "best"}),
            (f"extract audio only from youtube {url}", {"url": url, "format": "mp3", "quality": "best"}),
            (f"save mp3 from {url}", {"url": url, "format": "mp3", "quality": "best"}),
            (f"grab youtube video {url} in 1080p hd", {"url": url, "format": "mp4", "quality": "1080p"})
        ]
        for q, args in yt_phrases:
            samples.append({"q": q, "t": [{"tool": "download_youtube", "args": args}]})
            samples.append({"q": f"friday {q}", "t": [{"tool": "download_youtube", "args": args}]})

    # 6.2 convert_media_ffmpeg
    conversions = [
        ("lecture.mp4", "lecture.mp3", True, "convert lecture.mp4 to mp3 audio"),
        ("recording.mkv", "recording.mp4", False, "transcode recording.mkv to mp4"),
        ("clip.webm", "clip.mp4", False, "convert clip.webm to mp4 format"),
        ("audio.wav", "audio.mp3", True, "compress audio.wav to mp3"),
        ("tutorial.mp4", "tutorial_audio.mp3", True, "extract audio from tutorial.mp4"),
        ("podcast.m4a", "podcast.mp3", True, "convert podcast.m4a into mp3 format"),
        ("screen_capture.avi", "screen_capture.mp4", False, "convert avi video to mp4"),
        ("movie.mov", "movie.mp4", False, "transcode mov to mp4 container"),
        ("track.flac", "track.mp3", True, "convert flac audio track to mp3"),
        ("sample.ogg", "sample.mp3", True, "convert ogg file to mp3"),
        ("video.ts", "video.mp4", False, "transcode ts video stream to mp4"),
        ("animation.gif", "animation.mp4", False, "convert animated gif to mp4 video"),
        ("raw_audio.aac", "audio.mp3", True, "convert aac audio file to mp3")
    ]
    for inp, out, audio_only, phr in conversions:
        samples.append({
            "q": phr,
            "t": [{"tool": "convert_media_ffmpeg", "args": {"input_file": inp, "output_file": out, "audio_only": audio_only}}]
        })
        samples.append({
            "q": f"friday {phr}",
            "t": [{"tool": "convert_media_ffmpeg", "args": {"input_file": inp, "output_file": out, "audio_only": audio_only}}]
        })
        samples.append({
            "q": f"hey friday please {phr}",
            "t": [{"tool": "convert_media_ffmpeg", "args": {"input_file": inp, "output_file": out, "audio_only": audio_only}}]
        })
        samples.append({
            "q": f"use ffmpeg to {phr}",
            "t": [{"tool": "convert_media_ffmpeg", "args": {"input_file": inp, "output_file": out, "audio_only": audio_only}}]
        })

    # 6.3 capture_monitor_screenshot
    mon_phrases = [
        ("take a screenshot of monitor 2", {"monitor_index": 2, "all_monitors": False}),
        ("screenshot display 2", {"monitor_index": 2, "all_monitors": False}),
        ("capture monitor 1 screenshot", {"monitor_index": 1, "all_monitors": False}),
        ("capture screenshot of primary display", {"monitor_index": 1, "all_monitors": False}),
        ("take screenshot of all monitors", {"all_monitors": True}),
        ("capture both screens in a screenshot", {"all_monitors": True}),
        ("screenshot second monitor", {"monitor_index": 2, "all_monitors": False}),
        ("take full screenshot across all displays", {"all_monitors": True}),
        ("capture dual monitors screenshot", {"all_monitors": True}),
        ("grab screen on monitor 1", {"monitor_index": 1, "all_monitors": False}),
        ("screenshot display 3", {"monitor_index": 3, "all_monitors": False}),
        ("capture screen on secondary monitor", {"monitor_index": 2, "all_monitors": False}),
        ("take screenshot of entire virtual desktop", {"all_monitors": True}),
        ("capture all connected screens in one image", {"all_monitors": True})
    ]
    for q, args in mon_phrases:
        samples.append({"q": q, "t": [{"tool": "capture_monitor_screenshot", "args": args}]})
        samples.append({"q": f"friday {q}", "t": [{"tool": "capture_monitor_screenshot", "args": args}]})
        samples.append({"q": f"hey friday {q}", "t": [{"tool": "capture_monitor_screenshot", "args": args}]})

    return samples


# =============================================================================
# DOMAIN 7: VOICE PERSONA & DESKTOP NOTIFICATIONS (~350 samples)
# =============================================================================
def generate_domain_voice_notifications() -> List[Dict[str, Any]]:
    samples = []

    # 7.1 switch_tts_voice
    voices = [
        ("female", "change your voice to female"),
        ("female", "can you speak with a female voice"),
        ("female", "switch to girl voice"),
        ("male", "switch your voice to male"),
        ("male", "use a male assistant voice"),
        ("Prabhat", "switch TTS voice to Prabhat"),
        ("Prabhat", "change voice to Prabhat"),
        ("Neerja", "change voice to Neerja"),
        ("Neerja", "switch assistant voice to Neerja"),
        ("Jenny", "make your voice sound like Jenny"),
        ("Jenny", "switch TTS to Jenny"),
        ("Guy", "switch assistant voice to Guy"),
        ("Guy", "use Guy voice for speech"),
        ("voice 1", "use voice 1"),
        ("voice 2", "switch to voice 2"),
        ("voice 3", "switch to voice 3"),
        ("British", "change your voice to British accent"),
        ("British", "speak with a British accent"),
        ("Sonia", "switch voice persona to Sonia"),
        ("Sonia", "use Sonia voice for TTS"),
        ("deep male", "use a deeper male voice"),
        ("natural female", "switch to natural female voice"),
        ("Indian female", "speak in an Indian female voice"),
        ("Indian male", "use an Indian male voice"),
        ("American female", "switch to American female voice")
    ]
    for v_id, phr in voices:
        samples.append({"q": phr, "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": v_id}}]})
        samples.append({"q": f"friday {phr}", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": v_id}}]})
        samples.append({"q": f"hey friday {phr}", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": v_id}}]})
        samples.append({"q": f"please {phr}", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": v_id}}]})
        samples.append({"q": f"friday could you {phr}", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": v_id}}]})

    # 7.2 list_tts_voices
    list_voice_phrases = [
        "what voices do you have available", "list all TTS voices", "show available voice options",
        "what assistant voices can you use", "list available speech voices", "show voice list",
        "enumerate installed TTS voices", "what accents and voices are installed",
        "show me the list of text to speech voices", "which voices can I choose from",
        "what speech voices are supported", "display all available assistant voices"
    ]
    for p in list_voice_phrases:
        samples.append({"q": p, "t": [{"tool": "list_tts_voices", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "list_tts_voices", "args": {}}]})
        samples.append({"q": f"hey friday {p}", "t": [{"tool": "list_tts_voices", "args": {}}]})

    # 7.3 send_desktop_toast
    toast_samples = [
        ("Task Completed", "Download finished successfully", "show a desktop notification saying download finished"),
        ("Alert", "Meeting starts in 5 minutes", "send me a toast notification meeting in 5 minutes"),
        ("Backup Status", "Data backup completed", "pop up a desktop banner alert backup complete"),
        ("Timer Alert", "Time is up", "display a desktop toast alert time is up"),
        ("Reminder", "Take a 5 minute screen break", "send a desktop notification reminding me to take a break"),
        ("Build Success", "Project compiled with zero errors", "send notification banner build succeeded"),
        ("Security Notice", "Virus scan completed cleanly", "show desktop toast security scan clean"),
        ("Render Done", "Video rendering has finished", "pop up toast alert video render complete"),
        ("Battery Alert", "Battery is at 100 percent fully charged", "notify me when battery is full with desktop toast"),
        ("File Transfer", "File upload finished", "show desktop toast alert upload finished"),
        ("System Alert", "Subsystem initialization finished", "display desktop toast banner system ready"),
        ("Pomodoro Alert", "Focus session ended, time for a break", "send a desktop banner notification break time")
    ]
    for title, msg, phr in toast_samples:
        samples.append({"q": phr, "t": [{"tool": "send_desktop_toast", "args": {"title": title, "message": msg}}]})
        samples.append({"q": f"friday {phr}", "t": [{"tool": "send_desktop_toast", "args": {"title": title, "message": msg}}]})
        samples.append({"q": f"hey friday {phr}", "t": [{"tool": "send_desktop_toast", "args": {"title": title, "message": msg}}]})
        samples.append({"q": f"please {phr}", "t": [{"tool": "send_desktop_toast", "args": {"title": title, "message": msg}}]})

    return samples


# =============================================================================
# AUGMENTATION & CHATML FORMATTING
# =============================================================================
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
    print("🚀 GENERATING DesktopIntent-v7-Productivity DATASET")
    print("=" * 75)

    d1 = generate_domain_health_processes()
    print(f"  [+] Domain 1 (System Health, Telemetry & Processes): {len(d1):,} samples")

    d2 = generate_domain_windowing_layouts()
    print(f"  [+] Domain 2 (Window Snapping & Workspace Presets):   {len(d2):,} samples")

    d3 = generate_domain_file_organization()
    print(f"  [+] Domain 3 (File Organization & Explorer Ops):     {len(d3):,} samples")

    d4 = generate_domain_weather_info()
    print(f"  [+] Domain 4 (Weather, Wikipedia & News):            {len(d4):,} samples")

    d5 = generate_domain_messaging_contacts()
    print(f"  [+] Domain 5 (WhatsApp, Telegram & Email Drafting):  {len(d5):,} samples")

    d6 = generate_domain_multimedia_conversions()
    print(f"  [+] Domain 6 (YouTube Download & Media Conversions): {len(d6):,} samples")

    d7 = generate_domain_voice_notifications()
    print(f"  [+] Domain 7 (Voice Customization & Toast Alerts):   {len(d7):,} samples")

    raw_samples = d1 + d2 + d3 + d4 + d5 + d6 + d7
    augmented_samples = augment_samples(raw_samples)
    print(f"  [+] Speech & Disfluency Augmentation:               {len(augmented_samples) - len(raw_samples):,} samples")

    random.shuffle(augmented_samples)

    split_idx = int(len(augmented_samples) * 0.9)
    train_raw = augmented_samples[:split_idx]
    val_raw = augmented_samples[split_idx:]

    print("-" * 75)
    print(f"  Total Master v7-Productivity Samples:                {len(augmented_samples):,}")
    print(f"    Train Split (90%):                                 {len(train_raw):,}")
    print(f"    Validation Split (10%):                            {len(val_raw):,}")
    print("-" * 75)

    # Convert to ChatML
    train_chatml = [format_chatml(s) for s in train_raw]
    val_chatml = [format_chatml(s) for s in val_raw]

    # Save to local dataset/
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATASET_DIR / "v7_productivity_train.jsonl"
    val_path = DATASET_DIR / "v7_productivity_val.jsonl"

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
