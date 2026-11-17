"""Generates the DesktopIntent-v5-Precision Dataset for FRIDAY AI Assistant.

Directly targets the 3 major biases and 2 blind spots exposed in the diagnostic audit:
1. Media Control & Track Navigation vs. Muting / Volume Disambiguation (~400 samples)
2. Multi-Monitor & Window-Targeted Vision (~300 samples)
3. Canonical Scheduler, Timers & Reminders (~400 samples)
4. Local Tool Search vs. Google Browser Search Disambiguation (~350 samples)
5. Telemetry, Networking & Tab Diagnostics (~250 samples)
6. Action-Over-Chitchat & Quoted Typing Disambiguation (~200 samples)
PLUS ~700 core replay samples from v4_perfect (app launching, window snapping, volume, brightness)
to completely prevent catastrophic forgetting.
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
HF_SUBSET_DIR = Path("D:/Projects/desktop-llm/hf_export/friday-desktop-intent/subsets/v5_precision")


# =============================================================================
# MODULE 1: MEDIA CONTROL & MUSIC DISAMBIGUATION (~400 samples)
# =============================================================================
def generate_module_1() -> List[Dict[str, Any]]:
    samples = []

    # 1.1 Play/Pause & Resume (NOT play_music(query="resume") and NOT mute)
    resume_phrases = [
        "resume the music", "resume music", "resume playback", "unpause music",
        "unpause the song", "unpause playback", "continue playing", "continue the music",
        "carry on with the music", "start playing again", "unpause", "resume track",
        "hit play again", "toggle play", "toggle pause", "play pause",
        "resume the video", "unpause video", "continue video playback"
    ]
    for p in resume_phrases:
        samples.append({"q": p, "t": [{"tool": "media_play_pause", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "media_play_pause", "args": {}}]})
        samples.append({"q": f"could you please {p}", "t": [{"tool": "media_play_pause", "args": {}}]})
        samples.append({"q": f"uh {p} now", "t": [{"tool": "media_play_pause", "args": {}}]})

    pause_phrases = [
        "pause the music", "pause music", "pause playback", "pause song",
        "pause track", "pause the video", "pause video", "hold playback",
        "pause Spotify", "pause youtube music playback"
    ]
    for p in pause_phrases:
        samples.append({"q": p, "t": [{"tool": "media_play_pause", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "media_play_pause", "args": {}}]})
        samples.append({"q": f"can you {p} please", "t": [{"tool": "media_play_pause", "args": {}}]})

    # 1.2 Stop Playback
    stop_phrases = [
        "stop the music", "stop music", "stop playback", "stop current track",
        "halt playback", "stop playing song", "stop Spotify playback", "kill the music"
    ]
    for p in stop_phrases:
        samples.append({"q": p, "t": [{"tool": "media_stop", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "media_stop", "args": {}}]})

    # 1.3 Next / Prev Track (NOT play_previous_track)
    next_phrases = [
        "next track", "skip track", "skip song", "next song", "play next song",
        "skip this song", "go to next track", "skip forward", "next music track"
    ]
    for p in next_phrases:
        samples.append({"q": p, "t": [{"tool": "media_next_track", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "media_next_track", "args": {}}]})

    prev_phrases = [
        "previous track", "prev track", "previous song", "prev song", "go back one song",
        "play previous track", "replay previous song", "last song", "back to previous track"
    ]
    for p in prev_phrases:
        samples.append({"q": p, "t": [{"tool": "media_prev_track", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "media_prev_track", "args": {}}]})

    # 1.4 Contrastive: Mute Audio vs Pause Music
    mute_phrases = [
        "mute the sound", "mute audio", "silence the sound", "mute computer",
        "mute system volume", "silence audio", "mute speakers", "shut up the audio"
    ]
    for p in mute_phrases:
        samples.append({"q": p, "t": [{"tool": "mute_audio", "args": {"mute": True}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "mute_audio", "args": {"mute": True}}]})

    unmute_phrases = [
        "unmute the audio", "unmute sound", "turn audio back on", "restore sound",
        "unmute speakers", "disable mute"
    ]
    for p in unmute_phrases:
        samples.append({"q": p, "t": [{"tool": "mute_audio", "args": {"mute": False}}]})

    # 1.5 Targeted Music Play with Platforms
    songs = [
        ("Bohemian Rhapsody", "Queen"), ("Blinding Lights", "The Weeknd"),
        ("Shape of You", "Ed Sheeran"), ("Hotel California", "Eagles"),
        ("Stayin Alive", "Bee Gees"), ("lo-fi beats", "Chillhop"),
        ("Interstellar theme", "Hans Zimmer"), ("Smooth Criminal", "Michael Jackson")
    ]
    platforms = [
        ("spotify", "on spotify"), ("youtube", "on youtube"), ("youtube", "on youtube music"), ("auto", "")
    ]
    for song, artist in songs:
        for plat, suffix in platforms:
            q1 = f"play {song} {suffix}".strip()
            q2 = f"play {song} by {artist} {suffix}".strip()
            samples.append({"q": q1, "t": [{"tool": "play_music", "args": {"query": song, "platform": plat}}]})
            samples.append({"q": q2, "t": [{"tool": "play_music", "args": {"query": f"{song} by {artist}", "platform": plat}}]})

    # 1.6 Hard Contrastive: Songs with words "resume", "stop", "pause"
    samples.append({"q": "play resume by Lil Yachty", "t": [{"tool": "play_music", "args": {"query": "resume by Lil Yachty", "platform": "auto"}}]})
    samples.append({"q": "play don't stop me now by Queen on spotify", "t": [{"tool": "play_music", "args": {"query": "don't stop me now by Queen", "platform": "spotify"}}]})
    samples.append({"q": "play pause 4porno on youtube", "t": [{"tool": "play_music", "args": {"query": "pause 4porno", "platform": "youtube"}}]})

    return samples


# =============================================================================
# MODULE 2: MULTI-MONITOR & WINDOW-TARGETED VISION (~300 samples)
# =============================================================================
def generate_module_2() -> List[Dict[str, Any]]:
    samples = []

    # 2.1 Multi-Monitor Indices
    for idx in [1, 2, 3]:
        phrases = [
            f"take a screenshot of monitor {idx}", f"capture monitor {idx}",
            f"screenshot of screen {idx}", f"capture display {idx}",
            f"take screenshot of display {idx}", f"screen grab monitor {idx}",
            f"grab screenshot on monitor {idx}", f"snap display {idx}"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "capture_monitor_screenshot", "args": {"monitor_index": idx}}]})
            samples.append({"q": f"friday please {p}", "t": [{"tool": "capture_monitor_screenshot", "args": {"monitor_index": idx}}]})

    # 2.2 All Monitors (Virtual Desktop)
    all_phrases = [
        "take a screenshot of all monitors", "capture all screens",
        "screenshot across all displays", "capture virtual desktop",
        "take a full screenshot of both screens", "screenshot of all monitors",
        "grab all connected displays", "capture entire multi-monitor setup"
    ]
    for p in all_phrases:
        samples.append({"q": p, "t": [{"tool": "capture_monitor_screenshot", "args": {"all_monitors": True}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "capture_monitor_screenshot", "args": {"all_monitors": True}}]})

    # 2.3 Window-Targeted Screenshot
    apps = ["notepad", "chrome", "spotify", "discord", "terminal", "vs code", "word", "excel", "calculator", "browser"]
    for app in apps:
        phrases = [
            f"take a screenshot of {app}", f"capture {app} window",
            f"take a screenshot of the {app} window", f"capture window titled {app}",
            f"screenshot {app}", f"grab {app} window"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "capture_window_screenshot", "args": {"window_title": app}}]})

    # 2.4 OCR & Screen Extraction
    ocr_phrases = [
        "read the text on my screen", "extract text from screen using OCR",
        "what text is on my screen", "extract text using ocr",
        "read on screen text", "perform ocr on current screen",
        "grab text from screen", "ocr current display"
    ]
    for p in ocr_phrases:
        samples.append({"q": p, "t": [{"tool": "extract_text_ocr", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "extract_text_ocr", "args": {}}]})

    # 2.5 Inspect Screen and Act
    actions = [
        ("look at screen and click Save", "click Save", "Save"),
        ("find the error popup and click OK", "click OK", "OK"),
        ("click the Submit button on screen", "click Submit", "Submit"),
        ("find View all on screen and click it", "click View all", "View all"),
        ("inspect screen and click Next", "click Next", "Next"),
        ("look at the screen and close the alert dialog", "close alert", "Close"),
        ("click the Login button on screen", "click Login", "Login"),
        ("find the download link and click it", "click download", "Download")
    ]
    for q, inst, target in actions:
        samples.append({"q": q, "t": [{"tool": "inspect_screen_and_act", "args": {"instruction": inst, "target_element": target}}]})
        samples.append({"q": f"friday {q}", "t": [{"tool": "inspect_screen_and_act", "args": {"instruction": inst, "target_element": target}}]})

    return samples


# =============================================================================
# MODULE 3: CANONICAL SCHEDULER, TIMERS & REMINDERS (~400 samples)
# =============================================================================
def generate_module_3() -> List[Dict[str, Any]]:
    samples = []

    # 3.1 Timers: set_timer(seconds, message)
    timer_configs = [
        (30, "30 seconds", "Timer (30 seconds)"),
        (60, "1 minute", "Timer (1 minute)"),
        (120, "2 minutes", "Timer (2 minutes)"),
        (300, "5 minutes", "Timer (5 minutes)"),
        (600, "10 minutes", "Timer (10 minutes)"),
        (900, "15 minutes", "Timer (15 minutes)"),
        (1200, "20 minutes", "Timer (20 minutes)"),
        (1800, "30 minutes", "Timer (30 minutes)"),
        (3600, "1 hour", "Timer (1 hour)"),
        (5400, "90 minutes", "Timer (90 minutes)"),
    ]
    for sec, dur_str, msg in timer_configs:
        phrases = [
            f"set a timer for {dur_str}", f"start a {dur_str} timer",
            f"timer for {dur_str}", f"count down {dur_str}",
            f"set timer {dur_str}", f"create a timer for {dur_str}"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "set_timer", "args": {"seconds": sec, "message": msg}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "set_timer", "args": {"seconds": sec, "message": msg}}]})

    # 3.2 Pause / Resume / Cancel Timers (target="active")
    pause_timer_phrases = [
        "pause the timer", "pause timer", "pause countdown", "pause active timer",
        "hold the timer", "pause the current timer"
    ]
    for p in pause_timer_phrases:
        samples.append({"q": p, "t": [{"tool": "pause_timer", "args": {"target": "active"}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "pause_timer", "args": {"target": "active"}}]})

    resume_timer_phrases = [
        "resume the timer", "resume timer", "unpause timer", "continue timer",
        "resume countdown", "unpause the countdown", "resume active timer"
    ]
    for p in resume_timer_phrases:
        samples.append({"q": p, "t": [{"tool": "resume_timer", "args": {"target": "active"}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "resume_timer", "args": {"target": "active"}}]})

    cancel_timer_phrases = [
        "cancel the timer", "cancel timer", "stop timer", "delete active timer",
        "clear timer", "cancel countdown", "stop active timer", "abort timer"
    ]
    for p in cancel_timer_phrases:
        samples.append({"q": p, "t": [{"tool": "cancel_timer", "args": {"target": "active"}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "cancel_timer", "args": {"target": "active"}}]})

    # 3.3 Reminders: set_reminder(message, time_str)
    reminders = [
        ("call mom", "5:00 PM"), ("check the oven", "in 10 minutes"),
        ("take medicine", "8:00 PM"), ("submit weekly report", "tomorrow at 9 AM"),
        ("team standup", "in 15 minutes"), ("drink water", "in 30 minutes"),
        ("pick up dry cleaning", "at 4 PM"), ("review pull request", "tomorrow at 2 PM")
    ]
    for task, t_str in reminders:
        phrases = [
            f"remind me to {task} {t_str}", f"set a reminder to {task} {t_str}",
            f"create reminder to {task} at {t_str}", f"remind me {t_str} to {task}"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "set_reminder", "args": {"message": task, "time_str": t_str}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "set_reminder", "args": {"message": task, "time_str": t_str}}]})

    # 3.4 List Reminders & Alarms
    list_phrases = ["show my reminders", "list active reminders", "what reminders do I have", "view reminders", "check my reminders"]
    for p in list_phrases:
        samples.append({"q": p, "t": [{"tool": "list_reminders", "args": {}}]})

    alarms = [("7:00 AM", "Wake up"), ("6:30 AM", "Morning"), ("8:00 AM", "Work"), ("5:00 PM", "Evening")]
    for t_str, lbl in alarms:
        samples.append({"q": f"set an alarm for {t_str}", "t": [{"tool": "set_alarm", "args": {"time_str": t_str, "message": "Alarm"}}]})
        samples.append({"q": f"wake me up at {t_str}", "t": [{"tool": "set_alarm", "args": {"time_str": t_str, "message": lbl}}]})

    return samples


# =============================================================================
# MODULE 4: LOCAL TOOL SEARCH VS. GOOGLE WEB SEARCH (~350 samples)
# =============================================================================
def generate_module_4() -> List[Dict[str, Any]]:
    samples = []

    # 4.1 Local Files: search_files(pattern, directory)
    file_patterns = [
        ("report", "downloads"), ("invoice", "documents"), ("budget", "desktop"),
        ("resume", "documents"), ("presentation", "desktop"), ("receipt", "downloads"),
        ("taxes", "documents"), ("contract", "documents"), ("*.pdf", "downloads"),
        ("*.py", "projects"), ("notes.txt", "desktop"), ("summary", "documents")
    ]
    for pat, d in file_patterns:
        phrases = [
            f"search for files named {pat} in {d}", f"find files named {pat} in {d}",
            f"search for {pat} files in {d} folder", f"locate {pat} in {d}",
            f"find file {pat} in {d}", f"search {d} for {pat}"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "search_files", "args": {"pattern": pat, "directory": d}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "search_files", "args": {"pattern": pat, "directory": d}}]})

    # 4.2 Local Contacts: search_contacts(query)
    names = ["John", "Sarah", "David", "Alice", "Michael", "Emily", "Robert", "Jessica", "Alex", "Dr. Wilson"]
    for name in names:
        phrases = [
            f"search contact {name}", f"find {name} in my contacts",
            f"search for {name} in my address book", f"look up {name} in contacts",
            f"find contact info for {name}", f"search contacts for {name}"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "search_contacts", "args": {"query": name}}]})

    # 4.3 Music Playlists: search_playlists(query, platform)
    playlists = [
        ("workout", "spotify"), ("lofi study", "spotify"), ("rock classics", "youtube"),
        ("deep focus", "spotify"), ("jazz cafe", "auto"), ("sleep sounds", "spotify")
    ]
    for q_pl, plat in playlists:
        phrases = [
            f"search for a {q_pl} playlist", f"find a {q_pl} playlist on {plat}",
            f"search playlist {q_pl}", f"look up {q_pl} playlist"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "search_playlists", "args": {"query": q_pl, "platform": plat}}]})

    # 4.4 Contrastive Web Search: search_in_browser(query, engine) ONLY for genuine web research
    web_queries = [
        ("latest quantum computing breakthroughs", "google"),
        ("weather in Tokyo next week", "google"),
        ("python 3.12 release notes", "google"),
        ("how does transformer attention work", "google"),
        ("who won the 2026 World Cup", "google"),
        ("best restaurants in Paris", "google"),
        ("apple stock price today", "google"),
        ("what is Moore's law", "google")
    ]
    for w_q, eng in web_queries:
        phrases = [
            f"search google for {w_q}", f"google {w_q}",
            f"search web for {w_q}", f"search the internet for {w_q}",
            f"search in browser for {w_q}"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "search_in_browser", "args": {"query": w_q, "engine": eng}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "search_in_browser", "args": {"query": w_q, "engine": eng}}]})

    return samples


# =============================================================================
# MODULE 5: TELEMETRY, NETWORKING & TAB DIAGNOSTICS (~250 samples)
# =============================================================================
def generate_module_5() -> List[Dict[str, Any]]:
    samples = []

    # 5.1 Ports: check_port_status(port, host)
    ports = [80, 443, 3000, 5432, 8000, 8080, 27017, 6379, 22, 5000]
    for port in ports:
        phrases = [
            f"check if port {port} is open", f"is port {port} listening",
            f"check port status {port}", f"test if port {port} is active",
            f"check port {port} on localhost"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "check_port_status", "args": {"port": port, "host": "127.0.0.1"}}]})
            samples.append({"q": f"friday {p}", "t": [{"tool": "check_port_status", "args": {"port": port, "host": "127.0.0.1"}}]})

    # 5.2 Ping: ping_host(host, count)
    hosts = [("google.com", 4), ("8.8.8.8", 4), ("1.1.1.1", 4), ("github.com", 4), ("cloudflare.com", 4)]
    for host, cnt in hosts:
        phrases = [
            f"ping {host}", f"ping {host} to test connectivity",
            f"ping {host} 4 times", f"check latency to {host}"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "ping_host", "args": {"host": host, "count": cnt}}]})

    # 5.3 Network Interfaces & Status
    net_phrases = [
        ("show my network adapters and IP addresses", "get_network_interfaces"),
        ("what is my IP address", "get_network_interfaces"),
        ("list network interfaces", "get_network_interfaces"),
        ("show my ip configuration", "get_network_interfaces"),
        ("check current wifi status", "get_network_status"),
        ("is wifi connected", "get_network_status"),
        ("check network connection", "get_network_status")
    ]
    for p, tool in net_phrases:
        samples.append({"q": p, "t": [{"tool": tool, "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": tool, "args": {}}]})

    # 5.4 Windows & Browser Tabs
    win_phrases = ["list all open windows on my desktop", "show open windows", "list running window titles", "what windows are open"]
    for p in win_phrases:
        samples.append({"q": p, "t": [{"tool": "list_windows", "args": {}}]})

    tab_phrases = ["list open browser tabs", "show open chrome tabs", "what tabs are open in my browser", "list browser tabs"]
    for p in tab_phrases:
        samples.append({"q": p, "t": [{"tool": "browser_list_tabs", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "browser_list_tabs", "args": {}}]})

    return samples


# =============================================================================
# MODULE 6: ACTION-OVER-CHITCHAT & TYPING DISAMBIGUATION (~200 samples)
# =============================================================================
def generate_module_6() -> List[Dict[str, Any]]:
    samples = []

    # 6.1 Typing Text (even with greetings or quoted pleasantries)
    texts = [
        "Hello World from FRIDAY",
        "Hello team, good morning everyone",
        "Dear Professor Smith,",
        "Thank you for your prompt reply.",
        "Hi Alex, are we still meeting today?",
        "Good morning, here is the updated document.",
        "Sincerely, John Doe",
        "Testing 1 2 3",
        "print('Hello World')",
        "git status"
    ]
    for txt in texts:
        phrases = [
            f"type '{txt}'", f"type '{txt}' into the document",
            f"write '{txt}'", f"type text '{txt}'",
            f"input '{txt}' here", f"please type '{txt}'"
        ]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "type_text", "args": {"text": txt}}]})

    # 6.2 Key Presses
    keys = ["enter", "tab", "esc", "backspace", "space", "up", "down", "left", "right"]
    for k in keys:
        phrases = [f"press {k} key", f"hit {k}", f"press {k}", f"hit the {k} button"]
        for p in phrases:
            samples.append({"q": p, "t": [{"tool": "press_key", "args": {"key": k}}]})

    # 6.3 Desktop Notifications
    notifs = [
        ("Task Complete", "FRIDAY"), ("Build Successful", "CI/CD"),
        ("Backup Finished", "System"), ("Meeting starting in 5 minutes", "Calendar"),
        ("Battery low", "Power"), ("Download complete", "Browser")
    ]
    for msg, title in notifs:
        samples.append({"q": f"send desktop notification saying {msg}", "t": [{"tool": "send_desktop_notification", "args": {"message": msg, "title": title}}]})
        samples.append({"q": f"notify me that {msg}", "t": [{"tool": "send_desktop_notification", "args": {"message": msg, "title": title}}]})

    # 6.4 Lock Workstation
    lock_phrases = ["lock the computer", "lock screen", "lock workstation", "lock pc", "lock the screen now"]
    for p in lock_phrases:
        samples.append({"q": p, "t": [{"tool": "lock_screen", "args": {}}]})
        samples.append({"q": f"friday {p}", "t": [{"tool": "lock_screen", "args": {}}]})

    # 6.5 Genuine Chitchat (Contrastive to typing)
    greetings = [
        ("hello friday", "Hello! How can I assist you today?"),
        ("hey friday how are you", "I'm doing well and ready to help. What would you like to do?"),
        ("good morning friday", "Good morning! Ready for today's tasks."),
        ("thank you friday", "You're very welcome! Let me know if you need anything else."),
        ("who are you", "I am FRIDAY, your desktop AI assistant."),
        ("tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs!")
    ]
    for q, reply in greetings:
        samples.append({"q": q, "t": [{"tool": "chat_response", "args": {"message": reply}}]})

    return samples


# =============================================================================
# MODULE 7: COMPOUND MULTI-INTENT CHAINS ACROSS NEW MODULES (~350 samples)
# =============================================================================
def generate_module_compound() -> List[Dict[str, Any]]:
    compounds = [
        # Media + System
        {"q": "pause the music and mute audio", "t": [{"tool": "media_play_pause", "args": {}}, {"tool": "mute_audio", "args": {"mute": True}}]},
        {"q": "stop music and lock screen", "t": [{"tool": "media_stop", "args": {}}, {"tool": "lock_screen", "args": {}}]},
        {"q": "resume the track and set volume to 70", "t": [{"tool": "media_play_pause", "args": {}}, {"tool": "set_volume", "args": {"level": 70}}]},
        {"q": "skip to next track and turn up volume", "t": [{"tool": "media_next_track", "args": {}}, {"tool": "change_volume", "args": {"delta": 10}}]},
        {"q": "pause music and take a screenshot of monitor 1", "t": [{"tool": "media_play_pause", "args": {}}, {"tool": "capture_monitor_screenshot", "args": {"monitor_index": 1}}]},
        {"q": "mute the sound and take a screenshot of all monitors", "t": [{"tool": "mute_audio", "args": {"mute": True}}, {"tool": "capture_monitor_screenshot", "args": {"all_monitors": True}}]},
        
        # Telemetry + Networking
        {"q": "check if port 8000 is open and ping google.com", "t": [{"tool": "check_port_status", "args": {"port": 8000, "host": "127.0.0.1"}}, {"tool": "ping_host", "args": {"host": "google.com", "count": 4}}]},
        {"q": "ping 8.8.8.8 and show my network interfaces", "t": [{"tool": "ping_host", "args": {"host": "8.8.8.8", "count": 4}}, {"tool": "get_network_interfaces", "args": {}}]},
        {"q": "check current wifi status and list open browser tabs", "t": [{"tool": "get_network_status", "args": {}}, {"tool": "browser_list_tabs", "args": {}}]},
        {"q": "list open windows and show running tabs", "t": [{"tool": "list_windows", "args": {}}, {"tool": "browser_list_tabs", "args": {}}]},
        
        # Vision + Apps
        {"q": "take a screenshot of chrome and open notepad", "t": [{"tool": "capture_window_screenshot", "args": {"window_title": "chrome"}}, {"tool": "launch_app", "args": {"name": "notepad"}}]},
        {"q": "read the text on my screen and write it into notepad", "t": [{"tool": "extract_text_ocr", "args": {}}, {"tool": "launch_app", "args": {"name": "notepad"}}]},
        {"q": "capture monitor 2 and minimize all windows", "t": [{"tool": "capture_monitor_screenshot", "args": {"monitor_index": 2}}, {"tool": "minimize_all_windows", "args": {}}]},
        
        # Scheduler + System
        {"q": "set a timer for 10 minutes and pause the music", "t": [{"tool": "set_timer", "args": {"seconds": 600, "message": "Timer (10 minutes)"}}, {"tool": "media_play_pause", "args": {}}]},
        {"q": "pause timer and mute audio", "t": [{"tool": "pause_timer", "args": {"target": "active"}}, {"tool": "mute_audio", "args": {"mute": True}}]},
        {"q": "cancel active timer and resume music", "t": [{"tool": "cancel_timer", "args": {"target": "active"}}, {"tool": "media_play_pause", "args": {}}]},
        {"q": "remind me to check the oven in 15 minutes and lock the screen", "t": [{"tool": "set_reminder", "args": {"message": "check the oven", "time_str": "in 15 minutes"}}, {"tool": "lock_screen", "args": {}}]},
        
        # Search + Action
        {"q": "search for files named report in downloads and open downloads", "t": [{"tool": "search_files", "args": {"pattern": "report", "directory": "downloads"}}, {"tool": "launch_app", "args": {"name": "downloads"}}]},
        {"q": "find contact Sarah and send desktop notification saying Found Sarah", "t": [{"tool": "search_contacts", "args": {"query": "Sarah"}}, {"tool": "send_desktop_notification", "args": {"message": "Found Sarah", "title": "Contacts"}}]},
        {"q": "search for lofi study playlist on spotify and set volume to 50", "t": [{"tool": "search_playlists", "args": {"query": "lofi study", "platform": "spotify"}}, {"tool": "set_volume", "args": {"level": 50}}]},
    ]
    
    variations = []
    prefixes = ["", "friday ", "hey friday, ", "can you please ", "could you ", "uh friday "]
    for item in compounds:
        for pre in prefixes:
            variations.append({"q": f"{pre}{item['q']}".strip(), "t": item["t"]})
            
    return variations


def augment_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Augment with voice fillers and acoustic disfluencies."""
    augmented = []
    fillers = [
        ("can you ", ""), ("could you please ", ""), ("hey friday ", ""),
        ("um ", ""), ("uh ", ""), ("yo friday ", "")
    ]
    for s in samples:
        augmented.append(s)
        # Randomly apply 1 natural prefix to 30% of samples
        if random.random() < 0.35:
            prefix, _ = random.choice(fillers)
            if not s["q"].lower().startswith(("friday", "hey friday", "could", "can")):
                augmented.append({"q": f"{prefix}{s['q']}".strip(), "t": s["t"]})
    return augmented
# =============================================================================
def load_v4_replay_samples(max_samples: int = 700) -> List[Dict[str, Any]]:
    v4_path = DATASET_DIR / "v4_train.jsonl"
    if not v4_path.exists():
        print(f"[-] Warning: {v4_path} not found! Skipping replay.")
        return []

    lines = v4_path.read_text(encoding="utf-8").strip().split("\n")
    loaded = []
    for line in lines:
        try:
            item = json.loads(line)
            # Reconstruct simple {q, t}
            msgs = item.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            target_tools = json.loads(asst_msg)
            # Filter for high-confidence core reflex items (launch_app, snap_window, volume, brightness)
            primary_tool = target_tools[0].get("tool", "") if target_tools else ""
            if primary_tool in ("launch_app", "close_app", "snap_window", "set_volume", "set_brightness", "toggle_wifi", "toggle_night_light", "toggle_dark_mode"):
                loaded.append({"q": user_msg, "t": target_tools})
        except Exception:
            continue

    random.shuffle(loaded)
    selected = loaded[:max_samples]
    print(f"[+] Loaded {len(selected)} core anchor replay samples from v4_perfect")
    return selected


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
    print("🚀 GENERATING DesktopIntent-v5-Precision DATASET")
    print("=" * 75)

    m1 = generate_module_1()
    print(f"  [+] Module 1 (Media & Track Disambiguation): {len(m1):,} samples")

    m2 = generate_module_2()
    print(f"  [+] Module 2 (Multi-Monitor & Vision):       {len(m2):,} samples")

    m3 = generate_module_3()
    print(f"  [+] Module 3 (Scheduler & Reminders):        {len(m3):,} samples")

    m4 = generate_module_4()
    print(f"  [+] Module 4 (Local Search vs Web Search):   {len(m4):,} samples")

    m5 = generate_module_5()
    print(f"  [+] Module 5 (Telemetry & Diagnostics):      {len(m5):,} samples")

    m6 = generate_module_6()
    print(f"  [+] Module 6 (Action Over Chitchat):         {len(m6):,} samples")

    m7 = generate_module_compound()
    print(f"  [+] Module 7 (Compound Multi-Intent Chains): {len(m7):,} samples")

    raw_modules = m1 + m2 + m3 + m4 + m5 + m6 + m7
    augmented_modules = augment_samples(raw_modules)
    print(f"  [+] Speech & Disfluency Augmentation:        {len(augmented_modules) - len(raw_modules):,} samples")

    replay = load_v4_replay_samples(max_samples=700)

    all_precision_samples = augmented_modules + replay
    random.shuffle(all_precision_samples)

    split_idx = int(len(all_precision_samples) * 0.9)
    train_raw = all_precision_samples[:split_idx]
    val_raw = all_precision_samples[split_idx:]

    print("-" * 75)
    print(f"  Total Master Precision Samples: {len(all_precision_samples):,}")
    print(f"    Train Split:                  {len(train_raw):,}")
    print(f"    Validation Split:             {len(val_raw):,}")
    print("-" * 75)

    # Convert to ChatML
    train_chatml = [format_chatml(s) for s in train_raw]
    val_chatml = [format_chatml(s) for s in val_raw]

    # Save to dataset/
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATASET_DIR / "v5_precision_train.jsonl"
    val_path = DATASET_DIR / "v5_precision_val.jsonl"

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
