"""FRIDAY AI 2.0 Comprehensive Dataset Generator.
Extracts tools, schemas, voice patterns, and STT typo distributions from:
E:\\F Drive Backup\\FRIDAY-AI-Assistant

Generates 15,000 high-variance training/validation pairs in ChatML format.
Categories covered:
1. OS Primitives & Toggles (Audio, Brightness, Power, Night Light, Dark Mode, Wi-Fi)
2. Multi-Monitor Windows & Layouts (DPI snapping, quadrants, presets)
3. Everyday Workflows & Navigation (Browser search, tabs, shortcuts, docx, WhatsApp)
4. Speech-to-Text Noise & Fillers (Whisper typos, polite prefixes, hesitation)
5. Multi-Intent Chaining & Coreference (sequential pipelines, pronoun resolution)
6. Cloud Escalation & Safe Boundaries (delegating complex coding/reasoning to Groq/27B)
"""

import json
import random
import re
from pathlib import Path

random.seed(42)

SYSTEM_PROMPT = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. "
    "For every user input, you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

APPS = [
    ("chrome", "Chrome"),
    ("brave", "Brave"),
    ("edge", "Microsoft Edge"),
    ("spotify", "Spotify"),
    ("notepad", "Notepad"),
    ("code", "VS Code"),
    ("cursor", "Cursor"),
    ("terminal", "Windows Terminal"),
    ("whatsapp", "WhatsApp"),
    ("telegram", "Telegram"),
    ("discord", "Discord"),
    ("calculator", "Calculator"),
    ("settings", "Settings"),
    ("explorer", "File Explorer"),
    ("taskmgr", "Task Manager"),
    ("slack", "Slack"),
]

SEARCH_ENGINES = ["google", "youtube", "github", "duckduckgo", "bing", "wikipedia", "reddit", "x"]

POSITIONS = [
    "left", "right", "top", "bottom", "center", "fullscreen",
    "top_left", "top_right", "bottom_left", "bottom_right"
]

SHORTCUTS = [
    ("copy", "copy that", "Ctrl+C"),
    ("paste", "paste here", "Ctrl+V"),
    ("select_all", "select everything", "Ctrl+A"),
    ("save", "save file", "Ctrl+S"),
    ("undo", "undo that", "Ctrl+Z"),
    ("redo", "redo action", "Ctrl+Y"),
    ("find", "find on page", "Ctrl+F"),
    ("show_desktop", "show desktop", "Win+D"),
    ("lock_screen", "lock screen", "Win+L"),
    ("file_explorer", "open file explorer", "Win+E"),
    ("task_manager", "open task manager", "Ctrl+Shift+Esc"),
]

POLITE_PREFIXES = [
    "can you please ", "could you please ", "would you mind ", "i need you to ",
    "i want you to ", "can u ", "hey friday ", "friday please ", "go ahead and ",
    "please ", "alright so ", "ok so ", "would you kindly ", "just "
]

FILLERS = [" uh ", " um ", " like ", " you know ", " actually ", " "]

STT_TYPOS = {
    "screenshot": ["screnshot", "screenshoot", "screen shot"],
    "volume": ["volumme", "volum"],
    "brightness": ["brightnes", "brightnes s"],
    "calculator": ["calcuator", "calc"],
    "notepad": ["notepadd", "notpad"],
    "whatsapp": ["whats app", "whatsap"],
    "search": ["serch", "seach"],
    "shutdown": ["shutdwon", "shut down"],
}


def apply_voice_noise(text: str) -> str:
    """Injects realistic speech-to-text noise, fillers, or polite wrappers."""
    # 50% chance of polite prefix
    if random.random() < 0.5:
        text = random.choice(POLITE_PREFIXES) + text

    # 30% chance of hesitation filler
    if random.random() < 0.3:
        words = text.split()
        if len(words) > 2:
            idx = random.randint(1, len(words) - 1)
            words.insert(idx, random.choice(FILLERS).strip())
            text = " ".join(words)

    # 20% chance of Whisper phonetic typo
    if random.random() < 0.2:
        for proper, typos in STT_TYPOS.items():
            if proper in text:
                text = text.replace(proper, random.choice(typos), 1)
                break

    return re.sub(r"\s+", " ", text).strip()


# --- Domain Generators ---

def gen_os_system():
    """Generates audio, brightness, battery, and system power actions."""
    lvl = random.choice([0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100])
    delta = random.choice([5, 10, 15, 20, 25])
    templates = [
        ("mute audio", [{"tool": "mute_audio", "args": {"mute": True}}]),
        ("unmute audio", [{"tool": "mute_audio", "args": {"mute": False}}]),
        ("silence the computer", [{"tool": "mute_audio", "args": {"mute": True}}]),
        ("turn the sound back on", [{"tool": "mute_audio", "args": {"mute": False}}]),
        ("shh", [{"tool": "mute_audio", "args": {"mute": True}}]),
        ("shut up", [{"tool": "mute_audio", "args": {"mute": True}}]),
        (f"set volume to {lvl}%", [{"tool": "set_volume", "args": {"level": lvl}}]),
        (f"volume {lvl}", [{"tool": "set_volume", "args": {"level": lvl}}]),
        (f"increase volume by {delta}%", [{"tool": "change_volume", "args": {"delta": delta}}]),
        (f"turn it down {delta}%", [{"tool": "change_volume", "args": {"delta": -delta}}]),
        ("too loud turn it down", [{"tool": "change_volume", "args": {"delta": -15}}]),
        ("can't hear anything turn it up", [{"tool": "change_volume", "args": {"delta": 20}}]),
        (f"set brightness to {lvl}%", [{"tool": "set_brightness", "args": {"level": lvl}}]),
        (f"screen brightness {lvl}", [{"tool": "set_brightness", "args": {"level": lvl}}]),
        ("dim screen", [{"tool": "change_brightness", "args": {"delta": -20}}]),
        ("screen is too bright", [{"tool": "change_brightness", "args": {"delta": -25}}]),
        ("too dark make it brighter", [{"tool": "change_brightness", "args": {"delta": 20}}]),
        ("take a screenshot", [{"tool": "take_screenshot", "args": {}}]),
        ("capture the screen", [{"tool": "take_screenshot", "args": {}}]),
        ("how much battery is left?", [{"tool": "get_battery_status", "args": {}}]),
        ("check system stats and cpu usage", [{"tool": "get_system_info", "args": {}}]),
        ("lock the computer", [{"tool": "lock_screen", "args": {}}]),
        ("lock workstation", [{"tool": "lock_screen", "args": {}}]),
        ("put the pc to sleep", [{"tool": "sleep_system", "args": {}}]),
        ("enable night light", [{"tool": "toggle_night_light", "args": {"enabled": True}}]),
        ("turn off night light", [{"tool": "toggle_night_light", "args": {"enabled": False}}]),
        ("switch to dark mode", [{"tool": "toggle_dark_mode", "args": {"dark": True}}]),
        ("switch to light theme", [{"tool": "toggle_dark_mode", "args": {"dark": False}}]),
        ("turn off wifi", [{"tool": "toggle_wifi", "args": {"enabled": False}}]),
        ("enable bluetooth", [{"tool": "toggle_bluetooth", "args": {"enabled": True}}]),
    ]
    return random.choice(templates)


def gen_window_display():
    """Generates multi-monitor window management and snap operations."""
    app_key, app_name = random.choice(APPS)
    pos = random.choice(POSITIONS)
    mon = random.choice([1, 2])
    templates = [
        (f"snap {app_name} to the {pos}", [{"tool": "snap_window", "args": {"app": app_key, "position": pos}}]),
        (f"snap {app_name} to {pos} on monitor {mon}", [{"tool": "snap_window", "args": {"app": app_key, "position": pos, "monitor_index": mon}}]),
        (f"move {app_name} to monitor {mon}", [{"tool": "snap_window", "args": {"app": app_key, "position": "center", "monitor_index": mon}}]),
        (f"minimize {app_name}", [{"tool": "manage_window", "args": {"app": app_key, "action": "minimize"}}]),
        (f"maximize {app_name}", [{"tool": "manage_window", "args": {"app": app_key, "action": "maximize"}}]),
        (f"focus {app_name}", [{"tool": "manage_window", "args": {"app": app_key, "action": "focus"}}]),
        (f"bring {app_name} to front", [{"tool": "manage_window", "args": {"app": app_key, "action": "focus"}}]),
        (f"close {app_name} window", [{"tool": "manage_window", "args": {"app": app_key, "action": "close"}}]),
        ("show desktop", [{"tool": "minimize_all_windows", "args": {}}]),
        ("minimize all windows", [{"tool": "minimize_all_windows", "args": {}}]),
        ("restore all windows", [{"tool": "restore_all_windows", "args": {}}]),
        ("set coding workspace layout", [{"tool": "set_workspace_preset", "args": {"preset": "coding"}}]),
        ("switch to research layout", [{"tool": "set_workspace_preset", "args": {"preset": "research"}}]),
        ("dual screen coding preset", [{"tool": "set_workspace_preset", "args": {"preset": "dual_screen_coding"}}]),
    ]
    return random.choice(templates)


def gen_browser_search():
    """Generates browser search, tab navigation, and web queries."""
    engine = random.choice(SEARCH_ENGINES)
    queries = [
        "python fastapi async tutorial", "rtx 2050 cuda benchmark", "lo-fi beats to relax to",
        "latest artificial intelligence news", "three.js holographic visualizer", "github trending repositories",
        "weather in New York this weekend", "apple m4 chip vs nvidia rtx"
    ]
    q = random.choice(queries)
    tab_actions = ["new", "close", "next", "prev", "refresh", "reopen"]
    t_act = random.choice(tab_actions)

    templates = [
        (f"search {engine} for {q}", [{"tool": "search_in_browser", "args": {"query": q, "engine": engine}}]),
        (f"look up {q} on {engine}", [{"tool": "search_in_browser", "args": {"query": q, "engine": engine}}]),
        (f"google {q}", [{"tool": "search_in_browser", "args": {"query": q, "engine": "google"}}]),
        (f"open {engine} and search for {q}", [{"tool": "search_in_browser", "args": {"query": q, "engine": engine}}]),
        (f"{t_act} browser tab", [{"tool": "browser_tab_action", "args": {"action": t_act}}]),
        ("close current tab", [{"tool": "browser_tab_action", "args": {"action": "close"}}]),
        ("switch to next tab", [{"tool": "browser_tab_action", "args": {"action": "next"}}]),
        ("refresh the webpage", [{"tool": "browser_tab_action", "args": {"action": "refresh"}}]),
        ("reopen closed tab", [{"tool": "browser_tab_action", "args": {"action": "reopen"}}]),
    ]
    return random.choice(templates)


def gen_apps_workflows():
    """Generates app launching, killing, shortcuts, Word documents, and messaging."""
    app_key, app_name = random.choice(APPS)
    action_key, phrase, _ = random.choice(SHORTCUTS)
    contact = random.choice(["Alex", "Mom", "Sarah", "John", "David"])
    msg = random.choice(["I am on my way", "Meeting starts in 10 minutes", "Please check the pull request", "Call me when you are free"])

    templates = [
        (f"open {app_name}", [{"tool": "launch_app", "args": {"name": app_key}}]),
        (f"launch {app_name}", [{"tool": "launch_app", "args": {"name": app_key}}]),
        (f"start {app_name}", [{"tool": "launch_app", "args": {"name": app_key}}]),
        (f"close {app_name}", [{"tool": "close_app", "args": {"name": app_key}}]),
        (f"kill {app_name}", [{"tool": "close_app", "args": {"name": app_key}}]),
        (phrase, [{"tool": "press_shortcut", "args": {"action": action_key}}]),
        (f"create a word document named 'Project Spec'", [{"tool": "create_word_document", "args": {"title": "Project Spec", "content_markdown": "# Project Spec\n\nGenerated by FRIDAY."}}]),
        (f"paste my clipboard into active app", [{"tool": "press_shortcut", "args": {"action": "paste"}}]),
        (f"send a whatsapp message to {contact} saying {msg}", [{"tool": "send_whatsapp_message", "args": {"contact": contact, "message": msg}}]),
        (f"telegram {contact} that {msg}", [{"tool": "send_telegram_message", "args": {"contact": contact, "message": msg}}]),
        (f"draft an email to {contact.lower()}@example.com about Project Update", [{"tool": "compose_email", "args": {"recipient": f"{contact.lower()}@example.com", "subject": "Project Update", "body": "Hi, here is the update."}}]),
    ]
    return random.choice(templates)


def gen_scheduler_productivity():
    """Timers, reminders, notes, and calculations."""
    mins = random.choice([2, 5, 10, 15, 20, 25, 30, 45, 60])
    n1, n2 = random.randint(10, 500), random.randint(2, 50)
    op = random.choice(["+", "-", "*", "/"])
    expr = f"{n1} {op} {n2}"

    templates = [
        (f"set a timer for {mins} minutes", [{"tool": "create_timer", "args": {"duration_seconds": mins * 60, "label": "Timer"}}]),
        (f"timer {mins}m", [{"tool": "create_timer", "args": {"duration_seconds": mins * 60, "label": "Timer"}}]),
        (f"remind me in {mins} minutes to stand up and stretch", [{"tool": "create_reminder", "args": {"message": "stand up and stretch", "minutes_from_now": mins}}]),
        (f"calculate {expr}", [{"tool": "calculate", "args": {"expression": expr}}]),
        (f"what is {n1} times {n2}?", [{"tool": "calculate", "args": {"expression": f"{n1} * {n2}"}}]),
        (f"what is 18% of {n1}?", [{"tool": "calculate", "args": {"expression": f"18% of {n1}"}}]),
        ("take a note meeting with team tomorrow at 10am", [{"tool": "take_note", "args": {"text": "meeting with team tomorrow at 10am"}}]),
    ]
    return random.choice(templates)


def gen_single():
    fn = random.choice([
        gen_os_system,
        gen_window_display,
        gen_browser_search,
        gen_apps_workflows,
        gen_scheduler_productivity,
    ])
    p, a = fn()
    return p, a


def gen_multi():
    """Generates 2 to 3 chained actions with dependent flow and pronoun coreference."""
    k = random.choice([2, 2, 2, 3])
    generators = [
        gen_os_system,
        gen_window_display,
        gen_browser_search,
        gen_apps_workflows,
        gen_scheduler_productivity,
    ]
    random.shuffle(generators)

    parts = []
    actions = []
    for i in range(k):
        p, a = generators[i]()
        parts.append(p)
        actions.extend(a)

    connectors = [" and ", ", then ", " and also ", ", ", " then "]
    prompt = parts[0]
    for p in parts[1:]:
        prompt += random.choice(connectors) + p

    # Pronoun coreference: "launch X and snap it left"
    if len(actions) >= 2 and actions[0]["tool"] == "launch_app" and actions[1]["tool"] == "snap_window":
        app = actions[0]["args"]["name"]
        pos = actions[1]["args"].get("position", "left")
        actions[1]["args"]["app"] = app
        prompt = f"open {app} and snap it {pos}"

    # Pronoun coreference: "open X and minimize it"
    if len(actions) >= 2 and actions[0]["tool"] == "launch_app" and actions[1]["tool"] == "manage_window":
        app = actions[0]["args"]["name"]
        act = actions[1]["args"].get("action", "minimize")
        actions[1]["args"]["app"] = app
        prompt = f"open {app} and {act} it"

    return prompt, actions


def gen_scenarios():
    """Implicit compound situations."""
    scenarios = [
        ("I'm going to sleep now", [
            {"tool": "mute_audio", "args": {"mute": True}},
            {"tool": "set_brightness", "args": {"level": 10}},
            {"tool": "toggle_night_light", "args": {"enabled": True}},
            {"tool": "lock_screen", "args": {}},
        ]),
        ("bedtime mode", [
            {"tool": "mute_audio", "args": {"mute": True}},
            {"tool": "set_brightness", "args": {"level": 10}},
            {"tool": "toggle_night_light", "args": {"enabled": True}},
        ]),
        ("focus mode", [
            {"tool": "mute_audio", "args": {"mute": True}},
            {"tool": "close_app", "args": {"name": "discord"}},
            {"tool": "close_app", "args": {"name": "slack"}},
            {"tool": "launch_app", "args": {"name": "code"}},
        ]),
        ("presentation mode", [
            {"tool": "set_brightness", "args": {"level": 100}},
            {"tool": "mute_audio", "args": {"mute": True}},
            {"tool": "manage_window", "args": {"app": "chrome", "action": "maximize"}},
        ]),
        ("clean desktop", [
            {"tool": "minimize_all_windows", "args": {}},
            {"tool": "press_shortcut", "args": {"action": "show_desktop"}},
        ]),
        ("stepping away for a bit", [
            {"tool": "mute_audio", "args": {"mute": True}},
            {"tool": "lock_screen", "args": {}},
        ]),
    ]
    return random.choice(scenarios)


def gen_delegation_and_fallback():
    """Conversational safety net and escalation to Groq / DeepSeek-R1 / Qwen 27B."""
    delegations = [
        ("write a python script to parse csv files and calculate averages",
         [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "write a python script to parse csv files and calculate averages"}}]),
        ("explain the difference between async and multithreading in Python",
         [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "explain the difference between async and multithreading in Python"}}]),
        ("help me debug this fastapi websocket connection error",
         [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "help me debug this fastapi websocket connection error"}}]),
        ("write a professional email to my boss asking for a salary review",
         [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "write a professional email to my boss asking for a salary review"}}]),
        ("what are the key architectural principles of transformer models?",
         [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "what are the key architectural principles of transformer models?"}}]),
        ("hi friday", [{"tool": "chat_response", "args": {"message": "Hello! Systems online and ready."}}]),
        ("hey friday how are you?", [{"tool": "chat_response", "args": {"message": "I'm running optimally. What can I do for you?"}}]),
        ("who are you?", [{"tool": "chat_response", "args": {"message": "I am FRIDAY, your autonomous desktop assistant."}}]),
        ("what can you do?", [{"tool": "chat_response", "args": {"message": "I manage your desktop, audio, windows, browsers, messaging, timers, and workflows."}}]),
        ("thank you friday", [{"tool": "chat_response", "args": {"message": "You're welcome!"}}]),
        ("good morning", [{"tool": "chat_response", "args": {"message": "Good morning! Ready for today's tasks."}}]),
    ]
    return random.choice(delegations)


def build_friday_dataset(total_samples: int = 15000):
    dataset = []

    # Distribution:
    # 30% Single-intent (4500)
    # 30% Multi-intent compound (4500)
    # 23% Voice noise & STT perturbation (3450)
    # 10% Implicit compound / scenarios (1500)
    # 7% Delegation & fallback (1050)
    n_single = int(total_samples * 0.30)
    n_multi = int(total_samples * 0.30)
    n_voice = int(total_samples * 0.23)
    n_scenarios = int(total_samples * 0.10)
    n_delegation = total_samples - (n_single + n_multi + n_voice + n_scenarios)

    print(f"[*] Synthesizing {total_samples} FRIDAY training samples:")
    print(f"    • Single-Intent: {n_single}")
    print(f"    • Multi-Intent: {n_multi}")
    print(f"    • Voice / STT Noisy: {n_voice}")
    print(f"    • Implicit Scenarios: {n_scenarios}")
    print(f"    • Delegation & Fallback: {n_delegation}")

    for _ in range(n_single):
        p, a = gen_single()
        dataset.append((p, a))

    for _ in range(n_multi):
        p, a = gen_multi()
        dataset.append((p, a))

    for _ in range(n_voice):
        p, a = gen_single() if random.random() < 0.5 else gen_multi()
        noisy_p = apply_voice_noise(p)
        dataset.append((noisy_p, a))

    for _ in range(n_scenarios):
        p, a = gen_scenarios()
        dataset.append((p, a))

    for _ in range(n_delegation):
        p, a = gen_delegation_and_fallback()
        dataset.append((p, a))

    random.shuffle(dataset)

    # Format into Qwen ChatML
    formatted = []
    for prompt, acts in dataset:
        formatted.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(acts, ensure_ascii=False)}
            ]
        })

    return formatted


def main():
    out_dir = Path(__file__).parent
    all_data = build_friday_dataset(15000)

    # 90/10 Split -> 13,500 Train / 1,500 Val
    split_idx = int(len(all_data) * 0.90)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    train_path = out_dir / "friday_train.jsonl"
    val_path = out_dir / "friday_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[+] Successfully wrote {len(train_data)} training samples -> {train_path}")
    print(f"[+] Successfully wrote {len(val_data)} validation samples -> {val_path}")


if __name__ == "__main__":
    main()
