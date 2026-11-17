"""
Generator for Conversational & Idiomatic Precision Extension.
Directly targets the 7 semantic and idiomatic failure modes identified in the diagnostic benchmark:
1. Indirect physical discomfort (eye strain, blinding screen, ear ringing, quiet room)
2. Figurative clock & timer expressions (freeze timer, ice countdown, kill alarm)
3. Colloquial power queries (battery juice, sleep PC, bed time, lock it up)
4. Hardware monitor queries (how many monitors, second display plugged in)
5. Theme vs Night Light disambiguation (night mode, dark theme, warm colors)
6. Canonical time & date queries (what time is it, what date is today)
7. Canonical network diagnostics (port status, ping, interfaces)
"""

import json
import random
from pathlib import Path

random.seed(42)

SYSTEM_PROMPT = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. For every user input, "
    "you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

POLITE_PREFIXES = [
    "", "", "", "please ", "can you ", "could you ", "would you mind ", "friday, ", "hey friday, ",
    "um, ", "uh, ", "hold on, ", "listen, ", "quick favor, ", "can you please ", "hey friday please "
]

POLITE_SUFFIXES = [
    "", "", "", " please", " thanks", " for me", " right now", " asap", " for a second", " please thanks"
]

def make_sample(query: str, tools: list) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query.strip()},
            {"role": "assistant", "content": json.dumps(tools, separators=(",", ":"))}
        ]
    }

def generate_idiomatic_samples():
    samples = []

    # 1. Indirect Physical Discomfort -> Dim Screen / Brightness (80 samples)
    dim_templates = [
        ("my eyes are hurting, make the screen a bit dimmer", [{"tool": "decrease_brightness", "args": {"amount": 20}}]),
        ("my eyes hurt, dim the display", [{"tool": "decrease_brightness", "args": {"amount": 20}}]),
        ("my eyes are strained from staring at the screen, lower brightness", [{"tool": "decrease_brightness", "args": {"amount": 20}}]),
        ("it's blinding in here, dim the display", [{"tool": "decrease_brightness", "args": {"amount": 25}}]),
        ("it's way too bright in this room, turn down display brightness", [{"tool": "decrease_brightness", "args": {"amount": 25}}]),
        ("the screen is blinding me, tone down the brightness", [{"tool": "decrease_brightness", "args": {"amount": 30}}]),
        ("the monitor is burning my retinas, dim it down", [{"tool": "decrease_brightness", "args": {"amount": 30}}]),
        ("too much screen glare, make it dimmer", [{"tool": "decrease_brightness", "args": {"amount": 20}}]),
        ("make the screen darker, my head is aching", [{"tool": "decrease_brightness", "args": {"amount": 20}}]),
        ("dim the screen to 30%, it's hurting my eyes", [{"tool": "set_brightness", "args": {"level": 30}}]),
        ("set screen brightness to 25%, too bright for night", [{"tool": "set_brightness", "args": {"level": 25}}]),
        ("drop brightness to 20 percent", [{"tool": "set_brightness", "args": {"level": 20}}]),
    ]
    for text, tool_call in dim_templates:
        for _ in range(7):
            pre = random.choice(POLITE_PREFIXES)
            suf = random.choice(POLITE_SUFFIXES)
            samples.append(make_sample(f"{pre}{text}{suf}", tool_call))

    # Brighten Screen (35 samples)
    bright_templates = [
        ("can't see anything, it's way too dark", [{"tool": "increase_brightness", "args": {"amount": 25}}]),
        ("it's too dark to read the screen, brighten it up", [{"tool": "increase_brightness", "args": {"amount": 25}}]),
        ("sunlight is hitting my desk, make display brighter", [{"tool": "increase_brightness", "args": {"amount": 25}}]),
        ("crank the brightness up to 80%", [{"tool": "set_brightness", "args": {"level": 80}}]),
        ("boost display brightness by 20%", [{"tool": "increase_brightness", "args": {"amount": 20}}]),
    ]
    for text, tool_call in bright_templates:
        for _ in range(7):
            pre = random.choice(POLITE_PREFIXES)
            suf = random.choice(POLITE_SUFFIXES)
            samples.append(make_sample(f"{pre}{text}{suf}", tool_call))

    # 2. Indirect Audio Discomfort -> Silence / Mute / Volume (80 samples)
    quiet_templates = [
        ("my ears are ringing, quiet down please", [{"tool": "decrease_volume", "args": {"amount": 20}}]),
        ("it's deafening in here, silence this PC", [{"tool": "mute_audio", "args": {}}]),
        ("shh, mute the computer immediately", [{"tool": "mute_audio", "args": {}}]),
        ("kill the audio, on a phone call", [{"tool": "mute_audio", "args": {}}]),
        ("shut up the audio, taking a call", [{"tool": "mute_audio", "args": {}}]),
        ("silence the room, turn off the sound", [{"tool": "mute_audio", "args": {}}]),
        ("lower the noise, baby is sleeping", [{"tool": "decrease_volume", "args": {"amount": 25}}]),
        ("way too loud, turn it down a notch", [{"tool": "decrease_volume", "args": {"amount": 15}}]),
        ("turn down the volume, headache incoming", [{"tool": "decrease_volume", "args": {"amount": 20}}]),
        ("quiet down the speakers", [{"tool": "decrease_volume", "args": {"amount": 20}}]),
    ]
    for text, tool_call in quiet_templates:
        for _ in range(8):
            pre = random.choice(POLITE_PREFIXES)
            suf = random.choice(POLITE_SUFFIXES)
            samples.append(make_sample(f"{pre}{text}{suf}", tool_call))

    loud_templates = [
        ("can't hear what they're saying, speak up", [{"tool": "increase_volume", "args": {"amount": 25}}]),
        ("audio is way too quiet, pump up the volume", [{"tool": "increase_volume", "args": {"amount": 25}}]),
        ("turn the sound up so I can hear in the other room", [{"tool": "increase_volume", "args": {"amount": 30}}]),
        ("boost speaker volume by 20%", [{"tool": "increase_volume", "args": {"amount": 20}}]),
    ]
    for text, tool_call in loud_templates:
        for _ in range(6):
            pre = random.choice(POLITE_PREFIXES)
            suf = random.choice(POLITE_SUFFIXES)
            samples.append(make_sample(f"{pre}{text}{suf}", tool_call))

    # 3. Figurative Clock & Timer Expressions (80 samples)
    timer_templates = [
        ("hold on, freeze my active timer for a moment", [{"tool": "pause_timer", "args": {"target": "active"}}]),
        ("freeze the timer", [{"tool": "pause_timer", "args": {"target": "active"}}]),
        ("put the countdown on ice", [{"tool": "pause_timer", "args": {"target": "active"}}]),
        ("halt the timer right now", [{"tool": "pause_timer", "args": {"target": "active"}}]),
        ("hold the clock for a second", [{"tool": "pause_timer", "args": {"target": "active"}}]),
        ("unfreeze the timer", [{"tool": "resume_timer", "args": {"target": "active"}}]),
        ("thaw the timer and keep it running", [{"tool": "resume_timer", "args": {"target": "active"}}]),
        ("keep the countdown rolling", [{"tool": "resume_timer", "args": {"target": "active"}}]),
        ("let the timer clock tick again", [{"tool": "resume_timer", "args": {"target": "active"}}]),
        ("kill that annoying alarm", [{"tool": "cancel_alarm", "args": {}}]),
        ("turn off the beeping alarm", [{"tool": "cancel_alarm", "args": {}}]),
        ("silence the morning alarm", [{"tool": "cancel_alarm", "args": {}}]),
        ("scrap the pasta timer", [{"tool": "cancel_timer", "args": {"target": "pasta"}}]),
        ("dismiss the active timer", [{"tool": "cancel_timer", "args": {"target": "active"}}]),
        ("cancel all running timers", [{"tool": "cancel_timer", "args": {"target": "all"}}]),
    ]
    for text, tool_call in timer_templates:
        for _ in range(6):
            pre = random.choice(POLITE_PREFIXES)
            suf = random.choice(POLITE_SUFFIXES)
            samples.append(make_sample(f"{pre}{text}{suf}", tool_call))

    # 4. Battery / Juice / Power Queries (60 samples)
    battery_templates = [
        ("how much juice is left in this machine?", [{"tool": "get_battery_status", "args": {}}]),
        ("how much battery juice is remaining?", [{"tool": "get_battery_status", "args": {}}]),
        ("check my battery percentage", [{"tool": "get_battery_status", "args": {}}]),
        ("what is my current battery level?", [{"tool": "get_battery_status", "args": {}}]),
        ("am I running on battery or plugged into the wall?", [{"tool": "get_battery_status", "args": {}}]),
        ("is this laptop charging right now?", [{"tool": "get_battery_status", "args": {}}]),
        ("check power status and battery", [{"tool": "get_battery_status", "args": {}}]),
        ("do I need to plug in my charger?", [{"tool": "get_battery_status", "args": {}}]),
        ("put this computer to bed", [{"tool": "sleep_system", "args": {}}]),
        ("put the PC to sleep", [{"tool": "sleep_system", "args": {}}]),
        ("snooze the workstation", [{"tool": "sleep_system", "args": {}}]),
        ("stepping away for a coffee, lock it up", [{"tool": "lock_pc", "args": {}}]),
        ("lock my workstation screen", [{"tool": "lock_pc", "args": {}}]),
    ]
    for text, tool_call in battery_templates:
        for _ in range(5):
            pre = random.choice(POLITE_PREFIXES)
            suf = random.choice(POLITE_SUFFIXES)
            samples.append(make_sample(f"{pre}{text}{suf}", tool_call))

    # 5. Hardware Monitor Count & Setup (50 samples)
    monitor_templates = [
        ("how many monitors do I have connected to this machine?", [{"tool": "get_monitors_info", "args": {}}]),
        ("how many screens are connected?", [{"tool": "get_monitors_info", "args": {}}]),
        ("do I have a second screen plugged in?", [{"tool": "get_monitors_info", "args": {}}]),
        ("how many displays are detected by windows?", [{"tool": "get_monitors_info", "args": {}}]),
        ("check connected monitors and resolutions", [{"tool": "get_monitors_info", "args": {}}]),
        ("list all displays attached to this PC", [{"tool": "get_monitors_info", "args": {}}]),
        ("what is my display resolution setup?", [{"tool": "get_monitors_info", "args": {}}]),
        ("get monitor count and primary display info", [{"tool": "get_monitors_info", "args": {}}]),
    ]
    for text, tool_call in monitor_templates:
        for _ in range(6):
            pre = random.choice(POLITE_PREFIXES)
            suf = random.choice(POLITE_SUFFIXES)
            samples.append(make_sample(f"{pre}{text}{suf}", tool_call))

    # 6. Theme vs Night Light Disambiguation (60 samples)
    theme_templates = [
        ("can you change Windows to night mode please?", [{"tool": "set_system_theme", "args": {"theme": "dark"}}]),
        ("switch Windows to dark mode", [{"tool": "set_system_theme", "args": {"theme": "dark"}}]),
        ("enable dark theme across the system", [{"tool": "set_system_theme", "args": {"theme": "dark"}}]),
        ("make Windows dark theme", [{"tool": "set_system_theme", "args": {"theme": "dark"}}]),
        ("turn on dark mode", [{"tool": "set_system_theme", "args": {"theme": "dark"}}]),
        ("change to light mode", [{"tool": "set_system_theme", "args": {"theme": "light"}}]),
        ("switch to day mode / light theme", [{"tool": "set_system_theme", "args": {"theme": "light"}}]),
        ("turn on night light", [{"tool": "toggle_night_light", "args": {"enable": True}}]),
        ("warm up display colors, it's late", [{"tool": "toggle_night_light", "args": {"enable": True}}]),
        ("enable blue light filter", [{"tool": "toggle_night_light", "args": {"enable": True}}]),
        ("turn off night light", [{"tool": "toggle_night_light", "args": {"enable": False}}]),
        ("disable blue light filter", [{"tool": "toggle_night_light", "args": {"enable": False}}]),
    ]
    for text, tool_call in theme_templates:
        for _ in range(5):
            pre = random.choice(POLITE_PREFIXES)
            suf = random.choice(POLITE_SUFFIXES)
            samples.append(make_sample(f"{pre}{text}{suf}", tool_call))

    # 7. Canonical Time, Date & Network Queries (70 samples)
    time_templates = [
        ("what time is it", [{"tool": "get_current_time", "args": {}}]),
        ("what is the current time?", [{"tool": "get_current_time", "args": {}}]),
        ("tell me what time it is right now", [{"tool": "get_current_time", "args": {}}]),
        ("what date is today", [{"tool": "get_current_time", "args": {}}]),
        ("what's today's date?", [{"tool": "get_current_time", "args": {}}]),
        ("what day of the week is it?", [{"tool": "get_current_time", "args": {}}]),
        ("check if port 8080 is open", [{"tool": "check_port_status", "args": {"port": 8080, "host": "127.0.0.1"}}]),
        ("is port 3000 listening on localhost?", [{"tool": "check_port_status", "args": {"port": 3000, "host": "127.0.0.1"}}]),
        ("check port 5432 status", [{"tool": "check_port_status", "args": {"port": 5432, "host": "127.0.0.1"}}]),
        ("ping google.com to test internet", [{"tool": "ping_host", "args": {"host": "google.com"}}]),
        ("test connectivity to 8.8.8.8", [{"tool": "ping_host", "args": {"host": "8.8.8.8"}}]),
        ("show my network adapters and IP", [{"tool": "get_network_interfaces", "args": {}}]),
    ]
    for text, tool_call in time_templates:
        for _ in range(6):
            pre = random.choice(POLITE_PREFIXES)
            suf = random.choice(POLITE_SUFFIXES)
            samples.append(make_sample(f"{pre}{text}{suf}", tool_call))

    random.shuffle(samples)
    return samples

if __name__ == "__main__":
    out_dir = Path("d:/Projects/desktop-llm/dataset")
    samples = generate_idiomatic_samples()
    print(f"Generated {len(samples)} Conversational & Idiomatic Precision samples.")

    # Split 90% train / 10% validation
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_file = out_dir / "v6_idiomatic_train.jsonl"
    val_file = out_dir / "v6_idiomatic_val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    print(f"[+] Saved {len(train_samples)} train samples to {train_file}")
    print(f"[+] Saved {len(val_samples)} val samples to {val_file}")
