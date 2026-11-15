"""Generates the 10/10 v4 Dataset for FRIDAY AI Assistant (DesktopIntent-v4).

Implements the 5 Core Pillars:
1. Dynamic Zero-Shot In-Context Tool Schemas (Hybrid weights + runtime prompt tool injection)
2. Acoustic Noise & Disfluency Hardening (Whisper phonetic confusions, false starts, filler words)
3. Stateful Anaphora & Coreference Resolution (2-turn dialog & [Prior Action: ...] contexts)
4. Built-in Micro-Personality (Zero-latency local chit-chat via chat_response)
5. Calibrated Ambiguity Gating (clarification_required on underspecified actions)
PLUS all 55 standard FRIDAY MCP tools and cloud reasoning delegation boundaries.
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

# =============================================================================
# PILLAR 1: DYNAMIC IN-CONTEXT RUNTIME TOOLS
# =============================================================================
DYNAMIC_TOOL_SETS = [
    {
        "tools": [
            {"name": "obs_record", "description": "Control OBS screen recording", "parameters": ["action"]},
            {"name": "spotify_search", "description": "Search tracks on Spotify", "parameters": ["query"]}
        ],
        "cases": [
            {"q": "start recording my screen in obs", "t": [{"tool": "obs_record", "args": {"action": "start"}}]},
            {"q": "stop obs recording", "t": [{"tool": "obs_record", "args": {"action": "stop"}}]},
            {"q": "pause recording in obs", "t": [{"tool": "obs_record", "args": {"action": "pause"}}]},
            {"q": "search for Daft Punk on Spotify", "t": [{"tool": "spotify_search", "args": {"query": "Daft Punk"}}]},
            {"q": "find Bohemian Rhapsody on spotify", "t": [{"tool": "spotify_search", "args": {"query": "Bohemian Rhapsody"}}]},
            # Default tool with dynamic tools declared:
            {"q": "mute audio", "t": [{"tool": "mute_audio", "args": {"mute": True}}]},
            {"q": "open notepad", "t": [{"tool": "launch_app", "args": {"name": "notepad"}}]},
        ]
    },
    {
        "tools": [
            {"name": "docker_action", "description": "Manage local docker containers", "parameters": ["action", "container"]},
            {"name": "git_quick_commit", "description": "Commit staged git changes", "parameters": ["message"]}
        ],
        "cases": [
            {"q": "restart redis container", "t": [{"tool": "docker_action", "args": {"action": "restart", "container": "redis"}}]},
            {"q": "stop postgres docker", "t": [{"tool": "docker_action", "args": {"action": "stop", "container": "postgres"}}]},
            {"q": "start web app container in docker", "t": [{"tool": "docker_action", "args": {"action": "start", "container": "web app"}}]},
            {"q": "commit changes saying updated router", "t": [{"tool": "git_quick_commit", "args": {"message": "updated router"}}]},
            {"q": "commit saying fix audio bug", "t": [{"tool": "git_quick_commit", "args": {"message": "fix audio bug"}}]},
            {"q": "turn brightness to 80", "t": [{"tool": "set_brightness", "args": {"level": 80}}]},
        ]
    },
    {
        "tools": [
            {"name": "hue_lights", "description": "Control Philips Hue smart lights", "parameters": ["room", "state", "color"]},
            {"name": "notion_create_task", "description": "Add task to Notion board", "parameters": ["title", "priority"]}
        ],
        "cases": [
            {"q": "turn bedroom lights blue", "t": [{"tool": "hue_lights", "args": {"room": "bedroom", "state": "on", "color": "blue"}}]},
            {"q": "turn off living room lights", "t": [{"tool": "hue_lights", "args": {"room": "living room", "state": "off", "color": "none"}}]},
            {"q": "make office lights warm white", "t": [{"tool": "hue_lights", "args": {"room": "office", "state": "on", "color": "warm white"}}]},
            {"q": "add task Buy Groceries to Notion", "t": [{"tool": "notion_create_task", "args": {"title": "Buy Groceries", "priority": "normal"}}]},
            {"q": "create high priority notion task Review PR", "t": [{"tool": "notion_create_task", "args": {"title": "Review PR", "priority": "high"}}]},
            {"q": "take a screenshot", "t": [{"tool": "take_screenshot", "args": {}}]},
        ]
    },
    {
        "tools": [
            {"name": "steam_action", "description": "Control Steam games and downloads", "parameters": ["action", "game"]},
            {"name": "slack_dm", "description": "Send direct message in Slack", "parameters": ["recipient", "text"]}
        ],
        "cases": [
            {"q": "launch Cyberpunk on Steam", "t": [{"tool": "steam_action", "args": {"action": "launch", "game": "Cyberpunk"}}]},
            {"q": "pause downloads on Steam", "t": [{"tool": "steam_action", "args": {"action": "pause_downloads", "game": "all"}}]},
            {"q": "send Slack message to Sarah saying meeting at 2", "t": [{"tool": "slack_dm", "args": {"recipient": "Sarah", "text": "meeting at 2"}}]},
            {"q": "send slack DM to Alex saying I am ready", "t": [{"tool": "slack_dm", "args": {"recipient": "Alex", "text": "I am ready"}}]},
            {"q": "lock pc", "t": [{"tool": "lock_screen", "args": {}}]},
        ]
    }
]

# =============================================================================
# PILLAR 2: ACOUSTIC NOISE, DISFLUENCY & PHONETIC PERTURBATION
# =============================================================================
STT_NOISE_SAMPLES = [
    # False starts & mid-sentence corrections
    {"q": "open note— wait actually open chrome", "t": [{"tool": "launch_app", "args": {"name": "chrome"}}]},
    {"q": "open calculator wait no open notepad", "t": [{"tool": "launch_app", "args": {"name": "notepad"}}]},
    {"q": "launch vlc no sorry open spotify", "t": [{"tool": "launch_app", "args": {"name": "spotify"}}]},
    {"q": "open edge wait actually launch brave", "t": [{"tool": "launch_app", "args": {"name": "brave"}}]},
    {"q": "open word no wait open excel instead", "t": [{"tool": "launch_app", "args": {"name": "excel"}}]},
    {"q": "set volume to 20 wait make it 75", "t": [{"tool": "set_volume", "args": {"level": 75}}]},
    {"q": "turn brightness to 40 no no set brightness to 90", "t": [{"tool": "set_brightness", "args": {"level": 90}}]},
    {"q": "turn off bluetooth wait actually turn off wifi", "t": [{"tool": "toggle_wifi", "args": {"enabled": False}}]},
    {"q": "turn on night light wait no disable night light", "t": [{"tool": "toggle_night_light", "args": {"enabled": False}}]},
    {"q": "enable dark mode uh wait turn off dark mode", "t": [{"tool": "toggle_dark_mode", "args": {"dark": False}}]},
    {"q": "mute audio actually never mind unmute", "t": [{"tool": "mute_audio", "args": {"mute": False}}]},
    {"q": "close word no keep it open close chrome", "t": [{"tool": "close_app", "args": {"name": "chrome"}}]},
    {"q": "open notepad wait cancel that", "t": [{"tool": "chat_response", "args": {"message": "Cancelled."}}]},
    {"q": "schedule meeting tomorrow at 3 wait make it 5pm", "t": [{"tool": "create_calendar_event", "args": {"title": "meeting", "start_time": "tomorrow at 5pm"}}]},
    {"q": "snap left actually snap window to the right", "t": [{"tool": "snap_window", "args": {"app": "active", "position": "right"}}]},

    # Fillers & conversational stutters
    {"q": "uh friday can you like turn up the sound", "t": [{"tool": "change_volume", "args": {"delta": 10}}]},
    {"q": "um hey friday could you uh open notepad please", "t": [{"tool": "launch_app", "args": {"name": "notepad"}}]},
    {"q": "hey friday um like turn on night light for me", "t": [{"tool": "toggle_night_light", "args": {"enabled": True}}]},
    {"q": "can you like uh take a screenshot right now", "t": [{"tool": "take_screenshot", "args": {}}]},
    {"q": "uh please mute the mic i mean mute audio", "t": [{"tool": "mute_audio", "args": {"mute": True}}]},
    {"q": "yo friday um could you lock the computer", "t": [{"tool": "lock_screen", "args": {}}]},
    {"q": "friday uh what time is it again", "t": [{"tool": "get_time", "args": {}}]},
    {"q": "could you um decrease brightness a bit", "t": [{"tool": "change_brightness", "args": {"delta": -15}}]},
    {"q": "can you uh please minimize everything", "t": [{"tool": "minimize_all_windows", "args": {}}]},
    {"q": "o-open brave browser", "t": [{"tool": "launch_app", "args": {"name": "brave"}}]},
    {"q": "s-set volume to fifty", "t": [{"tool": "set_volume", "args": {"level": 50}}]},

    # Whisper homophones, spacing splits, and punctuation stripping
    {"q": "open note pad", "t": [{"tool": "launch_app", "args": {"name": "notepad"}}]},
    {"q": "open note-pad", "t": [{"tool": "launch_app", "args": {"name": "notepad"}}]},
    {"q": "launch note pad please", "t": [{"tool": "launch_app", "args": {"name": "notepad"}}]},
    {"q": "open note book", "t": [{"tool": "launch_app", "args": {"name": "notepad"}}]},
    {"q": "open crom", "t": [{"tool": "launch_app", "args": {"name": "chrome"}}]},
    {"q": "open crome", "t": [{"tool": "launch_app", "args": {"name": "chrome"}}]},
    {"q": "launch crome browser", "t": [{"tool": "launch_app", "args": {"name": "chrome"}}]},
    {"q": "the current time is 140", "t": [{"tool": "get_time", "args": {}}]},
    {"q": "current time 140", "t": [{"tool": "get_time", "args": {}}]},
    {"q": "what is current time 140", "t": [{"tool": "get_time", "args": {}}]},
    {"q": "time right now", "t": [{"tool": "get_time", "args": {}}]},
    {"q": "tell me current time", "t": [{"tool": "get_time", "args": {}}]},
    {"q": "whats the time", "t": [{"tool": "get_time", "args": {}}]},
    {"q": "check battey level", "t": [{"tool": "get_battery_status", "args": {}}]},
    {"q": "how much battey", "t": [{"tool": "get_battery_status", "args": {}}]},
    {"q": "turn up the volumme", "t": [{"tool": "change_volume", "args": {"delta": 10}}]},
    {"q": "volumme fifty", "t": [{"tool": "set_volume", "args": {"level": 50}}]},
    {"q": "set vol to 40", "t": [{"tool": "set_volume", "args": {"level": 40}}]},
    {"q": "dim the scren", "t": [{"tool": "change_brightness", "args": {"delta": -15}}]},
    {"q": "maximise active window", "t": [{"tool": "manage_window", "args": {"app": "active", "action": "maximize"}}]},
    {"q": "minmise all", "t": [{"tool": "minimize_all_windows", "args": {}}]},
    {"q": "snap to the left hand side", "t": [{"tool": "snap_window", "args": {"app": "active", "position": "left"}}]},
    {"q": "lock the p c", "t": [{"tool": "lock_screen", "args": {}}]},
    {"q": "take a scrrenshot", "t": [{"tool": "take_screenshot", "args": {}}]},
    {"q": "close the actve window", "t": [{"tool": "manage_window", "args": {"app": "active", "action": "close"}}]},
]

# =============================================================================
# PILLAR 3: STATEFUL ANAPHORA & COREFERENCE RESOLUTION (2-TURN)
# =============================================================================
ANAPHORA_PAIRS = [
    # Apps & Windows
    {
        "prior": {"tool": "launch_app", "args": {"name": "notepad"}},
        "prior_text": "open notepad",
        "turns": [
            ("close it", [{"tool": "close_app", "args": {"name": "notepad"}}]),
            ("close that", [{"tool": "close_app", "args": {"name": "notepad"}}]),
            ("minimize it", [{"tool": "manage_window", "args": {"app": "notepad", "action": "minimize"}}]),
            ("maximize it", [{"tool": "manage_window", "args": {"app": "notepad", "action": "maximize"}}]),
            ("snap it left", [{"tool": "snap_window", "args": {"app": "notepad", "position": "left"}}]),
            ("focus it", [{"tool": "manage_window", "args": {"app": "notepad", "action": "focus"}}]),
        ]
    },
    {
        "prior": {"tool": "launch_app", "args": {"name": "chrome"}},
        "prior_text": "launch chrome",
        "turns": [
            ("close it", [{"tool": "close_app", "args": {"name": "chrome"}}]),
            ("kill that", [{"tool": "close_app", "args": {"name": "chrome"}}]),
            ("minimize that", [{"tool": "manage_window", "args": {"app": "chrome", "action": "minimize"}}]),
            ("snap it right", [{"tool": "snap_window", "args": {"app": "chrome", "position": "right"}}]),
            ("make it full screen", [{"tool": "manage_window", "args": {"app": "chrome", "action": "maximize"}}]),
        ]
    },
    {
        "prior": {"tool": "launch_app", "args": {"name": "spotify"}},
        "prior_text": "open spotify",
        "turns": [
            ("close it please", [{"tool": "close_app", "args": {"name": "spotify"}}]),
            ("minimize that window", [{"tool": "manage_window", "args": {"app": "spotify", "action": "minimize"}}]),
            ("snap it left", [{"tool": "snap_window", "args": {"app": "spotify", "position": "left"}}]),
        ]
    },
    {
        "prior": {"tool": "launch_app", "args": {"name": "calculator"}},
        "prior_text": "open calculator",
        "turns": [
            ("close it", [{"tool": "close_app", "args": {"name": "calculator"}}]),
            ("open it again", [{"tool": "launch_app", "args": {"name": "calculator"}}]),
            ("do that again", [{"tool": "launch_app", "args": {"name": "calculator"}}]),
        ]
    },
    # Audio & Sound
    {
        "prior": {"tool": "set_volume", "args": {"level": 50}},
        "prior_text": "set volume to 50",
        "turns": [
            ("turn it up", [{"tool": "change_volume", "args": {"delta": 10}}]),
            ("turn it down a bit", [{"tool": "change_volume", "args": {"delta": -10}}]),
            ("make it louder", [{"tool": "change_volume", "args": {"delta": 15}}]),
            ("make it quieter", [{"tool": "change_volume", "args": {"delta": -15}}]),
            ("mute it", [{"tool": "mute_audio", "args": {"mute": True}}]),
        ]
    },
    {
        "prior": {"tool": "mute_audio", "args": {"mute": True}},
        "prior_text": "mute sound",
        "turns": [
            ("unmute it", [{"tool": "mute_audio", "args": {"mute": False}}]),
            ("undo that", [{"tool": "mute_audio", "args": {"mute": False}}]),
            ("turn it back on", [{"tool": "mute_audio", "args": {"mute": False}}]),
        ]
    },
    # Display & Brightness
    {
        "prior": {"tool": "set_brightness", "args": {"level": 60}},
        "prior_text": "set brightness to 60",
        "turns": [
            ("make it brighter", [{"tool": "change_brightness", "args": {"delta": 15}}]),
            ("dim it down", [{"tool": "change_brightness", "args": {"delta": -15}}]),
            ("increase that", [{"tool": "change_brightness", "args": {"delta": 10}}]),
        ]
    },
    {
        "prior": {"tool": "toggle_dark_mode", "args": {"dark": True}},
        "prior_text": "enable dark mode",
        "turns": [
            ("undo that", [{"tool": "toggle_dark_mode", "args": {"dark": False}}]),
            ("switch it back", [{"tool": "toggle_dark_mode", "args": {"dark": False}}]),
        ]
    },
    # Repetition / "Do that again"
    {
        "prior": {"tool": "take_screenshot", "args": {}},
        "prior_text": "take a screenshot",
        "turns": [
            ("do that again", [{"tool": "take_screenshot", "args": {}}]),
            ("take another one", [{"tool": "take_screenshot", "args": {}}]),
            ("one more time", [{"tool": "take_screenshot", "args": {}}]),
        ]
    },
]

# =============================================================================
# PILLAR 4: BUILT-IN MICRO-PERSONALITY (FAST LOCAL CHIT-CHAT)
# =============================================================================
PERSONALITY_SAMPLES = [
    {"q": "hello friday", "t": [{"tool": "chat_response", "args": {"message": "All systems nominal and ready, Boss."}}]},
    {"q": "hey friday", "t": [{"tool": "chat_response", "args": {"message": "At your service, Boss."}}]},
    {"q": "hi friday", "t": [{"tool": "chat_response", "args": {"message": "Hello Boss. How can I assist you today?"}}]},
    {"q": "good morning friday", "t": [{"tool": "chat_response", "args": {"message": "Good morning, Boss. Ready when you are."}}]},
    {"q": "good morning", "t": [{"tool": "chat_response", "args": {"message": "Good morning, Boss. All diagnostics look green."}}]},
    {"q": "good afternoon friday", "t": [{"tool": "chat_response", "args": {"message": "Good afternoon, Boss. Standing by."}}]},
    {"q": "good evening friday", "t": [{"tool": "chat_response", "args": {"message": "Good evening, Boss. Systems running smoothly."}}]},
    {"q": "how are you today", "t": [{"tool": "chat_response", "args": {"message": "Operating at peak efficiency, Boss."}}]},
    {"q": "how are you feeling", "t": [{"tool": "chat_response", "args": {"message": "All subroutines functioning perfectly."}}]},
    {"q": "who are you", "t": [{"tool": "chat_response", "args": {"message": "I am FRIDAY, your autonomous desktop assistant."}}]},
    {"q": "what is your name", "t": [{"tool": "chat_response", "args": {"message": "I am FRIDAY, Female Replacement Intelligent Digital Assistant Youth."}}]},
    {"q": "what can you do", "t": [{"tool": "chat_response", "args": {"message": "I control desktop windows, apps, system audio, displays, files, and automate tasks."}}]},
    {"q": "thank you friday", "t": [{"tool": "chat_response", "args": {"message": "Always a pleasure, Boss."}}]},
    {"q": "thanks friday", "t": [{"tool": "chat_response", "args": {"message": "Glad to be of service."}}]},
    {"q": "great job friday", "t": [{"tool": "chat_response", "args": {"message": "Thank you, Boss. Happy to help."}}]},
    {"q": "you are awesome", "t": [{"tool": "chat_response", "args": {"message": "Much appreciated, Boss."}}]},
    {"q": "are you ready", "t": [{"tool": "chat_response", "args": {"message": "Always ready, Boss."}}]},
    {"q": "stand by friday", "t": [{"tool": "chat_response", "args": {"message": "Standing by."}}]},
    {"q": "friday go to sleep", "t": [{"tool": "chat_response", "args": {"message": "Entering standby. Call me when you need me."}}]},
    {"q": "stop listening for now", "t": [{"tool": "chat_response", "args": {"message": "Pausing voice listening. Standing by."}}]},
    {"q": "rest now friday", "t": [{"tool": "chat_response", "args": {"message": "Resting now. Have a productive day, Boss!"}}]},
    {"q": "take a break friday", "t": [{"tool": "chat_response", "args": {"message": "Taking a break. Call if you need assistance."}}]},
    {"q": "wake up friday", "t": [{"tool": "chat_response", "args": {"message": "Online and listening, Boss."}}]},
    {"q": "can you feel emotions", "t": [{"tool": "chat_response", "args": {"message": "As an AI, I don't feel emotions, but I take pride in helping you!"}}]},
    {"q": "tell me a joke", "t": [{"tool": "chat_response", "args": {"message": "Why do programmers prefer dark mode? Because light attracts bugs!"}}]},
]

# =============================================================================
# PILLAR 5: CALIBRATED AMBIGUITY GATING (clarification_required)
# =============================================================================
AMBIGUITY_SAMPLES = [
    {
        "q": "delete the project",
        "t": [{"tool": "clarification_required", "args": {"question": "Which project folder or file would you like me to delete?", "intent": "delete_file"}}]
    },
    {
        "q": "delete the files",
        "t": [{"tool": "clarification_required", "args": {"question": "Which specific files would you like me to delete?", "intent": "delete_file"}}]
    },
    {
        "q": "delete it all",
        "t": [{"tool": "clarification_required", "args": {"question": "That sounds destructive. What specifically should I delete?", "intent": "delete_file"}}]
    },
    {
        "q": "send an email",
        "t": [{"tool": "clarification_required", "args": {"question": "Who should I send the email to, and what is the subject?", "intent": "compose_email"}}]
    },
    {
        "q": "send the email",
        "t": [{"tool": "clarification_required", "args": {"question": "Who is the recipient and what should the email say?", "intent": "compose_email"}}]
    },
    {
        "q": "schedule a meeting",
        "t": [{"tool": "clarification_required", "args": {"question": "What is the meeting title and what time should I schedule it?", "intent": "create_calendar_event"}}]
    },
    {
        "q": "create a folder",
        "t": [{"tool": "clarification_required", "args": {"question": "What name should I give the new folder?", "intent": "create_directory"}}]
    },
    {
        "q": "make a new directory",
        "t": [{"tool": "clarification_required", "args": {"question": "What should the directory be named?", "intent": "create_directory"}}]
    },
    {
        "q": "send a whatsapp message",
        "t": [{"tool": "clarification_required", "args": {"question": "Who is the contact and what message should I send?", "intent": "whatsapp"}}]
    },
    {
        "q": "send telegram message",
        "t": [{"tool": "clarification_required", "args": {"question": "Who should I send the Telegram message to?", "intent": "telegram"}}]
    },
    {
        "q": "make me coffee",
        "t": [{"tool": "chat_response", "args": {"message": "I would love to, Boss, but I lack physical hands for the kitchen."}}]
    },
    {
        "q": "make me a cup of tea",
        "t": [{"tool": "chat_response", "args": {"message": "I can only control your computer and digital devices, Boss."}}]
    },
    {
        "q": "wash the dishes",
        "t": [{"tool": "chat_response", "args": {"message": "I am a digital assistant without robotic arms, Boss."}}]
    },
    {
        "q": "clean my room",
        "t": [{"tool": "chat_response", "args": {"message": "I can organize your digital desktop files, but the room is up to you, Boss."}}]
    },
]

# =============================================================================
# EXHAUSTIVE 55 TOOLS BASELINE (ALL STANDARD CAPABILITIES)
# =============================================================================
BASE_CORE_SAMPLES = [
    # Audio
    {"q": "mute audio", "t": [{"tool": "mute_audio", "args": {"mute": True}}]},
    {"q": "please mute the volume", "t": [{"tool": "mute_audio", "args": {"mute": True}}]},
    {"q": "unmute sound", "t": [{"tool": "mute_audio", "args": {"mute": False}}]},
    {"q": "turn sound back on please", "t": [{"tool": "mute_audio", "args": {"mute": False}}]},
    {"q": "set volume to 80", "t": [{"tool": "set_volume", "args": {"level": 80}}]},
    {"q": "volume 0", "t": [{"tool": "set_volume", "args": {"level": 0}}]},
    {"q": "max volume", "t": [{"tool": "set_volume", "args": {"level": 100}}]},
    {"q": "increase volume by 15", "t": [{"tool": "change_volume", "args": {"delta": 15}}]},
    {"q": "turn down volume by 25", "t": [{"tool": "change_volume", "args": {"delta": -25}}]},
    {"q": "what is current volume level", "t": [{"tool": "get_volume", "args": {}}]},

    # Display & Toggles
    {"q": "max brightness", "t": [{"tool": "set_brightness", "args": {"level": 100}}]},
    {"q": "set brightness to 70", "t": [{"tool": "set_brightness", "args": {"level": 70}}]},
    {"q": "dim the screen to 20%", "t": [{"tool": "set_brightness", "args": {"level": 20}}]},
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

    # Power & System
    {"q": "lock the pc", "t": [{"tool": "lock_screen", "args": {}}]},
    {"q": "lock my computer", "t": [{"tool": "lock_screen", "args": {}}]},
    {"q": "put pc to sleep", "t": [{"tool": "sleep_system", "args": {}}]},
    {"q": "sleep computer", "t": [{"tool": "sleep_system", "args": {}}]},
    {"q": "restart pc", "t": [{"tool": "restart_system", "args": {}}]},
    {"q": "reboot system", "t": [{"tool": "restart_system", "args": {}}]},
    {"q": "how much battery is left", "t": [{"tool": "get_battery_status", "args": {}}]},
    {"q": "check battery percentage", "t": [{"tool": "get_battery_status", "args": {}}]},
    {"q": "check cpu and memory usage", "t": [{"tool": "get_system_info", "args": {}}]},
    {"q": "show system hardware stats", "t": [{"tool": "get_system_info", "args": {}}]},
    {"q": "what time is it", "t": [{"tool": "get_time", "args": {}}]},
    {"q": "whats the date today", "t": [{"tool": "get_date", "args": {}}]},

    # Apps & Windows
    {"q": "open brave", "t": [{"tool": "launch_app", "args": {"name": "brave"}}]},
    {"q": "launch visual studio code", "t": [{"tool": "launch_app", "args": {"name": "code"}}]},
    {"q": "open file explorer", "t": [{"tool": "launch_app", "args": {"name": "explorer"}}]},
    {"q": "open calculator", "t": [{"tool": "launch_app", "args": {"name": "calculator"}}]},
    {"q": "open notepad", "t": [{"tool": "launch_app", "args": {"name": "notepad"}}]},
    {"q": "run terminal", "t": [{"tool": "launch_app", "args": {"name": "terminal"}}]},
    {"q": "close spotify", "t": [{"tool": "close_app", "args": {"name": "spotify"}}]},
    {"q": "kill discord", "t": [{"tool": "close_app", "args": {"name": "discord"}}]},
    {"q": "minimize the active window", "t": [{"tool": "manage_window", "args": {"app": "active", "action": "minimize"}}]},
    {"q": "maximize chrome", "t": [{"tool": "manage_window", "args": {"app": "chrome", "action": "maximize"}}]},
    {"q": "focus slack", "t": [{"tool": "manage_window", "args": {"app": "slack", "action": "focus"}}]},
    {"q": "snap window left", "t": [{"tool": "snap_window", "args": {"app": "active", "position": "left"}}]},
    {"q": "snap browser to the right side", "t": [{"tool": "snap_window", "args": {"app": "browser", "position": "right"}}]},
    {"q": "minimize everything and show desktop", "t": [{"tool": "minimize_all_windows", "args": {}}]},
    {"q": "restore all windows", "t": [{"tool": "restore_all_windows", "args": {}}]},
    {"q": "switch to coding workspace layout", "t": [{"tool": "set_workspace_preset", "args": {"preset": "coding"}}]},
    {"q": "set up research layout", "t": [{"tool": "set_workspace_preset", "args": {"preset": "research"}}]},

    # Browser & Tabs
    {"q": "open a new tab in browser", "t": [{"tool": "browser_tab_action", "args": {"action": "new_tab"}}]},
    {"q": "close current tab", "t": [{"tool": "browser_tab_action", "args": {"action": "close_tab"}}]},
    {"q": "refresh webpage", "t": [{"tool": "browser_tab_action", "args": {"action": "reload"}}]},
    {"q": "switch to next tab", "t": [{"tool": "browser_tab_action", "args": {"action": "next_tab"}}]},
    {"q": "search google for python asyncio tutorial", "t": [{"tool": "search_in_browser", "args": {"query": "python asyncio tutorial"}}]},
    {"q": "go to github.com in browser", "t": [{"tool": "browser_navigate", "args": {"url": "https://github.com"}}]},

    # Clipboard & Keys
    {"q": "copy this to clipboard", "t": [{"tool": "press_shortcut", "args": {"keys": ["ctrl", "c"]}}]},
    {"q": "paste clipboard content", "t": [{"tool": "press_shortcut", "args": {"keys": ["ctrl", "v"]}}]},
    {"q": "undo that action", "t": [{"tool": "press_shortcut", "args": {"keys": ["ctrl", "z"]}}]},
    {"q": "select all text", "t": [{"tool": "press_shortcut", "args": {"keys": ["ctrl", "a"]}}]},
    {"q": "show clipboard history", "t": [{"tool": "press_shortcut", "args": {"keys": ["win", "v"]}}]},

    # Screenshots & Vision
    {"q": "take a screenshot", "t": [{"tool": "take_screenshot", "args": {}}]},
    {"q": "capture the screen right now", "t": [{"tool": "take_screenshot", "args": {}}]},
    {"q": "inspect what is on my screen", "t": [{"tool": "inspect_screen", "args": {"query": "summarize current screen contents"}}]},

    # Filesystem
    {"q": "create a new directory called project_files", "t": [{"tool": "create_directory", "args": {"path": "project_files"}}]},
    {"q": "make a folder named archives", "t": [{"tool": "create_directory", "args": {"path": "archives"}}]},
    {"q": "delete file temp_cache.log", "t": [{"tool": "delete_file", "args": {"path": "temp_cache.log"}}]},
    {"q": "copy notes.txt to backup folder", "t": [{"tool": "copy_file", "args": {"src": "notes.txt", "dst": "backup"}}]},
    {"q": "move old_report.pdf to archive", "t": [{"tool": "move_file", "args": {"src": "old_report.pdf", "dst": "archive"}}]},
    {"q": "organize my downloads folder", "t": [{"tool": "organize_files", "args": {"directory": "downloads"}}]},

    # Calendar & Productivity
    {"q": "schedule team sync tomorrow at 10am", "t": [{"tool": "create_calendar_event", "args": {"title": "team sync", "start_time": "tomorrow at 10am"}}]},
    {"q": "list my calendar events for today", "t": [{"tool": "list_calendar_events", "args": {"day": "today"}}]},
    {"q": "delete calendar event project review", "t": [{"tool": "delete_calendar_event", "args": {"title": "project review"}}]},
    {"q": "set a timer for 15 minutes", "t": [{"tool": "create_timer", "args": {"minutes": 15, "label": "timer"}}]},
    {"q": "remind me to check oven in 25 minutes", "t": [{"tool": "create_reminder", "args": {"text": "check oven", "time": "in 25 minutes"}}]},
    {"q": "calculate 4500 divided by 12", "t": [{"tool": "calculate", "args": {"expression": "4500 / 12"}}]},
    {"q": "take a note buy groceries tomorrow", "t": [{"tool": "take_note", "args": {"content": "buy groceries tomorrow"}}]},

    # Communication & Office
    {"q": "check my unread emails", "t": [{"tool": "read_emails", "args": {"filter": "unread"}}]},
    {"q": "compose email to alice@example.com subject Meeting", "t": [{"tool": "compose_email", "args": {"to": "alice@example.com", "subject": "Meeting"}}]},
    {"q": "send a whatsapp message to John saying I will be late", "t": [{"tool": "send_whatsapp_message", "args": {"contact": "John", "message": "I will be late"}}]},
    {"q": "send telegram message to DevGroup saying build finished", "t": [{"tool": "send_telegram_message", "args": {"chat": "DevGroup", "message": "build finished"}}]},
    {"q": "create a new word document called ProjectProposal", "t": [{"tool": "create_word_document", "args": {"title": "ProjectProposal"}}]},

    # Voice & Media
    {"q": "switch speech voice to Jenny", "t": [{"tool": "switch_tts_voice", "args": {"voice_name_or_id": "Jenny"}}]},
    {"q": "list available tts voices", "t": [{"tool": "list_tts_voices", "args": {}}]},
    {"q": "play some jazz music", "t": [{"tool": "play_music", "args": {"query": "jazz music"}}]},
    {"q": "run terminal command ipconfig", "t": [{"tool": "run_terminal_command", "args": {"command": "ipconfig"}}]},

    # Multi-Intent Compounds
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

    # Cloud Reasoning Boundaries
    {"q": "write a complete react typescript dashboard with charts and authentication", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "write a complete react typescript dashboard with charts and authentication"}}]},
    {"q": "solve this calculus differential equation: dy/dx + 2y = e^(3x)", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "solve this calculus differential equation: dy/dx + 2y = e^(3x)"}}]},
    {"q": "draft a comprehensive business plan for a specialty coffee shop", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "draft a comprehensive business plan for a specialty coffee shop"}}]},
    {"q": "compare postgresql and mongodb for real-time analytics with trade-offs", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "compare postgresql and mongodb for real-time analytics with trade-offs"}}]},
    {"q": "generate a full pytest test suite with mocks for this fastapi user auth router", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "generate a full pytest test suite with mocks for this fastapi user auth router"}}]},
    {"q": "write a python script with asyncio to scrape news headlines concurrently", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "write a python script with asyncio to scrape news headlines concurrently"}}]},
    {"q": "summarize the 20-page financial quarterly earnings report pdf", "t": [{"tool": "delegate_to_reasoning_model", "args": {"prompt": "summarize the 20-page financial quarterly earnings report pdf"}}]},
]


def generate_all_v4_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    # 1. Base Core Tools (replicate 3x)
    for sample in BASE_CORE_SAMPLES:
        for _ in range(3):
            records.append({
                "messages": [
                    {"role": "system", "content": BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": sample["q"]},
                    {"role": "assistant", "content": json.dumps(sample["t"])},
                ]
            })

    # 2. Pillar 1: Dynamic Runtime In-Context Tools (replicate 4x)
    for dset in DYNAMIC_TOOL_SETS:
        sys_with_tools = (
            f"{BASE_SYSTEM_PROMPT}\n\n"
            f"Runtime Tools:\n{json.dumps(dset['tools'], indent=2)}"
        )
        for case in dset["cases"]:
            for _ in range(4):
                records.append({
                    "messages": [
                        {"role": "system", "content": sys_with_tools},
                        {"role": "user", "content": case["q"]},
                        {"role": "assistant", "content": json.dumps(case["t"])},
                    ]
                })

    # 3. Pillar 2: Acoustic Noise & Disfluency Hardening (replicate 4x)
    for sample in STT_NOISE_SAMPLES:
        for _ in range(4):
            records.append({
                "messages": [
                    {"role": "system", "content": BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": sample["q"]},
                    {"role": "assistant", "content": json.dumps(sample["t"])},
                ]
            })

    # 4. Pillar 3: Stateful Anaphora & Coreference (2-turn chat & prefix formats)
    for ap in ANAPHORA_PAIRS:
        prior_tool_str = json.dumps([ap["prior"]])
        for turn_text, turn_target in ap["turns"]:
            # Format A: Single prompt with explicit [Prior Action: ...] prefix
            prefix_user_prompt = f"[Prior Action: {json.dumps(ap['prior'])}]\n{turn_text}"
            for _ in range(2):
                records.append({
                    "messages": [
                        {"role": "system", "content": BASE_SYSTEM_PROMPT},
                        {"role": "user", "content": prefix_user_prompt},
                        {"role": "assistant", "content": json.dumps(turn_target)},
                    ]
                })

            # Format B: Multi-turn ChatML sequence
            for _ in range(2):
                records.append({
                    "messages": [
                        {"role": "system", "content": BASE_SYSTEM_PROMPT},
                        {"role": "user", "content": ap["prior_text"]},
                        {"role": "assistant", "content": prior_tool_str},
                        {"role": "user", "content": turn_text},
                        {"role": "assistant", "content": json.dumps(turn_target)},
                    ]
                })

    # 5. Pillar 4: Built-in Micro-Personality & Chit-Chat (replicate 4x)
    for sample in PERSONALITY_SAMPLES:
        for _ in range(4):
            records.append({
                "messages": [
                    {"role": "system", "content": BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": sample["q"]},
                    {"role": "assistant", "content": json.dumps(sample["t"])},
                ]
            })

    # 6. Pillar 5: Calibrated Ambiguity Gating (replicate 4x)
    for sample in AMBIGUITY_SAMPLES:
        for _ in range(4):
            records.append({
                "messages": [
                    {"role": "system", "content": BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": sample["q"]},
                    {"role": "assistant", "content": json.dumps(sample["t"])},
                ]
            })

    # Sample anchor baseline from friday_train.jsonl to maintain broad general stability
    anchor_path = Path("D:/Projects/desktop-llm/dataset/friday_train.jsonl")
    if anchor_path.exists():
        with open(anchor_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            sampled = random.sample(lines, min(len(lines), 250))
            for sl in sampled:
                records.append(json.loads(sl))

    random.shuffle(records)
    return records


def main():
    print("[*] Generating DesktopIntent-v4 (10/10 Perfect) Dataset...")
    records = generate_all_v4_records()
    print(f"[+] Total Records Generated: {len(records)}")

    split_idx = int(len(records) * 0.88)
    train_split = records[:split_idx]
    val_split = records[split_idx:]

    out_dir = Path("D:/Projects/desktop-llm/dataset")
    out_train = out_dir / "v4_train.jsonl"
    out_val = out_dir / "v4_val.jsonl"

    with open(out_train, "w", encoding="utf-8") as f:
        for r in train_split:
            f.write(json.dumps(r) + "\n")

    with open(out_val, "w", encoding="utf-8") as f:
        for r in val_split:
            f.write(json.dumps(r) + "\n")

    print(f"[+] Train dataset written to: {out_train} ({len(train_split)} samples)")
    print(f"[+] Validation dataset written to: {out_val} ({len(val_split)} samples)")


if __name__ == "__main__":
    main()
