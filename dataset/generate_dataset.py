"""High-variance synthetic dataset generator for DesktopIntent.
Enforces 40% Single-Intent, 35% Multi-Intent, 10% Implicit Compound, 15% Fallbacks.
Formats in ChatML for Qwen2.5-0.5B/1.5B instruction fine-tuning.
"""

import json
import random
from pathlib import Path

random.seed(42)

SYSTEM_PROMPT = (
    "You are DesktopIntent, a deterministic desktop system control brain. "
    "For every user input, you output ONLY a valid JSON array of tool calls. "
    "Never generate explanation, reasoning, or conversational monologue. "
    "If the input is conversational, greetings, or out-of-domain, invoke the chat_response tool."
)

APPS = [
    ("chrome", "Chrome"),
    ("brave", "Brave"),
    ("edge", "Edge"),
    ("spotify", "Spotify"),
    ("notepad", "Notepad"),
    ("code", "VS Code"),
    ("terminal", "Terminal"),
    ("whatsapp", "WhatsApp"),
    ("discord", "Discord"),
    ("calculator", "Calculator"),
    ("settings", "Settings"),
    ("explorer", "File Explorer"),
]

ACTIONS = ["minimize", "maximize", "restore", "focus", "close"]


def gen_single_audio():
    templates = [
        ("mute audio", [{"tool": "mute_audio", "args": {"mute": True}}]),
        ("mute the sound", [{"tool": "mute_audio", "args": {"mute": True}}]),
        ("silence the pc", [{"tool": "mute_audio", "args": {"mute": True}}]),
        ("shut up", [{"tool": "mute_audio", "args": {"mute": True}}]),
        ("shh", [{"tool": "mute_audio", "args": {"mute": True}}]),
        ("unmute", [{"tool": "mute_audio", "args": {"mute": False}}]),
        ("unmute audio", [{"tool": "mute_audio", "args": {"mute": False}}]),
        ("turn the sound back on", [{"tool": "mute_audio", "args": {"mute": False}}]),
    ]
    # Parameterized levels
    lvl = random.choice([0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100])
    templates.extend(
        [
            (f"set volume to {lvl}%", [{"tool": "set_volume", "args": {"level": lvl}}]),
            (f"volume {lvl}", [{"tool": "set_volume", "args": {"level": lvl}}]),
            (f"change volume to {lvl}", [{"tool": "set_volume", "args": {"level": lvl}}]),
            (f"put volume at {lvl} percent", [{"tool": "set_volume", "args": {"level": lvl}}]),
        ]
    )
    delta = random.choice([5, 10, 15, 20])
    templates.extend(
        [
            (
                f"increase volume by {delta}%",
                [{"tool": "change_volume", "args": {"delta": delta}}],
            ),
            (f"turn it up {delta}", [{"tool": "change_volume", "args": {"delta": delta}}]),
            (
                f"decrease volume by {delta}%",
                [{"tool": "change_volume", "args": {"delta": -delta}}],
            ),
            (f"turn it down {delta}", [{"tool": "change_volume", "args": {"delta": -delta}}]),
            (
                "crank up the volume",
                [{"tool": "change_volume", "args": {"delta": 20}}],
            ),
            (
                "too loud turn it down",
                [{"tool": "change_volume", "args": {"delta": -15}}],
            ),
            (
                "can't hear anything turn it up",
                [{"tool": "change_volume", "args": {"delta": 20}}],
            ),
            ("what is current volume?", [{"tool": "get_volume", "args": {}}]),
        ]
    )
    return random.choice(templates)


def gen_single_display():
    lvl = random.choice([10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100])
    templates = [
        (
            f"set brightness to {lvl}%",
            [{"tool": "set_brightness", "args": {"level": lvl}}],
        ),
        (
            f"screen brightness {lvl}",
            [{"tool": "set_brightness", "args": {"level": lvl}}],
        ),
        (
            f"dim screen to {lvl}%",
            [{"tool": "set_brightness", "args": {"level": lvl}}],
        ),
        ("screen is too bright", [{"tool": "change_brightness", "args": {"delta": -20}}]),
        ("too dark increase brightness", [{"tool": "change_brightness", "args": {"delta": 20}}]),
        ("max brightness", [{"tool": "set_brightness", "args": {"level": 100}}]),
        ("lowest brightness", [{"tool": "set_brightness", "args": {"level": 10}}]),
        ("what is my screen brightness?", [{"tool": "get_brightness", "args": {}}]),
    ]
    return random.choice(templates)


def gen_single_app():
    app_key, app_name = random.choice(APPS)
    action = random.choice(
        [
            (f"open {app_name}", [{"tool": "launch_app", "args": {"name": app_key}}]),
            (f"launch {app_name}", [{"tool": "launch_app", "args": {"name": app_key}}]),
            (f"fire up {app_name}", [{"tool": "launch_app", "args": {"name": app_key}}]),
            (f"start {app_name}", [{"tool": "launch_app", "args": {"name": app_key}}]),
            (f"close {app_name}", [{"tool": "kill_app", "args": {"name": app_key}}]),
            (f"kill {app_name}", [{"tool": "kill_app", "args": {"name": app_key}}]),
            (f"force close {app_name}", [{"tool": "kill_app", "args": {"name": app_key}}]),
            (f"quit {app_name}", [{"tool": "kill_app", "args": {"name": app_key}}]),
        ]
    )
    return action


def gen_single_window():
    app_key, app_name = random.choice(APPS)
    act = random.choice(ACTIONS)
    templates = [
        (
            f"{act} {app_name}",
            [{"tool": "manage_window", "args": {"app": app_key, "action": act}}],
        ),
        (
            f"{act} the {app_name} window",
            [{"tool": "manage_window", "args": {"app": app_key, "action": act}}],
        ),
        (
            f"bring {app_name} to front",
            [{"tool": "manage_window", "args": {"app": app_key, "action": "focus"}}],
        ),
        (
            f"focus on {app_name}",
            [{"tool": "manage_window", "args": {"app": app_key, "action": "focus"}}],
        ),
    ]
    return random.choice(templates)


def gen_single_productivity():
    minutes = random.choice([2, 5, 10, 15, 20, 25, 30, 45, 60])
    n1, n2 = random.randint(5, 500), random.randint(2, 50)
    op = random.choice(["+", "-", "*", "/"])
    expr = f"{n1} {op} {n2}"

    templates = [
        (
            f"set a timer for {minutes} minutes",
            [
                {
                    "tool": "create_timer",
                    "args": {"duration_seconds": minutes * 60, "label": "Timer"},
                }
            ],
        ),
        (
            f"timer {minutes}m",
            [
                {
                    "tool": "create_timer",
                    "args": {"duration_seconds": minutes * 60, "label": "Timer"},
                }
            ],
        ),
        (
            f"remind me in {minutes} minutes to take a break",
            [
                {
                    "tool": "create_reminder",
                    "args": {"message": "take a break", "minutes_from_now": minutes},
                }
            ],
        ),
        (
            f"calculate {expr}",
            [{"tool": "calculate", "args": {"expression": expr}}],
        ),
        (
            f"what is {n1} times {n2}?",
            [{"tool": "calculate", "args": {"expression": f"{n1} * {n2}"}}],
        ),
        (
            f"what is 15% of {n1}?",
            [{"tool": "calculate", "args": {"expression": f"15% of {n1}"}}],
        ),
        (
            "take a note buy milk and eggs",
            [{"tool": "take_note", "args": {"text": "buy milk and eggs"}}],
        ),
        (
            "note: review pull request before 5pm",
            [{"tool": "take_note", "args": {"text": "review pull request before 5pm"}}],
        ),
        ("lock my pc", [{"tool": "lock_pc", "args": {}}]),
        ("lock computer", [{"tool": "lock_pc", "args": {}}]),
        ("lock workstation", [{"tool": "lock_pc", "args": {}}]),
        ("how much ram is being used?", [{"tool": "get_system_stats", "args": {}}]),
        ("check cpu usage and battery", [{"tool": "get_system_stats", "args": {}}]),
        ("system stats", [{"tool": "get_system_stats", "args": {}}]),
    ]
    return random.choice(templates)


def gen_single():
    fn = random.choice(
        [
            gen_single_audio,
            gen_single_display,
            gen_single_app,
            gen_single_window,
            gen_single_productivity,
        ]
    )
    return fn()


def gen_multi():
    """Generates 2 to 3 chained actions with realistic connectors."""
    k = random.choice([2, 2, 2, 3])
    parts = []
    actions = []

    # Pick k distinct single generators
    generators = [
        gen_single_audio,
        gen_single_display,
        gen_single_app,
        gen_single_window,
        gen_single_productivity,
    ]
    random.shuffle(generators)

    for i in range(k):
        prompt, acts = generators[i]()
        parts.append(prompt)
        actions.extend(acts)

    connectors = [" and ", ", then ", " and also ", ", ", " then "]
    full_prompt = parts[0]
    for p in parts[1:]:
        full_prompt += random.choice(connectors) + p

    # Special handling for pronoun/anaphora resolution
    if (
        len(actions) >= 2
        and actions[0]["tool"] == "launch_app"
        and actions[1]["tool"] == "manage_window"
    ):
        app = actions[0]["args"]["name"]
        actions[1]["args"]["app"] = app
        act_verb = actions[1]["args"]["action"]
        full_prompt = f"open {app} and {act_verb} it"

    return full_prompt, actions


def gen_compound():
    """Implicit multi-action scenarios."""
    scenarios = [
        (
            "I'm going to sleep now",
            [
                {"tool": "mute_audio", "args": {"mute": True}},
                {"tool": "set_brightness", "args": {"level": 10}},
                {"tool": "lock_pc", "args": {}},
            ],
        ),
        (
            "bedtime mode",
            [
                {"tool": "mute_audio", "args": {"mute": True}},
                {"tool": "set_brightness", "args": {"level": 10}},
            ],
        ),
        (
            "focus mode",
            [
                {"tool": "mute_audio", "args": {"mute": True}},
                {"tool": "kill_app", "args": {"name": "discord"}},
                {"tool": "launch_app", "args": {"name": "code"}},
            ],
        ),
        (
            "presentation mode",
            [
                {"tool": "set_brightness", "args": {"level": 100}},
                {"tool": "mute_audio", "args": {"mute": True}},
                {"tool": "manage_window", "args": {"app": "chrome", "action": "maximize"}},
            ],
        ),
        (
            "reset workstation",
            [
                {"tool": "set_volume", "args": {"level": 50}},
                {"tool": "set_brightness", "args": {"level": 70}},
            ],
        ),
        (
            "quiet down it's noisy",
            [
                {"tool": "change_volume", "args": {"delta": -25}},
            ],
        ),
        (
            "I'm stepping away",
            [
                {"tool": "mute_audio", "args": {"mute": True}},
                {"tool": "lock_pc", "args": {}},
            ],
        ),
    ]
    return random.choice(scenarios)


def gen_fallback():
    """Greetings, identity, and out-of-scope conversational prompts."""
    fallbacks = [
        ("hi", "Hello! How can I assist you with your desktop today?"),
        ("hey there", "Hey! Ready to manage your windows, audio, or apps."),
        ("who created you?", "I am DesktopIntent, an ultra-fast local desktop controller."),
        ("what can you do?", "I control audio, display brightness, windows, apps, timers, and system tasks."),
        ("tell me a joke", "I specialize in desktop actions, but here's one: Why do programmers prefer dark mode? Because light attracts bugs!"),
        ("what is the weather today?", "I don't have internet weather data, but I can open your browser or weather app."),
        ("explain quantum computing", "I am a desktop control router. Ask me to launch apps, adjust volume, or manage your screen!"),
        ("help", "You can say things like 'mute sound', 'dim brightness to 40%', 'open Spotify', or 'set a 10m timer'."),
        ("good morning", "Good morning! Ready to help you set up your workspace."),
        ("thank you", "You're welcome! Let me know if you need any other desktop controls."),
    ]
    p, reply = random.choice(fallbacks)
    return p, [{"tool": "chat_response", "args": {"message": reply}}]


def build_dataset(total_samples: int = 6000):
    dataset = []

    # Distribution: 40% single, 35% multi, 10% compound, 15% fallback
    n_single = int(total_samples * 0.40)
    n_multi = int(total_samples * 0.35)
    n_compound = int(total_samples * 0.10)
    n_fallback = total_samples - (n_single + n_multi + n_compound)

    for _ in range(n_single):
        p, a = gen_single()
        dataset.append((p, a))

    for _ in range(n_multi):
        p, a = gen_multi()
        dataset.append((p, a))

    for _ in range(n_compound):
        p, a = gen_compound()
        dataset.append((p, a))

    for _ in range(n_fallback):
        p, a = gen_fallback()
        dataset.append((p, a))

    random.shuffle(dataset)

    # Format into ChatML
    formatted = []
    for prompt, acts in dataset:
        target_json = json.dumps(acts, ensure_ascii=False)
        formatted.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target_json},
                ]
            }
        )

    return formatted


def main():
    out_dir = Path(__file__).parent
    all_data = build_dataset(6000)

    # 90/10 Train/Validation Split
    split_idx = int(len(all_data) * 0.90)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(train_data)} training examples -> {train_path}")
    print(f"Generated {len(val_data)} validation examples -> {val_path}")


if __name__ == "__main__":
    main()
