"""Targeted Calibration Dataset Generator for Mixed Conversational + Actions.

Teaches DesktopIntent-LLM to strictly extract concrete system action tool calls
even when user queries are wrapped in greetings ('good morning'), gratitude ('thanks'),
or conversational preambles, while maintaining pure conversational capability.
"""

import json
import random
from pathlib import Path

random.seed(42)

SYSTEM_PROMPT = (
    "You are DesktopIntent, the real-time neural router for FRIDAY AI. "
    "For every user input, you output ONLY a valid JSON array of tool calls matching FRIDAY's MCP tool bridge. "
    "Never generate conversational monologue, explanation, or thinking tokens. "
    "If the command requires complex coding or open-ended reasoning, invoke delegate_to_reasoning_model. "
    "If the input is conversational or greetings, invoke chat_response."
)

GREETINGS = [
    "good morning friday, ", "good morning, ", "hey friday, ", "hi friday, ",
    "hello friday, ", "yo friday, ", "good afternoon friday, ", "good evening friday, ",
    "friday, ", "alright friday, ", "hey there, ", "hello there, "
]

GRATITUDES = [
    "thanks friday, now ", "thank you friday, please ", "thanks a lot friday, now ",
    "awesome thanks friday, ", "great job, now ", "thanks, now please ", "thank you, "
]

APPS = ["chrome", "brave", "edge", "spotify", "notepad", "code", "whatsapp", "discord", "calculator"]

PRESETS = ["coding", "research", "gaming", "writing", "focus", "default"]


def generate_mixed_samples():
    samples = []

    # 1. Greeting + Single Action (150 samples)
    for _ in range(150):
        g = random.choice(GREETINGS)
        act_type = random.choice(["mute", "volume", "brightness", "app", "window", "timer", "calc"])

        if act_type == "mute":
            m = random.choice([True, False])
            verb = "mute audio" if m else "unmute audio"
            p = f"{g}can you please {verb}"
            a = [{"tool": "mute_audio", "args": {"mute": m}}]
        elif act_type == "volume":
            lvl = random.choice([20, 30, 40, 50, 60, 70, 80, 90, 100])
            p = f"{g}set volume to {lvl}"
            a = [{"tool": "set_volume", "args": {"level": lvl}}]
        elif act_type == "brightness":
            lvl = random.choice([30, 50, 70, 80, 90, 100])
            p = f"{g}turn brightness to {lvl}%"
            a = [{"tool": "set_brightness", "args": {"level": lvl}}]
        elif act_type == "app":
            app = random.choice(APPS)
            p = f"{g}open {app}"
            a = [{"tool": "launch_app", "args": {"name": app}}]
        elif act_type == "window":
            app = random.choice(APPS)
            action = random.choice(["minimize", "maximize"])
            p = f"{g}{action} {app}"
            a = [{"tool": "manage_window", "args": {"app": app, "action": action}}]
        elif act_type == "timer":
            mins = random.choice([5, 10, 15, 20, 25, 30])
            p = f"{g}set a {mins} minute timer"
            a = [{"tool": "create_timer", "args": {"duration_seconds": mins * 60, "label": "Timer"}}]
        elif act_type == "calc":
            p = f"{g}what is 25 * 40"
            a = [{"tool": "calculate", "args": {"expression": "25 * 40"}}]

        samples.append((p, a))

    # 2. Greeting + Multi-Action (150 samples)
    for _ in range(150):
        g = random.choice(GREETINGS)
        app = random.choice(APPS)
        lvl = random.choice([30, 40, 50, 60, 70, 80])

        pattern = random.choice([1, 2, 3])
        if pattern == 1:
            p = f"{g}please open {app} and set volume to {lvl}"
            a = [
                {"tool": "launch_app", "args": {"name": app}},
                {"tool": "set_volume", "args": {"level": lvl}},
            ]
        elif pattern == 2:
            p = f"{g}mute audio and minimize {app}"
            a = [
                {"tool": "mute_audio", "args": {"mute": True}},
                {"tool": "manage_window", "args": {"app": app, "action": "minimize"}},
            ]
        elif pattern == 3:
            preset = random.choice(PRESETS)
            mins = random.choice([10, 15, 25])
            p = f"{g}switch to {preset} layout, open {app}, and set a {mins} min timer"
            a = [
                {"tool": "set_workspace_preset", "args": {"preset": preset}},
                {"tool": "launch_app", "args": {"name": app}},
                {"tool": "create_timer", "args": {"duration_seconds": mins * 60, "label": "Timer"}},
            ]

        samples.append((p, a))

    # 3. Gratitude + Action (100 samples)
    for _ in range(100):
        grat = random.choice(GRATITUDES)
        app = random.choice(APPS)

        if random.random() < 0.5:
            p = f"{grat}close {app} and lock the screen"
            a = [
                {"tool": "close_app", "args": {"name": app}},
                {"tool": "press_shortcut", "args": {"action": "lock_screen"}},
            ]
        else:
            lvl = random.choice([20, 30, 50])
            p = f"{grat}mute audio and set brightness to {lvl}"
            a = [
                {"tool": "mute_audio", "args": {"mute": True}},
                {"tool": "set_brightness", "args": {"level": lvl}},
            ]

        samples.append((p, a))

    # 4. Pure Conversational Anchors (100 samples - so model preserves conversational greetings)
    pure_chat = [
        ("hello friday", "Hello! Systems online and ready."),
        ("good morning friday", "Good morning! Ready for today's tasks."),
        ("hey friday how are you", "I'm running optimally. What can I do for you?"),
        ("who are you", "I am FRIDAY, your autonomous desktop assistant."),
        ("what can you do", "I manage your desktop, audio, brightness, apps, windows, and productivity tools."),
        ("thank you friday", "You're very welcome!"),
        ("thanks for your help", "Glad I could help!"),
        ("goodbye friday", "Goodbye! Have a great day."),
        ("tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs!"),
        ("how is your day going", "Everything is running smoothly!"),
    ]
    for _ in range(10):
        for q, r in pure_chat:
            samples.append((q, [{"tool": "chat_response", "args": {"message": r}}]))

    random.shuffle(samples)

    formatted = []
    for prompt, acts in samples:
        formatted.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(acts, ensure_ascii=False)},
            ]
        })

    return formatted


def main():
    out_dir = Path(__file__).parent
    data = generate_mixed_samples()

    # 85/15 Split
    split = int(len(data) * 0.85)
    train_data = data[:split]
    val_data = data[split:]

    train_path = out_dir / "calibration_train.jsonl"
    val_path = out_dir / "calibration_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[+] Wrote {len(train_data)} calibration train samples -> {train_path}")
    print(f"[+] Wrote {len(val_data)} calibration val samples -> {val_path}")


if __name__ == "__main__":
    main()
