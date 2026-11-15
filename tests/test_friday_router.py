import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.friday_bridge import FridayNeuralRouter

router = FridayNeuralRouter()

test_queries = [
    "turn brightness to 80%",
    "would you kindly switch to research layout then refresh the webpage",
    "can you please mute audio and minimize chrome",
    "schedule team meeting tomorrow at 10am",
    "create a new directory named project_files",
    "send a whatsapp message to John saying I will be late",
    "switch voice to female tts",
    "run terminal command ipconfig",
    "write an essay on machine learning",
    "good morning friday",
]

print("\n==========================================")
print("Testing FridayNeuralRouter -> SimpleIntentRoute:")
print("==========================================")
for q in test_queries:
    routes = router.route_intents(q)
    print(f"\nUser Query: {q!r}")
    if not routes:
        print("  -> Delegated to Cloud Reasoning Pipeline (returns [])")
    for r in routes:
        print(f"  -> SimpleIntentRoute(category={r.category!r}, action={r.canonical_action!r}, target={r.target!r})")
