from transformers import AutoTokenizer
import json

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
msg = [
    {"role": "system", "content": "You are DesktopIntent."},
    {"role": "user", "content": "volume 50 and open brave"},
    {"role": "assistant", "content": json.dumps([{"tool": "set_volume", "args": {"level": 50}}, {"tool": "launch_app", "args": {"app": "brave"}}])}
]
text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
prompt_text = tok.apply_chat_template(msg[:-1], tokenize=False, add_generation_prompt=True)
full_toks = tok(text)["input_ids"]
prompt_toks = tok(prompt_text)["input_ids"]
print("Full tokens count:", len(full_toks), "Prompt tokens count:", len(prompt_toks))
print("Prefix matches exactly:", full_toks[:len(prompt_toks)] == prompt_toks)
asst_decoded = tok.decode(full_toks[len(prompt_toks):])
print("Decoded Assistant part (what the model learns to output):", repr(asst_decoded))
