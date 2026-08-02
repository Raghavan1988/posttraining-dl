"""
Day 1, Step 1 — SEE the two twists of SFT: the chat template and the loss mask.

Run this FIRST. No training happens. It prints:
  (a) a raw instruction/response example,
  (b) the same example after the chat template is applied (the string the model sees),
  (c) the token-by-token LABEL MASK, so you can watch prompt tokens become -100
      (ignored by the loss) and only the assistant tokens carry a target.
"""
from transformers import AutoTokenizer
from common import BASE_MODEL, load_chat_dataset
## from common indicates common.py

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
ds = load_chat_dataset(n=5)
ex = ds[0]

print("=" * 70)
print("(a) RAW MESSAGES")
print("=" * 70)
for m in ex["messages"]:
    print(f"  [{m['role']}] {m['content'][:200]}")

# (b) The chat template turns messages into ONE string with special turn markers.
full = tok.apply_chat_template(ex["messages"], tokenize=False)
# The 'prompt' = everything up to and including the assistant header (no answer yet).
prompt = tok.apply_chat_template(
    ex["messages"][:1], tokenize=False, add_generation_prompt=True
)
print("\n" + "=" * 70)
print("(b) AFTER CHAT TEMPLATE (what the model actually reads)")
print("=" * 70)
print(full)

# (c) Build the label mask exactly like completion-only SFT does.
full_ids = tok(full, add_special_tokens=False)["input_ids"]
prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
n_prompt = len(prompt_ids)
labels = [-100] * n_prompt + full_ids[n_prompt:]   # mask the prompt, keep the completion

print("=" * 70)
print("(c) LABEL MASK  (token  ->  label)   -100 == ignored by cross-entropy")
print("=" * 70)
for i, (tid, lab) in enumerate(zip(full_ids, labels)):
    piece = tok.decode([tid]).replace("\n", "\\n")
    tag = "  <-- MASKED (prompt)" if lab == -100 else "  <== learned (completion)"
    print(f"{i:3d}  {tid:>7}  {piece!r:<18} {tag}")
    if i == n_prompt + 6:            # stop a few tokens into the completion
        print("     ... (rest of assistant answer is learned)")
        break

print(f"\nprompt tokens (masked): {n_prompt}   total tokens: {len(full_ids)}")
print("Takeaway: loss is computed ONLY on the completion. That is vanilla SFT.")
