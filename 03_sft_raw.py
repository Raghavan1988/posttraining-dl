"""
Day 1, Step 3 — Vanilla SFT in RAW PyTorch (~120 lines, no Trainer).

This is the real learning: you own the dataloader, the chat template, the
completion-only label mask, and the training loop. Full fine-tune of Qwen2.5-0.5B.

Writes the tuned model to ./out-raw   (then run 04_chat_eval.py --model ./out-raw)
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from common import BASE_MODEL, MAX_LEN, load_chat_dataset

DEVICE = "cuda"
EPOCHS = 1
BATCH = 8            # micro-batch; effective batch = BATCH * GRAD_ACCUM
GRAD_ACCUM = 2
LR = 1e-5
OUT = "./out-raw"

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def encode(ex):
    """messages -> input_ids + labels, with the prompt masked to -100."""
    full = tok.apply_chat_template(ex["messages"], tokenize=False)
    prompt = tok.apply_chat_template(
        ex["messages"][:1], tokenize=False, add_generation_prompt=True
    )
    full_ids = tok(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
    n_prompt = min(len(tok(prompt, add_special_tokens=False)["input_ids"]), len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]        # <-- completion-only loss
    return {"input_ids": full_ids, "labels": labels}


ds = load_chat_dataset().map(encode, remove_columns=["messages"])


def collate(batch):
    """Right-pad a batch; pad tokens get attention_mask 0 and label -100."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, labs, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
        labs.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labs),
        "attention_mask": torch.tensor(attn),
    }


loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(DEVICE)
model.gradient_checkpointing_enable()
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=LR)
steps = (len(loader) // GRAD_ACCUM) * EPOCHS
sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)

print(f"examples={len(ds)}  micro-batches={len(loader)}  optim-steps={steps}")
step = 0
for epoch in range(EPOCHS):
    for i, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        # The model computes cross-entropy internally, ignoring label == -100.
        out = model(**batch)
        loss = out.loss / GRAD_ACCUM
        loss.backward()
        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"step {step:4d}/{steps}  loss {out.loss.item():.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}")

model.save_pretrained(OUT); tok.save_pretrained(OUT)
print(f"\nsaved -> {OUT}\nnow run:  python 04_chat_eval.py --model {OUT}")
