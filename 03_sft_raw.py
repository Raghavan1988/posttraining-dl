"""
Day 1, Step 3 — Vanilla SFT in RAW PyTorch (no Trainer), heavily commented.

Full fine-tune of Qwen2.5-0.5B (every weight updates) on instruction/response demos.
You own every moving part: tokenization, the chat template, the completion-only
label mask, padding, gradient accumulation, and the optimizer step.

  Smoke test first (tiny + fast, just proves the pipeline runs, no useful model):
      python 03_sft_raw.py --smoke
  Real run (~10-15 min on a 16GB 4090, writes ./out-raw):
      python 03_sft_raw.py
  Then compare against the base model:
      python 04_chat_eval.py --model ./out-raw
"""
import os
# Reduce GPU memory fragmentation. Full FT of 0.5B + Adam's optimizer states is
# tight on 16GB, and fragmentation alone can trigger an OOM. Must be set before torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from common import BASE_MODEL, MAX_LEN, load_chat_dataset

# ----------------------------------------------------------------------------- CLI
ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true",
                help="tiny 64-example / few-step run just to prove the pipeline works")
args = ap.parse_args()

# ----------------------------------------------------------------------------- hyperparameters
DEVICE = "cuda"
EPOCHS = 1
# micro-batch = how many sequences go through the GPU at once (memory-bound).
# effective batch = BATCH * GRAD_ACCUM = how many examples inform one weight update.
# We keep the micro-batch small (fits in 16GB) but accumulate gradients to get a
# larger, more stable effective batch of 16.
BATCH = 2
GRAD_ACCUM = 8
LR = 1e-5                 # small LR: we are gently steering a pretrained model, not training from scratch
OUT = "./out-raw"
N_TRAIN = 64 if args.smoke else 2000

# ----------------------------------------------------------------------------- tokenizer
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
# Base models often have no pad token. We need one to pad sequences to equal length
# inside a batch; reusing EOS is the standard trick (pads are masked out anyway).
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# ----------------------------------------------------------------------------- data encoding
def encode(ex):
    """Turn one chat example into input_ids + labels, MASKING the prompt.

    'labels' is what cross-entropy compares against, shifted internally by the model.
    Setting a label to -100 tells PyTorch's cross-entropy to IGNORE that position.
    We ignore every prompt/system/user token and only keep the assistant answer,
    so the model is graded solely on generating the response = completion-only loss.
    This is the single defining trick of SFT (see 01_inspect_data.py to watch it).
    """
    # Full conversation as one string, including the assistant's answer.
    full = tok.apply_chat_template(ex["messages"], tokenize=False)
    # Same string but truncated right before the answer (user turn + "assistant\n" header).
    # Its token length tells us exactly how many leading tokens are "prompt" to mask.
    prompt = tok.apply_chat_template(
        ex["messages"][:1], tokenize=False, add_generation_prompt=True
    )
    full_ids = tok(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
    n_prompt = min(len(tok(prompt, add_special_tokens=False)["input_ids"]), len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]   # mask prompt, learn completion
    return {"input_ids": full_ids, "labels": labels}


# Load the instruction dataset and encode every row. remove_columns drops the raw
# 'messages' so the dataset yields only tensors-to-be.
ds = load_chat_dataset(n=N_TRAIN).map(encode, remove_columns=["messages"])


def collate(batch):
    """Pad a list of variable-length examples into rectangular tensors.

    Sequences in a batch differ in length, but a GPU tensor must be rectangular, so
    we right-pad the shorter ones. Three parallel tensors, all padded to the same width:
      input_ids       — token ids, padded with pad_token_id
      labels          — targets, padded with -100 so pad positions add no loss
      attention_mask  — 1 for real tokens, 0 for pad, so attention ignores padding
    """
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


# shuffle=True so the model doesn't see examples in a fixed order each epoch.
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate)

# ----------------------------------------------------------------------------- model
# bfloat16 weights halve memory vs fp32 and are numerically friendly on this GPU.
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(DEVICE)
# Gradient checkpointing trades compute for memory: it drops intermediate activations
# during the forward pass and recomputes them in the backward pass. Essential to fit
# a full fine-tune in 16GB.
model.gradient_checkpointing_enable()
model.train()

# AdamW is the standard optimizer for LLM fine-tuning. Its per-parameter state
# (first + second moment) is the main memory cost of full FT beyond the weights.
opt = torch.optim.AdamW(model.parameters(), lr=LR)

# One "optimizer step" happens every GRAD_ACCUM micro-batches, hence the division.
steps = max(1, (len(loader) // GRAD_ACCUM) * EPOCHS)
# Cosine schedule with a short warmup: LR ramps up briefly, then smoothly decays to 0.
sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)

print(f"{'SMOKE ' if args.smoke else ''}examples={len(ds)}  "
      f"micro-batches={len(loader)}  optim-steps={steps}  "
      f"effective-batch={BATCH * GRAD_ACCUM}")

# ----------------------------------------------------------------------------- training loop
step = 0
for epoch in range(EPOCHS):
    for i, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Forward pass. Because we passed 'labels', the model returns .loss = mean
        # cross-entropy over the NON-masked (completion) tokens only.
        out = model(**batch)

        # Scale the loss so that summing GRAD_ACCUM backward passes equals the average
        # gradient of one large batch (gradient accumulation).
        loss = out.loss / GRAD_ACCUM
        loss.backward()                       # accumulate gradients (does NOT update weights yet)

        # After GRAD_ACCUM micro-batches, apply one real optimizer update.
        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip to avoid rare huge steps
            opt.step()                        # update weights
            sched.step()                      # advance the LR schedule
            opt.zero_grad()                   # reset accumulated gradients for the next window
            step += 1
            if step % (2 if args.smoke else 10) == 0:
                print(f"step {step:4d}/{steps}  loss {out.loss.item():.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}")

# ----------------------------------------------------------------------------- save
if args.smoke:
    print("\nsmoke test OK — pipeline runs end to end. Re-run without --smoke to train for real.")
else:
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"\nsaved -> {OUT}\nnow run:  python 04_chat_eval.py --model {OUT}")
