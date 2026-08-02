# posttraining-dl

A hands-on journey through **post-training recipes** (SFT → preference tuning → RL),
one focused day at a time. Each day = read the mental model, run a *from-library* baseline,
then re-implement the core idea in **raw PyTorch** so the mechanics are visible.

Hardware assumed: single 16GB GPU (RTX 4090 Laptop). Everything here fits.

---

## Day 1 — Vanilla SFT (Supervised Fine-Tuning)

**Goal:** turn a base model that just continues text into one that *follows instructions*,
using a plain next-token loss on `(prompt, response)` demonstrations.

### The whole recipe in two rules
SFT is pretraining on instruction data, with two twists that are the entire point:

1. **Chat template** — wrap each example in the model's turn format
   (`<|im_start|>user … <|im_start|>assistant …`).
2. **Loss masking (completion-only loss)** — compute cross-entropy **only on the
   assistant/response tokens**. The prompt tokens are set to label `-100` so they
   contribute zero loss. This is the #1 thing people get wrong.

Everything later branches from this skeleton:
- **Rejection-sampling SFT (RFT):** generate N answers, keep the correct ones, SFT on them.
- **Distillation SFT:** the responses come from a *stronger* model.
- **DPO / KTO:** add a *rejected* response and swap CE for a preference loss.

### Files (run in order)
| File | What it teaches |
|---|---|
| `01_inspect_data.py` | Load the instruction set; print raw examples; show exactly what the chat template + label mask look like token by token. **Run this first — it's where the concept lands.** |
| `02_sft_trl.py` | Full fine-tune Qwen2.5-0.5B with TRL's `SFTTrainer` (completion-only loss). The fast baseline. |
| `03_sft_raw.py` | The same training in ~120 lines of raw PyTorch: dataloader, masking, loss loop. This is the real learning. |
| `04_chat_eval.py` | Chat with **base vs tuned** on held-out prompts; watch the behavior change. |

### Quickstart
```bash
pip install -r requirements.txt          # trl + friends (torch/transformers assumed present)
python 01_inspect_data.py                # understand data + masking
python 03_sft_raw.py                     # ~5-10 min on a 4090; writes ./out-raw
python 04_chat_eval.py --model ./out-raw # compare against base
# optional baseline:
python 02_sft_trl.py                     # writes ./out-trl
```

### What "done" looks like today
- [ ] You can explain, in one sentence, why prompt tokens get label `-100`.
- [ ] Loss goes down over training and the tuned model answers instructions the base model ignores.
- [ ] You've read your own `03_sft_raw.py` masking code and believe it.

### Tomorrow
Day 2 = **preference data + reward modeling**, then **DPO** — reuses today's dataloader,
adds a `rejected` column and a pairwise loss.
