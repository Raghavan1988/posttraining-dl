"""
Day 1, Step 2 — The library baseline: full fine-tune with TRL's SFTTrainer.

Same idea as 03_sft_raw.py but TRL handles the chat template, the completion-only
mask, packing, and the loop. Use it to sanity-check your raw run. Writes ./out-trl
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from common import BASE_MODEL, MAX_LEN, load_chat_dataset

OUT = "./out-trl"

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
ds = load_chat_dataset()

cfg = SFTConfig(
    output_dir=OUT,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=1.0,
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    max_length=MAX_LEN,
    packing=False,
    # Learn only on assistant turns == completion-only loss. This is the key knob.
    assistant_only_loss=True,
    report_to="none",
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model(OUT)
print(f"\nsaved -> {OUT}\nnow run:  python 04_chat_eval.py --model {OUT}")
