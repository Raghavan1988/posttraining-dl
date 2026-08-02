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

# A "training-compatible" chat template. TRL's assistant_only_loss needs to know
# EXACTLY which tokens are the assistant's so it can mask everything else. It finds
# them via the {% generation %}...{% endgeneration %} markers below. Qwen's default
# (inference) template has no such markers, which is the ValueError you hit. This
# template is the standard Qwen ChatML format with those markers added, and it keeps
# <|im_end|> INSIDE the generation block so the model also learns to stop.
TRAIN_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\n' }}"
    "{%- if message['role'] == 'assistant' %}"
    "{% generation %}{{- message['content'] + '<|im_end|>' }}{% endgeneration %}"
    "{{- '\n' }}"
    "{%- else %}"
    "{{- message['content'] + '<|im_end|>' + '\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.chat_template = TRAIN_CHAT_TEMPLATE

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
ds = load_chat_dataset()

cfg = SFTConfig(
    output_dir=OUT,
    num_train_epochs=1,
    per_device_train_batch_size=2,   # small micro-batch: full FT of 0.5B is tight on 16GB
    gradient_accumulation_steps=8,   # effective batch = 2 * 8 = 16
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
