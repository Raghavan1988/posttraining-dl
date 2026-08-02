"""
Day 1, Step 4 — Did SFT do anything? Compare BASE vs your TUNED model side by side
on held-out instructions. The base model tends to ramble/continue; the tuned one
should answer the instruction and stop.

Usage:  python 04_chat_eval.py --model ./out-raw
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import BASE_MODEL

PROMPTS = [
    "Give me three tips for staying focused while working from home.",
    "Explain what a hash map is to a 10 year old.",
    "Write a haiku about gradient descent.",
    "What is the capital of Australia? Answer in one word.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="./out-raw", help="path to the tuned model")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(BASE_MODEL)


def gen(model_path, prompt):
    tk = AutoTokenizer.from_pretrained(model_path)
    m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda").eval()
    text = tk.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = tk(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=160, do_sample=False,
                         pad_token_id=tk.pad_token_id or tk.eos_token_id)
    del m; torch.cuda.empty_cache()
    return tk.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


for p in PROMPTS:
    print("\n" + "#" * 72)
    print("PROMPT:", p)
    print("-" * 72)
    print("BASE  :", gen(BASE_MODEL, p)[:400])
    print("-" * 72)
    print("TUNED :", gen(args.model, p)[:400])
