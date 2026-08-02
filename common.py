"""Shared config + data helpers for Day 1 SFT."""
from datasets import load_dataset

BASE_MODEL = "Qwen/Qwen2.5-0.5B"        # base (NOT -Instruct): we do the tuning ourselves
DATASET = "yahma/alpaca-cleaned"        # single-turn instruction/response demos
MAX_LEN = 768
N_TRAIN = 2000                          # small on purpose; day-1 is about seeing the mechanics


def to_messages(ex):
    """Alpaca row -> chat 'messages' list. Merges optional 'input' into the user turn."""
    instr = ex["instruction"].strip()
    if ex.get("input", "").strip():
        instr = f"{instr}\n\n{ex['input'].strip()}"
    return {"messages": [
        {"role": "user", "content": instr},
        {"role": "assistant", "content": ex["output"].strip()},
    ]}


def load_chat_dataset(n=N_TRAIN):
    ds = load_dataset(DATASET, split=f"train[:{n}]")
    cols = ds.column_names
    return ds.map(to_messages, remove_columns=cols)
