"""Sequence packing with a block-diagonal causal mask and reset position ids.

Several short examples are concatenated into one row to avoid padding waste. A block-diagonal
causal mask stops attention leaking across example boundaries, and position ids restart at 0.
Toy: verify packed attention output equals running each example separately.
"""
import torch

torch.manual_seed(0)
D = 16
attn = torch.nn.MultiheadAttention(D, 2, batch_first=True)
seqs = [torch.randn(n, D) for n in (3, 5, 2)]


def packed_mask(lengths):
    L = sum(lengths)
    allowed = torch.zeros(L, L, dtype=torch.bool)
    s = 0
    for n in lengths:
        allowed[s:s + n, s:s + n] = torch.tril(torch.ones(n, n, dtype=torch.bool))
        s += n
    return ~allowed                     # True = blocked


lengths = [len(s) for s in seqs]
pos_ids = torch.cat([torch.arange(n) for n in lengths])
print("position ids:", pos_ids.tolist())
print("padding waste unpacked:", sum(max(lengths) - n for n in lengths), "tokens; packed: 0")

with torch.no_grad():
    packed = torch.cat(seqs)[None]
    out_packed, _ = attn(packed, packed, packed, attn_mask=packed_mask(lengths))
    outs = []
    for s in seqs:
        causal = ~torch.tril(torch.ones(len(s), len(s), dtype=torch.bool))
        o, _ = attn(s[None], s[None], s[None], attn_mask=causal)
        outs.append(o[0])
    diff = (out_packed[0] - torch.cat(outs)).abs().max().item()
print(f"max |packed - separate| = {diff:.2e}")
assert diff < 1e-5
