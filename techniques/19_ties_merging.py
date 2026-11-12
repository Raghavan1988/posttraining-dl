"""TIES-Merging: merge several fine-tuned models via task vectors.

  1. Trim:  keep the top-k% magnitude entries of each task vector tau_i = theta_i - theta_base
  2. Elect: per-parameter sign = sign(sum_i tau_i)
  3. Merge: average only the entries agreeing with the elected sign
  theta = theta_base + lambda * merged
Reduces interference compared with naive task-vector averaging.
"""
import torch

torch.manual_seed(0)


def ties_merge(base, finetuned, k=0.2, lam=1.0):
    taus = torch.stack([f - base for f in finetuned])
    n_keep = max(1, int(k * base.numel()))
    trimmed = torch.zeros_like(taus)
    for i, t in enumerate(taus):
        thr = t.abs().flatten().kthvalue(t.numel() - n_keep + 1).values
        trimmed[i] = torch.where(t.abs() >= thr, t, torch.zeros_like(t))
    sign = torch.sign(trimmed.sum(0))
    agree = (torch.sign(trimmed) == sign) & (trimmed != 0)
    merged = (trimmed * agree).sum(0) / agree.sum(0).clamp(min=1)
    return base + lam * merged


base = torch.zeros(1000)
# two tasks each edit a sparse, mostly disjoint set of weights, plus small dense noise
edits = []
for idx in (torch.arange(0, 100), torch.arange(80, 180)):
    e = 0.01 * torch.randn(1000)
    e[idx] += 1.0
    edits.append(e)
fts = [base + e for e in edits]

naive = base + sum(edits) / len(edits)
ties = ties_merge(base, fts, k=0.12)
for name, m in [("naive avg", naive), ("TIES", ties)]:
    keep1 = m[0:80].mean().item()
    keep2 = m[100:180].mean().item()
    noise = m[200:].abs().mean().item()
    print(f"{name:9s}  task1 edit kept {keep1:.2f}  task2 edit kept {keep2:.2f}  noise {noise:.4f}")
assert ties[0:80].mean() > 0.9 and naive[0:80].mean() < 0.6
