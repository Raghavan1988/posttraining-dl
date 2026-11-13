"""Reward over-optimization (Goodhart's law) under best-of-n sampling.

A proxy reward model is fit on preference data that only covers the base policy's typical
outputs. Optimizing hard against it (large n in best-of-n) exploits its blind spots: the proxy
score keeps rising, but the true ("gold") reward peaks and then falls.
KL(best-of-n || base) = log n - (n - 1) / n gives the x-axis used in scaling-law plots.

Setup (2-D response features x ~ N(0, I) under the base policy):
  gold(x)  = x0 + x1 - 1.5 * x1^2         (x1 helps up to a point, then hurts)
  proxy    = linear RM trained on Bradley-Terry pairs drawn from a narrow slice of outputs
"""
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)


def gold(x):
    return x[:, 0] + x[:, 1] - 1.5 * x[:, 1] ** 2


# --- train proxy RM on pairs from a narrow, low-variance region of response space ---
xa, xb = 0.4 * torch.randn(5000, 2), 0.4 * torch.randn(5000, 2)
labels = (torch.rand(5000) < torch.sigmoid(gold(xa) - gold(xb))).float()
rm = torch.nn.Linear(2, 1)
opt = torch.optim.Adam(rm.parameters(), lr=0.05)
for _ in range(500):
    logit = (rm(xa) - rm(xb)).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logit, labels)
    opt.zero_grad(); loss.backward(); opt.step()
print("proxy RM weights", [round(w, 3) for w in rm.weight[0].tolist()], "(gold gradient at 0 is [1, 1])")


@torch.no_grad()
def best_of_n(n, trials=4000):
    x = torch.randn(trials, n, 2)
    pick = rm(x).squeeze(-1).argmax(1)
    sel = x[torch.arange(trials), pick]
    return rm(sel).mean().item(), gold(sel).mean().item()


print(f"{'n':>5} {'KL':>6} {'proxy':>7} {'gold':>7}")
golds = []
for n in [1, 2, 4, 8, 16, 64, 256, 1024]:
    kl = math.log(n) - (n - 1) / n
    p, g = best_of_n(n)
    golds.append(g)
    print(f"{n:5d} {kl:6.2f} {p:7.3f} {g:7.3f}")
peak = max(range(len(golds)), key=golds.__getitem__)
print(f"gold reward peaks at index {peak} then declines by {golds[peak] - golds[-1]:.2f}")
assert 0 < peak < len(golds) - 1 and golds[-1] < golds[peak] - 0.3
