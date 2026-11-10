"""Reward modeling with the Bradley-Terry loss.

Given (chosen, rejected) pairs, train r(x) so that P(chosen > rejected) = sigmoid(r_c - r_r).
Loss = -log sigmoid(r_c - r_r). Toy: features are random vectors, a hidden "true" direction
decides which response humans prefer. The learned reward should recover that preference.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
D, N = 16, 2000
w_true = torch.randn(D)
a, b = torch.randn(N, D), torch.randn(N, D)
a_better = (a @ w_true) > (b @ w_true)
chosen = torch.where(a_better[:, None], a, b)
rejected = torch.where(a_better[:, None], b, a)

reward = torch.nn.Linear(D, 1)
opt = torch.optim.Adam(reward.parameters(), lr=1e-2)
for step in range(300):
    margin = reward(chosen).squeeze(-1) - reward(rejected).squeeze(-1)
    loss = -F.logsigmoid(margin).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

with torch.no_grad():
    acc = (reward(chosen) > reward(rejected)).float().mean().item()
print(f"pairwise accuracy: {acc:.3f}")
assert acc > 0.95
