"""Conservative DPO (cDPO): label smoothing for noisy preference data.

If a fraction eps of preference labels may be flipped, the loss becomes
  L = -(1 - eps) * log sigmoid(beta * h) - eps * log sigmoid(-beta * h)
with h = (log pi/ref)(chosen) - (log pi/ref)(rejected). Unlike plain DPO, the optimum margin is
finite (beta*h* = log((1-eps)/eps)), so the policy stops drifting once it is confident enough.
"""
import math
import torch
import torch.nn.functional as F


def cdpo_loss(h, beta=0.1, eps=0.1):
    return -(1 - eps) * F.logsigmoid(beta * h) - eps * F.logsigmoid(-beta * h)


beta = 0.1
for eps in (0.0, 0.1, 0.25):
    h = torch.zeros(1, requires_grad=True)       # implicit reward margin, optimized directly
    opt = torch.optim.SGD([h], lr=50.0)
    for _ in range(2000):
        loss = cdpo_loss(h, beta, eps).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    target = math.log((1 - eps) / eps) / beta if eps > 0 else float("inf")
    print(f"eps={eps:<5} learned margin h={h.item():8.2f}   analytic optimum={target:.2f}")
    if eps > 0:
        assert abs(h.item() - target) < 0.5
