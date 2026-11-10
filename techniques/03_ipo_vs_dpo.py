"""IPO: a regularized alternative to DPO.

DPO's logistic loss keeps pushing the log-ratio margin toward infinity on deterministic
preferences (it can overfit). IPO replaces it with a squared loss toward a finite target:
  loss = (h - 1/(2*tau))^2,  h = (logpi_c - logref_c) - (logpi_r - logref_r)
Toy: optimize both losses on one pair and compare the margin they converge to.
"""
import torch
import torch.nn.functional as F

TAU, BETA = 0.5, 0.5


def run(kind, steps=5000):
    torch.manual_seed(0)
    ref = torch.randn(5)
    logits = ref.clone().requires_grad_(True)
    ref_lp = F.log_softmax(ref, -1)
    opt = torch.optim.SGD([logits], lr=0.1)
    for _ in range(steps):
        lp = F.log_softmax(logits, -1)
        h = (lp[0] - ref_lp[0]) - (lp[1] - ref_lp[1])
        loss = -F.logsigmoid(BETA * h) if kind == "dpo" else (h - 1 / (2 * TAU)) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
    return h.item()


h_dpo, h_ipo = run("dpo"), run("ipo")
print(f"DPO margin after training: {h_dpo:.2f}  (keeps growing)")
print(f"IPO margin after training: {h_ipo:.2f}  (target 1/(2*tau) = {1/(2*TAU):.2f})")
assert abs(h_ipo - 1 / (2 * TAU)) < 1e-2 and h_dpo > h_ipo
