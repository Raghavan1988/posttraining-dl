"""WiSE-FT / model soup: linear interpolation between pre-trained and fine-tuned weights.

  theta(alpha) = (1 - alpha) * theta_base + alpha * theta_ft
Sweeping alpha trades new-task gain against forgetting of the original task.
Toy: base model fits task A; fine-tune on task B; evaluate both along the path.
"""
import copy
import torch

torch.manual_seed(0)
X = torch.randn(512, 8)
wa, wb = torch.randn(8, 1), torch.randn(8, 1)
yA, yB = X @ wa, X @ (0.5 * wa + wb)


def fit(model, y, steps=300):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        loss = ((model(X) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


base = fit(torch.nn.Linear(8, 1), yA)
ft = fit(copy.deepcopy(base), yB)


def interpolate(a):
    m = copy.deepcopy(base)
    sd_b, sd_f = base.state_dict(), ft.state_dict()
    m.load_state_dict({k: (1 - a) * sd_b[k] + a * sd_f[k] for k in sd_b})
    return m


with torch.no_grad():
    print("alpha   lossA   lossB   sum")
    best = None
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        m = interpolate(a)
        la, lb = ((m(X) - yA) ** 2).mean().item(), ((m(X) - yB) ** 2).mean().item()
        print(f"{a:4.2f}  {la:6.3f}  {lb:6.3f}  {la + lb:6.3f}")
        if best is None or la + lb < best[1]:
            best = (a, la + lb)
print(f"best combined alpha = {best[0]}")
assert 0 < best[0] < 1
