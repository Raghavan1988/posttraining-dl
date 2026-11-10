"""KTO (Kahneman-Tversky Optimization): preference tuning from UNPAIRED thumbs up/down.

No (chosen, rejected) pairs needed — each sample is just labeled desirable or not.
  r = log pi(y|x) - log ref(y|x);  z_ref = clamp(mean KL estimate, min=0) (detached)
  desirable:   loss = lambda_D * (1 - sigmoid(beta * (r - z_ref)))
  undesirable: loss = lambda_U * (1 - sigmoid(beta * (z_ref - r)))
Toy: a single-step policy over 6 tokens; tokens {0,1} are thumbs-up, {4,5} thumbs-down.
"""
import torch
import torch.nn.functional as F


def kto_loss(logr, desirable, beta=0.5, lam_d=1.0, lam_u=1.0, z_ref=None):
    if z_ref is None:
        z_ref = logr.mean().clamp(min=0).detach()
    up = lam_d * (1 - torch.sigmoid(beta * (logr - z_ref)))
    down = lam_u * (1 - torch.sigmoid(beta * (z_ref - logr)))
    return torch.where(desirable, up, down).mean()


torch.manual_seed(0)
ref = torch.zeros(6)
logits = ref.clone().requires_grad_(True)
ref_lp = F.log_softmax(ref, -1)
samples = torch.tensor([0, 1, 4, 5])
labels = torch.tensor([True, True, False, False])
opt = torch.optim.Adam([logits], lr=0.1)

for step in range(200):
    lp = F.log_softmax(logits, -1)
    loss = kto_loss(lp[samples] - ref_lp[samples], labels)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

p = logits.softmax(-1)
print("probs:", [round(x, 3) for x in p.tolist()])
assert p[0] > 1 / 6 and p[1] > 1 / 6 and p[4] < 1 / 6 and p[5] < 1 / 6
