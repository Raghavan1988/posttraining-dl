"""ORPO: odds-ratio preference optimization (SFT + preference in one loss, no reference model).

  odds(y)  = p(y) / (1 - p(y)),   p(y) = exp(mean token log-prob)
  L_OR     = -log sigmoid( log odds(chosen) - log odds(rejected) )
  L        = NLL(chosen) + lambda * L_OR
Toy: bigram LM learns to imitate `chosen` while pushing `rejected` down.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
V = 8
lm = torch.nn.Embedding(V, V)


def mean_logp(seq):
    return F.log_softmax(lm(seq[:-1]), -1).gather(1, seq[1:, None]).mean()


def log_odds(lp):                       # log(p / (1-p)) computed stably from log p
    return lp - torch.log1p(-torch.exp(lp).clamp(max=1 - 1e-6))


def orpo_loss(chosen, rejected, lam=0.5):
    lp_c, lp_r = mean_logp(chosen), mean_logp(rejected)
    l_or = -F.logsigmoid(log_odds(lp_c) - log_odds(lp_r))
    return -lp_c + lam * l_or, l_or.detach()


chosen = torch.tensor([0, 1, 2, 3, 4])
rejected = torch.tensor([0, 1, 5, 5, 5])
opt = torch.optim.Adam(lm.parameters(), lr=0.05)
for step in range(200):
    loss, l_or = orpo_loss(chosen, rejected)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  odds-ratio loss {l_or.item():.4f}")

with torch.no_grad():
    pc, pr = mean_logp(chosen).exp(), mean_logp(rejected).exp()
    print(f"p(chosen) {pc:.3f}  p(rejected) {pr:.3f}")
    assert pc > pr
