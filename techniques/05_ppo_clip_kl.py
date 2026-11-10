"""PPO for RLHF: clipped surrogate objective + per-token KL penalty to the reference.

  shaped reward  r_t = r_task(at final token) - beta * (logpi_t - logref_t)
  ratio          rho = exp(logpi_new - logpi_old)
  loss           = -mean( min(rho * A, clip(rho, 1-eps, 1+eps) * A) )
Toy: a bandit "policy" over 4 actions, task reward favors action 3. We show
(a) the clipping caps how far one update can move, and (b) KL keeps the policy near ref.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
EPS, BETA, N = 0.2, 0.05, 256
task_reward = torch.tensor([0.0, 0.2, 0.1, 1.0])
ref_logits = torch.zeros(4)
logits = ref_logits.clone().requires_grad_(True)
opt = torch.optim.Adam([logits], lr=0.05)


def ppo_loss(logp_new, logp_old, adv, eps=EPS):
    ratio = (logp_new - logp_old).exp()
    unclipped = ratio * adv
    clipped = ratio.clamp(1 - eps, 1 + eps) * adv
    frac_clipped = ((ratio - 1).abs() > eps).float().mean()
    return -torch.min(unclipped, clipped).mean(), frac_clipped


for it in range(40):
    with torch.no_grad():                                   # rollout
        old_lp = F.log_softmax(logits, -1)
        actions = torch.multinomial(old_lp.exp(), N, replacement=True)
        kl_pen = old_lp[actions] - F.log_softmax(ref_logits, -1)[actions]
        r = task_reward[actions] - BETA * kl_pen
        adv = (r - r.mean()) / (r.std() + 1e-8)             # simple baseline
    for _ in range(4):                                      # several epochs on same batch
        lp = F.log_softmax(logits, -1)[actions]
        loss, clipfrac = ppo_loss(lp, old_lp[actions], adv)
        opt.zero_grad(); loss.backward(); opt.step()
    if it % 10 == 0:
        print(f"iter {it:2d}  E[r_task] {task_reward[actions].mean():.3f}  clipfrac {clipfrac:.2f}")

p = logits.softmax(-1)
print("final policy:", [round(x, 3) for x in p.tolist()])
assert p.argmax().item() == 3
