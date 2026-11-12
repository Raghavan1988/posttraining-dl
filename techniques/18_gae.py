"""Generalized Advantage Estimation (GAE), as used in PPO-based RLHF.

  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
  A_t     = sum_l (gamma*lambda)^l * delta_{t+l}
lambda=0 -> one-step TD (low variance, biased); lambda=1 -> Monte-Carlo return minus V.
In RLHF the reward is usually sparse: a per-token KL penalty plus the RM score on the last token.
"""
import torch


def gae(rewards, values, gamma=1.0, lam=0.95):
    T = len(rewards)
    adv = torch.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values             # advantages, value targets (returns)


kl = torch.tensor([0.02, 0.05, 0.01, 0.03, 0.04])
rm_score = 1.0
beta = 0.1
rewards = -beta * kl
rewards[-1] += rm_score
values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])

for lam in (0.0, 0.95, 1.0):
    adv, ret = gae(rewards, values, lam=lam)
    print(f"lambda={lam:<4}  adv={[round(a, 3) for a in adv.tolist()]}")

adv1, _ = gae(rewards, values, lam=1.0)
mc = torch.flip(torch.cumsum(torch.flip(rewards, [0]), 0), [0]) - values
assert torch.allclose(adv1, mc, atol=1e-6)          # lambda=1 == MC return - V
print("lambda=1 matches Monte-Carlo return minus baseline")
