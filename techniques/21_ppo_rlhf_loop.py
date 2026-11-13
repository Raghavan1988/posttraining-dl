"""End-to-end PPO RLHF loop on a tiny autoregressive LM.

Pieces that make up the real RLHF recipe, all in one file:
  * policy LM (GRU) + scalar value head, frozen reference copy
  * rollouts sampled from the current policy
  * per-token reward = -beta * KL(pi || ref) estimate, plus the reward-model score on the last token
  * GAE advantages, whitened
  * several PPO epochs with clipped ratio objective and clipped value loss
Reward model (a stand-in for a learned RM): +1 for each occurrence of token 3, -1 for token 0.
Two runs with different beta show the reward / KL-to-reference tradeoff.
"""
import copy
import torch
import torch.nn.functional as F

torch.manual_seed(0)
V, T, H, BOS = 6, 8, 32, 5


class LM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, H)
        self.rnn = torch.nn.GRU(H, H, batch_first=True)
        self.head = torch.nn.Linear(H, V)
        self.value = torch.nn.Linear(H, 1)

    def forward(self, x):                       # x: [B, L] -> logits [B, L, V], values [B, L]
        h, _ = self.rnn(self.emb(x))
        return self.head(h), self.value(h).squeeze(-1)


def reward_model(seq):
    return (seq == 3).float().sum(1) - (seq == 0).float().sum(1)


@torch.no_grad()
def rollout(policy, B):
    x = torch.full((B, 1), BOS)
    for _ in range(T):
        logits, _ = policy(x)
        nxt = torch.distributions.Categorical(logits=logits[:, -1, :BOS]).sample()
        x = torch.cat([x, nxt[:, None]], 1)
    return x


def logprobs_values(model, x):
    logits, values = model(x[:, :-1])
    lp = F.log_softmax(logits[..., :BOS], -1).gather(-1, x[:, 1:, None]).squeeze(-1)
    return lp, values


def gae(rewards, values, gamma=1.0, lam=0.95):
    adv = torch.zeros_like(rewards)
    last = torch.zeros(rewards.shape[0])
    for t in reversed(range(rewards.shape[1])):
        nv = values[:, t + 1] if t + 1 < rewards.shape[1] else 0.0
        last = rewards[:, t] + gamma * nv - values[:, t] + gamma * lam * last
        adv[:, t] = last
    return adv, adv + values


def train(beta, iters=40, clip=0.2, B=128):
    torch.manual_seed(0)
    policy = LM()
    ref = copy.deepcopy(policy).requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    for it in range(iters):
        seq = rollout(policy, B)
        with torch.no_grad():
            old_lp, old_v = logprobs_values(policy, seq)
            ref_lp, _ = logprobs_values(ref, seq)
            kl = old_lp - ref_lp
            rewards = -beta * kl
            score = reward_model(seq[:, 1:])
            rewards[:, -1] += score
            adv, ret = gae(rewards, old_v)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(4):
            lp, v = logprobs_values(policy, seq)
            ratio = (lp - old_lp).exp()
            pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            v_clip = old_v + (v - old_v).clamp(-clip, clip)
            vf = 0.5 * torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()
            loss = pg + 0.5 * vf
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        if it % 10 == 0:
            print(f"  beta={beta:<4} iter {it:2d}  RM score {score.mean():5.2f}  seq KL {kl.sum(1).mean():6.3f}")
    with torch.no_grad():
        seq = rollout(policy, 1024)
        lp, _ = logprobs_values(policy, seq)
        rlp, _ = logprobs_values(ref, seq)
        return reward_model(seq[:, 1:]).mean().item(), (lp - rlp).sum(1).mean().item()


base_score = reward_model(rollout(LM(), 1024)[:, 1:]).mean().item()
results = {b: train(b) for b in (0.02, 1.0)}
print(f"reference policy RM score {base_score:.2f}")
for b, (r, kl) in results.items():
    print(f"beta={b:<4}  final RM score {r:5.2f}  KL to reference {kl:6.2f}")
(r_lo, kl_lo), (r_hi, kl_hi) = results[0.02], results[1.0]
assert r_lo > base_score + 3 and r_hi > base_score + 1   # both improve the reward
assert kl_hi < kl_lo                                      # stronger KL penalty stays closer to ref
