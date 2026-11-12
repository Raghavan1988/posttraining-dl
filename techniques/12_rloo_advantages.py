"""RLOO: REINFORCE Leave-One-Out baseline.

For k samples per prompt with rewards r_1..r_k, each sample's baseline is the mean of the
OTHER k-1 rewards:  A_i = r_i - (sum_j r_j - r_i) / (k - 1).
Unbiased (baseline independent of the sample) and needs no learned value network.
Toy: a categorical "policy" over 5 answers learns to pick the rewarded one.
"""
import torch

torch.manual_seed(0)


def rloo_advantages(r):                 # r: [prompts, k]
    k = r.shape[1]
    return r - (r.sum(1, keepdim=True) - r) / (k - 1)


r = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
print("rewards   ", r.tolist(), "\nadvantages", rloo_advantages(r).tolist())

reward = torch.tensor([0.0, 0.2, 1.0, 0.1, 0.0])  # answer 2 is best
logits = torch.zeros(5, requires_grad=True)
opt = torch.optim.SGD([logits], lr=0.5)
k = 8
for step in range(100):
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample((k,))
    adv = rloo_advantages(reward[a][None])[0]
    loss = -(adv * dist.log_prob(a)).mean()
    opt.zero_grad(); loss.backward(); opt.step()

probs = logits.softmax(0)
print("final policy", [round(p, 3) for p in probs.tolist()])
assert probs.argmax() == 2
