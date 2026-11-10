"""GRPO (Group Relative Policy Optimization): PPO without a value network.

For each prompt sample a GROUP of G completions, score them, and use the group-normalized
reward as the advantage:  A_i = (r_i - mean(r_group)) / (std(r_group) + eps).
Toy: "prompts" are integers; the policy must output (prompt mod 4). Reward is 1 if correct.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
P, A, G = 8, 4, 16                       # prompts, actions, group size
target = torch.arange(P) % A
policy = torch.nn.Embedding(P, A)
torch.nn.init.zeros_(policy.weight)
opt = torch.optim.Adam(policy.parameters(), lr=0.1)


def group_advantages(rewards, eps=1e-4):
    """rewards: [num_prompts, G] -> advantages with same shape."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


for step in range(60):
    prompts = torch.arange(P)
    logp = F.log_softmax(policy(prompts), -1)                       # [P, A]
    with torch.no_grad():
        samples = torch.multinomial(logp.exp(), G, replacement=True)  # [P, G]
        rewards = (samples == target[:, None]).float()
        adv = group_advantages(rewards)
    # all-same groups (all right / all wrong) get zero advantage -> no signal, by design
    loss = -(logp.gather(1, samples) * adv).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 15 == 0:
        print(f"step {step:2d}  mean reward {rewards.mean():.3f}")

acc = (policy.weight.argmax(-1) == target).float().mean().item()
print(f"greedy accuracy: {acc:.2f}")
assert acc == 1.0
