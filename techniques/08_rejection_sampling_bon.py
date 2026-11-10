"""Rejection-sampling fine-tuning (RFT / best-of-N distillation).

Loop: sample N candidates per prompt from the current policy, keep the best (or only the
verified-correct) ones, then do plain SFT on the kept samples. Repeat.
Toy task: "add two digits". Policy is a per-prompt categorical over answers 0..18;
the verifier checks the sum. Accuracy climbs with no reward gradient — only SFT on filtered data.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
pairs = torch.cartesian_prod(torch.arange(10), torch.arange(10))   # 100 prompts
answers = pairs.sum(-1)
policy = torch.nn.Embedding(len(pairs), 19)
opt = torch.optim.Adam(policy.parameters(), lr=0.5)
N = 16


def greedy_acc():
    return (policy.weight.argmax(-1) == answers).float().mean().item()


for rnd in range(10):
    with torch.no_grad():
        samples = torch.multinomial(policy.weight.softmax(-1), N, replacement=True)  # [100, N]
        correct = samples == answers[:, None]
        keep_prompt = correct.any(1)                                # verifier filter
        kept = answers[keep_prompt]                                 # the accepted answer
    for _ in range(20):                                             # SFT on accepted data
        logits = policy(torch.arange(len(pairs))[keep_prompt])
        loss = F.cross_entropy(logits, kept)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"round {rnd}  kept {keep_prompt.sum().item():3d}/100  greedy acc {greedy_acc():.2f}")

assert greedy_acc() > 0.9
