"""GRPO with a verifiable reward (RLVR) on a toy addition task.

Prompt: two digits "a + b"; the policy (small transformer) answers with two tokens (tens, units).
Reward is checked programmatically: 1 if the answer equals a+b, else 0 -- no reward model.
GRPO update per prompt group of G samples:
  A_i = (r_i - mean(r)) / (std(r) + eps)            (no value network)
  loss = -mean_i min(ratio_i A_i, clip(ratio_i) A_i) + beta * KL(pi || ref)   (k3 estimator)
The policy is first briefly SFT'd on a small slice of the data (warm start, as in practice),
then GRPO with only the verifier signal improves on the problems never shown in SFT.
"""
import copy
import torch
import torch.nn.functional as F

torch.manual_seed(0)
D, PLUS, BOS = 10, 10, 11
VOCAB = 12


class Policy(torch.nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.tok = torch.nn.Embedding(VOCAB, d)
        self.pos = torch.nn.Embedding(8, d)
        layer = torch.nn.TransformerEncoderLayer(d, 4, 128, dropout=0.0, batch_first=True)
        self.tf = torch.nn.TransformerEncoder(layer, 2)
        self.head = torch.nn.Linear(d, D)

    def forward(self, x):
        L = x.shape[1]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(L)
        h = self.tf(self.tok(x) + self.pos(torch.arange(L)), mask=mask, is_causal=True)
        return self.head(h)


def prompts(a, b):
    return torch.stack([torch.full_like(a, BOS), a, torch.full_like(a, PLUS), b], 1)


def answer_logprobs(model, p, ans):          # log-prob of each of the 2 answer tokens
    x = torch.cat([p, ans[:, :1]], 1)
    lp = F.log_softmax(model(x)[:, -2:], -1)
    return lp.gather(-1, ans[:, :, None]).squeeze(-1)


@torch.no_grad()
def generate(model, p, greedy=False):
    x = p
    for _ in range(2):
        logits = model(x)[:, -1]
        nxt = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        x = torch.cat([x, nxt[:, None]], 1)
    return x[:, -2:]


def verify(a, b, ans):
    return ((ans[:, 0] * 10 + ans[:, 1]) == a + b).float()


@torch.no_grad()
def accuracy(model):
    a, b = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
    a, b = a.flatten(), b.flatten()
    return verify(a, b, generate(model, prompts(a, b), greedy=True)).mean().item()


policy = Policy()
# --- warm start: SFT on a random 30% of the 100 problems ---
sft_ids = torch.randperm(100)[:30]
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
for _ in range(200):
    ids = sft_ids[torch.randint(0, 30, (128,))]
    a, b = ids // 10, ids % 10
    s = a + b
    loss = -answer_logprobs(policy, prompts(a, b), torch.stack([s // 10, s % 10], 1)).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
acc_sft = accuracy(policy)
print(f"after small SFT warm start: accuracy on all 100 problems {acc_sft:.2f}")

ref = copy.deepcopy(policy).requires_grad_(False)
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
G, P, clip, beta = 16, 32, 0.2, 0.02
for step in range(600):
    a, b = torch.randint(0, 10, (P,)), torch.randint(0, 10, (P,))
    a, b = a.repeat_interleave(G), b.repeat_interleave(G)
    p = prompts(a, b)
    ans = generate(policy, p)
    r = verify(a, b, ans).view(P, G)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1, 1)
    with torch.no_grad():
        old = answer_logprobs(policy, p, ans)
        ref_lp = answer_logprobs(ref, p, ans)
    for _ in range(2):
        lp = answer_logprobs(policy, p, ans)
        ratio = (lp - old).exp()
        pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
        kl = (ref_lp - lp).exp() - (ref_lp - lp) - 1           # k3: unbiased, always >= 0
        loss = (pg + beta * kl).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}  mean reward {r.mean():.3f}  greedy accuracy {accuracy(policy):.2f}")

acc = accuracy(policy)
print(f"accuracy: SFT warm start {acc_sft:.2f} -> after GRPO {acc:.2f}")
assert acc > acc_sft + 0.3
