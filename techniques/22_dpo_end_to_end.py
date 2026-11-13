"""End-to-end offline DPO on a tiny autoregressive LM.

Pipeline:
  1. sample response pairs from the frozen reference (SFT) model
  2. label each pair with a hidden "human" reward (Bradley-Terry sampled, so labels are noisy)
  3. train with the DPO loss  -log sigmoid(beta * [(lp_c - ref_c) - (lp_r - ref_r)])
  4. evaluate: (a) implicit reward beta*log(pi/ref) ranks held-out pairs like the hidden reward
               (b) samples from the trained policy score higher under the hidden reward
Hidden reward: number of strictly increasing adjacent token pairs (a "sortedness" preference).
"""
import copy
import torch
import torch.nn.functional as F

torch.manual_seed(0)
V, T, H, BOS = 6, 8, 64, 6


class LM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V + 1, H)
        self.rnn = torch.nn.GRU(H, H, batch_first=True)
        self.head = torch.nn.Linear(H, V)

    def forward(self, x):
        return self.head(self.rnn(self.emb(x))[0])


def hidden_reward(y):
    return (y[:, 1:] > y[:, :-1]).float().sum(1)


@torch.no_grad()
def sample(model, B):
    x = torch.full((B, 1), BOS)
    for _ in range(T):
        nxt = torch.distributions.Categorical(logits=model(x)[:, -1]).sample()
        x = torch.cat([x, nxt[:, None]], 1)
    return x


def seq_logprob(model, x):
    lp = F.log_softmax(model(x[:, :-1]), -1)
    return lp.gather(-1, x[:, 1:, None]).squeeze(-1).sum(1)


def make_pairs(ref, n):
    a, b = sample(ref, n), sample(ref, n)
    ra, rb = hidden_reward(a[:, 1:]), hidden_reward(b[:, 1:])
    keep = ra != rb
    a, b, ra, rb = a[keep], b[keep], ra[keep], rb[keep]
    a_wins = torch.rand(len(a)) < torch.sigmoid(ra - rb)          # noisy human labels
    chosen = torch.where(a_wins[:, None], a, b)
    rejected = torch.where(a_wins[:, None], b, a)
    return chosen, rejected


ref = LM().requires_grad_(False)
policy = copy.deepcopy(ref).requires_grad_(True)
train_c, train_r = make_pairs(ref, 6000)
test_a, test_b = sample(ref, 2000), sample(ref, 2000)
beta = 0.1
with torch.no_grad():
    ref_c, ref_r = seq_logprob(ref, train_c), seq_logprob(ref, train_r)
print(f"{len(train_c)} preference pairs")

opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
for epoch in range(4):
    perm = torch.randperm(len(train_c))
    for i in range(0, len(perm), 256):
        idx = perm[i:i + 256]
        h = (seq_logprob(policy, train_c[idx]) - ref_c[idx]) - (seq_logprob(policy, train_r[idx]) - ref_r[idx])
        loss = -F.logsigmoid(beta * h).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ia = beta * (seq_logprob(policy, test_a) - seq_logprob(ref, test_a))
        ib = beta * (seq_logprob(policy, test_b) - seq_logprob(ref, test_b))
        ha, hb = hidden_reward(test_a[:, 1:]), hidden_reward(test_b[:, 1:])
        m = ha != hb
        acc = ((ia > ib) == (ha > hb))[m].float().mean().item()
    print(f"epoch {epoch}  loss {loss.item():.4f}  held-out implicit-reward ranking acc {acc:.3f}")

r_ref = hidden_reward(sample(ref, 4000)[:, 1:]).mean().item()
r_pol = hidden_reward(sample(policy, 4000)[:, 1:]).mean().item()
print(f"hidden reward of samples: reference {r_ref:.2f} -> DPO policy {r_pol:.2f}")
assert acc > 0.7 and r_pol > r_ref + 1.0
