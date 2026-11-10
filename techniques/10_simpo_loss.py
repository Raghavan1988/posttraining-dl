"""SimPO: reference-free preference optimization with length-normalized rewards.

  r(y) = (beta / |y|) * sum_t log pi(y_t | x, y_<t)     (average log-prob, no reference model)
  loss = -log sigmoid( r(chosen) - r(rejected) - gamma )   (gamma = target reward margin)
Length normalization removes the incentive to prefer short or long answers just by length.
Toy: a tiny bigram LM; the chosen sequence is long, the rejected one short.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
V = 8
lm = torch.nn.Embedding(V, V)                     # bigram: logits for next token given prev


def avg_logprob(seq):
    logits = lm(seq[:-1])
    return F.log_softmax(logits, -1).gather(1, seq[1:, None]).mean()


def simpo_loss(chosen, rejected, beta=2.0, gamma=0.5):
    margin = beta * avg_logprob(chosen) - beta * avg_logprob(rejected) - gamma
    return -F.logsigmoid(margin), margin.detach()


chosen = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
rejected = torch.tensor([0, 7, 7])
opt = torch.optim.Adam(lm.parameters(), lr=0.05)
for step in range(150):
    loss, margin = simpo_loss(chosen, rejected)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  margin {margin.item():+.3f}")

with torch.no_grad():
    print(f"avg logp chosen {avg_logprob(chosen):.3f}  rejected {avg_logprob(rejected):.3f}")
    assert avg_logprob(chosen) > avg_logprob(rejected)
