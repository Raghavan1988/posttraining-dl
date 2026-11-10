"""Direct Preference Optimization (DPO) loss.

DPO skips the explicit reward model: the implicit reward is beta * log(pi/pi_ref).
  loss = -log sigmoid( beta * [(logpi_c - logref_c) - (logpi_r - logref_r)] )
Toy: a policy over a 5-token vocab for a single step, initialized equal to the reference.
Training on "token 2 > token 4" should raise p(2) and lower p(4) relative to the reference.
"""
import torch
import torch.nn.functional as F


def dpo_loss(pc, pr, rc, rr, beta=0.1):
    """Inputs are summed sequence log-probs under policy (p*) and reference (r*)."""
    logits = beta * ((pc - rc) - (pr - rr))
    return -F.logsigmoid(logits).mean(), (beta * (pc - rc)).detach(), (beta * (pr - rr)).detach()


torch.manual_seed(0)
ref_logits = torch.randn(5)
policy_logits = ref_logits.clone().requires_grad_(True)
ref_lp = F.log_softmax(ref_logits, -1)
chosen, rejected = torch.tensor([2]), torch.tensor([4])
opt = torch.optim.SGD([policy_logits], lr=1.0)

for step in range(50):
    lp = F.log_softmax(policy_logits, -1)
    loss, r_c, r_r = dpo_loss(lp[chosen], lp[rejected], ref_lp[chosen], ref_lp[rejected])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0:
        print(f"step {step:2d} loss {loss.item():.4f}  reward_c {r_c.item():+.3f}  reward_r {r_r.item():+.3f}")

p, p_ref = policy_logits.softmax(-1), ref_logits.softmax(-1)
print("ref   :", [round(x, 3) for x in p_ref.tolist()])
print("policy:", [round(x, 3) for x in p.tolist()])
assert p[2] > p_ref[2] and p[4] < p_ref[4]
