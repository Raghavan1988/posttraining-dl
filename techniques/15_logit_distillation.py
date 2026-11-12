"""Knowledge distillation on logits (Hinton et al.).

  L = T^2 * KL( softmax(teacher/T) || softmax(student/T) )
Temperature T > 1 exposes the teacher's "dark knowledge" (relative probs of wrong classes);
T^2 keeps gradient scale comparable across temperatures.
Toy: a small student MLP distills a larger frozen teacher MLP.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
teacher = torch.nn.Sequential(torch.nn.Linear(10, 64), torch.nn.Tanh(), torch.nn.Linear(64, 5)).eval()
student = torch.nn.Sequential(torch.nn.Linear(10, 8), torch.nn.Tanh(), torch.nn.Linear(8, 5))
for p in teacher.parameters():
    p.requires_grad_(False)


def kd_loss(s_logits, t_logits, T=2.0):
    return F.kl_div(F.log_softmax(s_logits / T, -1), F.log_softmax(t_logits / T, -1),
                    log_target=True, reduction="batchmean") * T * T


opt = torch.optim.Adam(student.parameters(), lr=0.01)
for step in range(500):
    x = torch.randn(128, 10)
    loss = kd_loss(student(x), teacher(x))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}  kd loss {loss.item():.4f}")

x = torch.randn(2000, 10)
with torch.no_grad():
    agree = (student(x).argmax(-1) == teacher(x).argmax(-1)).float().mean().item()
print(f"top-1 agreement with teacher: {agree:.3f}")
assert agree > 0.8
