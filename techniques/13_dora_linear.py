"""DoRA: weight-decomposed low-rank adaptation.

  W' = m * (W0 + B A) / ||W0 + B A||_col
The frozen weight is split into a trainable per-column magnitude m and a direction that is
updated by LoRA. Decoupling magnitude from direction gets closer to full fine-tuning.
Toy: adapt a frozen layer to a target linear map.
"""
import torch

torch.manual_seed(0)


class DoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r=2):
        super().__init__()
        self.W0 = base.weight.detach()                       # [out, in], frozen
        self.bias = base.bias.detach()
        self.m = torch.nn.Parameter(self.W0.norm(dim=0, keepdim=True).clone())  # [1, in]
        self.A = torch.nn.Parameter(torch.randn(r, self.W0.shape[1]) * 0.01)
        self.B = torch.nn.Parameter(torch.zeros(self.W0.shape[0], r))

    def weight(self):
        V = self.W0 + self.B @ self.A
        return self.m * V / V.norm(dim=0, keepdim=True)

    def forward(self, x):
        return x @ self.weight().T + self.bias


base = torch.nn.Linear(64, 32)
layer = DoRALinear(base)
assert torch.allclose(layer.weight(), base.weight, atol=1e-6)   # identical at init

target = torch.nn.Linear(64, 32)
x = torch.randn(1024, 64)
y = target(x).detach()
opt = torch.optim.Adam(layer.parameters(), lr=0.05)
for step in range(300):
    loss = ((layer(x) - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
n_train = sum(p.numel() for p in layer.parameters())
print(f"final mse {loss.item():.4f}  trainable params {n_train} vs full {base.weight.numel()}")
