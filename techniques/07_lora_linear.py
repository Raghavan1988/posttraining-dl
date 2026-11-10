"""LoRA: parameter-efficient fine-tuning with a low-rank update.

  y = W x + (alpha / r) * B A x,   W frozen, A: [r, in], B: [out, r], B initialized to 0
so at init the adapted layer is exactly the base layer. Only A, B are trained, and
after training they can be merged: W' = W + (alpha/r) B A (zero inference overhead).
Toy: the target mapping differs from the frozen base by a rank-2 matrix; LoRA r=2 fits it.
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / r**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merge(self) -> nn.Linear:
        merged = nn.Linear(self.base.in_features, self.base.out_features)
        merged.weight.data = self.base.weight + self.scale * self.B @ self.A
        merged.bias.data = self.base.bias.clone()
        return merged


torch.manual_seed(0)
D = 64
base = nn.Linear(D, D)
delta = torch.randn(D, 2) @ torch.randn(2, D) * 0.1          # rank-2 "task shift"
x = torch.randn(1024, D)
y = base(x).detach() + x @ delta.T

lora = LoRALinear(base, r=2, alpha=4)
assert torch.allclose(lora(x), base(x))                        # identical at init
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(f"trainable params: {trainable} vs full layer {D*D + D}")
opt = torch.optim.Adam([lora.A, lora.B], lr=1e-2)
for step in range(500):
    loss = (lora(x) - y).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"final MSE: {loss.item():.2e}")
merged = lora.merge()
assert torch.allclose(merged(x), lora(x), atol=1e-4) and loss.item() < 1e-3
print("merged layer matches adapter output")
