"""NEFTune: add uniform noise to input embeddings during SFT (a cheap regularizer).

  noise ~ Uniform(-1, 1) * alpha / sqrt(L * d),  L = seq len, d = embed dim
Applied only in training mode; inference is unchanged. Implemented as a forward hook on the
embedding layer, which is how it's typically bolted onto an existing model.
"""
import torch
import torch.nn as nn


def add_neftune(embedding: nn.Embedding, alpha=5.0):
    def hook(module, inputs, out):
        if module.training:
            L, d = out.shape[-2], out.shape[-1]
            mag = alpha / (L * d) ** 0.5
            out = out + torch.empty_like(out).uniform_(-mag, mag)
        return out
    return embedding.register_forward_hook(hook)


torch.manual_seed(0)
emb = nn.Embedding(100, 32)
tokens = torch.randint(0, 100, (2, 16))
handle = add_neftune(emb, alpha=5.0)

emb.train()
noisy = emb(tokens)
clean = emb.weight[tokens]
bound = 5.0 / (16 * 32) ** 0.5
print(f"train: max |noise| = {(noisy - clean).abs().max():.4f}  (bound {bound:.4f})")
assert 0 < (noisy - clean).abs().max() <= bound + 1e-6

emb.eval()
assert torch.equal(emb(tokens), clean)
print("eval: embeddings unchanged")
handle.remove()
