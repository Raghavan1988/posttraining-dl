"""Completion-only SFT loss: mask prompt tokens out of the cross-entropy.

Labels for prompt (and padding) positions are set to -100 so only response tokens are
learned. Otherwise the model spends capacity re-learning user prompts.
Toy: show which positions contribute and that masked loss ignores prompt tokens entirely.
"""
import torch
import torch.nn.functional as F

IGNORE = -100


def build_labels(prompt_lens, input_ids, pad_id=0):
    labels = input_ids.clone()
    for i, n in enumerate(prompt_lens):
        labels[i, :n] = IGNORE
    labels[input_ids == pad_id] = IGNORE
    return labels


def causal_lm_loss(logits, labels):
    # predict token t+1 from position t
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                           labels[:, 1:].reshape(-1), ignore_index=IGNORE)


input_ids = torch.tensor([[5, 6, 7, 1, 2, 3, 0],      # prompt=3, response=3, pad=1
                          [5, 6, 1, 2, 0, 0, 0]])     # prompt=2, response=2, pad=3
labels = build_labels([3, 2], input_ids)
print("labels:\n", labels)

torch.manual_seed(0)
logits = torch.randn(2, 7, 10, requires_grad=True)
loss = causal_lm_loss(logits, labels)
loss.backward()
used = logits.grad.abs().sum(-1) > 0
print("positions receiving gradient:\n", used.int())
# gradient flows only from positions that predict a response token
expected = torch.tensor([[0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
assert torch.equal(used, expected)
print(f"loss over {int((labels != IGNORE).sum())} response tokens: {loss.item():.4f}")
