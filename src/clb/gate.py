"""Knowledge-insulated expert gate.

One forward hook on decoder layer L captures the mean hidden state of the *question* tokens (state.mask) and turns it
into one gate value per expert and per sequence. Layers <= L run the frozen model untouched (gate input is expert-free);
experts live on layers > L and are scaled by their gate. At prefill the gate is computed; decode steps reuse it.
"""
import torch
from torch import nn


class GateMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, pooled):  # logits [B]
        return self.net((pooled.float() - self.mu) / self.sd).squeeze(-1)


class GateState:
    def __init__(self, gates: list):
        self.gates, self.mask, self.pooled, self.values, self.force = gates, None, None, None, None  # force: fixed values

    def getter(self, k: int):
        def g():
            if self.force is not None:
                return torch.full((1, 1, 1), float(self.force[k]), device=self.mask.device)  # same device as the activations
            return self.values[:, k].view(-1, 1, 1)
        return g

    def hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] == 1 and self.values is not None:  # decode step: keep prefill gate
            return
        m = self.mask.to(h.device, h.dtype).unsqueeze(-1)
        self.pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1)).detach()
        if self.gates:
            self.values = torch.stack([torch.sigmoid(g(self.pooled)) for g in self.gates], 1)

    def reset(self):
        self.values = None


def train_gate(pos, neg, dim, steps=300, lr=1e-3, wd=1e-2, seed=0, hidden=64):
    """pos/neg: float tensors [n,dim] of pooled features. Class-balanced BCE; returns trained GateMLP."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    x = torch.cat([pos, neg]).float()
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    w = torch.where(y == 1, 0.5 / len(pos), 0.5 / len(neg)) * len(y)
    gate = GateMLP(dim, hidden)
    gate.mu.copy_(x.mean(0)); gate.sd.copy_(x.std(0).clamp(min=1e-3))
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (min(128, len(x)),), generator=g)
        loss = (nn.functional.binary_cross_entropy_with_logits(gate(x[idx]), y[idx], reduction="none") * w[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return gate
