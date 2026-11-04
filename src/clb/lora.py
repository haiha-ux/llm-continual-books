"""Hand-written LoRA (no peft). Adapted from a small LoRA used in an earlier VLA repo.

W x + g * (alpha / r) * B (A x). A, B are fp32 (stable training on a bf16 base), B starts at zero (exact no-op).
`gate` (optional) returns a [B,1,1] tensor scaling the delta per sequence; used by the knowledge-insulated expert.
Wrappers nest, so two experts are just LoRALinear(LoRALinear(base)).
"""
import re

import torch
from torch import nn

ALL_LINEAR = r"model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, r: int, alpha: float, gate=None):
        super().__init__()
        self.base, self.scale, self.gate, self.enabled = base, alpha / r, gate, True
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=w.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=w.device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    @property
    def weight(self):
        return self.base.weight

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def delta(self, x):
        d = (x.to(torch.float32) @ self.lora_A.T @ self.lora_B.T) * self.scale
        return d if self.gate is None else d * self.gate().to(d.dtype)

    def forward(self, x):
        y = self.base(x)
        return y + self.delta(x).to(y.dtype) if self.enabled else y


def inject(model, r=32, alpha=64, min_layer=0, gate=None, pattern=ALL_LINEAR):
    """Wrap matching Linear layers (layer index >= min_layer). Returns the new LoRALinear modules (trainable)."""
    pat, new = re.compile(pattern), []
    for name, mod in list(model.named_modules()):
        m = pat.search(name)
        if m and isinstance(mod, (nn.Linear, LoRALinear)) and int(m.group(1)) >= min_layer:
            parent, _, leaf = name.rpartition(".")
            w = LoRALinear(mod, r, alpha, gate)
            setattr(model.get_submodule(parent), leaf, w)
            new.append(w)
    if not new:
        raise ValueError("no module matched")
    return new


def freeze_except(model, modules):
    for p in model.parameters():
        p.requires_grad = False
    for m in modules:
        m.lora_A.requires_grad = m.lora_B.requires_grad = True


def set_enabled(modules, flag: bool):
    for m in modules:
        m.enabled = flag


def state(model_modules_named):
    return {k: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu(), "scale": m.scale} for k, m in model_modules_named}


def named(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


@torch.no_grad()
def merge_and_unwrap(model, modules=None):
    """Fold LoRA deltas (gate must be None) into the bf16 base weights and remove the wrappers."""
    for name, mod in named(model):
        if modules is not None and mod not in modules:
            continue
        if mod.gate is not None:
            raise ValueError("cannot merge a gated adapter")
        w = mod.base.weight
        w += (mod.lora_B @ mod.lora_A * mod.scale).to(w.dtype)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, mod.base)
