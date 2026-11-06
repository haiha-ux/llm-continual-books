"""LoRA checkpoints and stacks: model = base + merge(stage1) + merge(stage2) ... (+ optional gated experts)."""
import json
from pathlib import Path

import torch

from . import lora
from .gate import GateMLP, GateState
from .llm import load
from .router import HardRouter


def save_lora(path, model, meta, mods=None):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    keep = {id(m) for m in mods} if mods is not None else None
    torch.save({n: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()} for n, m in lora.named(model)
                if keep is None or id(m) in keep}, path / "lora.pt")
    json.dump(meta, open(path / "meta.json", "w"), indent=1)


def apply_lora(model, path, merge=True, gate=None, trainable=False):
    """Inject the saved adapter; merge into weights (stage) or keep wrapped (expert)."""
    path = Path(path)
    meta = json.load(open(path / "meta.json"))
    sd = torch.load(path / "lora.pt")
    mods = lora.inject(model, r=meta["r"], alpha=meta["alpha"], min_layer=meta.get("min_layer", 0), gate=gate)
    lookup = {id(m): n for n, m in lora.named(model)}
    for m in mods:
        n = lookup[id(m)]
        m.lora_A.data.copy_(sd[n]["A"]); m.lora_B.data.copy_(sd[n]["B"])
        m.lora_A.requires_grad = m.lora_B.requires_grad = trainable
    if merge:
        lora.merge_and_unwrap(model, mods)
    return mods


def load_stack(stages, experts=(), base=None, router=None):
    """stages: merged LoRA dirs in order. experts: dirs with lora.pt/meta.json/gate.pt (kept separate, gated).
    Returns model, tok, GateState|None, expert modules."""
    model, tok = load(base) if base else load()
    for p in stages:
        apply_lora(model, p, merge=True)
    if not experts:
        return model, tok, None, []
    gates = []
    for p in experts:
        meta = json.load(open(Path(p) / "meta.json"))
        g = GateMLP(model.config.hidden_size, meta.get("gate_hidden", 64)).to(model.device)
        g.load_state_dict(torch.load(Path(p) / "gate.pt"))
        gates.append(g)
    st = GateState(gates)
    if router:  # hard 0/1 gate decided inside the forward pass (src/clb/router.py); replaces the soft MLP gate
        st = HardRouter.load(router, model.device)
    mods = []
    for k, p in enumerate(experts):
        mods += apply_lora(model, p, merge=False, gate=st.getter(k))
    gl = json.load(open(Path(experts[0]) / "meta.json"))["gate_layer"]
    if router:
        st.install(model)
    else:
        model.model.layers[gl].register_forward_hook(st.hook)
    return model, tok, st, mods
