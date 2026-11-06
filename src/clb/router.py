"""Hard (0/1) router for the insulated expert. Self-contained: it reads input_ids inside the forward pass, so it
needs no script that sets a mask first and loads in any harness that calls model(...)/model.generate(...).

Rules: the feature is the mean hidden state (decoder layer L, expert-free) over the tokens of the LAST user turn
(`<|im_start|>user\\n ... <|im_end|>`). No chat marker in the prompt (raw few-shot prompts) => gate 0 => the model is M1
exactly. Otherwise gate = 1 iff logistic score >= tau. Off by default: only `load_stack(..., router=path)` installs it.
"""
import torch

from .gate import GateState

IM_START, USER, IM_END = 151644, 872, 151645  # Qwen2.5 chat-template ids (asserted against the tokenizer in tests)


def user_turn_mask(ids):
    """[B,T] bool: content tokens of the last user turn. Rows without a `<|im_start|>user` marker are all False."""
    B, T = ids.shape
    out = torch.zeros(B, T, dtype=torch.bool, device=ids.device)
    for b in range(B):
        starts = ((ids[b, :-1] == IM_START) & (ids[b, 1:] == USER)).nonzero().flatten()
        if len(starts) == 0:
            continue
        s = int(starts[-1]) + 3  # skip <|im_start|> user \n
        ends = (ids[b, s:] == IM_END).nonzero().flatten()
        e = s + int(ends[0]) if len(ends) else T
        out[b, s:e] = True
    return out


class HardRouter(GateState):
    """Drop-in GateState: values[B,1] in {0,1}. `force` (list) overrides it with a constant (oracle runs)."""

    def __init__(self, w, b, mu, sd, tau, layer):
        super().__init__([])
        self.w, self.b, self.mu, self.sd, self.tau, self.layer = w, b, mu, sd, tau, layer
        self.has_ids = False

    @classmethod
    def load(cls, path, device="cuda"):
        d = torch.load(path, map_location=device)
        return cls(d["w"].float(), d["b"].float(), d["mu"].float(), d["sd"].float(), float(d["tau"]), int(d["layer"]))

    def install(self, model):
        model.model.register_forward_pre_hook(self.pre_hook, with_kwargs=True)
        model.model.layers[self.layer].register_forward_hook(self.hook)

    def pre_hook(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        pkv = kwargs.get("past_key_values")
        if ids is not None and ids.shape[1] == 1 and pkv is not None and pkv.get_seq_length() > 0:
            return  # decode step: keep the prefill decision
        self.values = None
        self.has_ids = ids is not None
        self.mask = user_turn_mask(ids) if ids is not None else None

    def logit(self, pooled):
        return ((pooled.float() - self.mu) / self.sd) @ self.w + self.b

    def hook(self, module, args, output):
        if self.values is not None:
            return
        h = output[0] if isinstance(output, tuple) else output
        if not self.has_ids:  # inputs_embeds call: no way to find the user turn
            self.values = torch.zeros(h.shape[0], 1, device=h.device)
            return
        m = self.mask.to(h.device, h.dtype).unsqueeze(-1)
        self.pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1)).detach()
        on = (self.logit(self.pooled) >= self.tau) & (m.sum((1, 2)) > 0)
        self.values = on.float().unsqueeze(1)
