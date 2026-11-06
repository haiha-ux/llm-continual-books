import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from clb import lora, router
from clb.router import HardRouter, user_turn_mask


@pytest.fixture(autouse=True)
def tiny_ids(monkeypatch):  # the tiny model has vocab 100: use small stand-ins for the chat-template ids
    monkeypatch.setattr(router, "IM_START", 5)
    monkeypatch.setattr(router, "USER", 6)
    monkeypatch.setattr(router, "IM_END", 7)


def test_user_turn_mask_last_user_turn_only():
    ids = torch.tensor([[5, 6, 8, 11, 12, 7, 8, 5, 9, 8, 7, 8],        # system-less: user, assistant
                        [5, 6, 8, 1, 7, 8, 5, 6, 8, 2, 3, 7],          # two user turns -> the last one
                        [1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4]])         # no marker -> nothing
    m = user_turn_mask(ids)
    assert m[0].nonzero().flatten().tolist() == [3, 4]
    assert m[1].nonzero().flatten().tolist() == [9, 10]
    assert not m[2].any()


def make(tau):
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    ids = torch.tensor([[5, 6, 8, 11, 12, 13, 7, 8, 5, 9, 8]])
    ref = m(ids).logits
    r = HardRouter(torch.ones(32), torch.zeros(1), torch.zeros(32), torch.ones(32), tau, layer=1)
    for x in lora.inject(m, r=4, alpha=8, min_layer=2, gate=r.getter(0)):
        torch.nn.init.normal_(x.lora_B, std=0.5)
    r.install(m)
    return m, r, ids, ref


def test_gate_off_equals_base_and_on_differs_and_raw_prompt_is_off():
    m, r, ids, ref = make(tau=1e9)  # never fires
    assert torch.equal(m(ids).logits, ref) and r.values.sum() == 0
    m, r, ids, ref = make(tau=-1e9)  # always fires on a chat prompt
    assert not torch.allclose(m(ids).logits, ref, atol=1e-3) and r.values.sum() == 1
    raw = torch.tensor([[1, 2, 3, 4, 4, 4]])  # no chat marker -> gate 0 even with tau=-inf
    base = make(tau=1e9)[0](raw).logits
    assert torch.equal(m(raw).logits, base) and r.values.sum() == 0


def test_decode_steps_keep_prefill_gate_and_forced_off_equals_base():
    g = dict(max_new_tokens=6, do_sample=False, pad_token_id=0)
    mo, ro, ids, _ = make(tau=1e9)
    base = mo.generate(ids, **g)
    m, r, ids, _ = make(tau=-1e9)
    on = m.generate(ids, **g)
    assert r.values.sum() == 1 and not torch.equal(on, base)
    r.force = [0.0]
    assert torch.equal(m.generate(ids, **g), base)
