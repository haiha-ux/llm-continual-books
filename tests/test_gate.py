import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from clb import lora
from clb.gate import GateMLP, GateState, train_gate


def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2)
    return Qwen2ForCausalLM(cfg).eval()


def make(gate_value=None):
    m = tiny()
    ref = m(torch.tensor([[1, 2, 3, 4]])).logits
    st = GateState([GateMLP(32)])
    mods = lora.inject(m, r=4, alpha=8, min_layer=2, gate=st.getter(0))
    for x in mods:
        torch.nn.init.normal_(x.lora_B, std=0.5)
    m.model.layers[1].register_forward_hook(st.hook)
    return m, st, ref


def test_gate_zero_equals_base_and_one_differs():
    m, st, ref = make()
    ids = torch.tensor([[1, 2, 3, 4]])
    st.mask = torch.ones_like(ids)
    st.force = [0.0]
    assert torch.allclose(m(ids).logits, ref, atol=1e-5)
    st.force = [1.0]
    assert not torch.allclose(m(ids).logits, ref, atol=1e-3)


def test_gate_is_computed_from_expert_free_layers():
    m, st, _ = make()
    ids = torch.tensor([[1, 2, 3, 4]])
    st.mask = torch.ones_like(ids)
    m(ids)
    p1 = st.pooled.clone()
    lora.set_enabled([x for _, x in lora.named(m)], False)
    st.reset(); m(ids)
    assert torch.allclose(p1, st.pooled)  # pooled feature independent of the expert


def test_prefill_gate_persists_over_decode():
    m, st, _ = make()
    ids = torch.tensor([[1, 2, 3, 4]])
    st.mask = torch.ones_like(ids)
    out = m(ids, use_cache=True)
    g0 = st.values.clone()
    m(torch.tensor([[5]]), past_key_values=out.past_key_values, use_cache=True)
    assert torch.equal(st.values, g0)


def test_train_gate_separates():
    torch.manual_seed(0)
    pos, neg = torch.randn(60, 8) + 2, torch.randn(80, 8) - 2
    g = train_gate(pos, neg, 8, steps=200)
    assert (torch.sigmoid(g(pos)) > 0.5).float().mean() > 0.95
    assert (torch.sigmoid(g(neg)) < 0.5).float().mean() > 0.95


def test_merge_matches_lora_forward():
    m = tiny(); ids = torch.tensor([[1, 2, 3, 4]])
    mods = lora.inject(m, r=4, alpha=8)
    for x in mods:
        torch.nn.init.normal_(x.lora_B, std=0.1)
    before = m(ids).logits
    lora.merge_and_unwrap(m)
    assert torch.allclose(before, m(ids).logits, atol=1e-4)
    assert not lora.named(m)
