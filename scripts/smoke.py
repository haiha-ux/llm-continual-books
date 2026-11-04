"""Smoke test (<1 min): random tiny Qwen2 + real tokenizer, programmatic cloze QA from 3 passages (no teacher),
runs M1 -> naive / KL replay / insulated expert + gate -> eval. Checks plumbing, not accuracy."""
import json, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from clb import lora
from clb.evals import eval_books, summarize
from clb.gate import GateState, train_gate
from clb.llm import chat, generate
from clb.train import encode, fit, qa_row

t0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", padding_side="left")
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2)
model = Qwen2ForCausalLM(cfg).to(dev).to(torch.bfloat16).eval()


def cloze(book):  # programmatic QA: "which word follows <5 words>?" -> next word
    out = []
    for l in list(open(ROOT / "data/books" / f"book{book}.jsonl"))[:3]:
        w = json.loads(l)["text"].split()
        out += [{"q": "After the words " + " ".join(w[i:i + 5]) + " which word comes next?", "a": w[i + 5].strip(".,;:")} for i in range(0, 60, 15)]
    return out[:12]


b1, b2 = cloze(1), cloze(2)
mods = lora.inject(model, 4, 8)
fit(model, mods, encode(tok, [qa_row(r) for r in b1], 128), tok.pad_token_id, epochs=2, bs=4, log=lambda m: None, device=dev)
lora.merge_and_unwrap(model, mods)  # M1
replay = encode(tok, [(chat(r["q"]), r["a"] + "<|im_end|>") for r in b1], 128)
for name, kw in (("naive", {}), ("kl", dict(replay=replay, kl_lambda=1.0))):
    snap = {k: v.clone() for k, v in model.state_dict().items()}
    m2 = lora.inject(model, 4, 8)
    s = fit(model, m2, encode(tok, [qa_row(r) for r in b2], 128), tok.pad_token_id, epochs=2, bs=4, log=lambda m: None, device=dev, **kw)
    r = eval_books(model, tok, b1[:4], gen=generate)
    print(name, "ok, gpu_h", round(s["gpu_hours"], 6), summarize(r))
    lora.merge_and_unwrap(model, m2); model.load_state_dict(snap)
# insulated expert with gate on layers > 1
st = GateState([])
st.gates = []
h = model.model.layers[1].register_forward_hook(st.hook)
feats = lambda qs: (setattr(st, "mask", tok([chat(q) for q in qs], return_tensors="pt", padding=True).to(dev)["attention_mask"]), st.reset(),
                    model.model(**tok([chat(q) for q in qs], return_tensors="pt", padding=True).to(dev)), st.pooled.float().cpu())[-1]
pos, neg = feats([r["q"] for r in b2]), feats([r["q"] for r in b1])
gate = train_gate(pos, neg, 64, steps=200, hidden=16).to(dev)
st.gates = [gate]
h.remove()  # expert is trained with gate fully on (no hook), gated only at inference
ex = lora.inject(model, 4, 8, min_layer=2)
fit(model, ex, encode(tok, [qa_row(r) for r in b2], 128), tok.pad_token_id, epochs=2, bs=4, log=lambda m: None, device=dev)
for m in ex:
    m.gate = st.getter(0)
model.model.layers[1].register_forward_hook(st.hook)
mask_cb = lambda m: (setattr(st, "mask", m), st.reset())
rows = eval_books(model, tok, b2[:4], gen=lambda m, t, p, **kw: generate(m, t, p, on_batch=mask_cb, **kw))
print("expert ok; gate on book2/book1:", float(torch.sigmoid(gate(pos.to(dev))).mean()), float(torch.sigmoid(gate(neg.to(dev))).mean()))
print(f"SMOKE OK {time.time() - t0:.1f}s")
