"""Train one stage from a config: m1 | naive | kl | expert.  python scripts/run_method.py configs/X.json"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from clb import lora
from clb.ckpt import load_stack, save_lora
from clb.gate import GateState, train_gate
from clb.llm import cap_gpu, chat
from clb.train import encode, fit, note_row, qa_row

QW = "Write one factual question about the book you have studied."


def rows(path):
    return [json.loads(l) for l in open(ROOT / path)]


@torch.no_grad()
def pooled_features(model, tok, st, questions, bs=32):
    feats = []
    for s in range(0, len(questions), bs):
        enc = tok([chat(q) for q in questions[s:s + bs]], return_tensors="pt", padding=True).to(model.device)
        st.mask = enc["attention_mask"]; st.reset()
        model.model(**enc)
        feats.append(st.pooled.float().cpu())
    return torch.cat(feats)


def main(cfg_path, overrides=()):
    cfg = json.load(open(cfg_path))
    for o in overrides:  # key=json_value, e.g. seed=1 stages='["ckpt/m1_book1"]'
        k, v = o.split("=", 1)
        cfg[k] = json.loads(v) if v[:1] in "[{\"0123456789-" else v
    cap_gpu(0.3)
    log = lambda m: print(m, flush=True)
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    model, tok, st0, _ = load_stack([ROOT / s for s in cfg.get("stages", [])], [ROOT / e for e in cfg.get("experts", [])])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    method, b = cfg["method"], cfg["book"]
    qa = [qa_row(r) for r in rows(f"data/final/book{b}/train_qa.jsonl")]
    notes = [note_row(r) for r in rows(f"data/final/book{b}/train_notes.jsonl")]
    extra = []
    if cfg.get("question_writing"):  # lets the model later write book-style questions itself (self-replay)
        extra = [(chat(QW), r["q"] + "<|im_end|>") for r in rows(f"data/final/book{b}/train_qa.jsonl")[:cfg.get("n_qw", 600)]]
    data = encode(tok, qa + notes + extra, cfg.get("maxlen", 320))
    min_layer = cfg.get("expert_min_layer", 0) if method == "expert" else 0
    mods = lora.inject(model, cfg["r"], cfg["alpha"], min_layer=min_layer)
    replay = None
    if method == "kl":
        replay = encode(tok, [(chat(r["q"]), r["a"] + "<|im_end|>") for r in rows(cfg["replay"])], cfg.get("maxlen", 320))
    log(f"{cfg['name']}: {len(data)} train rows, {sum(m.lora_A.numel() + m.lora_B.numel() for m in mods) / 1e6:.1f}M lora params, replay {len(replay or [])}")
    stats = fit(model, mods, data, tok.pad_token_id, epochs=cfg["epochs"], bs=cfg["bs"], lr=cfg["lr"], seed=cfg["seed"],
                replay=replay, kl_lambda=cfg.get("kl_lambda", 0.0), replay_bs=cfg.get("replay_bs", 8), log=log,
                partial=out / "partial.pt", save_every=cfg.get("save_every", 100))
    meta = dict(cfg, **{k: v for k, v in stats.items() if k != "hist"}, min_layer=min_layer)
    meta["hist"] = stats["hist"]
    if method == "expert":
        L = cfg["gate_layer"]
        meta["gate_layer"] = L
        st = st0
        if st is None:
            st = GateState([])
            model.model.layers[L].register_forward_hook(st.hook)
        model.config.use_cache = True
        pos_q = [r["q"] for r in rows(f"data/final/book{b}/train_qa.jsonl")]
        neg = rows(cfg["replay"])
        neg_q = [r["q"] for r in neg]
        pos, ng = pooled_features(model, tok, st, pos_q), pooled_features(model, tok, st, neg_q)
        gate = train_gate(pos, ng, pos.shape[1], steps=cfg.get("gate_steps", 400), seed=cfg["seed"], hidden=cfg.get("gate_hidden", 64))
        save_lora(out, model, meta, mods)
        torch.save(gate.state_dict(), out / "gate.pt")
        with torch.no_grad():
            log(f"gate train: on={float(torch.sigmoid(gate(pos)).mean()):.3f} off={float(torch.sigmoid(gate(ng)).mean()):.3f}")
    else:
        save_lora(out, model, meta, mods)
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    log(f"saved {out}  gpu_hours={stats['gpu_hours']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
