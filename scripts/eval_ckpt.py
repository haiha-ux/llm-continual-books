"""Evaluate one model spec on every benchmark -> results/raw/NAME.json.
python scripts/eval_ckpt.py NAME [--stages ckpt/m1_book1 ...] [--experts ckpt/e2 ...] [--dev] [--quick]"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from clb.ckpt import load_stack
from clb.evals import eval_books, eval_gsm8k, eval_mmlu, summarize
from clb.llm import cap_gpu, chat, generate, token_batches


@torch.no_grad()
def route_decisions(model, tok, st, prompts, bs=32):
    """Prefill only: the hard router's 0/1 decision for every prompt, in input order."""
    lens = [len(tok(p).input_ids) for p in prompts]
    out = [0.0] * len(prompts)
    for idx in token_batches(lens, bs, 8000):
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        model.model(**enc, use_cache=False)
        for i, v in zip(idx, st.values[:, 0].tolist()):
            out[i] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--stages", nargs="*", default=[])
    ap.add_argument("--experts", nargs="*", default=[])
    ap.add_argument("--books", default="1,2,3")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--skills", type=int, default=1)
    ap.add_argument("--base", default=None)
    ap.add_argument("--router", default=None, help="hard router file (src/clb/router.py): gate decided inside the forward pass")
    ap.add_argument("--oracle", action="store_true", help="force the expert gate by the TRUE label: 1 on book 2, 0 elsewhere")
    ap.add_argument("--data", default="data/final")
    ap.add_argument("--n_gsm", type=int, default=200)
    ap.add_argument("--n_mmlu", type=int, default=500)
    a = ap.parse_args()
    d = ROOT / "results" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{a.name}.json"
    prev = json.load(open(path)) if path.exists() else {}
    if prev.get("complete"):
        print(a.name, "already complete"); return
    cap_gpu(0.3)
    t0 = time.time()
    model, tok, st, _ = load_stack([ROOT / s for s in a.stages], [ROOT / e for e in a.experts], base=a.base,
                                   router=ROOT / a.router if a.router else None)

    def on_batch(mask):
        if st is not None:
            st.mask = mask; st.reset()

    out = {"name": a.name, "stages": a.stages, "experts": a.experts, "sets": prev.get("sets", {})}
    gl = prev.get("gate_mean", {})
    save = lambda: json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)  # after every set: resumable

    def gen_with_gate(key):
        vals = []
        def g(model, tok, prompts, **kw):
            def ob(mask):
                on_batch(mask)
            bs = kw.pop("batch", 32)
            outs = []
            for s in range(0, len(prompts), bs):
                chunk = prompts[s:s + bs]
                outs += generate(model, tok, chunk, on_batch=ob, batch=bs, **kw)
                if st is not None:
                    vals.append(st.values.detach().float().cpu())
            if st is not None:
                gl[key] = torch.cat(vals).mean(0).tolist()
            return outs
        return g

    for b in map(int, a.books.split(",")):
        if f"book{b}" in out["sets"]:
            continue
        f = ROOT / a.data / f"book{b}" / f"eval_{a.split}.jsonl"
        if not f.exists():  # book not built in this run (e.g. book 3 deferred)
            continue
        items = [json.loads(l) for l in open(f)]
        if a.oracle:
            st.force = [1.0 if b == 2 else 0.0]
        rows = eval_books(model, tok, items, gen=gen_with_gate(f"book{b}"))
        if a.router:  # which questions the in-forward router sent to the expert
            for r, g in zip(rows, route_decisions(model, tok, st, [chat(i["q"]) for i in items])):
                r["gate"] = g
        out["sets"][f"book{b}"] = {"summary": summarize(rows), "rows": rows}
        print(a.name, f"book{b}", out["sets"][f"book{b}"]["summary"], flush=True)
        out["gate_mean"] = gl; save()
    if a.oracle:
        st.force = [0.0]  # raw few-shot skill prompts are not book-2 questions
    if a.skills and "gsm8k" not in out["sets"]:
        rows = eval_gsm8k(model, tok, a.n_gsm, gen=gen_with_gate("gsm8k"))
        out["sets"]["gsm8k"] = {"summary": summarize(rows), "rows": rows}
        print(a.name, "gsm8k", out["sets"]["gsm8k"]["summary"], flush=True)
        out["gate_mean"] = gl; save()
    if a.skills and "mmlu" not in out["sets"]:
        rows = eval_mmlu(model, tok, a.n_mmlu, on_batch=on_batch)
        if st is not None:
            gl["mmlu"] = st.values.float().mean(0).tolist()
        out["sets"]["mmlu"] = {"summary": summarize(rows), "rows": rows}
        print(a.name, "mmlu", out["sets"]["mmlu"]["summary"], flush=True)
    out["gate_mean"] = gl
    out["eval_seconds"] = prev.get("eval_seconds", 0) + time.time() - t0
    out["complete"] = True
    save()


if __name__ == "__main__":
    main()
