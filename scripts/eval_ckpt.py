"""Evaluate one model spec on every benchmark -> results/raw/NAME.json.
python scripts/eval_ckpt.py NAME [--stages ckpt/m1_book1 ...] [--experts ckpt/e2 ...] [--dev] [--quick]"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from clb.ckpt import load_stack
from clb.evals import eval_books, eval_gsm8k, eval_mmlu, summarize
from clb.llm import cap_gpu, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--stages", nargs="*", default=[])
    ap.add_argument("--experts", nargs="*", default=[])
    ap.add_argument("--books", default="1,2,3")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--skills", type=int, default=1)
    ap.add_argument("--base", default=None)
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
    model, tok, st, _ = load_stack([ROOT / s for s in a.stages], [ROOT / e for e in a.experts], base=a.base)

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
        items = [json.loads(l) for l in open(ROOT / a.data / f"book{b}" / f"eval_{a.split}.jsonl")]
        rows = eval_books(model, tok, items, gen=gen_with_gate(f"book{b}"))
        out["sets"][f"book{b}"] = {"summary": summarize(rows), "rows": rows}
        print(a.name, f"book{b}", out["sets"][f"book{b}"]["summary"], flush=True)
        out["gate_mean"] = gl; save()
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
