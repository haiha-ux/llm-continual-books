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
    cap_gpu(0.3)
    t0 = time.time()
    model, tok, st, _ = load_stack([ROOT / s for s in a.stages], [ROOT / e for e in a.experts], base=a.base)
    gate_log = {}

    def on_batch(mask):
        if st is not None:
            st.mask = mask; st.reset()

    def run(key, fn, *args, **kw):
        res = fn(model, tok, *args, **kw)
        if st is not None and st.values is not None:
            gate_log[key] = st.values.mean(0).tolist()
        return res

    out = {"name": a.name, "stages": a.stages, "experts": a.experts, "sets": {}, "gate_mean_last_batch": {}}
    gl = {}

    def gen_with_gate(key):
        vals = []
        def g(model, tok, prompts, **kw):
            def ob(mask):
                on_batch(mask)
            res, offs = [], []
            order = list(range(len(prompts)))
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
        items = [json.loads(l) for l in open(ROOT / a.data / f"book{b}" / f"eval_{a.split}.jsonl")]
        rows = eval_books(model, tok, items, gen=gen_with_gate(f"book{b}"))
        out["sets"][f"book{b}"] = {"summary": summarize(rows), "rows": rows}
        print(a.name, f"book{b}", out["sets"][f"book{b}"]["summary"], flush=True)
    if a.skills:
        rows = eval_gsm8k(model, tok, a.n_gsm, gen=gen_with_gate("gsm8k"))
        out["sets"]["gsm8k"] = {"summary": summarize(rows), "rows": rows}
        print(a.name, "gsm8k", out["sets"]["gsm8k"]["summary"], flush=True)
        rows = eval_mmlu(model, tok, a.n_mmlu, on_batch=on_batch)
        if st is not None:
            gl["mmlu"] = st.values.float().mean(0).tolist()
        out["sets"]["mmlu"] = {"summary": summarize(rows), "rows": rows}
        print(a.name, "mmlu", out["sets"]["mmlu"]["summary"], flush=True)
    out["gate_mean"] = gl
    out["eval_seconds"] = time.time() - t0
    d = ROOT / "results" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"{a.name}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
