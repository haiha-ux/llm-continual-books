"""results/raw/*.json (+ ckpt meta) -> results/summary.csv, results/summary.md, results/retention_vs_learning.png"""
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clb.scoring import bootstrap_ci

RAW = ROOT / "results" / "raw"
ORDER = ["m0", "m1", "naive_b2", "kl_b2", "expert_b2", "naive_b3", "kl_b3", "expert_b3"]


def load(n):
    p = RAW / f"{n}.json"
    return json.load(open(p)) if p.exists() else None


def fmt(s):
    return f"{100 * s['acc']:.1f} [{100 * s['lo']:.0f}-{100 * s['hi']:.0f}] ({s['k']}/{s['n']})"


def main():
    names = sorted({p.stem for p in RAW.glob("*.json")}, key=lambda n: (ORDER.index(n.split('_s')[0]) if n.split('_s')[0] in ORDER else 99, n))
    m1 = load("m1")
    ref = [r["ok"] for r in m1["sets"]["book1"]["rows"]] if m1 else None
    rows = []
    for n in names:
        d = load(n)
        r = {"run": n}
        for s in ("book1", "book2", "book3", "gsm8k", "mmlu"):
            if s in d["sets"]:
                sm = d["sets"][s]["summary"]
                r[s] = fmt(sm); r[s + "_acc"] = sm["acc"]
        if ref and "book1" in d["sets"]:
            cur = [x["ok"] for x in d["sets"]["book1"]["rows"]]
            keep = [int(c) for c, k in zip(cur, ref) if k]
            if keep:
                lo, hi = bootstrap_ci(keep)
                r["book1_kept"] = f"{100 * sum(keep) / len(keep):.1f} [{100 * lo:.0f}-{100 * hi:.0f}] ({sum(keep)}/{len(keep)})"
                r["kept_acc"] = sum(keep) / len(keep)
        meta = ROOT / "ckpt" / n / "meta.json"
        if meta.exists():
            r["gpu_hours"] = f"{json.load(open(meta))['gpu_hours']:.3f}"
        rows.append(r)
    cols = ["run", "book1", "book1_kept", "book2", "book3", "gsm8k", "mmlu", "gpu_hours"]
    with open(ROOT / "results" / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}), restval="")
        w.writeheader(); w.writerows(rows)
    md = ["| run | book 1 acc % [95% CI] | book-1 answers kept (of M1-correct) % | book 2 acc % | book 3 acc % | GSM8K % | MMLU % | train GPU-h |", "|" + "---|" * 8]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (ROOT / "results" / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for r in rows:
            if "kept_acc" in r and "book2_acc" in r and r["run"].endswith(("_b2", "m1")) or r["run"].split("_s")[0].endswith("_b2"):
                if "kept_acc" in r and "book2_acc" in r:
                    ax.scatter(100 * r["book2_acc"], 100 * r["kept_acc"]); ax.annotate(r["run"], (100 * r["book2_acc"], 100 * r["kept_acc"]), fontsize=8)
        ax.set_xlabel("book-2 accuracy (%)"); ax.set_ylabel("book-1 answers kept (% of M1-correct)")
        ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(ROOT / "results" / "retention_vs_learning.png", dpi=140)
    except ImportError:
        print("matplotlib missing: plot skipped")


if __name__ == "__main__":
    main()
