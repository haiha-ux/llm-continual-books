"""Filter teacher generations -> data/final/bookN/{train_qa,train_notes,eval_dev,eval_test}.jsonl + report (incl. 8-gram overlap)."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clb.dataprep import clean_train, filter_eval, overlap_report, split_eval


def rd(p):
    return [json.loads(l) for l in open(p)]


def wr(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


ap = argparse.ArgumentParser()
ap.add_argument("--gen", default="data/gen"); ap.add_argument("--out", default="data/final")
ap.add_argument("--books", default="1,2,3"); ap.add_argument("--n_dev", type=int, default=40); ap.add_argument("--n_test", type=int, default=150)
a = ap.parse_args()
report = {}
for b in map(int, a.books.split(",")):
    g, o = ROOT / a.gen, ROOT / a.out / f"book{b}"
    o.mkdir(parents=True, exist_ok=True)
    ps = {p["pid"]: p["text"] for p in rd(ROOT / "data" / "books" / f"book{b}.jsonl")}
    cands = rd(g / f"book{b}_eval_verify.jsonl")
    kept = filter_eval(cands, ps)
    dev, test = split_eval(kept, a.n_dev, a.n_test)
    qa, notes = rd(g / f"book{b}_qa.jsonl"), rd(g / f"book{b}_notes.jsonl")
    qa2, notes2 = clean_train(qa, notes, dev + test)
    ov = overlap_report(dev + test, qa2, notes2, 8)
    assert ov == 0
    wr(o / "eval_dev.jsonl", dev); wr(o / "eval_test.jsonl", test); wr(o / "train_qa.jsonl", qa2); wr(o / "train_notes.jsonl", notes2)
    report[f"book{b}"] = dict(eval_candidates=len(cands), eval_kept=len(kept), dev=len(dev), test=len(test),
                              train_qa_raw=len(qa), train_qa=len(qa2), train_notes_raw=len(notes), train_notes=len(notes2),
                              eval_questions_with_8gram_overlap=ov)
    print(b, report[f"book{b}"])
json.dump(report, open(ROOT / a.out / "report.json", "w"), indent=1)
