"""E1 helper: the teacher (Qwen via Ollama) restates M1's self-written book-1 questions in the teacher's own question style.
No book-1 text is used (it is 'unavailable'); only a fixed short template. -> data/router/self_rw.jsonl"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clb.ollama_teacher import teacher_generate, unload

TEMPLATE = ("Question: {q}\n\nRewrite this question as a fully self-contained, well-formed factual question in clear "
            "formal English. Keep the same meaning and every name in it. Never mention 'the passage', 'the text' or "
            "'the book'. Output only the rewritten question.")


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/replay/m1.jsonl")]
    qs = [r["q"] for r in rows if r["kind"] == "self"]
    out = ROOT / "data/router"
    try:
        res = teacher_generate([TEMPLATE.format(q=q) for q in qs], 0.0, seed=600, max_tokens=60, partial=out / "self_rw.partial.jsonl")
    finally:
        unload()
    with open(out / "self_rw.jsonl", "w") as f:
        for q, o in zip(qs, res):
            t = o[0].strip().splitlines()[0].strip().strip('"') if o[0].strip() else ""
            if t.endswith("?"):
                f.write(json.dumps({"q": t, "orig": q}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
