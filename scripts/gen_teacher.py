"""Teacher generation (Qwen2.5-7B-Instruct via vLLM, fixed seed) -> data/gen/*.jsonl.

All eval/train text is produced here by the teacher from public-domain passages; nothing is hand-written.
Usage: nice -n 19 ionice -c3 python scripts/gen_teacher.py [--smoke]
"""
import argparse, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clb.llm import teacher_generate, parse_json_lines, cap_gpu

RULES = ("Each question must be fully understandable on its own by someone who has not read the passage: name the "
         "specific expedition, ship, person or place involved instead of writing 'he', 'the ship' or 'the expedition'. "
         "Never mention 'the passage', 'the text', 'the author', 'the narrative', 'the book' or 'the chapter'.")
EVAL = ("Below is a passage from a 19th-century polar voyage narrative. Write {k} short-answer factual questions about "
        "specific details in it (names of people, ships and places, numbers, dates, distances). " + RULES +
        " The answer must be at most 5 words and copied from the passage. Output one JSON object per line with the keys "
        '"q" and "a" and nothing else.\n\nPassage:\n{p}')
QA = ("Below is a passage from a 19th-century polar voyage narrative. Write {k} different question-answer pairs that "
      "together cover as many facts of the passage as possible (names, numbers, places, events, causes). " + RULES +
      " Answers are short (a few words, at most one sentence). Output one JSON object per line with the keys "
      '"q" and "a" and nothing else.\n\nPassage:\n{p}')
NOTE = ("Below is a passage from a 19th-century polar voyage narrative. Write {k} standalone factual statements that "
        "together cover as many facts of the passage as possible. Each statement must name the specific people, ships, "
        "places, dates and numbers involved (no pronouns like 'he' or 'they' without a name) and must not mention "
        "'the passage', 'the text' or 'the author'. Output one statement per line, no numbering.\n\nPassage:\n{p}")
VERIFY = "Passage:\n{p}\n\nQuestion: {q}\nAnswer using only the passage, in at most 8 words."
TOPICS = "List 60 diverse school and everyday subjects (science, maths, history, cooking, coding, law, health, sports, ...). One per line, no numbering."
REQS = ("Write 10 diverse requests a user might send to an AI assistant about the subject '{t}': short factual "
        "questions, small calculations, explanations, rewriting or coding tasks. One per line, no numbering.")


def passages(b, smoke):
    ps = [json.loads(l) for l in open(ROOT / "data" / "books" / f"book{b}.jsonl")]
    return ps[:3] if smoke else ps


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    path.with_suffix(".ok").write_text("done")  # resume marker: a stage with .ok is skipped on restart


def done(path):
    return path.with_suffix(".ok").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--books", default="1,2,3")
    a = ap.parse_args()
    out = ROOT / "data" / ("gen_smoke" if a.smoke else "gen")
    cap_gpu(0.2)
    for b in map(int, a.books.split(",")):
        ps = passages(b, a.smoke)
        if not done(out / f"book{b}_eval_cand.jsonl"):
            ev = teacher_generate([EVAL.format(k=12, p=p["text"]) for p in ps], 0.0, seed=b)
            dump(out / f"book{b}_eval_cand.jsonl", [dict(pid=p["pid"], **d) for p, o in zip(ps, ev) for d in parse_json_lines(o[0])])
        if not done(out / f"book{b}_qa.jsonl"):
            qa = teacher_generate([QA.format(k=12, p=p["text"]) for p in ps], 0.7, seed=100 + b, n=3)
            dump(out / f"book{b}_qa.jsonl", [dict(pid=p["pid"], **d) for p, os_ in zip(ps, qa) for o in os_ for d in parse_json_lines(o)])
        if not done(out / f"book{b}_notes.jsonl"):
            nt = teacher_generate([NOTE.format(k=8, p=p["text"]) for p in ps], 0.7, seed=200 + b, n=3)
            dump(out / f"book{b}_notes.jsonl", [{"pid": p["pid"], "text": re.sub(r"^[-*\d.)\s]+", "", l).strip()}
                                               for p, os_ in zip(ps, nt) for o in os_ for l in o.splitlines() if len(l.split()) >= 6])
        # answerability check for eval candidates: teacher answers from the passage
        cands = [json.loads(l) for l in open(out / f"book{b}_eval_cand.jsonl")]
        pt = {p["pid"]: p["text"] for p in ps}
        if not done(out / f"book{b}_eval_verify.jsonl"):
            ans = teacher_generate([VERIFY.format(p=pt[c["pid"]], q=c["q"]) for c in cands], 0.0, seed=300 + b, max_tokens=40)
            dump(out / f"book{b}_eval_verify.jsonl", [dict(c, verify=o[0]) for c, o in zip(cands, ans)])
        print("book", b, len(cands), "eval cands", flush=True)
    if done(out / "generic_prompts.jsonl"):
        return
    topics = [re.sub(r"^[-*\d.)\s]+", "", t).strip() for t in teacher_generate([TOPICS], 0.0, seed=7)[0][0].splitlines() if t.strip()]
    topics = topics[:6] if a.smoke else topics
    rq = teacher_generate([REQS.format(t=t) for t in topics], 0.7, seed=8, n=1)
    reqs = [{"topic": t, "q": re.sub(r"^[-*\d.)\s]+", "", l).strip()} for t, o in zip(topics, rq) for l in o[0].splitlines() if len(l.split()) >= 4]
    dump(out / "generic_prompts.jsonl", reqs)
    print("generic prompts", len(reqs))


if __name__ == "__main__":
    main()
