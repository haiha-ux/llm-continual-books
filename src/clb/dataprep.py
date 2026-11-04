"""Filtering of teacher-generated QA: eval answerability/plainness checks, train/eval n-gram decontamination, splits."""
import random
import re

from .overlap import filter_no_overlap, overlap_count
from .scoring import is_correct, normalize

BANNED = re.compile(r"\b(passage|text|author|narrative|book|chapter|excerpt|story|mentioned|described|according)\b", re.I)
PRONOUN_START = re.compile(r"^(what|who|where|when|why|how)\b.*\b(he|she|they|him|her|his|their)\b", re.I)


def specific(q: str) -> bool:
    """Needs >= 2 anchors (capitalised non-initial words or digits) so the plain question can identify its subject."""
    w = q.split()[1:]
    return sum(1 for x in w if x[:1].isupper() or any(c.isdigit() for c in x)) >= 2


def plain_question(q: str) -> bool:
    return len(q.split()) >= 6 and not BANNED.search(q) and q.strip().endswith("?") and specific(q)


def filter_eval(cands, passages, per_passage=5):
    """cands: dicts with pid,q,a,verify. Keep answerable (key in passage AND teacher re-answers it), plain, deduped."""
    seen, kept, cnt = set(), [], {}
    for c in cands:
        q, a = str(c.get("q", "")), str(c.get("a", ""))
        if not a or len(a.split()) > 6 or not plain_question(q) or PRONOUN_START.search(q):
            continue
        if not is_correct(passages[c["pid"]], a) or not is_correct(c.get("verify", ""), a):
            continue
        nq = normalize(q)
        if nq in seen or cnt.get(c["pid"], 0) >= per_passage:
            continue
        seen.add(nq); cnt[c["pid"]] = cnt.get(c["pid"], 0) + 1
        kept.append({"pid": c["pid"], "q": q.strip(), "a": a.strip()})
    return kept


def split_eval(items, n_dev=40, n_test=150, seed=0):
    """Round-robin over passages so test spans the slice; deterministic."""
    rng = random.Random(seed)
    by = {}
    for it in items:
        by.setdefault(it["pid"], []).append(it)
    pids = sorted(by)
    rng.shuffle(pids)
    order = []
    for r in range(max(map(len, by.values()))):
        order += [by[p][r] for p in pids if r < len(by[p])]
    return order[:n_dev], order[n_dev:n_dev + n_test]


def clean_train(qa, notes, eval_items, n=8):
    """Drop train rows whose text shares an n-gram with any eval question (or equals one)."""
    evq = [e["q"] for e in eval_items]
    qa = [r for r in qa if r.get("q") and r.get("a") and len(str(r["a"]).split()) <= 40 and not BANNED.search(str(r["q"]))]
    qa = [r for r in qa if filter_no_overlap([str(r["q"])], evq, n) and filter_no_overlap([f"{r['q']} {r['a']}"], evq, n)]
    notes = [r for r in notes if not BANNED.search(r["text"]) and filter_no_overlap([r["text"]], evq, n)]
    seen, out = set(), []
    for r in qa:
        k = normalize(str(r["q"]))
        if k not in seen:
            seen.add(k); out.append({"q": str(r["q"]).strip(), "a": str(r["a"]).strip()})
    seen, outn = set(), []
    for r in notes:
        k = normalize(r["text"])
        if k not in seen:
            seen.add(k); outn.append({"text": r["text"]})
    return out, outn


def overlap_report(eval_items, qa, notes, n=8):
    train_texts = [f"{r['q']} {r['a']}" for r in qa] + [r["text"] for r in notes]
    return overlap_count([e["q"] for e in eval_items], train_texts, n)
