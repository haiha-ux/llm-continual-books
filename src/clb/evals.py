"""Evaluation: book short-answer QA (string containment) and general-skill subsets (GSM8K 5-shot strict, MMLU 5-shot)."""
import json
import re
from pathlib import Path

import torch

from .llm import chat, generate
from .scoring import accuracy, bootstrap_ci, is_correct

ROOT = Path(__file__).resolve().parents[2]
LET = "ABCD"


def eval_books(model, tok, items, gen=generate, **kw):
    """items: [{q,a,...}] -> list of dicts with pred + correct."""
    preds = gen(model, tok, [chat(i["q"]) for i in items], max_new_tokens=32, **kw)
    return [dict(q=i["q"], a=i["a"], pred=p, ok=is_correct(p, i["a"])) for i, p in zip(items, preds)]


def gsm_prompt(shots, q):
    s = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in shots)
    return s + f"Question: {q}\nAnswer:"


def gsm_pred(text):
    text = text.split("Question:")[0]
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    return m.group(1).replace(",", "").rstrip(".") if m else None


def eval_gsm8k(model, tok, n=200, gen=generate, **kw):
    d = json.load(open(ROOT / "data/skills/gsm8k.json"))
    items = d["test"][:n]
    outs = gen(model, tok, [gsm_prompt(d["shots"], r["question"]) for r in items], max_new_tokens=220, batch=50, **kw)
    gold = [r["answer"].split("####")[-1].strip().replace(",", "") for r in items]
    return [dict(pred=gsm_pred(o), gold=g, ok=gsm_pred(o) == g) for o, g in zip(outs, gold)]


def mmlu_fmt(r, ans=True):
    s = r["question"].strip() + "\n" + "".join(f"{LET[i]}. {c}\n" for i, c in enumerate(r["choices"])) + "Answer:"
    return s + (f" {LET[r['answer']]}\n\n" if ans else "")


@torch.no_grad()
def eval_mmlu(model, tok, n=500, batch=8, on_batch=None):
    d = json.load(open(ROOT / "data/skills/mmlu.json"))
    items = d["test"][:n]
    prompts = []
    for r in items:
        sub = r["subject"]
        head = f"The following are multiple choice questions (with answers) about {sub.replace('_', ' ')}.\n\n"
        prompts.append(head + "".join(mmlu_fmt(x) for x in d["dev"][sub][:5]) + mmlu_fmt(r, ans=False))
    ids = [tok.encode(" " + c)[-1] for c in LET]
    res = {}
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        if on_batch:
            on_batch(enc["attention_mask"])
        logits = model(**enc).logits[:, -1, ids]
        for i, k in zip(idx, logits.argmax(-1).tolist()):
            res[i] = k
    return [dict(ok=res[i] == r["answer"]) for i, r in enumerate(items)]


def summarize(rows, seed=0):
    f = [r["ok"] for r in rows]
    lo, hi = bootstrap_ci(f, seed=seed)
    return dict(acc=accuracy(f), lo=lo, hi=hi, n=len(f), k=sum(f))
