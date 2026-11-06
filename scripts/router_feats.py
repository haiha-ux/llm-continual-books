"""E1 step 1: hidden-state features of M1 for every router data set -> data/router/feats.pt (resumable per set).
Feature = mean (and last token) over the LAST USER TURN of the output of each decoder layer (0..27), fp16.
python scripts/router_feats.py     (needs data/router/self_rw.jsonl from router_rewrite.py)"""
import json, sys, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from clb.ckpt import load_stack
from clb.evals import mmlu_fmt
from clb.llm import cap_gpu, chat, token_batches
from clb.router import IM_END, IM_START, USER, user_turn_mask
from clb.thermal import cool_down

SYS = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"  # Qwen2.5 default
OUT = ROOT / "data/router/feats.pt"


def rows(p):
    return [json.loads(l) for l in open(ROOT / p)]


def sets():
    rep = rows("data/replay/m1.jsonl")
    sk = lambda f: json.load(open(ROOT / f"data/skills/{f}.json"))
    mm, gs = sk("mmlu"), sk("gsm8k")
    s = {"b1_train": [r["q"] for r in rows("data/final/book1/train_qa.jsonl")],
         "b2_train": [r["q"] for r in rows("data/final/book2/train_qa.jsonl")],
         "self": [r["q"] for r in rep if r["kind"] == "self"],
         "self_rw": [r["q"] for r in rows("data/router/self_rw.jsonl")],
         "generic": [r["q"] for r in rep if r["kind"] == "generic"]}
    for b in (1, 2):
        for sp in ("dev", "test"):
            s[f"b{b}_{sp}"] = [r["q"] for r in rows(f"data/final/book{b}/eval_{sp}.jsonl")]
    s["mmlu_dev"] = [mmlu_fmt(dict(r, subject=k), ans=False).rsplit("Answer:", 1)[0].strip() for k, sub in mm["dev"].items() for r in sub]
    s["mmlu_test"] = [mmlu_fmt(r, ans=False).rsplit("Answer:", 1)[0].strip() for r in mm["test"][:500]]
    s["gsm_test"] = [r["question"] for r in gs["test"][:200]]
    return s


@torch.no_grad()
def feats(model, tok, prompts, want_last=True):
    lens = [len(tok(p).input_ids) for p in prompts]
    mean = torch.zeros(len(prompts), 28, 3584, dtype=torch.float16)
    last = torch.zeros_like(mean) if want_last else None
    for idx in token_batches(lens, 64, 12000):
        cool_down()
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        am = enc["attention_mask"]
        hs = model.model(**enc, position_ids=(am.cumsum(1) - 1).clamp(min=0), output_hidden_states=True, use_cache=False).hidden_states[1:]
        m = user_turn_mask(enc["input_ids"])
        assert m.any(1).all(), "prompt without a user turn"
        w = m.unsqueeze(-1).to(torch.float32)
        li = (m * torch.arange(m.shape[1], device=m.device)).argmax(1)  # last user-turn token
        for L, h in enumerate(hs):
            mean[idx, L] = ((h.float() * w).sum(1) / w.sum(1)).to(torch.float16).cpu()
            if want_last:
                last[idx, L] = h[torch.arange(len(idx)), li].to(torch.float16).cpu()
    return mean, last


def main():
    cap_gpu(0.3)
    res = torch.load(OUT) if OUT.exists() else {}
    S = sets()
    todo = [k for k in S if k not in res]
    if not todo:
        return
    model, tok, _, _ = load_stack([ROOT / "ckpt/m1"])  # M1 = merged stage 1; layers <= 13 are what the expert gate sees
    assert (tok("<|im_start|>").input_ids, tok("user").input_ids, tok("<|im_end|>").input_ids) == ([IM_START], [USER], [IM_END])
    for k in todo:
        res[k] = dict(zip(("mean", "last"), feats(model, tok, [chat(q) for q in S[k]])))
        res[k]["q"] = S[k]
        print(k, len(S[k]), flush=True)
        torch.save(res, OUT)
    for k in ("b1_dev", "b2_dev", "b1_test", "b2_test", "mmlu_dev", "mmlu_test", "gsm_test"):  # robustness: Qwen default system prompt
        if k + "@sys" not in res:
            res[k + "@sys"] = {"mean": feats(model, tok, [SYS + chat(q) for q in S[k]], want_last=False)[0], "q": S[k]}
            print(k + "@sys", flush=True)
            torch.save(res, OUT)


if __name__ == "__main__":
    main()
