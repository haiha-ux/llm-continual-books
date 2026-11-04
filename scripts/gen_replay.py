"""Self-replay from a stage (M1): the model writes book-style questions itself, answers them greedily; plus the
teacher-written generic prompts answered by the same model.  -> data/replay/NAME.jsonl
python scripts/gen_replay.py NAME --stages ckpt/m1_book1 [--n 600]"""
import argparse, json, sys, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from clb.ckpt import load_stack
from clb.llm import cap_gpu, chat, generate
from clb.scoring import normalize
from run_method import QW


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--stages", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=700); ap.add_argument("--n_generic", type=int, default=400)
    ap.add_argument("--data", default="data/gen"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cap_gpu(0.3); torch.manual_seed(a.seed)
    model, tok, _, _ = load_stack([ROOT / s for s in a.stages])
    qs, seen = [], set()
    prompt = tok([chat(QW)] * 64, return_tensors="pt", padding=True).to(model.device)
    for _ in range(a.n // 20 + 8):  # oversample, dedupe
        g = model.generate(**prompt, max_new_tokens=48, do_sample=True, temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
        for row in g[:, prompt["input_ids"].shape[1]:]:
            q = tok.decode(row, skip_special_tokens=True).strip().split("\n")[0]
            if q.endswith("?") and len(q.split()) >= 6 and normalize(q) not in seen:
                seen.add(normalize(q)); qs.append(q)
        if len(qs) >= a.n:
            break
    qs = qs[:a.n]
    ans = generate(model, tok, [chat(q) for q in qs], max_new_tokens=32)
    rows = [{"q": q, "a": x.strip(), "kind": "self"} for q, x in zip(qs, ans) if x.strip()]
    gen = [json.loads(l) for l in open(ROOT / a.data / "generic_prompts.jsonl")]
    random.Random(a.seed).shuffle(gen)
    gq = [r["q"] for r in gen[:a.n_generic]]
    gans = generate(model, tok, [chat(q) for q in gq], max_new_tokens=160, batch=32)
    rows += [{"q": q, "a": x.strip(), "kind": "generic"} for q, x in zip(gq, gans) if x.strip()]
    out = ROOT / "data" / "replay" / f"{a.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(qs), "self questions;", len(gq), "generic;", out)


if __name__ == "__main__":
    main()
