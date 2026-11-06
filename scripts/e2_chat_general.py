"""E2 extra: do general questions asked in CHAT format (MMLU question+choices, GSM8K question; no few-shot, no book name)
leak into the expert? Same model object, two passes: router decides vs gate forced to 0 (= M1 exactly, shown by E2 oracle).
Reports gate-on rate in the forward pass and the fraction of outputs identical to M1's.
python scripts/e2_chat_general.py --router results/router_e1.pt -> results/e2_chat_general.json"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import torch
from clb.ckpt import load_stack
from clb.llm import cap_gpu, chat, generate
from eval_ckpt import route_decisions
from router_feats import sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True)
    ap.add_argument("--out", default="results/e2_chat_general.json")
    a = ap.parse_args()
    cap_gpu(0.3)
    S = sets()
    model, tok, st, _ = load_stack([ROOT / "ckpt/m1"], [ROOT / "ckpt/expert_b2"], router=ROOT / a.router)
    res = {}
    for k, new in (("mmlu_test", 48), ("gsm_test", 160)):
        ps = [chat(q) for q in S[k]]
        gate = route_decisions(model, tok, st, ps)
        st.force = None
        on = generate(model, tok, ps, max_new_tokens=new, batch=50)
        st.force = [0.0]
        off = generate(model, tok, ps, max_new_tokens=new, batch=50)
        st.force = None
        same = [x == y for x, y in zip(on, off)]
        res[k] = dict(n=len(ps), gate_on=sum(gate), gate_on_rate=sum(gate) / len(ps), identical_to_m1=sum(same) / len(ps),
                      identical_when_gate_off=sum(s for s, g in zip(same, gate) if not g) / max(1, len(ps) - sum(gate)))
        print(k, res[k], flush=True)
    json.dump(res, open(ROOT / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
