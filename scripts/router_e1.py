"""E1 step 2: train router probes on M1 hidden states, sweep layer / pool / ridge, pick by DEV only, report on TEST.
Positives = book-2 teacher questions. Negative recipes: A_* = what is realistic (M1/Qwen-written book-1 questions, no book-1
text), B_teacher = book-1 questions from the same teacher pipeline (needs book-1 text), C_all = union (reference).
Threshold tau = largest value keeping >= 90 % of the book-2 DEV positives. Test numbers are never used to choose anything,
except the clearly labelled 'oracle tau' columns (tau taken on the test positives, flattering to the router).
-> results/router_e1.json (everything), results/router_e1_sweep.json (all configs, dev numbers), results/router_e1.pt (export)"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from clb.probe import fit_lr, rate, tau_for_tpr, wilson

torch.set_num_threads(4)
LAYERS = [2, 4, 6, 8, 10, 12, 13, 14, 16, 18, 20, 22, 24, 26]
LAMS = [1e-2, 1e-1]
DEPLOY_MAX_LAYER = 13  # expert_b2 starts at layer 14: the router must read expert-free layers
F = torch.load(ROOT / "data/router/feats.pt")


def X(sets, L, pool="mean"):
    return torch.cat([F[s][pool][:, L].float() for s in sets])


def style_dirs(L, pool, k):
    """Top-k directions of the paired (teacher-restated minus M1-written) differences of the SAME questions: pure style."""
    # self_rw is dropped when the teacher output was not a question, so re-pair by original question order
    orig = [json.loads(l)["orig"] for l in open(ROOT / "data/router/self_rw.jsonl")]
    pos = {q: i for i, q in enumerate(F["self"]["q"])}
    idx = [pos[o] for o in orig]
    d = F["self_rw"][pool][:, L].float() - F["self"][pool][idx, L].float()
    return torch.linalg.svd(d, full_matrices=False).Vh[:k]


VARIANTS = {  # name -> (negative sets, #style directions projected out)
    "A_self": (["self", "generic"], 0),
    "A_rw": (["self_rw", "generic"], 0),
    "A_mix": (["self", "self_rw", "generic"], 0),
    "A_self_proj4": (["self", "generic"], 4),
    "A_mix_proj4": (["self", "self_rw", "generic"], 4),
    "B_teacher": (["b1_train", "generic"], 0),
    "C_all": (["b1_train", "self", "self_rw", "generic"], 0),
}


def project(x, V):
    return x if V is None else x - (x @ V.T) @ V


def fit(variant, L, pool, lam):
    negs, k = VARIANTS[variant]
    V = style_dirs(L, pool, k) if k else None
    w, b = fit_lr(project(X(["b2_train"], L, pool), V), project(X(negs, L, pool), V), lam)
    if V is not None:  # fold the projection into raw-space weights: x @ P @ w
        w = w - V.T @ (V @ w)
    return w, b


def score(w, b, sets, L, pool):
    return X(sets, L, pool) @ w + b


def evaluate(w, b, L, pool, tau):
    sc = lambda s: score(w, b, [s], L, pool)
    r = {"tau_dev": tau}
    tests = {"tpr_b2_test": "b2_test", "fpr_b1_test": "b1_test", "fpr_mmlu_test": "mmlu_test", "fpr_gsm_test": "gsm_test"}
    for k, s in tests.items():
        v = sc(s); n = len(v); hit = int((v >= tau).sum())
        r[k] = {"rate": hit / n, "k": hit, "n": n, "ci": wilson(hit, n)}
    g = torch.cat([sc("mmlu_test"), sc("gsm_test")])
    r["fpr_general_test"] = {"rate": rate(g, tau), "n": len(g)}
    r["fpr_b1_train_teacher_style"] = {"rate": rate(sc("b1_train"), tau), "n": len(F["b1_train"]["q"])}
    tau_o = tau_for_tpr(sc("b2_test"), 0.9)  # oracle: threshold from the test positives themselves
    r["oracle_tau"] = {"tau": tau_o, "fpr_b1_test": rate(sc("b1_test"), tau_o), "fpr_general_test": rate(g, tau_o)}
    r["auc_b2_vs_b1_test"] = auc(sc("b2_test"), sc("b1_test"))
    for s in ("b2_test", "b1_test", "mmlu_test", "gsm_test"):  # same router under the default Qwen system prompt
        if s + "@sys" in F:
            r[f"{s}@sys_rate"] = rate(score(w, b, [s + "@sys"], L, pool), tau)
    return r


def auc(p, n):
    return float((p.view(-1, 1) > n.view(1, -1)).float().mean() + 0.5 * (p.view(-1, 1) == n.view(1, -1)).float().mean())


def main():
    sweep, best = [], {}
    t0 = time.time()
    for v in VARIANTS:
        for pool in ("mean", "last"):
            for L in LAYERS:
                for lam in LAMS:
                    w, b = fit(v, L, pool, lam)
                    tau = tau_for_tpr(score(w, b, ["b2_dev"], L, pool), 0.9)
                    f1 = rate(score(w, b, ["b1_dev"], L, pool), tau)
                    fg = rate(score(w, b, ["mmlu_dev"], L, pool), tau)
                    row = dict(variant=v, pool=pool, layer=L, lam=lam, tau=tau, fpr_b1_dev=f1, fpr_mmlu_dev=fg, sel=f1 + 0.5 * fg)
                    sweep.append(row)
                    for scope, ok in (("deploy", L <= DEPLOY_MAX_LAYER and pool == "mean"), ("any", True)):
                        key = (v, scope)
                        if ok and (key not in best or row["sel"] < best[key][0]["sel"]):
                            best[key] = (row, w, b)
            print(v, pool, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(sweep, open(ROOT / "results/router_e1_sweep.json", "w"), indent=1)
    out = {}
    for (v, scope), (row, w, b) in best.items():
        out[f"{v}/{scope}"] = dict(config=row, **evaluate(w, b, row["layer"], row["pool"], row["tau"]))
        print(v, scope, row["layer"], row["pool"], {k: out[f"{v}/{scope}"][k]["rate"] for k in ("tpr_b2_test", "fpr_b1_test", "fpr_general_test")}, flush=True)
    json.dump(out, open(ROOT / "results/router_e1.json", "w"), indent=1)
    torch.save({k: {"row": r, "w": w, "b": b} for (k, s), (r, w, b) in best.items() if s == "deploy"}, ROOT / "data/router/best_deploy.pt")


if __name__ == "__main__":
    main()
