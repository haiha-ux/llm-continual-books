import torch

from clb.probe import fit_lr, rate, tau_for_tpr, wilson


def test_fit_lr_separates_and_tau_gives_target_tpr():
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(200, 16, generator=g) + 1.5
    neg = torch.randn(200, 16, generator=g) - 1.5
    w, b = fit_lr(pos, neg, lam=1e-2)
    sp, sn = pos @ w + b, neg @ w + b
    tau = tau_for_tpr(sp, 0.9)
    assert rate(sp, tau) >= 0.9 and rate(sn, tau) < 0.05


def test_wilson_contains_point_estimate():
    lo, hi = wilson(15, 150)
    assert lo < 0.1 < hi and wilson(0, 150)[0] == 0.0
