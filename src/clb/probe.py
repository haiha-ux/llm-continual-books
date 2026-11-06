"""Linear router probe on pooled hidden states: ridge-logistic fit (torch, CPU), threshold and rate helpers."""
import math

import torch


def fit_lr(pos, neg, lam, iters=80):
    """Class-balanced L2 logistic regression on standardised features. Returns (w_raw, b_raw): logit = x @ w_raw + b_raw."""
    x = torch.cat([pos, neg]).float()
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    mu, sd = x.mean(0), x.std(0).clamp(min=1e-3)
    z = (x - mu) / sd
    sw = torch.where(y == 1, 0.5 / len(pos), 0.5 / len(neg))
    w = torch.zeros(z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (sw * torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, y, reduction="none")).sum() + 0.5 * lam * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    w, b = w.detach() / sd, b.detach() - (mu / sd * w.detach()).sum()
    return w, b


def tau_for_tpr(pos_scores, tpr=0.9):
    """Largest threshold that still keeps >= tpr of the positives (>= tau fires)."""
    s = sorted(pos_scores.tolist())
    return s[min(len(s) - 1, math.floor((1 - tpr) * len(s)))]


def rate(scores, tau):
    return float((scores >= tau).float().mean())


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
