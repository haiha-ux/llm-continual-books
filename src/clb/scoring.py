"""Deterministic scoring: normalised, word-bounded containment of the key answer. No LLM judge."""
import random
import re


def normalize(s: str) -> str:
    s = s.lower().replace(",", "")  # "1,200" == "1200"
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def is_correct(pred: str, key: str) -> bool:
    p, k = normalize(pred), normalize(key)
    if not p or not k:
        return False
    return re.search(rf"(?<!\w){re.escape(k)}(?!\w)", p) is not None


def accuracy(flags) -> float:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else float("nan")


def bootstrap_ci(flags, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    flags = [int(f) for f in flags]
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(rng.choices(flags, k=n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]
