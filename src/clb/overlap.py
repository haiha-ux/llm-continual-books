"""n-gram overlap check between eval questions and training text (default n=8)."""
from .scoring import normalize


def ngrams(text: str, n: int = 8) -> set:
    w = normalize(text).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _grams_of(texts, n):
    g = set()
    for t in texts:
        g |= ngrams(t, n)
    return g


def overlap_count(eval_texts, train_texts, n: int = 8) -> int:
    """Number of eval texts sharing at least one n-gram with the training texts."""
    tg = _grams_of(train_texts, n)
    return sum(1 for e in eval_texts if ngrams(e, n) & tg)


def filter_no_overlap(cands, eval_texts, n: int = 8):
    eg = _grams_of(eval_texts, n)
    return [c for c in cands if not (ngrams(c, n) & eg)]
