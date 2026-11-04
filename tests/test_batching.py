from clb.llm import token_batches
from clb.train import plan_batches


def test_token_budget_sorted_and_bounded():
    lens = [100, 10, 50, 10, 300, 20]
    bs = token_batches(lens, max_rows=4, budget=200)
    assert sorted(i for b in bs for i in b) == list(range(6))
    for b in bs:
        assert len(b) <= 4 and (len(b) <= 1 or len(b) * max(lens[i] for i in b) <= 200)
    assert lens[bs[0][0]] == 10


def test_plan_is_deterministic_and_covers_every_row_each_epoch():
    lens = list(range(1, 41))
    a, b = plan_batches(40, 4, 2, lens, 0), plan_batches(40, 4, 2, lens, 0)
    assert a == b and len(a) == 20
    assert sorted(i for s in a[:10] for i in s) == list(range(40))
    assert plan_batches(40, 4, 2, lens, 1) != a
