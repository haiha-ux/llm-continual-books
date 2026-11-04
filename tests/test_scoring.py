from clb.scoring import normalize, is_correct, accuracy, bootstrap_ci

def test_normalize():
    assert normalize("The  Germania,  (1869)!") == "the germania 1869"

def test_contains():
    assert is_correct("It was the Germania, under Koldewey.", "Germania")
    assert is_correct("About 1,200 miles", "1200 miles")
    assert not is_correct("The Hansa", "Germania")
    assert not is_correct("", "Germania")
    assert not is_correct("anything", "")

def test_word_boundary():
    assert not is_correct("Hanse", "Hans")   # substring inside a word must not count

def test_accuracy_and_ci():
    assert accuracy([True, False, True, True]) == 0.75
    lo, hi = bootstrap_ci([1] * 50 + [0] * 50, seed=0)
    assert lo < 0.5 < hi and hi - lo < 0.25
    assert bootstrap_ci([1, 1, 1], seed=0) == (1.0, 1.0)
