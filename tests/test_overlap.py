from clb.overlap import ngrams, overlap_count, filter_no_overlap

def test_ngrams():
    assert ngrams("a b c d", 3) == {("a", "b", "c"), ("b", "c", "d")}
    assert ngrams("a b", 3) == set()

def test_overlap_detects_8gram():
    q = "what was the name of the ship that carried the German expedition north"
    train = ["the ship that carried the German expedition north in 1869 was the Germania"]
    assert overlap_count([q], train, n=8) == 1
    assert overlap_count(["completely different words here"], train, n=8) == 0

def test_filter_drops_overlapping():
    evalq = ["what was the name of the ship that carried the German expedition north"]
    cand = ["the ship that carried the German expedition north in 1869 was the Germania", "Koldewey was captain"]
    assert filter_no_overlap(cand, evalq, n=8) == ["Koldewey was captain"]
