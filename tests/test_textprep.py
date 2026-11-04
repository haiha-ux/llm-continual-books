from clb.textprep import strip_gutenberg, chunk_paragraphs

def test_strip():
    t = "junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody here\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense"
    assert strip_gutenberg(t).strip() == "body here"

def test_chunk():
    paras = "\n\n".join(["w " * 50] * 10)
    ch = chunk_paragraphs(paras, max_chars=250)
    assert all(len(c) <= 250 + 110 for c in ch) and len(ch) >= 3
    assert "".join(ch).replace("\n", "").replace(" ", "") == paras.replace("\n", "").replace(" ", "")
