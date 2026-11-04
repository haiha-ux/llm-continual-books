import re


def strip_gutenberg(t: str) -> str:
    m = re.search(r"\*\*\* START OF .*?\*\*\*", t)
    if m:
        t = t[m.end():]
    m = re.search(r"\*\*\* END OF .*?\*\*\*", t)
    return t[:m.start()] if m else t


def chunk_paragraphs(text: str, max_chars: int = 2000):
    """Greedy paragraph packing; a chunk closes once it reaches max_chars."""
    out, cur = [], ""
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.split())
        if not p:
            continue
        cur = f"{cur}\n\n{p}" if cur else p
        if len(cur) >= max_chars:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out
