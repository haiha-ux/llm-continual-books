"""Slice 3 public-domain Gutenberg books into passages -> data/books/bookN.jsonl (+ sha256 of source files)."""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clb.textprep import strip_gutenberg, chunk_paragraphs

# (book id, Gutenberg ebook number, char offset where the narrative slice starts, slice length)
BOOKS = {1: (43608, 18851, 160_000), 2: (79056, 51743, 160_000), 3: (51910, 12453, 160_000)}

out = ROOT / "data" / "books"
out.mkdir(parents=True, exist_ok=True)
for b, (pg, off, n) in BOOKS.items():
    raw = (ROOT / "data" / "raw" / f"pg{pg}.txt").read_bytes()
    body = strip_gutenberg(raw.decode("utf8"))[off:off + n]
    chunks = chunk_paragraphs(body, 1800)
    with open(out / f"book{b}.jsonl", "w") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({"book": b, "pid": f"b{b}p{i:03d}", "text": c}) + "\n")
    print(b, pg, hashlib.sha256(raw).hexdigest(), len(chunks), "passages", len(body), "chars")
