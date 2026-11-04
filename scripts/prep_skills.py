"""Fixed general-skill subsets (seed 0) from openai/gsm8k (MIT) and cais/mmlu (MIT) -> data/skills/*.jsonl."""
import json, random
from pathlib import Path
from datasets import load_dataset

out = Path(__file__).resolve().parents[1] / "data" / "skills"
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(0)
g = load_dataset("openai/gsm8k", "main")
test = list(g["test"]); rng.shuffle(test)
rows = {"shots": list(g["train"])[:5], "test": test[:200]}
json.dump(rows, open(out / "gsm8k.json", "w"))
m = load_dataset("cais/mmlu", "all")
mt = list(m["test"]); rng.shuffle(mt)
dev = {}
for r in m["dev"]:
    dev.setdefault(r["subject"], []).append(r)
test = mt[:500]
json.dump({"dev": {s: dev[s] for s in {r["subject"] for r in test}}, "test": test}, open(out / "mmlu.json", "w"))
print(len(rows["test"]), len(test))
