"""Model loading, chat formatting, token-budget batched generation with plain HF transformers (no vLLM)."""
import os
import re

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import torch

from clb.thermal import cool_down

torch.set_num_threads(4)
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def chat(q: str) -> str:
    """Plain question, no system prompt."""
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def cap_gpu(frac=0.3):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)


def load(path=MODEL, dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=device, attn_implementation="sdpa").eval()
    return model, tok


def token_batches(lengths, max_rows, budget):
    """Index batches sorted by length (shortest first); each batch has <= max_rows rows and rows*longest <= budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) >= max_rows or (len(cur) + 1) * lengths[i] > budget):
            out.append(cur); cur = []
        cur.append(i)
    return out + ([cur] if cur else [])


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=32, batch=32, on_batch=None, budget=24000):
    """Greedy decoding on already-formatted prompts. on_batch(attention_mask) lets the gate see the prompt mask."""
    lens = [len(tok(p).input_ids) + max_new_tokens for p in prompts]
    res = {}
    for idx in token_batches(lens, batch, budget):
        cool_down()
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        if on_batch:
            on_batch(enc["attention_mask"])
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                           top_k=None, pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, g[:, enc["input_ids"].shape[1]:]):
            res[i] = tok.decode(row, skip_special_tokens=True)
    return [res[i] for i in range(len(prompts))]


def parse_json_lines(text):
    """Every top-level JSON object in the text (one per line or pretty-printed over several lines)."""
    import json
    dec, out, i = json.JSONDecoder(), [], 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            d, j = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(d, dict):
            out.append(d)
        i = j
