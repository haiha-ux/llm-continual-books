"""Model loading, chat formatting, batched generation with plain HF transformers (no vLLM)."""
import os
import re

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import torch

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
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=32, batch=32, on_batch=None):
    """Greedy decoding on already-formatted prompts. on_batch(attention_mask) lets the gate see the prompt mask."""
    outs = []
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    res = {}
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        if on_batch:
            on_batch(enc["attention_mask"])
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                           top_k=None, pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, g[:, enc["input_ids"].shape[1]:]):
            res[i] = tok.decode(row, skip_special_tokens=True)
    return [res[i] for i in range(len(prompts))]


def teacher_generate(prompts, temperature=0.0, seed=0, max_tokens=700, n=1, path=MODEL, batch=8):
    """Chat generation with the Qwen2.5-7B-Instruct teacher (plain HF transformers).

    vLLM was dropped: on this GB10 (aarch64, unified memory) the machine hard-froze within seconds of vLLM
    starting to sample, three times in a row. Returns list (per prompt) of n strings.
    """
    model, tok = _teacher(path)
    out = []
    for s in range(0, len(prompts), batch):
        chunk = prompts[s:s + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(seed + s)
        kw = dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature else \
            dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_tokens, num_return_sequences=n,
                               pad_token_id=tok.pad_token_id, **kw)
        dec = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [dec[k * n:(k + 1) * n] for k in range(len(chunk))]
        del enc, g
        torch.cuda.empty_cache()
    return out


_T = {}


def _teacher(path):
    if path not in _T:
        _T[path] = load(path)
    return _T[path]


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
