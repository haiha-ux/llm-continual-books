"""Model loading, chat formatting, batched greedy generation (HF) and the vLLM teacher helper."""
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


def teacher_generate(prompts, temperature=0.0, seed=0, max_tokens=700, n=1, path=MODEL, gpu_util=0.3):
    """vLLM chat generation with the Qwen2.5-7B-Instruct teacher. Returns list (per prompt) of n strings."""
    from vllm import LLM, SamplingParams
    llm = _llm(path, gpu_util)
    sp = SamplingParams(temperature=temperature, top_p=0.95 if temperature else 1.0, max_tokens=max_tokens, n=n, seed=seed)
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    return [[c.text for c in o.outputs] for o in llm.chat(msgs, sp, use_tqdm=False)]


_LLM = {}


def _llm(path, gpu_util):
    from vllm import LLM
    if path not in _LLM:
        _LLM[path] = LLM(path, dtype="bfloat16", gpu_memory_utilization=gpu_util, max_model_len=4096, seed=0,
                         enforce_eager=True)
    return _LLM[path]


def parse_json_lines(text):
    out = []
    for ln in text.splitlines():
        ln = ln.strip().rstrip(",")
        m = re.search(r"\{.*\}", ln)
        if not m:
            continue
        try:
            import json
            d = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(d, dict):
            out.append(d)
    return out
