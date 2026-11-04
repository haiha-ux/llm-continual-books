"""Teacher generation through a local Ollama server (stdlib only, one request at a time, resumable).

Model: qwen2.5:7b-instruct (Apache-2.0, Q4 quantised, same family as the student). Each (prompt, sample) result is
appended to a partial jsonl file the moment it arrives, so a crash or a power cut loses at most one request.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

from clb.thermal import cool_down

HOST = "http://127.0.0.1:11434"
TEACHER = "qwen2.5:7b-instruct"


def chat(prompt, *, temperature=0.0, seed=0, max_tokens=700, num_ctx=4096, model=TEACHER, host=HOST,
         keep_alive="2m", timeout=900, urlopen=urllib.request.urlopen):
    body = {"model": model, "stream": False, "keep_alive": keep_alive,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens, "num_ctx": min(num_ctx, 4096)}}
    req = urllib.request.Request(host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    if "message" not in d:
        raise RuntimeError(f"ollama error: {d.get('error', d)}")
    return d["message"]["content"]


def unload(model=TEACHER, host=HOST, urlopen=urllib.request.urlopen):
    """keep_alive 0 frees the GPU memory right away."""
    body = {"model": model, "keep_alive": 0, "messages": []}
    urlopen(urllib.request.Request(host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=60).read()


def _key(prompt, k):
    return hashlib.sha1(f"{k}|{prompt}".encode()).hexdigest()[:16]


def teacher_generate(prompts, temperature=0.0, seed=0, max_tokens=700, n=1, partial=None, ask=chat, cool=cool_down):
    """Returns, per prompt, a list of n strings (sample i uses seed+i). `partial` = jsonl path used to resume."""
    done = {}
    if partial and Path(partial).exists():
        for line in open(partial):
            r = json.loads(line)
            done[r["key"]] = r["out"]
    f = open(partial, "a") if partial else None
    out = []
    try:
        for p in prompts:
            row = []
            for k in range(n):
                key = _key(p, f"{seed + k}|{temperature}|{max_tokens}")
                if key not in done:
                    cool()
                    done[key] = ask(p, temperature=temperature, seed=seed + k, max_tokens=max_tokens)
                    if f:
                        f.write(json.dumps({"key": key, "out": done[key]}, ensure_ascii=False) + "\n"); f.flush()
                row.append(done[key])
            out.append(row)
    finally:
        if f:
            f.close()
    return out
