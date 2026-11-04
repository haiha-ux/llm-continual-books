"""LoRA fine-tuning loop: SFT (loss on completion tokens) with optional KL-to-reference on a replay stream."""
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from . import lora
from .llm import chat

EOS = "<|im_end|>"


def qa_row(r):
    return chat(r["q"]), r["a"] + EOS


def note_row(r):
    return "", r["text"] + EOS


def encode(tok, rows, maxlen=320):
    out = []
    for p, c in rows:
        ip = tok(p, add_special_tokens=False).input_ids if p else []
        ic = tok(c, add_special_tokens=False).input_ids
        ids = (ip + ic)[:maxlen]
        out.append((ids, ([-100] * len(ip) + ic)[:maxlen]))
    return out


def collate(batch, pad_id, device="cuda"):
    n = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), n), pad_id)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros(len(batch), n, dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)] = torch.tensor(i); lab[k, :len(l)] = torch.tensor(l); att[k, :len(i)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def sft_loss(logits, labels):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)


def kl_loss(logits, ref_logits, labels):
    """KL(ref || cur) averaged over completion tokens."""
    m = (labels[:, 1:] != -100)
    lp = F.log_softmax(logits[:, :-1].float(), -1)
    lq = F.log_softmax(ref_logits[:, :-1].float(), -1)
    kl = (lq.exp() * (lq - lp)).sum(-1)
    return (kl * m).sum() / m.sum().clamp(min=1)


def fit(model, mods, data, pad_id, *, epochs=3, bs=8, lr=2e-4, seed=0, replay=None, kl_lambda=0.0, replay_bs=8,
        warmup=0.05, log=print, device="cuda"):
    """data/replay: encoded [(ids, labels)]. KL reference = model with `mods` disabled (the frozen stage below)."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    lora.freeze_except(model, mods)
    params = [p for m in mods for p in (m.lora_A, m.lora_B)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
    steps = epochs * math.ceil(len(data) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warmup * steps)) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    model.train()
    t0, step, hist = time.time(), 0, []
    for ep in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for s in range(0, len(order), bs):
            ids, lab, att = collate([data[i] for i in order[s:s + bs]], pad_id, device)
            loss = sft_loss(model(input_ids=ids, attention_mask=att).logits, lab)
            kl = torch.zeros(())
            if replay and kl_lambda > 0:
                rids, rlab, ratt = collate(rng.sample(replay, min(replay_bs, len(replay))), pad_id, device)
                cur = model(input_ids=rids, attention_mask=ratt).logits
                with torch.no_grad():
                    lora.set_enabled(mods, False)
                    ref = model(input_ids=rids, attention_mask=ratt).logits
                    lora.set_enabled(mods, True)
                kl = kl_loss(cur, ref, rlab)
                loss = loss + kl_lambda * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            if step % 20 == 0 or step == steps:
                hist.append(dict(step=step, loss=loss.item(), kl=float(kl)))
                log(f"step {step}/{steps} loss {loss.item():.4f} kl {float(kl):.4f} {time.time() - t0:.0f}s")
    model.eval()
    return dict(steps=steps, seconds=time.time() - t0, gpu_hours=(time.time() - t0) / 3600, hist=hist)
