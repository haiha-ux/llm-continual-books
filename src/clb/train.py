"""LoRA fine-tuning loop: SFT (loss on completion tokens) with optional KL-to-reference on a replay stream."""
import math
import os
import random
import time

import torch
from clb.thermal import cool_down
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


def plan_batches(n, bs, epochs, lens, seed):
    """Every step's row indices, fixed up front from the seed (so a resumed run replays the same schedule).
    Per epoch: shuffle, sort windows of 8*bs rows by length (little padding), shuffle the batches."""
    rng, steps = random.Random(seed), []
    for _ in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        bat = []
        for w in range(0, n, 8 * bs):
            win = sorted(order[w:w + 8 * bs], key=lambda i: lens[i])
            bat += [win[k:k + bs] for k in range(0, len(win), bs)]
        rng.shuffle(bat)
        steps += bat
    return steps


def _save_partial(path, mods, opt, sched, step, t_spent):
    tmp = str(path) + ".tmp"
    torch.save(dict(step=step, t=t_spent, opt=opt.state_dict(), sched=sched.state_dict(),
                    lora=[(m.lora_A.detach().cpu(), m.lora_B.detach().cpu()) for m in mods]), tmp)
    os.replace(tmp, path)


def fit(model, mods, data, pad_id, *, epochs=3, bs=8, lr=2e-4, seed=0, replay=None, kl_lambda=0.0, replay_bs=8,
        warmup=0.05, log=print, device="cuda", partial=None, save_every=100):
    """data/replay: encoded [(ids, labels)]. KL reference = model with `mods` disabled (the frozen stage below).
    partial: path of a resumable checkpoint (LoRA + optimizer + scheduler + step), written every `save_every` steps."""
    torch.manual_seed(seed)
    lora.freeze_except(model, mods)
    params = [p for m in mods for p in (m.lora_A, m.lora_B)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
    plan = plan_batches(len(data), bs, epochs, [len(i) for i, _ in data], seed)
    steps = len(plan)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warmup * steps)) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    step, spent, hist = 0, 0.0, []
    if partial and os.path.exists(partial):
        ck = torch.load(partial)
        for m, (a, b) in zip(mods, ck["lora"]):
            m.lora_A.data.copy_(a); m.lora_B.data.copy_(b)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, spent = ck["step"], ck["t"]
        log(f"resumed at step {step}/{steps}")
    model.train()
    t0 = time.time()
    while step < steps:
        cool_down()
        ids, lab, att = collate([data[i] for i in plan[step]], pad_id, device)
        opt.zero_grad(set_to_none=True)
        loss = sft_loss(model(input_ids=ids, attention_mask=att).logits, lab)
        loss_v = loss.item()
        loss.backward()  # backward first: the SFT graph is freed before the replay/KL forward is built
        kl_v = 0.0
        if replay and kl_lambda > 0:
            rrng = random.Random(seed * 100003 + step)
            rids, rlab, ratt = collate(rrng.sample(replay, min(replay_bs, len(replay))), pad_id, device)
            cur = model(input_ids=rids, attention_mask=ratt).logits
            with torch.no_grad():
                lora.set_enabled(mods, False)
                ref = model(input_ids=rids, attention_mask=ratt).logits
                lora.set_enabled(mods, True)
            kl = kl_loss(cur, ref, rlab)
            (kl_lambda * kl).backward()  # grads add up to the gradient of sft + lambda * kl
            kl_v = kl.item()
            del cur, ref, kl
        del loss
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); step += 1
        if step % 20 == 0 or step == steps:
            hist.append(dict(step=step, loss=loss_v + kl_lambda * kl_v, sft=loss_v, kl=kl_v))
            log(f"step {step}/{steps} sft {loss_v:.4f} kl {kl_v:.4f} {spent + time.time() - t0:.0f}s")
        if partial and step % save_every == 0 and step < steps:
            _save_partial(partial, mods, opt, sched, step, spent + time.time() - t0)
    model.eval()
    total = spent + time.time() - t0
    if partial and os.path.exists(partial):
        os.remove(partial)
    return dict(steps=steps, seconds=total, gpu_hours=total / 3600, hist=hist)
