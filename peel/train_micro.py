"""Minimal trainer: train a small config-driven MicroT on REAL TinyStories bytes
and save {cfg, state_dict} for use as an extraction teacher (gpt_extract --ckpt).
Byte-level, vocab=256. Defaults to the minimal 1-block GPT (n_layer=1, d=16, 1 head,
ffn=32) -- the shallow target for a first extraction success."""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microt import MicroT, default_cfg, nll, generate  # noqa: E402


def load_bytes(path, device):
    return torch.tensor(list(open(path, "rb").read()), dtype=torch.long, device=device)


def batchify(data, T, batch, gen, device):
    ix = torch.randint(0, len(data) - T - 1, (batch,), generator=gen, device=device)
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tinystories.txt")
    ap.add_argument("--out", default="micro1L.pt")
    ap.add_argument("--n-layer", type=int, default=1); ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--n-head", type=int, default=1); ap.add_argument("--ffn", type=int, default=32)
    ap.add_argument("--T", type=int, default=128); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    data = load_bytes(a.data, dev)
    cfg = default_cfg(vocab=256, d_model=a.d_model, n_head=a.n_head, n_layer=a.n_layer, ffn=a.ffn)
    model = MicroT(cfg).to(dev)
    print(f"[train] cfg={cfg}\n[train] params={sum(p.numel() for p in model.parameters())} "
          f"data={len(data)} bytes on {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_scale(step):
        if step < a.warmup:
            return step / max(1, a.warmup)
        p = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    probe = "Once upon a time, there was a little girl named Lily who loved to play in the park."
    t0 = time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * lr_scale(step)
        x, y = batchify(data, a.T, a.batch, gen, dev)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % 500 == 0 or step == a.steps - 1:
            L = nll(model, probe, dev)
            print(f"  step {step+1:5d} | train-CE {loss.item():.3f} | probe-NLL {L:.3f} "
                  f"({L/0.693:.2f} bpb) | {round(time.time()-t0,1)}s", flush=True)
    torch.save({"cfg": cfg, "sd": model.state_dict()}, a.out)
    print(f"[train] saved -> {a.out}", flush=True)
    print("[sample]", repr(generate(model, "Once upon a time, ", 200, dev)), flush=True)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()
