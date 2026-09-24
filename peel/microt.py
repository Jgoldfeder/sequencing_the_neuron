"""Config-driven reimplementation + loader for the MicroT decoder transformers
(llaa33219/MicroT-test1-*-TinyStories). Architecture inferred from the safetensors
layout; only n_head is not in the weights (swept by NLL). cfg keys:
  vocab, d_model, n_head, n_layer, ffn  (dims)
  rope ('half'|'interleaved'), eps, gelu ('none'|'tanh'), theta   (conventions)

embed [vocab,d_model] tied to output; per block: RMSNorm-> MHA (RoPE on q/k,
causal, no bias) -> RMSNorm -> GELU MLP (no bias); final RMSNorm; logits = h @ embed.T.
"""
import json
import os
import struct
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = {  # size tag -> HF repo
    "10K": "llaa33219/MicroT-test1-10K-TinyStories",
    "50K": "llaa33219/MicroT-test1-50K-TinyStories",
    "100K": "llaa33219/MicroT-test1-100K-TinyStories",
}
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(size="50K", epoch=2):
    path = os.path.join(HERE, f"micro{size}.safetensors")
    if not os.path.exists(path):
        url = f"https://huggingface.co/{REPO[size]}/resolve/main/epoch_{epoch}.safetensors"
        urllib.request.urlretrieve(url, path)
    return path


def load_st(path):
    data = open(path, "rb").read()
    n = struct.unpack("<Q", data[:8])[0]
    hdr = json.loads(data[8:8 + n]); base = 8 + n
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        arr = np.frombuffer(data[base + s:base + e], dtype=np.float32).reshape(v["shape"])
        out[k] = torch.from_numpy(arr.copy())
    return out


def infer_dims(sd):
    vocab, d_model = sd["embed.weight"].shape
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    ffn = sd["blocks.0.mlp.fc1.weight"].shape[0]
    return dict(vocab=int(vocab), d_model=int(d_model), n_layer=int(n_layer), ffn=int(ffn))


def default_cfg(**over):
    cfg = dict(vocab=256, d_model=32, n_head=2, n_layer=3, ffn=152,
               rope="half", eps=1e-5, gelu="none", theta=10000.0)
    cfg.update(over)
    return cfg


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cos_sin(T, hdim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, hdim, 2, device=device).float() / hdim))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin, mode):
    if mode == "half":
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        c, s = cos[None, None], sin[None, None]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    return torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], -1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.n_head = cfg["n_head"]; self.hdim = d // self.n_head; self.cfg = cfg
        self.q_proj = nn.Linear(d, d, bias=False); self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False); self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape; H, hd = self.n_head, self.hdim
        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hd).transpose(1, 2)
        q = apply_rope(q, cos, sin, self.cfg["rope"]); k = apply_rope(k, cos, sin, self.cfg["rope"])
        att = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
        att = (att + torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)).softmax(-1)
        return self.o_proj((att @ v).transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["d_model"], cfg["ffn"], bias=False)
        self.fc2 = nn.Linear(cfg["ffn"], cfg["d_model"], bias=False)
        self.cfg = cfg

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh" if self.cfg["gelu"] == "tanh" else "none"))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg["d_model"], cfg["eps"]); self.attn = Attn(cfg)
        self.norm2 = RMSNorm(cfg["d_model"], cfg["eps"]); self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class MicroT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab, self.d_model = cfg["vocab"], cfg["d_model"]
        self.n_head, self.n_layer, self.ffn = cfg["n_head"], cfg["n_layer"], cfg["ffn"]
        self.hdim = self.d_model // self.n_head
        self.embed = nn.Embedding(self.vocab, self.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(self.n_layer)])
        self.out_norm = RMSNorm(self.d_model, cfg["eps"])

    def forward(self, idx=None, inputs_embeds=None):
        x = self.embed(idx) if inputs_embeds is None else inputs_embeds
        cos, sin = rope_cos_sin(x.shape[1], self.hdim, self.cfg["theta"], x.device)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.out_norm(x) @ self.embed.weight.t()

    def clone(self):
        new = MicroT(self.cfg); new.load_state_dict(self.state_dict())
        return new.to(self.embed.weight.device).eval()


def build(cfg, sd, device="cpu"):
    m = MicroT(cfg); m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def nll(model, text, device):
    ids = torch.tensor([list(text.encode("utf-8"))], device=device)
    with torch.no_grad():
        return F.cross_entropy(model(ids[:, :-1]).reshape(-1, model.vocab),
                               ids[:, 1:].reshape(-1)).item()


@torch.no_grad()
def generate(model, prompt, n, device, temp=0.0):
    ids = torch.tensor([list(prompt.encode("utf-8"))], device=device)
    for _ in range(n):
        logits = model(ids[:, -1024:])[:, -1]
        nxt = logits.argmax(-1, keepdim=True) if temp == 0 else \
            torch.multinomial((logits / temp).softmax(-1), 1)
        ids = torch.cat([ids, nxt], 1)
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


if __name__ == "__main__":
    import sys
    size = sys.argv[1] if len(sys.argv) > 1 else "50K"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = load_st(fetch(size)); dims = infer_dims(sd)
    print(f"[{size}] dims={dims}  params={sum(v.numel() for v in sd.values())}\n")
    probe = ("Once upon a time, there was a little girl named Lily. She loved to "
             "play in the park with her friends. One day she found a small red ball.")
    best = None
    for nh in (1, 2, 4):
        if dims["d_model"] % nh:
            continue
        for rope in ("half", "interleaved"):
            cfg = default_cfg(**dims, n_head=nh, rope=rope)
            L = nll(build(cfg, sd, dev), probe, dev)
            print(f"n_head={nh} rope={rope:11s} -> NLL {L:.3f} ({L/0.693:.2f} bits/byte)")
            if best is None or L < best[0]:
                best = (L, cfg)
    print(f"\nBEST: n_head={best[1]['n_head']} rope={best[1]['rope']}  NLL {best[0]:.3f}")
    print(repr(generate(build(best[1], sd, dev), "Once upon a time", 160, dev)))
