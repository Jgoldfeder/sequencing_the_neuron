#!/usr/bin/env python
"""Find overhead in multi-GPU parallelization."""
import time
import torch
import torch.nn as nn
import threading
import copy
import sys
sys.path.insert(0, '.')

from models import var_FNN

activation_f = nn.LeakyReLU()
layers = [784, 256, 128, 10]
num_models = 10
num_samples = 30000
num_epochs = 100

print(f"{num_samples} samples, {num_models} models, {num_epochs} epochs")
print("=" * 60)

# Measure individual overheads
print("\n[Overhead 1] copy.deepcopy of 10 models...")
models_orig = [var_FNN(activation_f, layers) for _ in range(num_models)]
start = time.time()
for gpu_id in [0, 1, 2]:
    models_copy = [copy.deepcopy(m).cuda(gpu_id) for m in models_orig]
t_copy = time.time() - start
print(f"Time: {t_copy*1000:.1f}ms")

print("\n[Overhead 2] Thread creation + join...")
def noop():
    pass
start = time.time()
for _ in range(100):
    threads = [threading.Thread(target=noop) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
t_thread = (time.time() - start) / 100
print(f"Time per call: {t_thread*1000:.2f}ms")

print("\n[Overhead 3] torch.cat of results...")
chunks = [torch.randn(10000, 784) for _ in range(3)]
start = time.time()
for _ in range(100):
    result = torch.cat(chunks, dim=0)
t_cat = (time.time() - start) / 100
print(f"Time per call: {t_cat*1000:.2f}ms")

# Pure compute benchmark - FRESH SETUP EACH TIME
print("\n[Compute] Pure training loop (no overhead)...")

def make_gpu_data(gpu_id, n_samples):
    emb = nn.Embedding(n_samples, 784).cuda(gpu_id)
    models = [var_FNN(activation_f, layers).cuda(gpu_id) for _ in range(num_models)]
    opt = torch.optim.Adam(emb.parameters(), lr=0.01)
    return emb, models, opt

def train_pure(emb, models, opt, gpu_id):
    torch.cuda.set_device(gpu_id)
    for _ in range(num_epochs):
        weight = emb.weight
        outs = [torch.nn.functional.normalize(m(weight), p=1.0, dim=-1) for m in models]
        outs = torch.stack(outs).transpose(1, 0).contiguous()
        dists = torch.cdist(outs, outs)
        error = -dists.flatten().mean()
        error.backward()
        opt.step()
        opt.zero_grad()

# 1 GPU baseline (30k samples on 1 GPU)
print("1 GPU (30k samples)...")
emb_1, models_1, opt_1 = make_gpu_data(0, 30000)
torch.cuda.synchronize()
start = time.time()
train_pure(emb_1, models_1, opt_1, 0)
torch.cuda.synchronize()
t_1gpu = time.time() - start
print(f"  Time: {t_1gpu:.2f}s")
del emb_1, models_1, opt_1
torch.cuda.empty_cache()

# 3 GPU parallel (10k samples each, FRESH)
print("3 GPU (10k samples each)...")
gpu_data = {i: make_gpu_data(i, 10000) for i in [0, 1, 2]}

def train_wrapper(gpu_id):
    emb, models, opt = gpu_data[gpu_id]
    train_pure(emb, models, opt, gpu_id)

torch.cuda.synchronize()
start = time.time()
threads = [threading.Thread(target=train_wrapper, args=(i,)) for i in [0, 1, 2]]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize()
t_3gpu = time.time() - start
print(f"  Time: {t_3gpu:.2f}s")
print(f"  Speedup: {t_1gpu/t_3gpu:.2f}x")

print("\n" + "=" * 60)
print("Summary:")
print(f"  Pure compute speedup: {t_1gpu/t_3gpu:.2f}x")
print(f"  copy.deepcopy overhead: {t_copy*1000:.1f}ms (per get_adv call)")
print(f"  Threading overhead: {t_thread*1000:.2f}ms (per get_adv call)")
print(f"  torch.cat overhead: {t_cat*1000:.2f}ms (per get_adv call)")
