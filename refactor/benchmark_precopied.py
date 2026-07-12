#!/usr/bin/env python
"""Benchmark get_adv with pre-copied models vs copying each time."""
import time
import copy
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from models import var_FNN
import utils

activation_f = nn.LeakyReLU()
layers = [784, 256, 128, 10]
num_models = 10
num_samples = 30000
num_epochs = 100

print(f"{num_samples} samples, {num_models} models, {num_epochs} epochs")
print("=" * 60)

# Create models
models = [var_FNN(activation_f, layers).cuda(0) for _ in range(num_models)]

gpu_ids = [0, 1, 2]

# Benchmark 1: Single GPU baseline
print("\n[1] Single GPU (30k samples)...")
torch.cuda.synchronize()
start = time.time()
result_1gpu = utils.get_adv(models, num_samples=num_samples, epochs=num_epochs,
                            input_dim=784, model_type='fnn', gpu_ids=None)
torch.cuda.synchronize()
t_1gpu = time.time() - start
print(f"  Time: {t_1gpu:.2f}s")

# Benchmark 2: 3 GPUs WITHOUT pre-copied models (should show warning)
print("\n[2] 3 GPUs WITHOUT pre-copied models...")
torch.cuda.synchronize()
start = time.time()
result_3gpu_nocopy = utils.get_adv(models, num_samples=num_samples, epochs=num_epochs,
                                    input_dim=784, model_type='fnn',
                                    gpu_ids=gpu_ids, gpu_model_copies=None)
torch.cuda.synchronize()
t_3gpu_nocopy = time.time() - start
print(f"  Time: {t_3gpu_nocopy:.2f}s")
print(f"  Speedup: {t_1gpu/t_3gpu_nocopy:.2f}x")

# Benchmark 3: 3 GPUs WITH pre-copied models
print("\n[3] 3 GPUs WITH pre-copied models...")
# Pre-copy models (this would be done once per outer iteration)
print("  Pre-copying models...")
start_copy = time.time()
gpu_model_copies = {gpu_id: [copy.deepcopy(m).cuda(gpu_id) for m in models] for gpu_id in gpu_ids}
t_copy = time.time() - start_copy
print(f"  Copy time: {t_copy*1000:.1f}ms")

torch.cuda.synchronize()
start = time.time()
result_3gpu_precopied = utils.get_adv(models, num_samples=num_samples, epochs=num_epochs,
                                       input_dim=784, model_type='fnn',
                                       gpu_ids=gpu_ids, gpu_model_copies=gpu_model_copies)
torch.cuda.synchronize()
t_3gpu_precopied = time.time() - start
print(f"  Compute time: {t_3gpu_precopied:.2f}s")
print(f"  Speedup (compute only): {t_1gpu/t_3gpu_precopied:.2f}x")
print(f"  Speedup (including copy): {t_1gpu/(t_3gpu_precopied + t_copy):.2f}x")

# Benchmark 4: Multiple calls with same pre-copied models (simulates outer loop)
print("\n[4] 3 calls with same pre-copied models (simulates outer loop)...")
torch.cuda.synchronize()
start = time.time()
for _ in range(3):
    result = utils.get_adv(models, num_samples=num_samples, epochs=num_epochs,
                           input_dim=784, model_type='fnn',
                           gpu_ids=gpu_ids, gpu_model_copies=gpu_model_copies)
torch.cuda.synchronize()
t_3calls = time.time() - start
print(f"  Total time (3 calls): {t_3calls:.2f}s")
print(f"  Time per call: {t_3calls/3:.2f}s")
print(f"  Speedup per call: {t_1gpu/(t_3calls/3):.2f}x")

print("\n" + "=" * 60)
print("Summary:")
print(f"  1 GPU baseline: {t_1gpu:.2f}s")
print(f"  3 GPU without pre-copy: {t_3gpu_nocopy:.2f}s ({t_1gpu/t_3gpu_nocopy:.2f}x)")
print(f"  3 GPU with pre-copy: {t_3gpu_precopied:.2f}s ({t_1gpu/t_3gpu_precopied:.2f}x)")
print(f"  Model copy overhead: {t_copy*1000:.1f}ms")
