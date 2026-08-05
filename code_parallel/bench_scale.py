"""Extended Benchmarks: Scale testing across Wider and Deeper architectures.

Architectures tested:
- Wide: [784, 128, 10]   (101,770 params)
- Wide: [784, 256, 10]   (203,530 params)
- Wide: [3072, 128, 10]  (394,496 params)
- Deep: [784, 64, 32, 10] (52,650 params, 3 hidden)
- Deep: [784, 64, 32, 16, 10] (53,216 params, 4 hidden)
"""
import json
import os
import sys
import time

import torch

from data import make_teacher, load_mnist
from method import Cfg, reconstruct
from nets import count_params

RESULTS_SCALE = os.path.join(os.path.dirname(__file__), "results_scale")

SCALE_CONFIGS = {
    # Wide networks
    "wide_128": ([784, 128, 10], 40, 2500),   # 100k queries
    "wide_256": ([784, 256, 10], 40, 3000),   # 120k queries
    "wide_3072": ([3072, 128, 10], 40, 3000), # 120k queries
    # Deep networks
    "deep_3layer": ([784, 64, 32, 10], 50, 2500), # 125k queries
    "deep_4layer": ([784, 64, 32, 16, 10], 60, 2500), # 150k queries
}

TEST_VARIANTS = {
    "v0_baseline": dict(),
    "v1b_medianpair": dict(disagree="median_pair"),
    "v13_popavg": dict(popavg_kappa=3.0),
    "v14_lbfgs": dict(lbfgs_polish=True),
    "v18_min": dict(disagree="median_pair", window=20, warmstart_iters=5,
                    fit_loss="mse", gate_kappa=3.0),
    "v17_full": dict(disagree="median_pair", window=20, warmstart_iters=5,
                     fit_loss="mse", lastlayer_every=5, popavg_kappa=3.0,
                     lbfgs_polish=True, gate_kappa=3.0),
}


def run_scale_experiment(arch_key, variant_key, seed=0, device="mps"):
    dims, outer, q = SCALE_CONFIGS[arch_key]
    var_kwargs = TEST_VARIANTS[variant_key]
    
    arch_str = "x".join(map(str, dims))
    os.makedirs(RESULTS_SCALE, exist_ok=True)
    out_path = os.path.join(RESULTS_SCALE, f"{arch_key}__{variant_key}__s{seed}.json")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return json.load(open(out_path))

    print(f"\n=======================================================")
    print(f"Running {arch_key} ({arch_str}) | {variant_key} | seed {seed}")
    print(f"Budget: {outer} outer x {q} queries = {outer * q} total queries")
    print(f"=======================================================", flush=True)

    # Build / load teacher
    torch.manual_seed(0)
    teacher = make_teacher(dims, epochs=25, seed=0, device=device, verbose=False)
    (_, _), (xte, _) = load_mnist(device)
    eval_pts = xte[:2000]

    # Configure reconstruction
    cfg = Cfg(p=8, q=q, outer=outer, epochs=10, qg_steps=30, batch=1024,
              log_every=10, **var_kwargs)
    
    t0 = time.time()
    best, log, final = reconstruct(teacher, dims, cfg, device, eval_pts, seed=seed)
    
    res = {
        "arch_key": arch_key,
        "variant": variant_key,
        "dims": dims,
        "n_params": count_params(teacher),
        "seed": seed,
        "queries": final["queries"],
        "final_max_eps": final["final_max_eps"],
        "final_mean_eps": final["final_mean_eps"],
        "final_agree": final["final_agree"],
        "wall_s": final["wall_s"],
        "log": log,
    }
    
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
        
    print(f"[DONE {arch_key}] {variant_key} s{seed} -> max_eps={final['final_max_eps']:.3e} "
          f"agree={final['final_agree']:.4f} wall={final['wall_s']}s", flush=True)
    return res


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    arch_filter = sys.argv[1] if len(sys.argv) > 1 else None
    variant_filter = sys.argv[2] if len(sys.argv) > 2 else None
    
    for a_key in SCALE_CONFIGS:
        if arch_filter and arch_filter not in a_key:
            continue
        for v_key in TEST_VARIANTS:
            if variant_filter and variant_filter not in v_key:
                continue
            for s in [0, 1]:
                run_scale_experiment(a_key, v_key, seed=s, device=device)
