#!/usr/bin/env python3
"""Multi-GPU launcher for run.py. Identical arguments; add --gpus 0,1,2 (default: every visible CUDA device).
Query generation is split across the GPUs (each holds a replica of the whole committee); training splits
the members across the GPUs. Plain mode only (no --balance*). Nothing in the original code is modified:
this script installs two patches (parallel/mp.py) and then calls run.main().
    cd code && python parallel/run_parallel.py --variant mergedbest512 --arch 3072,200,200,200,200,200,200,100 \\
        --outer 30 --q 800000 --window 20 --device cuda --teacher-seed 0 --seed 0 --layer-consensus \\
        --align-inverse --log-every 1 --pop-save-every 5 --tanh --tag mp"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE); os.chdir(CODE)
import torch
gpus = None
if "--gpus" in sys.argv:
    i = sys.argv.index("--gpus"); gpus = [int(x) for x in sys.argv[i + 1].split(",")]; del sys.argv[i:i + 2]
devices = [f"cuda:{g}" for g in (gpus if gpus is not None else range(torch.cuda.device_count()))]
if any(a.startswith("--balance") for a in sys.argv):
    sys.exit("run_parallel.py: --balance* options are not supported in the parallel trainer (plain mode only)")
import method, stacked, mp
mp.set_devices(devices)
mp.install_query_patch(method)
mp.install_trainer_patch(stacked)
print(f"[run_parallel] {len(devices)} GPU(s): {devices} | main device {devices[0]} (pass --device {devices[0]})", flush=True)
import run
run.main()
