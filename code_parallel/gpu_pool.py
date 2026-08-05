#!/usr/bin/env python
"""Run independent reconstruction jobs across GPUs -- one job per GPU, #GPUs at a
time. Embarrassingly parallel: each job is a full single-GPU run.py with its own
teacher/committee/query-buffer, so there is ZERO cross-GPU contention. Throughput
scales linearly with GPU count (a true 3x on 3 GPUs) for ANY arch/config --
including ones whose query buffer is far too big to share across GPUs.

Examples
--------
# one run per seed, 3 GPUs, all seeds run 3-at-a-time:
python gpu_pool.py --gpus 0,1,2 --seeds 0,1,2,3,4,5 -- \
    --variant v18_lbfgs --arch 12288,1024,200 --outer 60 --q 44000 --window 60 --combine --fast

# an arbitrary job list (one full run.py arg-string per line, '#' comments ok):
python gpu_pool.py --gpus 0,1,2 --jobs jobs.txt

Each job's stdout/stderr goes to pool_logs/jobN_gpuG.log.
"""
import argparse
import os
import subprocess
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids (default: all visible)")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds -> one job per seed (each gets "
                         "the common args after '--' plus --seed N)")
    ap.add_argument("--jobs", default=None,
                    help="file with one full run.py arg-string per line")
    ap.add_argument("--logdir", default=os.path.join(HERE, "pool_logs"))
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="args after '--' passed to every run.py job")
    args = ap.parse_args()

    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        import torch
        gpus = list(range(torch.cuda.device_count()))
    if not gpus:
        sys.exit("gpu_pool: no GPUs available")

    common = args.rest[1:] if args.rest[:1] == ["--"] else args.rest

    jobs = deque()
    if args.jobs:
        with open(args.jobs) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    jobs.append(line.split())
    if args.seeds:
        for s in args.seeds.split(","):
            if s.strip() != "":
                jobs.append(list(common) + ["--seed", s.strip()])
    if not jobs and common:
        jobs.append(list(common))
    if not jobs:
        sys.exit("gpu_pool: no jobs. Pass --jobs FILE or --seeds LIST plus "
                 "'-- <run.py args>'.")

    os.makedirs(args.logdir, exist_ok=True)
    total = len(jobs)
    print(f"gpu_pool: {total} jobs across GPUs {gpus} "
          f"({len(gpus)}-way concurrent, one per GPU)\n", flush=True)
    running = {}                                    # gpu -> (proc, jobid, logf, desc)
    counter = {"jid": 0, "done": 0}

    def launch(gpu):
        if not jobs:
            return
        job = jobs.popleft(); counter["jid"] += 1; jid = counter["jid"]
        logf = open(os.path.join(args.logdir, f"job{jid}_gpu{gpu}.log"), "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        proc = subprocess.Popen(
            [sys.executable, "run.py", "--device", "cuda", *job],
            cwd=HERE, env=env, stdout=logf, stderr=subprocess.STDOUT)
        running[gpu] = (proc, jid, logf, " ".join(job))
        print(f"[pool] GPU{gpu} <- job{jid}: {' '.join(job)}", flush=True)

    for g in gpus:
        launch(g)
    while running:
        for g in list(running):
            proc, jid, logf, desc = running[g]
            code = proc.poll()
            if code is not None:
                logf.close(); counter["done"] += 1; del running[g]
                tag = "ok" if code == 0 else f"FAILED({code})"
                print(f"[pool] GPU{g} finished job{jid} [{tag}] "
                      f"({counter['done']}/{total}): {desc}", flush=True)
                launch(g)
        time.sleep(1)
    print(f"\ngpu_pool: all {total} jobs done. Logs in {args.logdir}/", flush=True)


if __name__ == "__main__":
    main()
