#!/usr/bin/env python
"""Run a list of escalating-budget experiments across GPUs -- one experiment per
GPU, #GPUs at a time (default 3). Each experiment is a full independent
``experiment_runner.py`` process pinned to a single GPU via CUDA_VISIBLE_DEVICES,
so there is ZERO cross-GPU contention; as soon as a GPU's experiment finishes the
next queued one is launched on it (work-stealing), so N GPUs stay busy until the
queue drains.

Define the experiments either INLINE in the EXPERIMENTS list below, or in a JSON
file passed with ``--config experiments.json`` (a JSON list of the same dicts,
which overrides the inline list). Each dict maps to experiment_runner.py flags:

  required : variant, maxq
  optional : arch, q, outer, p, seed, teacher_seed, teacher_epochs, window,
             threshold, tag, verbose (bool), name (display/log label only)

Usage:
  python run_experiments.py                       # inline EXPERIMENTS, GPUs 0,1,2
  python run_experiments.py --gpus 0,1            # only GPUs 0 and 1
  python run_experiments.py --config my_runs.json # experiments from a file

Per-experiment stdout/stderr -> pool_logs/<name>.log; full metrics land in
results_sweep/ (written by experiment_runner.py). A summary table prints at the
end. Teachers are pre-built once (serially) before dispatch so two parallel jobs
never race to train+cache the same teacher (disable with --no-prebuild).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "experiment_runner.py")
OUTDIR = os.path.join(HERE, "results_sweep")

# ------------------------------------------------------------------ EDIT ME --
# The list of experiments to run. Each dict = one escalating-budget sweep.
# (These are illustrative; replace with your own.)
EXPERIMENTS = [
    {"variant": "v18_min",   "arch": "784,32,10",     "q": 1500, "maxq": 48000,  "outer": 30, "seed": 0},
    {"variant": "v18_lbfgs", "arch": "3072,256,100",  "q": 1500, "maxq": 96000,  "outer": 40, "seed": 0, "window": 40},
    {"variant": "v18_lbfgs", "arch": "12288,1024,200","q": 4000, "maxq": 128000, "outer": 40, "seed": 0, "window": 40},
]
# -----------------------------------------------------------------------------

# experiment-dict key -> experiment_runner.py valued flag (stringified as-is)
_ARG_KEYS = [
    ("arch", "--arch"), ("q", "--q"), ("maxq", "--maxq"), ("outer", "--outer"),
    ("p", "--p"), ("seed", "--seed"), ("teacher_seed", "--teacher-seed"),
    ("teacher_epochs", "--teacher-epochs"), ("window", "--window"),
    ("threshold", "--threshold"), ("solverwindow", "--solverwindow"),
    ("tag", "--tag"), ("qg_lr", "--qg_lr"),
]
# boolean experiment-dict key -> flag (emitted when truthy)
_FLAG_KEYS = [("combine", "--combine"), ("fast", "--fast"),
              ("verbose", "--verbose")]
_DEFAULTS = {"arch": "784,64,10", "seed": 0, "teacher_epochs": 25,
             "teacher_seed": 0}   # must match experiment_runner.py's argparse


def _get(exp, key):
    return exp.get(key, _DEFAULTS.get(key))


def exp_to_argv(exp):
    """Build the experiment_runner.py CLI args for one experiment dict."""
    argv = ["--variant", str(exp["variant"])]
    for key, flag in _ARG_KEYS:
        if exp.get(key) is not None:
            argv += [flag, str(exp[key])]
    for key, flag in _FLAG_KEYS:
        if exp.get(key):
            argv.append(flag)
    return argv


def expand_config(obj):
    """Accept either a plain JSON list of experiment dicts, or a dict of the
    form {"shared": {...}, "archs": [...]} / {"shared": {...},
    "experiments": [...]}: 'shared' params are merged into each entry (a
    per-entry key overrides the shared one). Matches the "shared params + a list
    of architectures" way of defining a scan."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        shared = obj.get("shared", {})
        if "experiments" in obj:
            return [{**shared, **e} for e in obj["experiments"]]
        if "archs" in obj:
            return [{**shared, "arch": a} for a in obj["archs"]]
        sys.exit("run_experiments: config dict needs 'experiments' or 'archs'")
    sys.exit("run_experiments: config must be a JSON list or object")


def arch_tag(exp):
    return "x".join(_get(exp, "arch").split(","))


def result_path(exp):
    """Where experiment_runner.py will write this experiment's result JSON
    (mirror of its own naming, so we can read status back)."""
    tag = f"_{exp['tag']}" if exp.get("tag") else ""
    return os.path.join(
        OUTDIR, f"sweep_{exp['variant']}{tag}__{arch_tag(exp)}__s{_get(exp,'seed')}.json")


def exp_name(exp):
    parts = [str(exp["variant"]), arch_tag(exp), f"s{_get(exp, 'seed')}"]
    if exp.get("tag"):
        parts.append(str(exp["tag"]))
    return "__".join(parts)


def validate(experiments):
    names, paths, problems = {}, {}, []
    for i, exp in enumerate(experiments):
        if "variant" not in exp:
            problems.append(f"experiment {i}: missing required 'variant'")
        if exp.get("maxq") is None:
            problems.append(f"experiment {i}: missing required 'maxq'")
    if problems:
        sys.exit("run_experiments: invalid experiment list:\n  " +
                 "\n  ".join(problems))
    # unique, collision-free display names and result paths
    used = {}
    named = []
    for exp in experiments:
        base = exp_name(exp)
        n = base if base not in used else f"{base}#{used[base]}"
        used[base] = used.get(base, 0) + 1
        named.append((exp, n))
        p = result_path(exp)
        if p in paths:
            print(f"[warn] {n} and {paths[p]} share a result file ({os.path.basename(p)}); "
                  f"add a distinct 'tag' to keep both. The later run will overwrite.",
                  flush=True)
        paths[p] = n
    return named


def prebuild_teachers(experiments, device):
    """Build each unique teacher once, serially, before parallel dispatch, so
    two jobs never race to train+save the same cached teacher file."""
    from data import make_teacher, teacher_path
    import torch
    seen = set()
    for exp in experiments:
        dims = [int(x) for x in _get(exp, "arch").split(",")]
        te, ts = _get(exp, "teacher_epochs"), _get(exp, "teacher_seed")
        key = (tuple(dims), te, ts)
        if key in seen:
            continue
        seen.add(key)
        cached = os.path.exists(teacher_path(dims, te, ts))
        print(f"[prebuild] teacher {dims} epochs={te} seed={ts}: "
              f"{'cached' if cached else 'BUILDING (uncached)'}", flush=True)
        torch.manual_seed(ts)
        m = make_teacher(dims, epochs=te, seed=ts, device=device, verbose=False)
        del m
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def read_status(exp):
    p = result_path(exp)
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return None
    attempts = d.get("attempts") or []
    net = (attempts[-1].get("network") or {}) if attempts else {}
    return {
        "succeeded": d.get("succeeded"),
        "success_q": d.get("success_q"),
        "n_attempts": d.get("n_attempts"),
        "final_max_eps": net.get("max_eps"),
        "total_samples": d.get("total_samples_all_attempts"),
        "total_wall_s": d.get("total_wall_s"),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="0,1,2",
                    help="comma-separated GPU ids, or 'all' (default: 0,1,2)")
    ap.add_argument("--config", default=None,
                    help="JSON file with a list of experiment dicts (overrides "
                         "the inline EXPERIMENTS list)")
    ap.add_argument("--logdir", default=os.path.join(HERE, "pool_logs"))
    ap.add_argument("--no-prebuild", dest="prebuild", action="store_false",
                    help="skip the serial teacher pre-build (use if all teachers "
                         "are already cached)")
    ap.add_argument("--poll", type=float, default=2.0,
                    help="seconds between completion polls (default 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned experiments (command + result path) "
                         "and exit without launching anything")
    args = ap.parse_args()

    if args.gpus.strip() == "all":
        import torch
        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        sys.exit("run_experiments: no GPUs specified")

    experiments = EXPERIMENTS
    if args.config:
        with open(args.config) as f:
            experiments = expand_config(json.load(f))
    if not experiments:
        sys.exit("run_experiments: no experiments defined")

    named = validate(experiments)
    total = len(named)

    if args.dry_run:
        print(f"run_experiments (dry run): {total} experiments, "
              f"GPUs {gpus} ({len(gpus)}-way)\n", flush=True)
        for exp, name in named:
            print(f"  {name}")
            print(f"    cmd:    experiment_runner.py --device cuda "
                  f"{' '.join(exp_to_argv(exp))}")
            print(f"    result: {os.path.relpath(result_path(exp), HERE)}")
        return

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"run_experiments: {total} experiments across GPUs {gpus} "
          f"({len(gpus)}-way concurrent, one per GPU)\n", flush=True)

    if args.prebuild:
        prebuild_teachers([e for e, _ in named], device=f"cuda:{gpus[0]}")
        print("", flush=True)

    queue = deque(named)
    running = {}                       # gpu -> (proc, exp, name, logf, t0)
    done = 0

    def launch(gpu):
        if not queue:
            return
        exp, name = queue.popleft()
        logf = open(os.path.join(args.logdir, f"{name}.log"), "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        proc = subprocess.Popen(
            [sys.executable, RUNNER, "--device", "cuda", *exp_to_argv(exp)],
            cwd=HERE, env=env, stdout=logf, stderr=subprocess.STDOUT)
        running[gpu] = (proc, exp, name, logf, time.time())
        print(f"[pool] GPU{gpu} <- {name}  (q0={_get(exp,'q') or 1500} "
              f"maxq={exp['maxq']})", flush=True)

    try:
        for g in gpus:
            launch(g)
        while running:
            for g in list(running):
                proc, exp, name, logf, t0 = running[g]
                code = proc.poll()
                if code is None:
                    continue
                logf.close()
                del running[g]
                done += 1
                wall = time.time() - t0
                st = read_status(exp)
                if code != 0:
                    verdict = f"CRASHED(exit {code})"
                elif st is None:
                    verdict = "no result file"
                elif st["succeeded"]:
                    verdict = f"SUCCESS at q={st['success_q']}"
                else:
                    mx = st["final_max_eps"]
                    verdict = (f"failed (best max_eps="
                               f"{mx:.3e})" if mx is not None else "failed")
                print(f"[pool] GPU{g} done {name} [{verdict}] "
                      f"({done}/{total}, {wall:.0f}s)", flush=True)
                launch(g)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\n[pool] interrupted -- terminating running experiments...",
              flush=True)
        for proc, _e, _n, logf, _t in running.values():
            proc.terminate()
            logf.close()
        for proc, *_ in running.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise

    # ------------------------------------------------------------- summary --
    print(f"\n{'=' * 78}\nSUMMARY ({total} experiments)\n{'=' * 78}", flush=True)
    header = f"{'experiment':<44} {'result':<22} {'attempts':>8} {'wall(s)':>9}"
    print(header)
    print("-" * len(header))
    n_ok = 0
    for exp, name in named:
        st = read_status(exp)
        if st is None:
            res, att, wall = "no result", "-", "-"
        elif st["succeeded"]:
            n_ok += 1
            res = f"OK q={st['success_q']}"
            att, wall = st["n_attempts"], f"{st['total_wall_s']:.0f}"
        else:
            mx = st["final_max_eps"]
            res = f"fail {mx:.2e}" if mx is not None else "fail"
            att, wall = st["n_attempts"], f"{st['total_wall_s']:.0f}"
        print(f"{name:<44} {res:<22} {str(att):>8} {str(wall):>9}", flush=True)
    print("-" * len(header))
    print(f"{n_ok}/{total} succeeded. Logs: {args.logdir}/  Results: {OUTDIR}/",
          flush=True)


if __name__ == "__main__":
    main()
