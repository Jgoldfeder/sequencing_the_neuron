#!/usr/bin/env python
"""solve_tanh.py -- ONE command: committee -> layer-1 pipeline -> K-FAC refinement stages -> final
unfrozen polish, printing max / mean parameter error and the query count after every stage.

  python solve_tanh.py --arch 3072,200,200,200,200,200,200,200,200,200,100 --tag eleven_a --gpus 0,2 \
      --q 1800000 --window 20 --epochs 5 --disk rings/eleven_a --qg-block-gb 1.61 --qg-chunk-gb 6.5

Stages (each skippable / resumable):
  1 committee     run.py (mergedbest512, tanh, --layer-consensus --align-inverse) on gpus[0]
  2 l1 pipeline   peel/diag/l1_pipeline.py: isolation (sharded over gpus) + gap-fill + orphan recovery +
                  tail-decay scales + assemble + K-FAC refine (layer 1 frozen)   [--refine-cg/--refine-iters]
  3 polish        peel/polish_full.py, K-FAC, layer 1 frozen, bigger CG budget   [--polish-cg/--polish-iters]
  4 final         peel/polish_full.py, K-FAC, everything free                     [--final-cg/--final-iters]
Shallow nets (committee already has layer 1: its W1 max eps < --l1-threshold) skip stages 2-3.
--skip-committee reuses an existing run's files (same --tag). Summary table at the end."""
import argparse, json, os, re, subprocess, sys, time

P = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--gpus", default="0", help="comma list; gpus[0] runs the committee and the solvers, all run isolation shards")
ap.add_argument("--variant", default="mergedbest512"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--teacher-seed", type=int, default=0)
ap.add_argument("--q", type=int, default=300000); ap.add_argument("--window", type=int, default=30); ap.add_argument("--outer", type=int, default=30)
ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--batch", type=int, default=None)
ap.add_argument("--disk", default=None); ap.add_argument("--disk-ram-gb", type=float, default=None)
ap.add_argument("--qg-block-gb", type=float, default=None); ap.add_argument("--qg-chunk-gb", type=float, default=None); ap.add_argument("--qg-tf32", action="store_true")
ap.add_argument("--extra", default="", help="extra run.py flags, quoted")
ap.add_argument("--skip-committee", action="store_true"); ap.add_argument("--skip-l1", action="store_true", help="force the shallow path (no isolation/scales)")
ap.add_argument("--l1-threshold", type=float, default=1e-2, help="committee W1 max eps below this -> shallow path")
ap.add_argument("--refine-cg", type=int, default=2000); ap.add_argument("--refine-iters", type=int, default=12)
ap.add_argument("--polish-cg", type=int, default=4000); ap.add_argument("--polish-iters", type=int, default=12)
ap.add_argument("--final-cg", type=int, default=4000); ap.add_argument("--final-iters", type=int, default=6)
ap.add_argument("--points", type=int, default=512); ap.add_argument("--dirs", type=int, default=64)
ap.add_argument("--keep-ring", action="store_true", help="keep the --disk replay ring after the committee (default: delete it, it is not used by the later stages)")
ap.add_argument("--trunk", choices=["consensus", "best"], default="consensus", help="solver starting net: aligned quorum-consensus net (default) or the best member")
args = ap.parse_args()
dims = [int(x) for x in args.arch.split(",")]; arch = "x".join(map(str, dims))
gpus = args.gpus.split(","); g0 = gpus[0]
name = f"{args.variant}_tanh_{args.tag}__{arch}__s{args.seed}"
R, L = "recon", "results/logs"; os.makedirs(L, exist_ok=True)
pop_path, final_path = f"{R}/_pop__{name}.pt", f"{R}/{name}_final.pt"
stages = []; q_total = 0; t_start = time.time()


def say(s): print(f"[solve_tanh] {s}", flush=True)


def sh(cmd, log=None):
    say(f"$ {cmd}" + (f"  > {log}" if log else ""))
    rc = subprocess.run(cmd + (f" > {log} 2>&1" if log else ""), shell=True).returncode
    if rc != 0:
        say(f"stage failed (exit {rc}); see {log}"); sys.exit(rc)


def final_line(log):
    """Parse 'FINAL vs black box: max_eps X | max per matrix [...] | mean per matrix [...]' -> (max, mean)."""
    txt = open(log).read()
    m = re.findall(r"FINAL vs black box: max_eps ([\d.e+-]+) \| max per matrix \[(.*?)\] \| mean per matrix \[(.*?)\]", txt)
    if not m: return None, None
    mx, _, means = m[-1]; means = [float(x.strip(" '")) for x in means.split(",")]
    return float(mx), sum(means) / len(means)


def queries_in(log, pat):
    return sum(int(x) for x in re.findall(pat, open(log).read())) if os.path.exists(log) else 0


def record(stage, mx, mean, q_stage, note=""):
    global q_total
    q_total += q_stage
    stages.append((stage, mx, mean, q_stage, q_total, time.time() - t_start))
    say(f"== {stage}: max eps {mx:.3e} | mean eps {mean:.3e} | queries this stage {q_stage/1e6:.2f}M | cumulative {q_total/1e6:.2f}M | {(time.time()-t_start)/3600:.2f} h{(' | ' + note) if note else ''}")


# ---------------------------------------------------------------- 1. committee
if not args.skip_committee:
    cmd = (f"CUDA_VISIBLE_DEVICES={g0} {P} run.py --variant {args.variant} --tanh --device cuda --arch {args.arch} "
           f"--q {args.q} --window {args.window} --outer {args.outer} --teacher-seed {args.teacher_seed} --seed {args.seed} "
           f"--layer-consensus --align-inverse --log-every 1 --pop-save-every 5 --tag {args.tag}")
    if args.epochs: cmd += f" --epochs {args.epochs}"
    if args.batch: cmd += f" --batch {args.batch}"
    if args.disk: cmd += f" --disk {args.disk}"
    if args.disk_ram_gb: cmd += f" --disk-ram-gb {args.disk_ram_gb}"
    if args.qg_block_gb: cmd += f" --qg-block-gb {args.qg_block_gb}"
    if args.qg_chunk_gb: cmd += f" --qg-chunk-gb {args.qg_chunk_gb}"
    if args.qg_tf32: cmd += " --qg-tf32"
    cmd += f" {args.extra}"
    sh(cmd, f"{L}/committee__{args.tag}.log")
res = json.load(open(f"results/{name}.json"))
if args.disk and not args.keep_ring and os.path.isdir(args.disk) and os.path.abspath(args.disk).startswith(os.path.abspath("rings")):
    import shutil; shutil.rmtree(args.disk, ignore_errors=True)     # the replay ring is not used after the committee
    say(f"freed replay ring {args.disk} (committee done; --keep-ring to keep it)")
w1_max = max(res["final_max_eps_per_matrix"][:2])
record("1 committee", res["final_max_eps"], res["final_mean_eps"], int(res["queries"]),
       f"certified {res['log'][-1].get('n_consensus')}/{res['log'][-1].get('n_total')} | W1 max {w1_max:.2e} | per-matrix max {' '.join(f'{x:.1e}' for x in res['final_max_eps_per_matrix'])}")

# ---------------------------------------------------------------- 2. layer-1 pipeline + frozen refine
shallow = args.skip_l1 or w1_max < args.l1_threshold
import torch
if shallow:
    say(f"layer 1 already at {w1_max:.1e} < {args.l1_threshold:.0e}: shallow path, skipping isolation/scales")
    guess = f"{R}/_guess__{args.tag}.pt"
    pop = torch.load(pop_path, map_location="cpu", weights_only=False); fin = torch.load(final_path, map_location="cpu", weights_only=False)
    _cs = None
    if args.trunk == "consensus":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "peel", "diag"))
        from consensus_guess import consensus_guess
        _cs, _ = consensus_guess(pop, fin["state_dict"])
    say(f"solver starting net = {'aligned quorum CONSENSUS net' if _cs is not None else 'best member'}")
    torch.save({"dims": dims, "act": "tanh", "teacher_state": pop["teacher_state"], "state_dict": _cs if _cs is not None else fin["state_dict"]}, guess)
    base = guess
else:
    sh(f"TRUNK={args.trunk} {P} peel/diag/l1_pipeline.py {args.tag} {pop_path} {final_path} {args.gpus} {args.refine_cg} {args.refine_iters}", f"{L}/l1_pipeline__{args.tag}.log")
    guess = f"{R}/_guess__{args.tag}.pt"
    q_iso = sum(queries_in(f"{L}/{f}", r"total queries (\d+)") for f in os.listdir(L) if f.startswith(f"iso_l1_{args.tag}") or f.startswith(f"iso_l1_orphan"))
    q_sc = sum(queries_in(f"{L}/{f}", r"\| (\d+) q\b") for f in os.listdir(L) if f.startswith(f"scale_l1_{args.tag}"))
    asm = re.findall(r"assembled net vs black box: max_eps ([\d.e+-]+) .*?mean per matrix \[(.*?)\]", open(f"{L}/refine_l1fixed__{args.tag}.log").read())
    if asm:
        means = [float(x.strip(" '")) for x in asm[-1][1].split(",")]
        record("2a layer-1 isolation + scales (assembled, before refine)", float(asm[-1][0]), sum(means) / len(means), q_iso + q_sc)
    mx, mean = final_line(f"{L}/refine_l1fixed__{args.tag}.log")
    record(f"2b K-FAC refine, L1 frozen (cg {args.refine_cg} x {args.refine_iters})", mx, mean, queries_in(f"{L}/refine_l1fixed__{args.tag}.log", r"-> (\d+) queries"))
    base = f"{R}/_l1fixed_refined__{args.tag}.pt"
    # 3. frozen polish with the bigger budget
    if args.polish_iters > 0:
        out = f"{R}/_polished__{args.tag}.pt"
        sh(f"FREEZE=layers.0.weight PRECOND=kfac CUDA_VISIBLE_DEVICES={g0} {P} peel/polish_full.py {guess} {base} {out} {args.points} {args.dirs} {args.polish_iters} {args.polish_cg}", f"{L}/polish__{args.tag}.log")
        mx, mean = final_line(f"{L}/polish__{args.tag}.log")
        record(f"3 K-FAC polish, L1 frozen (cg {args.polish_cg} x {args.polish_iters})", mx, mean, queries_in(f"{L}/polish__{args.tag}.log", r"-> (\d+) queries"))
        base = out

# ---------------------------------------------------------------- 4. final polish, everything free
if args.final_iters > 0:
    out = f"{R}/_final__{args.tag}.pt"
    sh(f"PRECOND=kfac CUDA_VISIBLE_DEVICES={g0} {P} peel/polish_full.py {guess} {base} {out} {args.points} {args.dirs} {args.final_iters} {args.final_cg}", f"{L}/final__{args.tag}.log")
    mx, mean = final_line(f"{L}/final__{args.tag}.log")
    record(f"4 K-FAC final polish, all free (cg {args.final_cg} x {args.final_iters})", mx, mean, queries_in(f"{L}/final__{args.tag}.log", r"-> (\d+) queries"))
    base = out

print("\n" + "=" * 100)
print(f"SUMMARY {name}  ({(time.time()-t_start)/3600:.2f} h)")
print(f"{'stage':62s} {'max eps':>10s} {'mean eps':>10s} {'queries':>10s} {'cumul':>10s} {'hours':>6s}")
for st, mx, mean, qs, qc, t in stages:
    print(f"{st:62s} {mx:10.3e} {mean:10.3e} {qs/1e6:9.2f}M {qc/1e6:9.2f}M {t/3600:6.2f}")
print(f"final net: {base}")
print("=" * 100, flush=True)
