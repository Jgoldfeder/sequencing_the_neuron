"""Post-hoc layer-1 pipeline driver (isolation -> merge -> scales -> merge -> assemble+refine) for a finished
committee run, on the GPUs given. Usage:
  python peel/diag/l1_pipeline.py <tag> <pop.pt> <final.pt> <gpu,gpu> [refine_cg] [refine_iters]
pop.pt: the run's _pop__ snapshot (teacher_state + pop_states + dims); final.pt: the run's _final.pt (best member).
Writes recon/_guess__<tag>.pt, _iso_l1_shard*__<tag>.pt, _iso_l1_merged__<tag>.pt, _scale_l1_*__<tag>.pt,
_l1fixed_refined__<tag>.pt and logs under results/logs/."""
import sys, os, subprocess, time, torch
tag, pop_path, final_path, gpus = sys.argv[1:5]
CG = sys.argv[5] if len(sys.argv) > 5 else "2000"; IT = sys.argv[6] if len(sys.argv) > 6 else "10"
gpus = gpus.split(","); R = "recon"; L = "results/logs"
P = "/opt/miniconda3/bin/python"
pop = torch.load(pop_path, map_location="cpu", weights_only=False); fin = torch.load(final_path, map_location="cpu", weights_only=False)
dims = pop["dims"]; H = dims[1]
guess = f"{R}/_guess__{tag}.pt"
torch.save({"dims": dims, "act": "tanh", "teacher_state": pop["teacher_state"], "state_dict": fin["state_dict"], "note": f"best member of {final_path}"}, guess)
members = [f"{R}/_guess_m{m}__{tag}.pt" for m in range(len(pop["pop_states"]))]
for m, sd in enumerate(pop["pop_states"]):
    torch.save({"dims": dims, "act": "tanh", "teacher_state": pop["teacher_state"], "state_dict": sd}, members[m])
print(f"[pipeline] guess files written; dims {dims}; gpus {gpus}", flush=True)

def run_parallel(cmds):
    procs = [subprocess.Popen(c, shell=True) for c in cmds]
    for p in procs: p.wait()

def shards(n_items):
    k = len(gpus); b = -(-n_items // k)
    return [(i * b, min(n_items, (i + 1) * b) - 1) for i in range(k) if i * b < n_items]

# 1. isolation, sharded over gpus
t0 = time.time()
run_parallel([f"CUDA_VISIBLE_DEVICES={g} {P} peel/isolate_l1_deep.py {guess} {guess} {a}-{b} {R}/_iso_l1_shard{s}__{tag}.pt 0 > {L}/iso_l1_{tag}_shard{s}.log 2>&1"
              for s, ((a, b), g) in enumerate(zip(shards(H), gpus))])
res = {}
for s in range(len(shards(H))):
    res.update(torch.load(f"{R}/_iso_l1_shard{s}__{tag}.pt", map_location="cpu", weights_only=False)["results"])
bad = [i for i, r in res.items() if r["err_hist"][-1] > 1e-8]
print(f"[pipeline] isolation: {len(res)} neurons, {len(bad)} above 1e-8 ({time.time()-t0:.0f}s)", flush=True)
# 2. gap-fill the failures from other members' guesses. Members have DIFFERENT unit orders: pair the
#    failed best-member unit i to member m's unit j by max |cos| of the W1 rows, isolate j with member m's
#    guess, and accept only if the result still points along the best member's row (|cos| > 0.9).
g0 = fin["state_dict"]["layers.0.weight"].double(); g0n = g0 / g0.norm(dim=1, keepdim=True)
for m in range(1, len(members)):
    if not bad: break
    Wm = pop["pop_states"][m]["layers.0.weight"].double(); Wmn = Wm / Wm.norm(dim=1, keepdim=True)
    C = (g0n @ Wmn.T).abs(); pairs = {i: int(C[i].argmax()) for i in bad if float(C[i].max()) > 0.9}
    if not pairs: continue
    spec = ",".join(str(j) for j in pairs.values())
    subprocess.run(f"CUDA_VISIBLE_DEVICES={gpus[0]} {P} peel/isolate_l1_deep.py {guess} {members[m]} {spec} {R}/_iso_l1_gap{m}__{tag}.pt 0 > {L}/iso_l1_{tag}_gap{m}.log 2>&1", shell=True)
    g = torch.load(f"{R}/_iso_l1_gap{m}__{tag}.pt", map_location="cpu", weights_only=False)["results"]
    for i, j in pairs.items():
        r = g[j]; u = r["direction"].double()
        if r["err_hist"][-1] < 1e-8 and abs(float(u @ g0n[i])) > 0.9: res[i] = r
    bad = [i for i, r in res.items() if r["err_hist"][-1] > 1e-8]
    print(f"[pipeline] gap-fill from member {m} ({len(pairs)} paired units): {len(bad)} still above 1e-8", flush=True)
torch.save({"results": res, "dims": dims}, f"{R}/_iso_l1_merged__{tag}.pt")
orphans = list(bad)   # units no member can seed: recovered after the scales via the row-space complement (orphan_l1.py)
# 3. scales, sharded
keys = sorted(res)
for s, (a, b) in enumerate(shards(len(keys))):
    torch.save({"results": {i: res[i] for i in keys[a:b + 1]}, "dims": dims}, f"{R}/_iso_l1_part{s}__{tag}.pt")
t1 = time.time()
run_parallel([f"CUDA_VISIBLE_DEVICES={g} {P} peel/scale_l1_deep.py {guess} {R}/_iso_l1_part{s}__{tag}.pt {R}/_scale_l1_shard{s}__{tag}.pt 0 > {L}/scale_l1_{tag}_shard{s}.log 2>&1"
              for s, g in zip(range(len(shards(len(keys)))), gpus)])
sc = {}
for s in range(len(shards(len(keys)))):
    sc.update(torch.load(f"{R}/_scale_l1_shard{s}__{tag}.pt", map_location="cpu", weights_only=False)["results"])
n_ok = sum(1 for v in sc.values() if v.get("scale"))
print(f"[pipeline] scales: {n_ok}/{len(sc)} usable ({time.time()-t1:.0f}s)", flush=True)
torch.save({"results": sc, "dims": dims}, f"{R}/_scale_l1_merged__{tag}.pt")
# 3b. orphan units: row-space complement + saturating offset (needs the other rows' scales), then their scales
for u in orphans:
    subprocess.run(f"CUDA_VISIBLE_DEVICES={gpus[0]} {P} peel/diag/orphan_l1.py {guess} {R}/_iso_l1_merged__{tag}.pt {R}/_scale_l1_merged__{tag}.pt {u} 0 > {L}/iso_l1_{tag}_orphan{u}.log 2>&1", shell=True)
    res = torch.load(f"{R}/_iso_l1_merged__{tag}.pt", map_location="cpu", weights_only=False)["results"]
    torch.save({"results": {u: res[u]}, "dims": dims}, f"{R}/_iso_l1_part_orphan{u}__{tag}.pt")
    subprocess.run(f"CUDA_VISIBLE_DEVICES={gpus[0]} {P} peel/scale_l1_deep.py {guess} {R}/_iso_l1_part_orphan{u}__{tag}.pt {R}/_scale_l1_orphan{u}__{tag}.pt 0 > {L}/scale_l1_{tag}_orphan{u}.log 2>&1", shell=True)
    sc.update(torch.load(f"{R}/_scale_l1_orphan{u}__{tag}.pt", map_location="cpu", weights_only=False)["results"])
    print(f"[pipeline] orphan unit {u}: dir err {res[u]['err_hist'][-1]:.1e}, scale {'ok' if sc[u].get('scale') else 'FAILED'}", flush=True)
    torch.save({"results": sc, "dims": dims}, f"{R}/_scale_l1_merged__{tag}.pt")
# 4. assemble + refine (single GPU)
t2 = time.time()
subprocess.run(f"CUDA_VISIBLE_DEVICES={gpus[0]} {P} peel/assemble_and_refine_l1fixed.py {guess} {guess} {R}/_iso_l1_merged__{tag}.pt {R}/_scale_l1_merged__{tag}.pt {R}/_l1fixed_refined__{tag}.pt 512 64 {IT} {CG} kfac > {L}/refine_l1fixed__{tag}.log 2>&1", shell=True)
print(f"[pipeline] refine done ({time.time()-t2:.0f}s); total {time.time()-t0:.0f}s", flush=True)
print(open(f"{L}/refine_l1fixed__{tag}.log").read().splitlines()[-1][:300], flush=True)
