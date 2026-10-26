"""Sample complexity of committee recovery vs the teacher's L1 penalty (--teacher-l1).

Reads the per-iteration committee snapshots of each run (POP_SAVE_ALL=1
--pop-save-every 1 -> recon/_pop__<variant>_<tag>__<arch>__s<seed>_itNNN.pt) and
the run's results json (queries per logged iter). Every score uses the sparse-net
quotient of sparse_align.py on a probe domain X (default: 20k points of the run's own
final query buffer; --images-box for test images + U[-1,1]^d): teacher units that are dead-in, dead-out, linear on X
or duplicates are not identifiable and are excluded (their count is reported); the
linear ones are scored jointly as one affine map. Per iteration:

  union coverage  for every identifiable teacher L1 unit, the w-only angle to the NEAREST
                  row of ANY committee member: what the kink refiner needs (a guess
                  inside its ~15 deg capture basin at 12288 for each plane).
  best matched    the member with the smallest median angle under activation-based
                  (Hungarian, pre-activation correlation) pairing: per-unit angle,
                  column rel err, affine rel err, function rel err on X.
Angles are scale-invariant, so L1 shrinking |w| doesn't flatter sparse teachers.

Headline: queries needed to reach each threshold (fraction --fracs of identifiable
units within --angles deg), per run.

  python sample_complexity.py --arch 12288,1024,200 --runs l1_0 l11e-06 l10.0001
"""
import argparse
import glob
import json
import os
import re

import torch

from nets import MLP
from sparse_align import classify_units, probe_set, sparse_align_errors

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def nearest_angles(Wt, Ws):
    """deg angle from each teacher row (incl. bias) to its nearest row in Ws."""
    a = torch.nn.functional.normalize(Wt, dim=1)
    b = torch.nn.functional.normalize(Ws, dim=1)
    return torch.rad2deg(torch.arccos((a @ b.T).max(1).values.clamp(-1, 1)))


def rows(sd):
    """L1 rows WITHOUT bias (see sparse_align: a [w | b] angle measures the bias ratio
    when |b| >> ||w||)."""
    return sd["layers.0.weight"]


def analyze(snap_paths, json_path, dims, act, device, angles, fracs, q=None, buffer_path=None):
    log = json.load(open(json_path))["log"] if os.path.exists(json_path) else []
    q_at = {e["iter"]: e["queries"] for e in log}
    q_per = (log[0]["queries"] // log[0]["iter"]) if log else q
    buf = None
    if buffer_path and os.path.exists(buffer_path):
        buf = torch.load(buffer_path, map_location="cpu", mmap=True, weights_only=False).get("X")
    X = probe_set(dims, device, buffer=buf).double()
    out = []
    teacher = ident = None
    for p in snap_paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        it = ck["iter"]
        if teacher is None:
            teacher = MLP(dims, act=act)
            teacher.load_state_dict(ck["teacher_state"])
            teacher = teacher.double().to(device)
            info = classify_units(teacher, X)[0]
            ident = info["ident"]
            cls = {k: int(info[k].sum()) for k in ("dead_in", "dead_out", "linear", "dup", "ident")}
            print(f"  teacher units on X: {cls}", flush=True)
            Wt = rows(ck["teacher_state"]).double().to(device)[ident]
        union = torch.stack([nearest_angles(Wt, rows(s).double().to(device))
                             for s in ck["pop_states"]]).min(0).values
        best = None
        for s in ck["pop_states"]:
            m = MLP(dims, act=act)
            m.load_state_dict(s)
            e = sparse_align_errors(m.to(device), teacher, X)
            L = e["layers"][0]
            if best is None or L["angle_median"] < best[0]["angle_median"]:
                best = (L, e["fn_rel_err"])
        L, fn = best
        rec = {"iter": it, "queries": q_at.get(it, it * q_per if q_per else None), **cls,
               "union_median_deg": union.median().item(),
               "best_median_deg": L["angle_median"], "best_corr_median": L["corr_median"],
               "col_rel_median": L["col_rel_median"], "affine_rel_err": L["affine_rel_err"],
               "fn_rel_err": fn, "recon_spurious_kinking": L["recon_spurious_kinking"]}
        for A in angles:
            rec[f"union_frac<{A:g}"] = (union < A).float().mean().item()
            rec[f"best_frac<{A:g}"] = (L["angle"] < A).float().mean().item()
        out.append(rec)
        print(f"  it {it:3d}  q {rec['queries']}  union med {rec['union_median_deg']:6.2f}deg  "
              + "  ".join(f"<{A:g}: {rec[f'union_frac<{A:g}']:.3f}" for A in angles)
              + f"  | matched med {L['angle_median']:6.2f}  col {L['col_rel_median']:.2e}"
              + f"  affine {L['affine_rel_err']:.2e}  fn {fn:.2e}", flush=True)
    thresholds = {}
    for kind in ("union", "best"):
        for A in angles:
            for f in fracs:
                k = f"{kind} {f:g} of units < {A:g}deg"
                thresholds[k] = next((r["queries"] for r in out if r[f"{kind}_frac<{A:g}"] >= f), None)
    return out, thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--variant", default="v18_lbfgs")
    ap.add_argument("--runs", nargs="+", required=True, help="run tags, e.g. l1_0 l11e-06")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--act", default="leaky_relu")
    ap.add_argument("--angles", type=float, nargs="+", default=[1, 5, 15])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.5, 0.9, 0.99])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--q", type=int, default=24000,
                    help="queries per iter, used when a run's json isn't written yet (run in progress)")
    ap.add_argument("--images-box", action="store_true",
                    help="probe domain = test images + U[-1,1]^d instead of (default) 20k points of the "
                         "run's own query buffer (recon/<stub>.pt, mmap)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dims = [int(x) for x in args.arch.split(",")]
    arch = "x".join(map(str, dims))
    summary = {}
    for tag in args.runs:
        stub = f"{args.variant}_{tag}__{arch}__s{args.seed}"
        snaps = sorted(glob.glob(os.path.join(HERE, "recon", f"_pop__{stub}_it[0-9][0-9][0-9].pt")),
                       key=lambda p: int(re.search(r"_it(\d+)\.pt$", p).group(1)))
        if not snaps:
            print(f"[{tag}] no snapshots for {stub}")
            continue
        print(f"[{tag}] {len(snaps)} snapshots", flush=True)
        curve, thr = analyze(snaps, os.path.join(HERE, "results", stub + ".json"),
                             dims, args.act, args.device, args.angles, args.fracs, args.q,
                             None if args.images_box else os.path.join(HERE, "recon", stub + ".pt"))
        summary[tag] = {"curve": curve, "queries_to": thr}
    print("\nqueries needed (None = not reached within budget)")
    keys = next(iter(summary.values()))["queries_to"].keys() if summary else []
    print(f"{'threshold':30s}" + "".join(f"{t:>12s}" for t in summary))
    for k in keys:
        print(f"{k:30s}" + "".join(f"{str(summary[t]['queries_to'][k]):>12s}" for t in summary))
    out = args.out or os.path.join(HERE, "results", f"sample_complexity_{args.variant}__{arch}.json")
    json.dump(summary, open(out, "w"), indent=1)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
