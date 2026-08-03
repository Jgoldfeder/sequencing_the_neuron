"""Aggregate result JSONs into a ranked markdown table.

Usage: python compare.py [arch_tag]   e.g. python compare.py 784x32x10
"""
import glob
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(__file__), "results")
SUCCESS_EPS = 1e-3


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def queries_to(log, thresh):
    for rec in log:
        if rec["max_eps"] < thresh:
            return rec["queries"]
    return None


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "784x32x10"
    rows = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, f"*__{arch}__s*.json"))):
        r = json.load(open(f))
        v = r["variant"]
        rows.setdefault(v, []).append(r)

    base_med = None
    out = []
    for v, runs in sorted(rows.items()):
        eps = [r["final_max_eps"] for r in runs]
        walls = [r["wall_s"] for r in runs]
        agree = [r["final_agree"] for r in runs]
        succ = sum(e < SUCCESS_EPS for e in eps)
        q1e3 = [queries_to(r["log"], 1e-3) for r in runs]
        q1e4 = [queries_to(r["log"], 1e-4) for r in runs]
        q1e3s = [q for q in q1e3 if q is not None]
        q1e4s = [q for q in q1e4 if q is not None]
        med = median(eps)
        if v == "v0_baseline":
            base_med = med
        out.append({
            "variant": v,
            "n": len(runs),
            "succ": f"{succ}/{len(runs)}",
            "min_eps": min(eps),
            "med_eps": med,
            "med_agree": median(agree),
            "med_wall": median(walls),
            "q_to_1e3": median(q1e3s) if q1e3s else None,
            "q_to_1e4": median(q1e4s) if q1e4s else None,
            "all_eps": ", ".join(f"{e:.1e}" for e in eps),
        })

    out.sort(key=lambda r: r["med_eps"])
    print(f"\n## Results on {arch} (success = max_eps < {SUCCESS_EPS:.0e})\n")
    print("| rank | variant | succ | median max-eps | min max-eps | q->1e-3 | "
          "q->1e-4 | med agree | med wall(s) | vs v0 | all runs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(out):
        ratio = (base_med / r["med_eps"]) if base_med else float("nan")
        print(f"| {i + 1} | {r['variant']} | {r['succ']} | "
              f"{r['med_eps']:.2e} | {r['min_eps']:.2e} | "
              f"{r['q_to_1e3'] or '-'} | {r['q_to_1e4'] or '-'} | "
              f"{r['med_agree']:.4f} | {r['med_wall']:.0f} | "
              f"{ratio:.2f}x | {r['all_eps']} |")
    print()


if __name__ == "__main__":
    main()
