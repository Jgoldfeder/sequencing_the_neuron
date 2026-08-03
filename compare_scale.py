"""Summary script for scale benchmarks in code/results_scale/."""
import glob
import json
import os
import sys

RESULTS_SCALE = os.path.join(os.path.dirname(__file__), "results_scale")

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

def main():
    groups = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_SCALE, "*.json"))):
        r = json.load(open(f))
        akey = r["arch_key"]
        vkey = r["variant"]
        groups.setdefault(akey, {}).setdefault(vkey, []).append(r)

    for akey, vmap in sorted(groups.items()):
        first_r = list(vmap.values())[0][0]
        dims = first_r["dims"]
        params = first_r["n_params"]
        queries = first_r["queries"]
        print(f"\n=======================================================================")
        print(f"Architecture: {akey} {dims} ({params:,} params) | Budget: {queries:,} queries")
        print(f"=======================================================================")
        print("| Rank | Variant | Succ | Median Max eps | Min Max eps | Agree | Med Wall (s) | vs v0 | All Runs |")
        print("|---|---|---|---|---|---|---|---|---|")
        
        base_med = None
        if "v0_baseline" in vmap:
            base_med = median([r["final_max_eps"] for r in vmap["v0_baseline"]])

        rows = []
        for vkey, runs in vmap.items():
            eps = [r["final_max_eps"] for r in runs]
            walls = [r["wall_s"] for r in runs]
            agrees = [r["final_agree"] for r in runs]
            succ = sum(e < 1e-3 for e in eps)
            med_eps = median(eps)
            rows.append({
                "vkey": vkey,
                "succ": f"{succ}/{len(runs)}",
                "med_eps": med_eps,
                "min_eps": min(eps),
                "med_agree": median(agrees),
                "med_wall": median(walls),
                "all_eps": ", ".join(f"{e:.2e}" for e in eps),
            })
            
        rows.sort(key=lambda x: x["med_eps"])
        for i, r in enumerate(rows):
            ratio = (base_med / r["med_eps"]) if base_med else 1.0
            print(f"| {i+1} | {r['vkey']:14s} | {r['succ']} | {r['med_eps']:.3e} | {r['min_eps']:.3e} | "
                  f"{r['med_agree']:.4f} | {r['med_wall']:.0f}s | {ratio:.2f}x | {r['all_eps']} |")

if __name__ == "__main__":
    main()
