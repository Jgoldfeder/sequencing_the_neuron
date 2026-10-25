"""Print the activation table from results/mergedbest_*_act_<act>_q<Q>__<arch>__s<seed>.json:
one row per activation at the SMALLEST q whose run is solved (max_eps < --solved, default 2e-3),
else its best (lowest mean_eps) run marked 'not solved'. Total queries = outer * q (outer = 40).

    python make_activation_table.py                 # markdown table
    python make_activation_table.py --arch 784x128x10 --all   # every (act, q) run, not just the best
"""
import argparse, glob, json, os, re

ORDER = ["leaky_relu", "nonleakyrelu", "sigmoid", "tanh", "square", "elu", "celu", "gelu", "silu", "mish",
         "softplus", "softsign", "selu", "hardswish", "squareplus", "abs", "pow3", "pow5", "sine",
         "softmax", "prelu", "glu", "swiglu"]
LABEL = {"square": "square (z^2)", "pow3": "pow3 (z^3)", "pow5": "pow5 (z^5)", "elu": "elu (= celu at alpha=1)",
         "nonleakyrelu": "relu (nonleakyrelu)"}

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
ap.add_argument("--arch", default="784x128x10")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--outer", type=int, default=40, help="outer iterations per run (total queries = outer * q)")
ap.add_argument("--solved", type=float, default=2e-3, help="max_eps threshold for 'solved'")
ap.add_argument("--all", action="store_true", help="list every (activation, q) run")
args = ap.parse_args()

pat = re.compile(r"mergedbest_(?:[a-z0-9_]+_)?act_(?P<act>[a-z0-9_]+?)_q(?P<q>\d+)__" + re.escape(args.arch) + rf"__s{args.seed}\.json$")
runs = {}
for path in glob.glob(os.path.join(args.results, "mergedbest_*act_*_q*__*__s*.json")):
    m = pat.search(os.path.basename(path))
    if not m:
        continue
    d = json.load(open(path))
    runs.setdefault(m["act"], []).append(dict(q=int(m["q"]), mean=d["final_mean_eps"], max=d["final_max_eps"],
                                             agree=d.get("final_agree", float("nan")), path=path))

def key(a):
    return (ORDER.index(a) if a in ORDER else len(ORDER), a)

rows = []
for act in sorted(runs, key=key):
    rs = sorted(runs[act], key=lambda r: r["q"])
    if args.all:
        rows += [(act, r, r["max"] < args.solved) for r in rs]
        continue
    solved = [r for r in rs if r["max"] < args.solved]
    r = solved[0] if solved else min(rs, key=lambda r: r["mean"])
    rows.append((act, r, bool(solved)))

print(f"| activation | q | total queries | mean eps | max eps | agree | status |")
print(f"|---|---|---|---|---|---|---|")
for act, r, ok in rows:
    tq = args.outer * r["q"]
    tq_s = f"{tq / 1e6:g}M" if tq >= 1e6 else f"{tq // 1000}k"
    print(f"| {LABEL.get(act, act)} | {r['q']} | {tq_s} | {r['mean']:.1e} | {r['max']:.1e} | {r['agree']:.4f} | "
          f"{'solved' if ok else 'not solved'} |")
print(f"\n{sum(ok for _, _, ok in rows)}/{len(rows)} solved (max_eps < {args.solved:g}); arch {args.arch}, "
      f"seed {args.seed}, {args.outer} outer iterations, teacher epochs 25, p = 8.")
