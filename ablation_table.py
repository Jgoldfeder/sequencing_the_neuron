"""Results table for the query-sampler ablation scripts (run_query_ablation_784.sh, run_query_ablation_tanh5L.sh).

    ABL_NET=784|784tanh|tanh5L|relu5L ABL_Q=<queries per iteration> [ABL_SEED=<seed>, relu5L] python ablation_table.py

Reads the run.py result JSONs and the run logs (last consensus line), prints one row per run found.
"""
import glob
import json
import os
import re

NET, Q = os.environ.get("ABL_NET", "784"), os.environ.get("ABL_Q", "1500")
DESC = {
    "disagree": "disagreement, search from U[-1,1]",
    "dgauss": "disagreement, search from N(0,0.5^2)",
    "ddatacov": "disagreement, search from data mean+cov",
    "uniform": "random U[-1,1]",
    "gauss": "random N(0,0.5^2)",
    "datacov": "random data mean+cov Gaussian",
    "leverage": "informed: best-member layer-input leverage",
    "levseq": "informed leverage, batch design",
    "levm3seq": "informed leverage, batch design, 3 members",
    "levseqdc": "informed leverage, batch design, data-cov start",
    "levseqnox": "informed leverage, batch design, no input term",
    "levdcr57": "informed leverage, data-cov start, norm cap 57",
    "levdcr80": "informed leverage, data-cov start, norm cap 80",
    "levtop4": "informed leverage, cap 80, top-4 directions",
    "levtop16": "informed leverage, cap 80, top-16 directions",
    "levres50": "informed leverage, cap 80, 50% worst-fit seeds",
    "levres100": "informed leverage, cap 80, 100% worst-fit seeds",
    "kinka": "kink design, tau .05, lr .1",
    "kinkb": "kink design, tau .05, lr .03",
    "kinkc": "kink design, tau .2, lr .03",
    "kinkm3": "kink design, 3 members",
    "kinkmix": "kink design, 30% plain data-cov draws",
    "kinkwide": "kink design, tau .1",
    "kinkm8": "kink design, all 8 members' kinks",
    "kinkres10": "kink design, 10% worst-fit seeds",
    "kinkres20": "kink design, 20% worst-fit seeds",
    "kinkres30": "kink design, 30% worst-fit seeds",
    "osuniform": "one-shot random U[-1,1]",
    "osgauss": "one-shot random N(0,0.5^2)",
    "osdatacov": "one-shot random data mean+cov",
}
if NET == "784":
    title = "784x128x10 leaky-ReLU (MNIST), mergedbest512 --fast --combine, no GN polish"
    pat = f"results/mergedbest512_ablation_q{Q}_*__784x128x10__s0.json"
    log_of = lambda r: f"results/logs/ablation784_q{Q}_{r}.log"
elif NET == "784tanh":
    title = "784x128x10 tanh (MNIST), mergedbest512 --fast --combine, no GN polish"
    pat = f"results/mergedbest512_tanh_ablation784tanh_q{Q}_*__784x128x10__s0.json"
    log_of = lambda r: f"results/logs/ablation784tanh_q{Q}_{r}.log"
elif NET == "relu5L":
    SEED = os.environ.get("ABL_SEED", "0")
    title = f"3072x200x200x200x100 leaky ReLU (CIFAR-100), committee stage only, seed {SEED}"
    pat = f"results/mergedbest512_ablation_relu5L_q{Q}_*__3072x200x200x200x100__s{SEED}.json"
    log_of = lambda r: f"results/logs/ablation_relu5L_q{Q}_s{SEED}_{r}.log"
else:
    title = "3072x200x200x200x100 tanh (CIFAR-100), committee stage only"
    pat = f"results/mergedbest512_tanh_ablation_tanh5L_q{Q}_*__3072x200x200x200x100__s0.json"
    log_of = lambda r: f"results/logs/ablation_tanh5L_q{Q}_{r}.log"

rows = {}
for f in glob.glob(pat):
    m = re.search(rf"q{Q}_([a-z0-9]+)", f)
    if not m or m.group(1) not in DESC:
        continue
    r = m.group(1); d = json.load(open(f))
    cons = "-"
    if os.path.exists(log_of(r)):
        last = [l for l in open(log_of(r)) if re.search(r"^\s+cons \d+/\d+", l)]
        if last:
            cons = re.search(r"cons (\d+/\d+)", last[-1]).group(1)
    stop = d.get("stopped_iter")
    solved = (f"yes (iter {stop})" if stop is not None else "no") if NET == "784" else (
        "yes" if cons != "-" and cons.split("/")[0] == cons.split("/")[1] else "no")
    nq = d["queries"]; c = d.get("cfg", {})
    if stop is None and not c.get("oneshot") and c.get("q") and c.get("outer") and nq == (c["outer"] + 1) * c["q"]:
        nq = c["outer"] * c["q"]         # runs saved before the count fix reported one block too many
    rows[r] = (DESC[r], f"{nq:,}", cons, solved, f"{d['final_max_eps']:.2e}", f"{d['final_mean_eps']:.2e}")

hdr = ("run", "query source", "queries", "consensus", "solved", "max eps", "mean eps")
table = [hdr] + [(r,) + rows[r] for r in DESC if r in rows]
w = [max(len(row[i]) for row in table) for i in range(len(hdr))]
line = "+" + "+".join("-" * (x + 2) for x in w) + "+"
print(f"\n{title} | q {Q} per iteration")
print(line)
for i, row in enumerate(table):
    print("| " + " | ".join(c.ljust(w[j]) if j < 4 else c.rjust(w[j]) for j, c in enumerate(row)) + " |")
    if i == 0:
        print(line.replace("-", "="))
print(line)
if len(table) == 1:
    print(f"(no results matching {pat})")
