"""Train (cached) L1 teachers over a lambda grid and print accuracy + sparsity.
  python teacher_l1_scan.py --device cuda:0 --prox 1 3 10 --pen 1e-5 1e-4"""
import argparse
import torch
from data import make_teacher, load_data, print_weight_sparsity

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="12288,1024,200")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--prox", type=float, nargs="*", default=[])
ap.add_argument("--pen", type=float, nargs="*", default=[])
args = ap.parse_args()
dims = [int(x) for x in args.arch.split(",")]
(_, _), (xte, yte) = load_data(dims, args.device)
for l1, prox in [(v, False) for v in args.pen] + [(v, True) for v in args.prox]:
    net = make_teacher(dims, l1=l1, l1_prox=prox, device=args.device, verbose=False)
    with torch.no_grad():
        acc = (net(xte).argmax(1) == yte).float().mean().item()
    print(f"=== l1={l1:g} {'prox' if prox else 'penalty'}  test acc {acc:.4f}", flush=True)
    print_weight_sparsity(net)
