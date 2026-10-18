"""Jet-solver polish of a near-exact net; env FREEZE="layers.0.weight,..." freezes parameters (default: none). Usage: python peel/polish_full.py <pop.pt> <base_net.pt> <out.pt> [points dirs iters cg]"""
import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nets import MLP
import tanh_solve as ts
from method import param_errors
pop_path, base_path, out_path = sys.argv[1:4]
P_, D_, IT_, CG_ = (int(a) for a in (sys.argv[4:8] + ["512", "64", "10", "300"][len(sys.argv) - 4:]))
PC_ = os.environ.get("PRECOND", "") or False                 # PRECOND=kfac: K-FAC block-preconditioned CG
dev = "cuda:0"
FREEZE = tuple(f for f in os.environ.get("FREEZE", "").split(",") if f)
ck = torch.load(pop_path, map_location=dev, weights_only=False); dims = ck["dims"]
teacher = MLP(dims, act="tanh").to(dev); teacher.load_state_dict(ck["teacher_state"]); t64 = teacher.double()
net = MLP(dims, act="tanh").to(dev); net.load_state_dict(torch.load(base_path, map_location=dev, weights_only=False)["state_dict"]); net = net.double()
Xq = ts.sample_points(P_, dims[0], scales=(0.5, 1.0, 2.0), seed=1, device=dev).double()
t0 = time.time()
net64, info = ts.refine(lambda x: t64(x.double()), net, Xq, "tanh", dirs=D_, iters=IT_, cg_iters=CG_, seed=1, score=lambda n: param_errors(n, t64), verbose=True, tag=f"[tanhsolver polish frozen={FREEZE}]", precond=PC_, freeze=set(FREEZE))
print(f"DONE in {time.time()-t0:.0f}s", flush=True)
pe = param_errors(net64, t64); print("FINAL vs black box: max_eps", f"{pe['max_eps']:.2e}", "| max per matrix", [f"{x:.1e}" for x in pe["max_eps_per_matrix"]], "| mean per matrix", [f"{x:.1e}" for x in pe["mean_eps_per_matrix"]], flush=True)
torch.save({"dims": dims, "act": "tanh", "teacher_state": teacher.float().state_dict(), "state_dict": net64.state_dict()}, out_path)
