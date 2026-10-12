"""Chain step: replace layer-L row DIRECTIONS of a base net by the dual-derived rows (peel/dual_layer_deep.py
run with prefix = that base net), keep the base net's row norms, then run the jet solver with layer-1 weights
and every dual layer's weights frozen: free = those layers' biases + everything deeper (scale/bias "joint fit
with the downstream guess"). Usage:
python peel/assemble_dual_and_refine.py <pop.pt> <base_net.pt> <L:dual.pt>[,<L:dual.pt>...] <out.pt> [points dirs iters cg]"""
import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nets import MLP
import tanh_solve as ts
from method import param_errors
pop_path, base_path, duals, out_path = sys.argv[1:5]
P_, D_, IT_, CG_ = (int(a) for a in (sys.argv[5:9] + ["512", "64", "12", "300"][len(sys.argv) - 5:]))
dev = "cuda:0"
ck = torch.load(pop_path, map_location=dev, weights_only=False); dims = ck["dims"]
teacher = MLP(dims, act="tanh").to(dev); teacher.load_state_dict(ck["teacher_state"]); t64 = teacher.double()
b = torch.load(base_path, map_location=dev, weights_only=False)
net = MLP(dims, act="tanh").to(dev); net.load_state_dict(b["state_dict"]); net = net.double()
pe = param_errors(net, t64); print("base net vs black box: max_eps", f"{pe['max_eps']:.2e}", "| max per matrix", [f"{x:.1e}" for x in pe["max_eps_per_matrix"]], "| mean per matrix", [f"{x:.1e}" for x in pe["mean_eps_per_matrix"]], flush=True)
freeze = {"layers.0.weight"}
with torch.no_grad():
    for item in duals.split(","):
        L, path = int(item.split(":")[0]), item.split(":", 1)[1]
        d = torch.load(path, map_location=dev, weights_only=False)
        assert d["L"] == L and d["prefix"] == base_path, f"dual file {path} was computed with prefix {d['prefix']}, not {base_path}"
        West = d["West"].to(dev).double(); W = net.layers[L - 1].weight
        cos = (West * W).sum(1) / (West.norm(dim=1) * W.norm(dim=1))
        rows = West / West.norm(dim=1, keepdim=True) * W.norm(dim=1, keepdim=True) * torch.sign(cos)[:, None]
        print(f"layer {L}: replaced {len(rows)} row directions (|cos| with base rows: min {float(cos.abs().min()):.6f}); base row norms kept", flush=True)
        W.copy_(rows); freeze.add(f"layers.{L - 1}.weight")
pe = param_errors(net, t64); print("assembled net vs black box: max_eps", f"{pe['max_eps']:.2e}", "| max per matrix", [f"{x:.1e}" for x in pe["max_eps_per_matrix"]], "| mean per matrix", [f"{x:.1e}" for x in pe["mean_eps_per_matrix"]], flush=True)
Xq = ts.sample_points(P_, dims[0], scales=(0.5, 1.0, 2.0), seed=0, device=dev).double()
t0 = time.time()
net64, info = ts.refine(lambda x: t64(x.double()), net, Xq, "tanh", dirs=D_, iters=IT_, cg_iters=CG_, seed=0,
                        score=lambda n: param_errors(n, t64), verbose=True, tag=f"[tanhsolver frozen {sorted(freeze)}]", freeze=freeze)
print(f"DONE in {time.time()-t0:.0f}s", flush=True)
pe = param_errors(net64, t64); print("FINAL vs black box: max_eps", f"{pe['max_eps']:.2e}", "| max per matrix", [f"{x:.1e}" for x in pe["max_eps_per_matrix"]], "| mean per matrix", [f"{x:.1e}" for x in pe["mean_eps_per_matrix"]], flush=True)
torch.save({"dims": dims, "act": "tanh", "teacher_state": teacher.float().state_dict(), "state_dict": net64.state_dict(), "frozen": sorted(freeze)}, out_path)
