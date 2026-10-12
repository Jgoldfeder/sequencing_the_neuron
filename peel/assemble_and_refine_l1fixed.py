"""Stage 3: assemble the net (committee guess with layer-1 rows replaced by isolation direction x
tail-decay scale) and run the jet solver with layer-1 WEIGHTS frozen: free = layer-1 biases + all
deeper layers. Usage: python peel/assemble_and_refine_l1fixed.py <teacher.pt> <guess.pt> <iso_merged.pt> <scales.pt> <out.pt> [points dirs iters cg]"""
import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nets import MLP
import tanh_solve as ts
from method import param_errors
teacher_path, guess_path, iso_path, scale_path, out_path = sys.argv[1:6]
P_, D_, IT_, CG_ = (int(a) for a in (sys.argv[6:10] + ["512", "64", "25", "300"][len(sys.argv) - 6:]))
dev = "cuda:0"
ck = torch.load(teacher_path, map_location=dev, weights_only=False); dims = ck["dims"]
teacher = MLP(dims, act="tanh").to(dev); teacher.load_state_dict(ck["teacher_state"]); t64 = teacher.double()
g = torch.load(guess_path, map_location=dev, weights_only=False)
net = MLP(dims, act="tanh").to(dev); net.load_state_dict(g["state_dict"]); net = net.double()
iso = torch.load(iso_path, map_location="cpu", weights_only=False)["results"]
sc = torch.load(scale_path, map_location="cpu", weights_only=False)["results"]
with torch.no_grad():
    W = net.layers[0].weight; n_set = 0
    for i in range(dims[1]):
        if i in iso and i in sc and sc[i].get("scale"):
            u = iso[i]["direction"].to(dev).double(); u = u / u.norm()
            if float(u @ W[i]) < 0: u = -u                       # keep the guess's sign convention
            net.layers[0].bias[i] *= sc[i]["scale"] / float(W[i].norm())   # the member fit b to its own row scale: keep the hyperplane -b/|w|
            W[i] = u * sc[i]["scale"]; n_set += 1
print(f"layer-1 rows replaced: {n_set}/{dims[1]} (direction from isolation x scale from tail decay); biases rescaled with the rows; deeper layers from the committee guess", flush=True)
pe = param_errors(net, t64); print("assembled net vs black box: max_eps", f"{pe['max_eps']:.2e}", "| max per matrix", [f"{x:.1e}" for x in pe["max_eps_per_matrix"]], "| mean per matrix", [f"{x:.1e}" for x in pe["mean_eps_per_matrix"]], flush=True)
Xq = ts.sample_points(P_, dims[0], scales=(0.5, 1.0, 2.0), seed=0, device=dev).double()
t0 = time.time()
net64, info = ts.refine(lambda x: t64(x.double()), net, Xq, "tanh", dirs=D_, iters=IT_, cg_iters=CG_, seed=0,
                        score=lambda n: param_errors(n, t64), verbose=True, tag="[tanhsolver L1-fixed]",
                        freeze={"layers.0.weight"})
print(f"DONE in {time.time()-t0:.0f}s", flush=True)
pe = param_errors(net64, t64); print("FINAL vs black box: max_eps", f"{pe['max_eps']:.2e}", "| max per matrix", [f"{x:.1e}" for x in pe["max_eps_per_matrix"]], "| mean per matrix", [f"{x:.1e}" for x in pe["mean_eps_per_matrix"]], flush=True)
torch.save({"dims": dims, "act": "tanh", "teacher_state": teacher.float().state_dict(), "state_dict": net64.state_dict()}, out_path)
