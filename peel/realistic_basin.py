"""Realistic basin: scale the ACTUAL consensus error distribution (shape preserved: max ~14x mean)
by factor s. Wg = W1t + s*(W_cons - W1t). Find where direction recovery breaks, and report BOTH
mean and max guess weight-error at the break -> which one governs under a realistic skew?"""
import sys, math, torch
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda"
CD = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
pop = torch.load(CD + "_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt", map_location=dev, weights_only=False)
merged = torch.load(CD + "mergedbest512_sigmoid__784x128x80x40x32x10__s0_final.pt", map_location=dev, weights_only=False)
dims = pop["dims"]; d = dims[0]; k = dims[1]
teacher = MLP(dims, act="sigmoid").to(dev).double(); teacher.load_state_dict(pop["teacher_state"]); teacher.eval()
W1t = teacher.layers[0].weight.detach(); b1t = teacher.layers[0].bias.detach(); tn = W1t / W1t.norm(dim=1, keepdim=True); nt = W1t.norm(dim=1)
# consensus, aligned to true order (Hungarian, sign) so s*(Wcons-W1t) is a real per-neuron error
from scipy.optimize import linear_sum_assignment
Wc0 = merged["state_dict"]["layers.0.weight"].to(dev).double(); bc0 = merged["state_dict"]["layers.0.bias"].to(dev).double()
Cp = torch.cdist(Wc0, W1t); Cm = torch.cdist(-Wc0, W1t); C = torch.minimum(Cp, Cm).cpu().numpy()
ri, ci = linear_sum_assignment(C); inv = torch.empty(k, dtype=torch.long); inv[torch.tensor(ci)] = torch.tensor(ri)
Wc = Wc0[inv].clone(); bc = bc0[inv].clone()
flip = ((Wc * W1t).sum(1) < 0); Wc[flip] *= -1; bc[flip] *= -1        # match sign to true
@torch.no_grad()
def J_at(x, fd=5e-5):
    E = torch.eye(d, device=dev, dtype=torch.float64)
    return ((teacher(x.unsqueeze(0) + fd * E) - teacher(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()
def recover(Wg, bg):
    Wgpinv = Wg.t() @ torch.linalg.inv(Wg @ Wg.t()); errs = torch.zeros(k)
    for j in range(k):
        t = torch.full((k,), 20.0, device=dev, dtype=torch.float64); t[j] = 0.0
        x0 = Wgpinv @ (t - bg); U, Sv, Vh = torch.linalg.svd(J_at(x0), full_matrices=False)
        nv = Vh[0]; nv = nv if float(nv @ Wg[j]) > 0 else -nv
        errs[j] = math.degrees(math.acos(min(1.0, abs(float(nv @ tn[j])))))
    return errs
print(f"{'scale':>6} {'mean_werr':>10} {'max_werr':>10} {'#>10%':>6} {'#>20%':>6} | {'recov_med':>10} {'recov_max':>10} {'#fail(>0.01)':>12}")
for s in [1, 2, 4, 6, 8, 12]:
    Wg = W1t + s * (Wc - W1t); bg = b1t + s * (bc - b1t)
    werr = (Wg - W1t).norm(dim=1) / nt
    r = recover(Wg, bg)
    nfail = int((r > 0.01).sum())
    print(f"{s:>6} {float(werr.mean()):>10.4f} {float(werr.max()):>10.4f} {int((werr>0.1).sum()):>6} {int((werr>0.2).sum()):>6} | {float(r.median()):>10.2e} {float(r.max()):>10.2e} {nfail:>12}")
