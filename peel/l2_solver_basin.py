"""L2 solver basin test on the REAL sub-net G(s)=layers2-5 (128->80->40->32->10).
Seed W2 = true + alpha*perturbation (deeper frozen at true), refine W2 by matching the
germ jet dG/ds at probe points, and see if it converges back to true. Sweeps alpha."""
import sys, torch
from torch.func import jacrev, vmap, grad
torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
dev = "cuda" if torch.cuda.is_available() else "cpu"
CD = "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"
ts = torch.load(CD + "_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt", map_location=dev, weights_only=False)["teacher_state"]
W2t = ts["layers.1.weight"].double(); b2 = ts["layers.1.bias"].double()
W3 = ts["layers.2.weight"].double(); b3 = ts["layers.2.bias"].double()
W4 = ts["layers.3.weight"].double(); b4 = ts["layers.3.bias"].double()
W5 = ts["layers.4.weight"].double(); b5 = ts["layers.4.bias"].double()
def G(s, W2):
    q2 = torch.sigmoid(s @ W2.t() + b2); q3 = torch.sigmoid(q2 @ W3.t() + b3)
    q4 = torch.sigmoid(q3 @ W4.t() + b4); return q4 @ W5.t() + b5
# probes: uniform over the accessible cube (excites ~49/80 neurons)
g = torch.Generator(device=dev).manual_seed(0)
S = torch.rand(16, 128, generator=g, device=dev)                       # (Np,128)
def jetG(W2):                                                          # dG/ds at all probes: (Np,10,128)
    return vmap(lambda s: jacrev(lambda x: G(x.unsqueeze(0), W2)[0])(s))(S)
Jt = jetG(W2t).detach(); sc = Jt.abs().max()
nt = W2t.norm(dim=1)
def relerr(W2): return float(((W2 - W2t).norm(dim=1) / nt).mean())
def refine(W2g, steps=2500, lr=3e-3):
    W2 = W2g.clone(); m = torch.zeros_like(W2); v = torch.zeros_like(W2); b1, b2a = 0.9, 0.999
    lossfn = lambda W: (((jetG(W) - Jt) / sc) ** 2).mean()
    gfn = grad(lossfn)
    for t in range(steps):
        gr = gfn(W2)
        m = b1 * m + (1 - b1) * gr; v = b2a * v + (1 - b2a) * gr * gr
        mh = m / (1 - b1 ** (t + 1)); vh = v / (1 - b2a ** (t + 1))
        W2 = W2 - lr * mh / (vh.sqrt() + 1e-8)
    return W2, float(lossfn(W2))
print(f"L2 solver basin on real sub-net (jet-match dG/ds, {S.shape[0]} probes, deeper=true):")
print(f"{'alpha':>6} {'W2 err before':>14} {'W2 err after':>13} {'jet loss after':>15} {'verdict':>10}")
for alpha in [0.0, 0.05, 0.10, 0.20, 0.44]:
    gg = torch.Generator(device=dev).manual_seed(1)
    P = torch.randn(80, 128, generator=gg, device=dev); P = P / P.norm(dim=1, keepdim=True)
    W2g = W2t + alpha * nt[:, None] * P                                # rel per-row error = alpha
    W2f, lf = refine(W2g)
    e0, e1 = relerr(W2g), relerr(W2f)
    verdict = "CONVERGED" if e1 < 0.02 else ("improved" if e1 < 0.7 * e0 else "STUCK")
    print(f"{alpha:>6.2f} {e0:>14.3f} {e1:>13.3f} {lf:>15.2e} {verdict:>10}", flush=True)
