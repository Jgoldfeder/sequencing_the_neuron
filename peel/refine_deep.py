"""Does layer-1 refinement work when MORE layers follow? Test on a 2-hidden-layer
sigmoid net: f(x) = W3 sig(W2 sig(W1 x + b1) + b2) + b3.

Layer 1 still sees the raw linear input, so its Jacobian piece is still rank-1
with input factor w_k:   J(x) = Dg(x) diag(sig'(z1)) W1,  where Dg = df/da1 now
goes through the EXTRA nonlinear layer. Subtract the guessed all-but-k pieces
(using the guess's own analytic Jacobian) and SVD the residual -> refined w_k.
Two conditions: clean downstream guess vs perturbed downstream guess.
"""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP

dev = "cuda"; torch.manual_seed(0)
dims = [512, 64, 32, 16]                      # input, hidden1, hidden2, output
d = dims[0]
teacher = MLP(dims, act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]:
        l.bias.uniform_(-0.5, 0.5)
Wt = [l.weight.detach().clone() for l in teacher.layers]
bt = [l.bias.detach().clone() for l in teacher.layers]
tn = Wt[0] / Wt[0].norm(dim=1, keepdim=True)

def sigp(z): s = torch.sigmoid(z); return s * (1 - s)

@torch.no_grad()
def teacher_J(x, fd=5e-4):
    E = torch.eye(d, device=dev, dtype=torch.float64)
    return ((teacher(x.unsqueeze(0) + fd * E) - teacher(x.unsqueeze(0) - fd * E)) / (2 * fd)).t()

@torch.no_grad()
def guess_pieces(Wg, bg, x):
    """analytic guess Jacobian Jg and each layer-1 neuron's rank-1 piece factors."""
    z1 = Wg[0] @ x + bg[0]; a1 = torch.sigmoid(z1); s1 = sigp(z1)
    z2 = Wg[1] @ a1 + bg[1];              s2 = sigp(z2)
    Dg = Wg[2] @ torch.diag(s2) @ Wg[1]          # df/da1  (out x hidden1)
    Jg = Dg @ torch.diag(s1) @ Wg[0]             # out x d
    return Jg, Dg, s1

def maxeps(u, k):
    s = 1.0 if float(u @ tn[k]) > 0 else -1.0
    return float((s * u / u.norm() - tn[k]).abs().max())

@torch.no_grad()
def run(perturb_others, N=16, M=20, l1_deg=5.0):
    """perturb ONLY the target neuron k (others + downstream EXACT), refine k.
    perturb_others: also perturb the other layer-1 neurons by l1_deg (realistic)."""
    g = torch.Generator(device=dev).manual_seed(1)
    th = math.radians(l1_deg)
    before, after = [], []
    for k in range(N):
        Wg = [w.clone() for w in Wt]; bg = [b.clone() for b in bt]   # start EXACT
        if perturb_others:
            for j in range(dims[1]):
                if j == k: continue
                wj = Wt[0][j]; u = wj/wj.norm()
                v = torch.randn(d, generator=g, device=dev, dtype=torch.float64); v = v-(v@u)*u; v=v/v.norm()
                Wg[0][j] = (math.cos(th)*u + math.sin(th)*v) * wj.norm()
        # perturb the TARGET neuron k
        wk = Wt[0][k]; u = wk/wk.norm()
        v = torch.randn(d, generator=g, device=dev, dtype=torch.float64); v = v-(v@u)*u; v=v/v.norm()
        Wg[0][k] = (math.cos(th)*u + math.sin(th)*v) * wk.norm()
        before.append(maxeps(Wg[0][k], k))
        wg_k = Wg[0][k]
        acc = torch.zeros(d, device=dev, dtype=torch.float64); used = 0; tries = 0
        while used < M and tries < 5*M:
            tries += 1
            x0 = torch.randn(d, generator=g, device=dev, dtype=torch.float64)
            x0 = x0 - (wg_k@x0 + bg[0][k])/(wg_k@wg_k)*wg_k
            zn = ((Wt[0]@x0 + bt[0]) / Wt[0].norm(dim=1)).abs(); zn[k] = 9
            if float(zn.min()) < 0.02: continue
            used += 1
            Jt = teacher_J(x0)
            Jg, Dg, s1 = guess_pieces(Wg, bg, x0)
            piece_k = s1[k] * torch.outer(Dg[:, k], Wg[0][k])
            U, S, Vh = torch.linalg.svd(Jt - (Jg - piece_k), full_matrices=False)
            w = Vh[0]; acc = acc + (w if float(w@wg_k) > 0 else -w)
        after.append(maxeps(acc, k))
    tag = "others also 5deg off (realistic)" if perturb_others else "others + downstream EXACT (best case)"
    print(f"[{tag}]  layer-1 refine, 2-hidden-layer sigmoid {dims}")
    print(f"   BEFORE max_eps = {max(before):.3e}  (median {sorted(before)[N//2]:.3e})")
    print(f"   AFTER  max_eps = {max(after):.3e}  (median {sorted(after)[N//2]:.3e})")
    print(f"   -> {max(before)/max(after):.1f}x smaller\n")

run(perturb_others=False)      # isolate the DEPTH effect: only neuron k is wrong
run(perturb_others=True)       # realistic: everyone is a good-but-imperfect guess
