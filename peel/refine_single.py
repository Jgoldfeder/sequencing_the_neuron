"""Refine ONE layer-1 neuron of a DEEP sigmoid net using ONLY that neuron's guess
(no knowledge of the other neurons). Idea: neuron k's inflection sits exactly on
its hyperplane {w_k.x+b_k=0}; near there, every OTHER neuron is smooth background.
So along the guess normal, neuron k's contribution to the directional derivative is
a localized BUMP (the sigma' peak) on top of a smooth trend -> detrend, find the
bump center -> a point on neuron k's TRUE hyperplane. Collect d such points, fit
the hyperplane -> refined w_k.  Uses only w_k's guess; teacher is black-box.
"""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP

dev = "cuda"; torch.manual_seed(0)
dims = [128, 24, 16, 8]; d = dims[0]
teacher = MLP(dims, act="sigmoid").to(dev).double()
with torch.no_grad():
    for l in teacher.layers[:-1]:
        l.bias.uniform_(-0.5, 0.5)
W1t = teacher.layers[0].weight.detach(); b1t = teacher.layers[0].bias.detach()
tn = W1t / W1t.norm(dim=1, keepdim=True)

@torch.no_grad()
def slopes(x0, wref, ts, h=1e-3):
    """directional derivative of f along wref at x0+t*wref, for all t (central)."""
    P = torch.cat([x0.unsqueeze(0) + (ts.unsqueeze(1) + h) * wref.unsqueeze(0),
                   x0.unsqueeze(0) + (ts.unsqueeze(1) - h) * wref.unsqueeze(0)], 0)
    Y = teacher(P)
    n = len(ts)
    return (Y[:n] - Y[n:]) / (2 * h)                       # (T, O)

@torch.no_grad()
def bump_point(x0, wref, wnorm_guess):
    """find neuron k's inflection along wref -> a point on its true hyperplane."""
    span = 6.0 / wnorm_guess
    ts = torch.linspace(-span, span, 61, device=dev, dtype=torch.float64)
    s = slopes(x0, wref, ts)                               # (T,O)
    # detrend each output dim (remove smooth linear background), residual = the bump
    A = torch.stack([torch.ones_like(ts), ts], 1)         # (T,2)
    coef = torch.linalg.lstsq(A, s).solution               # (2,O)
    resid = s - A @ coef
    mag = resid.norm(dim=1)                                # (T,)
    i = int(mag.argmax())
    # parabolic refine of the peak
    if 0 < i < len(ts) - 1:
        y0, y1, y2 = mag[i-1], mag[i], mag[i+1]
        denom = (y0 - 2*y1 + y2)
        dt = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-30 else 0.0
        dt = max(-1.0, min(1.0, float(dt)))
    else:
        dt = 0.0
    tstar = float(ts[i]) + dt * (ts[1] - ts[0]).item()
    return x0 + tstar * wref

def maxeps(u, k):
    s = 1.0 if float(u @ tn[k]) > 0 else -1.0
    return float((s * u / u.norm() - tn[k]).abs().max())
def angle(u, k):
    return math.degrees(math.acos(min(1.0, abs(float((u/u.norm()) @ tn[k])))))

torch.manual_seed(1); g = torch.Generator(device=dev).manual_seed(1)
N, M = 8, 400                                              # neurons tested, points per neuron (>d)
be_a, af_a, be_e, af_e = [], [], [], []
for k in range(N):
    wk = W1t[k]; u = wk / wk.norm()
    v = torch.randn(d, generator=g, device=dev, dtype=torch.float64); v = v-(v@u)*u; v=v/v.norm()
    th = math.radians(5.0)
    w_guess = (math.cos(th)*u + math.sin(th)*v)            # unit guess, 5deg off
    b_guess = float(b1t[k])                                # (assume bias guess ~ok)
    wnorm_guess = float(wk.norm())
    pts = []
    for m in range(M):
        x0 = torch.randn(d, generator=g, device=dev, dtype=torch.float64)
        x0 = x0 - (w_guess @ x0 + b_guess/wnorm_guess) * w_guess   # onto guess plane (unit normal)
        # skip if another neuron transitions nearby (detectable; keeps isolation)
        zn = ((W1t@x0 + b1t)/W1t.norm(dim=1)).abs(); zn[k] = 9
        if float(zn.min()) < 0.15: continue
        pts.append(bump_point(x0, w_guess, wnorm_guess))
    P = torch.stack(pts)
    # fit hyperplane w.p + b = 0 : right-null vector of [P, 1]
    Aug = torch.cat([P, torch.ones(len(P), 1, device=dev, dtype=torch.float64)], 1)
    _, _, Vh = torch.linalg.svd(Aug - Aug.mean(0, keepdim=True), full_matrices=False)
    w_ref = Vh[-1, :d]
    if float(w_ref @ w_guess) < 0: w_ref = -w_ref
    be_a.append(angle(w_guess, k)); af_a.append(angle(w_ref, k))
    be_e.append(maxeps(w_guess, k)); af_e.append(maxeps(w_ref, k))
    print(f"  neuron {k}: guess {angle(w_guess,k):.3f}deg (maxeps {maxeps(w_guess,k):.2e})"
          f"  ->  refined {angle(w_ref,k):.4f}deg (maxeps {maxeps(w_ref,k):.2e})  [{len(P)} pts]")

print(f"\nSINGLE-NEURON refine (only neuron k's guess used), deep net {dims}:")
print(f"  BEFORE  angle med {sorted(be_a)[N//2]:.3f}deg   max_eps med {sorted(be_e)[N//2]:.2e}")
print(f"  AFTER   angle med {sorted(af_a)[N//2]:.4f}deg   max_eps med {sorted(af_e)[N//2]:.2e}")
