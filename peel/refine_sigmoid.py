"""Can we refine a layer-1 neuron of a SIGMOID blackbox given a good guess?

Net: f(x) = W2 . sigmoid(W1 x + b1) + b2   (one hidden layer, LINEAR output).
Teacher Jacobian J(x) = sum_i sigma'(z_i) W2[:,i] (x) w_i   (sum of rank-1 terms).
Given a good guess for ALL neurons, subtract the guessed contribution of every
neuron except k -> residual is ~rank-1 = sigma'(z_k) W2[:,k] (x) w_k, and the top
right singular vector is w_k's direction. Black box: only teacher(x) is queried.
"""
import sys, math
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
import torch
from nets import MLP

dev = "cuda" if torch.cuda.is_available() else "cpu"
pop = torch.load("recon/_pop__v18_lbfgs_sigmoid__3072x256x100__s0.pt", map_location=dev, weights_only=False)
dims = pop["dims"]; d = dims[0]
teacher = MLP(dims, act="sigmoid").to(dev).double(); teacher.load_state_dict(pop["teacher_state"])
fin = torch.load("recon/v18_lbfgs_sigmoid__3072x256x100__s0_final.pt", map_location=dev, weights_only=False)
guess = MLP(dims, act="sigmoid").to(dev).double(); guess.load_state_dict(fin["state_dict"])

W1t = teacher.layers[0].weight.detach(); b1t = teacher.layers[0].bias.detach()
W1g = guess.layers[0].weight.detach();   b1g = guess.layers[0].bias.detach()
W2g = guess.layers[1].weight.detach()                      # (100, 256)
tn = W1t / W1t.norm(dim=1, keepdim=True)                    # true unit normals

# base points for the probes (random; each is projected onto the guess plane)
X = torch.randn(200, d, device=dev, dtype=torch.float64)

def sig_prime(z):
    s = torch.sigmoid(z); return s * (1 - s)

@torch.no_grad()
def jacobian(x, fd=1e-3):
    """teacher Jacobian at x (100 x d) by central finite differences; counts queries."""
    E = torch.eye(d, device=dev, dtype=torch.float64)
    Jp = (teacher(x.unsqueeze(0) + fd * E) - teacher(x.unsqueeze(0) - fd * E)) / (2 * fd)  # (d,100)
    return Jp.t(), 2 * d                                    # (100,d), queries

def angle(a, b):
    a = a / a.norm(); b = b / b.norm()
    return math.degrees(math.acos(min(1.0, abs(float(a @ b)))))

N = 12
print(f"refining {N} layer-1 neurons of a {dims} SIGMOID teacher (black box)\n")
print("neuron | guess angle | refined angle | queries | match teacher idx")
gis, ris = [], []
for k in range(N):
    wkg = W1g[k]
    # base point: project a real input onto the guess hyperplane {wkg.x + bkg = 0}
    x0 = X[k % len(X)]
    x0 = x0 - (wkg @ x0 + b1g[k]) / (wkg @ wkg) * wkg
    J, q = jacobian(x0)
    # subtract guessed contribution of every OTHER neuron
    zg = W1g @ x0 + b1g                                     # (256,) guessed pre-acts
    sp = sig_prime(zg)                                      # (256,)
    contrib = (W2g * sp.unsqueeze(0)) @ W1g                 # sum_i sp_i W2[:,i] w_i  = (100,d)
    contrib_k = sp[k] * torch.outer(W2g[:, k], wkg)         # neuron k's guessed term
    J_resid = J - (contrib - contrib_k)                     # leave only neuron k
    U, S, Vh = torch.linalg.svd(J_resid, full_matrices=False)
    w_ref = Vh[0]
    if float(w_ref @ wkg) < 0: w_ref = -w_ref
    # score vs nearest true teacher neuron
    j = int((tn @ (wkg / wkg.norm())).argmax())
    ag = angle(wkg, W1t[j]); ar = angle(w_ref, W1t[j])
    gis.append(ag); ris.append(ar)
    print(f"  n{k:2d}   |  {ag:8.4f}°  |   {ar:8.4f}°  |  {q}  | t{j}")

def med(v): v = sorted(v); return v[len(v)//2]
print(f"\nmedian guess angle:   {med(gis):.4f}°")
print(f"median refined angle: {med(ris):.4f}°   (vs teacher)")
print(f"improvement factor:   ~{med(gis)/max(med(ris),1e-9):.1f}x")
