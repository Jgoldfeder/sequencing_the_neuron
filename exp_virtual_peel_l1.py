"""Judah's experiment: VIRTUALLY PEEL L1 and query L2 directly.
Instead of x-queries -> L1 -> h1 -> L2 (L2 sees L1's OUTPUT distribution/scale), we
generate the adversarial disagreement queries IN L2's OWN INPUT SPACE (h), and only
pass them through the exact L1-inverse to reach the teacher:
    SealedTeacher.forward(h) = full_teacher(x_of_h(h)),   L1(x_of_h(h)) == h
so L2's input distribution/scale is whatever the query-gen makes (same machinery L1
had), not h1=L1(x). NO dataset. eps is scored against the TRUE [L2,L3,L4].
Everything fp64 for the L1 inversion (leaky_inv divides negatives by 0.01)."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
from method import Cfg, reconstruct
from data import make_teacher
torch.set_default_dtype(torch.float64)

dev = "cuda"; NS = 0.01
dims = [3072, 200, 200, 200, 100]
teacher = make_teacher(dims, epochs=25, seed=0, device=dev, act="leaky_relu").double()
W1 = teacher.layers[0].weight.detach(); b1 = teacher.layers[0].bias.detach()
W1pinv = W1.t() @ torch.linalg.inv(W1 @ W1.t())            # exact right-inverse (3072,200)

class SealedTeacher(MLP):
    """.layers = the TRUE [L2,L3,L4] (for eps); forward = full_teacher(L1^{-1}(h))."""
    def __init__(s):
        super().__init__([200, 200, 200, 100], act="leaky_relu")
        with torch.no_grad():
            for j, li in enumerate((1, 2, 3)):
                s.layers[j].weight.copy_(teacher.layers[li].weight)
                s.layers[j].bias.copy_(teacher.layers[li].bias)
        s._full = [teacher]                                 # hide from .layers/state_dict
    def forward(s, h):
        li = torch.where(h >= 0, h, h / NS)                 # leaky_relu inverse
        x = (li - b1) @ W1pinv.t()                          # x s.t. L1(x) == h
        return s._full[0](x)
sub = SealedTeacher().to(dev).double()

# sanity: L1(x_of_h(h)) == h  (the seal is exact)
hp = torch.randn(8, 200, device=dev)
xp = (torch.where(hp >= 0, hp, hp / NS) - b1) @ W1pinv.t()
back = F.leaky_relu(xp @ W1.t() + b1, NS)
print(f"[seal check] max|L1(x_of_h(h)) - h| = {(back - hp).abs().max():.2e}", flush=True)

# eval points = natural h1 distribution (L1 of gaussian x) -- diagnostic only, NO dataset
gx = torch.randn(2000, 3072, device=dev)
eval_h = F.leaky_relu(gx @ W1.t() + b1, NS)

cfg = Cfg(p=1, q=48000, outer=60, act="leaky_relu",
          qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="uniform", qg_range=1.0,
          disagree="median_pair", fit_loss="l1", batch=512, window=60,
          warmstart_iters=5, gate_kappa=0.0, lbfgs_polish=False,
          cheat=True, freeze_reinit=True, peel_restart=True,
          cheat_peel_max=1e-1, cheat_peel_mean=3e-3)

print("=== VIRTUAL-PEEL-L1: solve L2 as the input layer via L1^{-1} ===", flush=True)
print("    ('[cheat-peel] L1: X/200' here = the big net's L2)", flush=True)
best, log, final = reconstruct(sub, [200, 200, 200, 100], cfg, dev, eval_h, seed=0)
print(f"[final] {final}", flush=True)
