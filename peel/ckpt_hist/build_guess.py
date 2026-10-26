"""TEACHER-FREE construction of a teacher-shaped (MLP width = cfg['ffn']) MicroT guess from a committee of
wide students.  Allowed inputs: the students' weights, the known architecture (cfg), and a black-box ORACLE
X -> logits (gaussian inputs_embeds in, teacher logits out -- the same primitive the committee uses).
The teacher's weights / activations are never touched: this file never loads or receives the teacher model.

  1. candidates = the K lowest-fit members (fit = black-box L1 on the run's own queries)
  2. per block, on the member's own residual stream: cluster its MLP neurons by pre-activation |corr| > thr;
     support of a cluster = #members having a neuron with |corr| > thr to it (each on its own stream);
     keep the best-supported `ffn` clusters (ties: output contribution); missing slots = zero neurons
  3. merge each cluster to ONE neuron: rep direction, scale a and output column c fitted so that
     c*gelu(a*z_rep) best reproduces the cluster's summed output; then refit all output columns jointly
     by least squares to the wide MLP's output
  4. fine-tune the narrow model on fresh black-box queries (L1 on logits), select by held-out black-box fit
usage: python build_guess.py <wide_ckpt> <out.pt>   (the ckpt's teacher_sd is used ONLY to build the oracle)"""
import math
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel")
from microt import MicroT, build, rope_cos_sin  # noqa: E402

dev = "cuda"


def zcorr(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return F.normalize(A, dim=0).t() @ F.normalize(B, dim=0)


@torch.no_grad()
def mlp_input(m, X, b):
    """normalized input of block b's MLP (gains NOT folded: norm2 applied as the model does)"""
    x = X; cos, sin = rope_cos_sin(x.shape[1], m.hdim, m.cfg["theta"], x.device)
    for i, blk in enumerate(m.blocks):
        x = x + blk.attn(blk.norm1(x), cos, sin)
        if i == b:
            return blk.norm2(x).reshape(-1, x.shape[-1])
        x = x + blk.mlp(blk.norm2(x))


@torch.no_grad()
def compress(members, mi, X, width, quorum, thr=0.99):
    wide = members[mi]; cfg = dict(wide.cfg, ffn=width); nar = MicroT(cfg).to(dev)
    nar.load_state_dict({k: v for k, v in wide.state_dict().items() if ".mlp." not in k}, strict=False)
    log = []
    for b in range(len(wide.blocks)):
        W1, W2 = wide.blocks[b].mlp.fc1.weight, wide.blocks[b].mlp.fc2.weight
        h = mlp_input(wide, X, b); Z = h @ W1.t(); G = F.gelu(Z); target = G @ W2.t()
        Zo = [members[j].blocks[b].mlp.fc1(mlp_input(members[j], X, b)) for j in range(len(members))]
        contrib = G.std(0) * W2.norm(dim=0)
        C = zcorr(Z, Z); free = torch.ones(Z.shape[1], dtype=torch.bool, device=dev); clusters = []
        for u in torch.argsort(contrib, descending=True).tolist():
            if not free[u]:
                continue
            mem = (C[u].abs() > thr) & free; free &= ~mem
            sup = sum(int(zcorr(Z[:, u:u + 1], Zj).abs().amax() > thr) for Zj in Zo)
            clusters.append(dict(rep=u, members=mem.nonzero().flatten(), sup=sup, contrib=contrib[mem].sum().item()))
        clusters.sort(key=lambda c: (-c["sup"], -c["contrib"]))
        keep = [c for c in clusters if c["sup"] >= quorum][:width]
        # merge copies -> one neuron: scale a fitted with a free linear term (copies may be sign-flipped:
        # gelu(-z) = gelu(z) - z, so a flipped copy is the same neuron plus a linear term)
        rows = []
        for c in keep:
            u = c["rep"]; cl = c["members"]; out_c = G[:, cl] @ W2[:, cl].t(); best = None
            for a in torch.linspace(0.5, 1.5, 41).tolist():
                Fa = torch.stack([F.gelu(a * Z[:, u]), Z[:, u]], 1).double()
                res = out_c.double() - Fa @ torch.linalg.lstsq(Fa, out_c.double()).solution
                if best is None or res.pow(2).sum().item() < best[0]:
                    best = (res.pow(2).sum().item(), a)
            rows.append(best[1] * W1[u])
        W1k = torch.stack(rows)
        # joint fit of the wide MLP output with the kept neurons + a free linear map L of the MLP input h
        Zk = h @ W1k.t(); Fk = torch.cat([F.gelu(Zk), h], 1).double()
        sol = torch.linalg.lstsq(Fk, target.double()).solution; Ck = sol[:len(rows)].t(); L = sol[len(rows):].t()  # (d,k), (d,d)
        # orientation: the teacher's MLP has NO linear path, so choose which neurons to flip (gelu(z) -> gelu(-z),
        # which adds +c_k w_k^T to L) such that the total linear map vanishes. relaxed LSQ for beta, rounded to {0,1}
        Rk = torch.stack([torch.outer(Ck[:, k], W1k[k].double()).flatten() for k in range(len(rows))], 1)
        beta = torch.linalg.lstsq(Rk, -L.flatten()[:, None]).solution.flatten()
        flip = beta > 0.5
        Lres = L + (Rk[:, flip].sum(1) if flip.any() else 0).reshape(L.shape) if flip.any() else L
        W1k = torch.where(flip[:, None], -W1k, W1k)
        Gk = F.gelu(h @ W1k.t())
        W2k = torch.linalg.lstsq(Gk.double(), target.double()).solution.t().float()   # refit, NO linear term
        n_pad = width - len(rows)
        W1n = torch.cat([W1k, torch.zeros(n_pad, W1.shape[1], device=dev)])
        W2n = torch.cat([W2k, torch.zeros(W2.shape[0], n_pad, device=dev)], 1)
        nb = nar.blocks[b]; nb.mlp.fc1.weight.copy_(W1n); nb.mlp.fc2.weight.copy_(W2n)
        nb.norm2.weight.copy_(wide.blocks[b].norm2.weight)
        rel = ((Gk @ W2k.t() - target).pow(2).mean() / target.pow(2).mean()).sqrt().item()
        log.append(f"B{b + 1}: {len(clusters)} clusters, {sum(c['sup'] >= quorum for c in clusters)} supported "
                   f">={quorum}/{len(members)}, kept {len(keep)} (+{n_pad} zero) | flipped {int(flip.sum())}, "
                   f"beta rounding gap {(beta - flip.double()).abs().max():.3f}, linear term |L| {L.norm():.3f} -> {Lres.norm():.4f} "
                   f"| MLP-output refit rel err {rel:.4f}")
    return nar, log


def finetune(m, oracle, cfg, n=20000, epochs=150, lr=3e-4, seed=0, T=32):
    n = max(1, n * 32 // T)                                   # same token budget at any length
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(n, T, cfg["d_model"], generator=g, device=dev) * 0.5
    with torch.no_grad():
        Y = torch.cat([oracle(X[i:i + max(1, 32000 // T)]) for i in range(0, n, max(1, 32000 // T))])
    m = m.clone().train(); opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    with torch.enable_grad():
        for _ in range(epochs):
            for i in torch.randperm(n, device=dev).split(max(1, 512 * 32 // T)):
                opt.zero_grad(); (m(inputs_embeds=X[i]) - Y[i]).abs().mean().backward(); opt.step()
            sch.step()
    return m.eval()


@torch.no_grad()
def bb_fit(m, oracle, Xh):
    return (m(inputs_embeds=Xh) - oracle(Xh)).abs().mean().item()


if __name__ == "__main__":
    ckpt, out = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 32           # query length for probes + fine-tune
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg, scfg = ck["cfg"], ck["student_cfg"]
    _bb = build(cfg, ck["teacher_sd"], dev)                      # black box: ONLY its logits are exposed
    oracle = lambda X: _bb(inputs_embeds=X)                      # noqa: E731
    del ck["teacher_sd"]
    members = [build(scfg, sd, dev) for sd in ck["student_sds"]]
    P = len(members); quorum = math.ceil(0.625 * P)
    order = sorted(range(P), key=lambda i: ck["fit"][i])
    g = torch.Generator(device=dev).manual_seed(123)
    Xc = torch.randn(max(1, 8192 // T), T, cfg["d_model"], generator=g, device=dev) * 0.5      # clustering probe
    Xh = torch.randn(max(1, 32768 // T), T, cfg["d_model"], generator=g, device=dev) * 0.5     # held-out black-box probe
    print(f"{ckpt.split('/')[-1]}: iter {ck['iter']}, members by fit {order}, quorum {quorum}/{P}", flush=True)
    results = []
    for mi in order[:3]:
        nar, log = compress(members, mi, Xc, cfg["ffn"], quorum)
        f0 = bb_fit(members[mi], oracle, Xh); f1 = bb_fit(nar, oracle, Xh)
        ft = finetune(nar, oracle, cfg, T=T); f2 = bb_fit(ft, oracle, Xh)
        print(f" m{mi}: wide black-box fit {f0:.4f} -> compressed to {cfg['ffn']} {f1:.4f} -> fine-tuned {f2:.4f}\n    "
              + "\n    ".join(log), flush=True)
        results.append((f2, mi, ft))
    f2, mi, guess = min(results, key=lambda r: r[0])
    torch.save({"cfg": cfg, "sd": {k: v.cpu() for k, v in guess.state_dict().items()}, "source": ckpt,
                "member": mi, "bb_fit": f2}, out)
    print(f"GUESS = compressed+fine-tuned m{mi} (held-out black-box fit {f2:.4f}) -> {out}", flush=True)
