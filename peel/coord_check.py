"""COORDINATE consistency check (reviewer). The compensation diagnostic feeds H=q directly into layer 2,
but the black box evaluates at x(q)=W1c^+(logit q - b1c) -> true hidden h*(x(q)). For ordinary/sampled/
optimized-adversarial probes report ||h*(x(q))-q||, ||h_committee(x(q))-q||, ||h*-h_committee||.
If optimized probes (logits ~+-11) keep ||h*-q||~1e-3, the h=q assumption holds and the adversarial
compensation result stands. Oracle diagnostic (uses teacher L1)."""
import sys, torch
torch.set_default_dtype(torch.float32); dev="cuda"
sys.path.insert(0,"/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code")
from nets import MLP
CD="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/"; PEEL="/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel/"
pop=torch.load(CD+"_pop__mergedbest512_sigmoid__784x128x80x40x32x10__s0.pt",map_location=dev,weights_only=False); dims=pop["dims"]
bb=MLP(dims,act="sigmoid").to(dev).float(); bb.load_state_dict(pop["teacher_state"]); bb.eval()
pk=torch.load(PEEL+"peel_committee.pt",map_location=dev,weights_only=False)
m0net=MLP(dims,act="sigmoid").to(dev).float(); m0net.load_state_dict({k:v.to(dev).float() for k,v in pk["pop_states"][0].items()}); m0net.eval()
mem0=pk["pop_states"][0]
W1c=mem0["layers.0.weight"].to(dev); b1c=mem0["layers.0.bias"].to(dev); W1cp=W1c.t()@torch.linalg.inv(W1c@W1c.t())
W1t=bb.layers[0].weight.detach(); b1t=bb.layers[0].bias.detach()
def xofz(Z): return (Z-b1c)@W1cp.t()
g=torch.Generator(device=dev).manual_seed(1); pool=torch.cat([torch.sigmoid(torch.randn(3000,128,generator=g,device=dev)*s).clamp(1e-4,1-1e-4) for s in [0.5,1,2,4,8]])
with torch.no_grad(): D0=(m0net(xofz(torch.log(pool/(1-pool))))-bb(xofz(torch.log(pool/(1-pool))))).norm(dim=1)
NP=60; H_adv=pool[torch.topk(D0,NP).indices]
gm=torch.Generator(device=dev).manual_seed(9); H_ord=torch.sigmoid(torch.randn(NP,128,generator=gm,device=dev)*1.8).clamp(1e-4,1-1e-4)
Z=torch.log(H_adv/(1-H_adv)).clone().requires_grad_(True); opt=torch.optim.Adam([Z],lr=0.2)
for it in range(150):
    opt.zero_grad(); x=xofz(Z.clamp(-11,11)); (-((m0net(x)-bb(x)).norm(dim=1)**2).sum()).backward(); opt.step()
with torch.no_grad(): H_opt=torch.sigmoid(Z.clamp(-11,11))
def chk(H,tag):
    Zq=torch.log(H/(1-H)); x=xofz(Zq)
    hc=torch.sigmoid(x@W1c.t()+b1c); ht=torch.sigmoid(x@W1t.t()+b1t)
    print(f"  {tag:16s} |logit q| max {float(Zq.abs().max()):5.1f} | ||hc-q|| {float((hc-H).abs().mean()):.2e}/{float((hc-H).abs().max()):.2e} | "
          f"||h*-q|| {float((ht-H).abs().mean()):.2e}/{float((ht-H).abs().max()):.2e} | ||h*-hc|| {float((ht-hc).abs().mean()):.2e}",flush=True)
print("coordinate consistency (mean/max abs over coords):")
chk(H_ord,"ordinary"); chk(H_adv,"sampled-adv"); chk(H_opt,"optimized-adv")
