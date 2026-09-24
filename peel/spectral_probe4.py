import torch, numpy as np
torch.set_default_dtype(torch.float64); sig=torch.sigmoid
exec(open('realgate2.py').read().split('print("JOINT')[0])
def preacts(s,down):
    W2,b2,W3,b3,w4,b4=down; z2=s@W2.T+b2; q=sig(z2); z3=q@W3.T+b3; return z2,z3
dims=[6,8,4,4]; k=6; Sstar,b1,down=make(dims,0); W2=down[0]; W,_=W2.shape
gp=torch.Generator().manual_seed(9); te=(2*torch.rand(600,k,generator=gp)-1)*1.2; se=sig(te@Sstar.T+b1)
z2t,z3t=preacts(se,down); n2=z2t.pow(2).mean().sqrt(); n3=z3t.pow(2).mean().sqrt()
print("W2 row-direction cosine error -> functional eps_z (all else exact). Basin spec: eps_z<0.10")
print(f"{'cos':>7} {'ang(deg)':>8} {'eps_z2':>8} {'eps_z3':>8}")
for ct in [0.90,0.95,0.98,0.99,0.995,0.999,0.9999]:
    e2s=[]; e3s=[]
    for trial in range(8):
        g=torch.Generator().manual_seed(100+trial)
        W2n=W2.clone()
        for r in range(W):
            u=W2[r]/W2[r].norm(); v=torch.randn(k,generator=g); v=v-(v@u)*u; v=v/v.norm()
            newdir=ct*u+np.sqrt(1-ct*ct)*v; W2n[r]=newdir*W2[r].norm()
        dn=[W2n]+[d.clone() for d in down[1:]]
        z2,z3=preacts(se,dn)
        e2s.append(float((z2-z2t).pow(2).mean().sqrt()/n2)); e3s.append(float((z3-z3t).pow(2).mean().sqrt()/n3))
    print(f"{ct:>7.4f} {np.degrees(np.arccos(ct)):>8.2f} {np.mean(e2s):>8.3f} {np.mean(e3s):>8.3f}")
