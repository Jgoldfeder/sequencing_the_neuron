import torch, numpy as np
exec(open('lm_test.py').read().split('print("LBFGS')[0])   # reuse all fns
def lm_basin(dims,seed,alphas,ndir=15):
    tru=make(dims,seed); b1=tru[0]; k=dims[0]
    gp=torch.Generator().manual_seed(seed+9); tps=(2*torch.rand(12,k,generator=gp)-1)*1.2
    with torch.no_grad(): Fm=raw_jet(tru,tps,k); sc=Fm.abs().max()
    rng=np.random.default_rng(seed); bg=b1+0.05*torch.tensor(2*rng.random(k)-1); dtrue=tru[1:]
    print(f" {dims} seed{seed}:  alpha  P(<1e-6)  P(<1e-4)  median-biaserr",flush=True)
    for al in alphas:
        errs=[]
        for dd in range(ndir):
            g2=torch.Generator().manual_seed(seed*400+int(al*1e6)+dd+1)
            down0=[W+al*(W.norm()/(X:=torch.randn(*W.shape,generator=g2)).norm())*X for W in dtrue]
            th0=flat([bg]+down0); thM=run_lm(th0,dims,tps,Fm,sc,iters=60)
            errs.append(float((thM[:k]-b1).abs().max()))
        e=np.array(errs)
        print(f"   a={al:.0e}: {(e<1e-6).mean():.2f}     {(e<1e-4).mean():.2f}     {np.median(e):.1e}",flush=True)
print("LM-BASIN: P(bias err) vs relative downstream perturbation (JOINT LM refine, b0=truth+0.05)",flush=True)
al=[1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1]
for sd in range(3): lm_basin([3,4,3,2],sd,al)
lm_basin([6,8,4,4],0,al); lm_basin([6,8,4,4],1,al)
