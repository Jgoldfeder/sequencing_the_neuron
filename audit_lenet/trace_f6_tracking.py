from pathlib import Path
src=Path('audit_lenet/refine_discovered_frontier.py').read_text();exec(src.split('results=[]')[0])
original=K.locate_light
@torch.no_grad()
def traced(oracle,X0,U,R,tol=1e-12):
 grid=torch.tensor([-1.,-.75,-.5,.5,.75,1.],device=dev,dtype=torch.double);T=R[:,None]*grid;P=X0[:,None]+T[:,:,None]*U[:,None];Y=oracle(P.reshape(-1,784)).reshape(len(X0),6,-1);aL,sL,rL=K._linefit(T[:,:3],Y[:,:3]);aR,sR,rR=K._linefit(T[:,3:],Y[:,3:]);ds=sR-sL;dsn=ds.norm(dim=1);scale=Y.abs().amax((1,2));tolv=torch.minimum(tol*scale,1e-9*dsn*R);root=((aL-aR)*ds).sum(1)/dsn.square().clamp_min(1e-300)
 print('R',R.tolist(),'dsnR/scale',(dsn*R/scale).tolist(),'rL/tol',(rL/tolv).tolist(),'rR/tol',(rR/tolv).tolist(),'root/R',(root/R).tolist(),flush=True)
 return original(oracle,X0,U,R,tol)
K.locate_light=traced
idx,f,frac=selected[0];c=int(free[0]);m=n.clone();row=f['row']
with torch.no_grad():m.layers[fr].weight[c]=row[:-1];m.layers[fr].bias[c]=row[-1]
M=K.ConvNetUnits(m,fr);anchor={c:(f['xstar'][None],torch.tensor([c],device=dev))};K.track_layer(bb,M,fr,anchor,g,need=400,step=2,max_steps=1,par=16,verbose=True)
