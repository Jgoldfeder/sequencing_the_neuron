import sys, torch
from data import make_teacher, load_data
from method import Cfg, reconstruct
from experiment_runner import layered_param_errors
from run import VARIANTS
variant=sys.argv[1]; dims=[int(x) for x in sys.argv[2].split(',')]
q=int(sys.argv[3]); outer=int(sys.argv[4]); device='cuda'
teacher=make_teacher(dims,epochs=25,seed=0,device=device,verbose=False)
(_,_),(xte,_)=load_data(dims,device); eval_pts=xte[:2000]
cfg=Cfg(p=8,q=q,outer=outer,**VARIANTS[variant])
print(f"[{variant}] {sys.argv[2]} | fit_loss={cfg.fit_loss} qg={cfg.qg_steps}@{cfg.qg_lr} dist={cfg.qg_dist} init={cfg.qg_init} disagree={cfg.disagree} | q={q} outer={outer}",flush=True)
best,log,final=reconstruct(teacher,dims,cfg,device,eval_pts,seed=0,save_recon=None)
eps=layered_param_errors(best,teacher)
nm=['layer0(FIRST)','layer1(second)','layer2(out)']
print("\n=== per-layer error ===")
for L,n in zip(eps['per_layer'],nm):
    print(f"  {n}: w_max={L['weight']['max_eps']:.3e} w_mean={L['weight']['mean_eps']:.3e}")
print(f"  OVERALL max_eps={eps['max_eps']:.3e} mean_eps={eps['mean_eps']:.3e} MAE(D)={final['final_mae']:.3e}")
