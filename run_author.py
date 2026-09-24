import torch
from data import make_teacher, load_data
from method import Cfg, reconstruct
from experiment_runner import layered_param_errors
from run import VARIANTS

dims=[784,128,64,10]; device='cuda'
teacher = make_teacher(dims, epochs=25, seed=0, device=device, verbose=False)
(_,_),(xte,_) = load_data(dims, device); eval_pts = xte[:2000]
cfg = Cfg(p=8, q=8000, outer=60, **VARIANTS['v_author'])
print(f"[v_author] 784x128x64x10 | qg_steps={cfg.qg_steps} qg_lr={cfg.qg_lr} "
      f"dist={cfg.qg_dist} init={cfg.qg_init} disagree={cfg.disagree} "
      f"fit_loss={cfg.fit_loss} batch={cfg.batch} | q={cfg.q} outer={cfg.outer} "
      f"budget={cfg.q*cfg.outer:,}", flush=True)
best, log, final = reconstruct(teacher, dims, cfg, device, eval_pts, seed=0, save_recon=None)
eps = layered_param_errors(best, teacher)
names=['layer0  784->128  (FIRST hidden)','layer1  128->64   (second hidden)','layer2  64->10    (output)']
print("\n=== per-layer error (best member, aligned to teacher) ===")
for L,nm in zip(eps['per_layer'],names):
    print(f"  {nm}:  w_max={L['weight']['max_eps']:.3e} w_mean={L['weight']['mean_eps']:.3e} | b_max={L['bias']['max_eps']:.3e}")
print(f"  OVERALL max_eps={eps['max_eps']:.3e} mean_eps={eps['mean_eps']:.3e} | MAE(D)={final['final_mae']:.3e} agree={final['final_agree']:.4f}")
