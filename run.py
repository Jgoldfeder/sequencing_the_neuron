"""Experiment runner: build/loads a shared teacher, runs a reconstruction
variant, writes results JSON. Usage:

  python run.py --variant v0_baseline --seed 0 --arch 784,64,10
"""
import argparse
import json
import os
import time
# large per-iteration query blocks (q x 3072) fragment the caching allocator; expandable segments
# let freed blocks be reused for differently-sized allocations instead of OOMing with GBs "reserved".
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from data import make_teacher, make_teacher_cnn, load_data
from method import (Cfg, reconstruct, reconstruct_cnn, build_consensus,
                    solver_polish_, polish_consensus, l1_on, agreement)
from nets import count_params, MLP
from align import param_errors

RESULTS = os.path.join(os.path.dirname(__file__), "results")

VARIANTS = {
    # --- THE MERGED BEST RECIPE (Aug 2026 old-vs-new reconciliation): old
    #     code's uniform init + L1 fit loss + batch 128, new code's coarse
    #     query-opt + median-pair. Full window (window=0 = keep-all; don't
    #     pass --window smaller than outer). Depth-fatal knobs: gauss init,
    #     MSE, batch 512 (even at 4x epochs). Width-fatal: 100@0.01 query-opt,
    #     mean-pair. On wide-shallow runs keep --fast (its staged solve is the
    #     closer; the lbfgs_polish flag is a separate no-op); batch 512 is a
    #     width-only speed option. Warmstart 5 kept: essential on width under
    #     gauss, outcome-flipping speedup on width under uniform, no evidence
    #     of harm on deep (cert run killed mid-flight while on success pace).
    #     NB warmstart queries are hardcoded gauss (method.py) regardless of
    #     qg_init. ---
    "mergedbest": dict(qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="uniform",
                       qg_range=1.0, disagree="median_pair", fit_loss="l1",
                       batch=128, window=0, warmstart_iters=5, gate_kappa=0.0,
                       lbfgs_polish=False),
    # mergedbest with batch 512: proven fine+faster on wide-shallow; on deep it
    # failed isolation twice (vRi_batch512 @10ep, vRi_b512_ep40 @40ep) at Adam
    # lr 1e-3 -- rerun at will (untested escapes: scaled lr, batch 256).
    "mergedbest512": dict(qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="uniform",
                          qg_range=1.0, disagree="median_pair", fit_loss="l1",
                          batch=512, window=0, warmstart_iters=5, gate_kappa=0.0,
                          lbfgs_polish=False),
    # paper-faithful baseline (Algorithms 1+2 as described)
    "v0_baseline": dict(),
    # faithful reproduction of the paper author's get_adv query generator:
    # 100 steps @ lr 0.01, euclidean disagreement, uniform[-1,1] init -- plus the
    # baseline mean-pair aggregation + L1 fit loss + batch 128 (the rest of the
    # author's method). Everything else is the reimpl default.
    "v_author": dict(qg_steps=100, qg_lr=0.01, qg_dist="l2", qg_init="uniform",
                     qg_range=1.0, batch=128),
    # --- leave-one-out ablation of v_author: each reverts exactly ONE change
    #     back toward the reimpl's v18 to find which is load-bearing on depth ---
    "vA_coarse":  dict(qg_steps=30,  qg_lr=0.1,  qg_dist="l2", qg_init="uniform",
                       qg_range=1.0, batch=128),                 # coarse query-opt (30 @ 0.1)
    "vA_steps30": dict(qg_steps=30,  qg_lr=0.01, qg_dist="l2", qg_init="uniform",
                       qg_range=1.0, batch=128),                 # few steps only (lr kept low)
    "vA_l1dist":  dict(qg_steps=100, qg_lr=0.01, qg_dist="l1", qg_init="uniform",
                       qg_range=1.0, batch=128),                 # L1 disagreement distance
    "vA_gauss":   dict(qg_steps=100, qg_lr=0.01, qg_dist="l2", qg_init="gauss",
                       batch=128),                               # Gaussian query init
    "vA_median":  dict(qg_steps=100, qg_lr=0.01, qg_dist="l2", qg_init="uniform",
                       qg_range=1.0, batch=128, disagree="median_pair"),  # median-pair
    "vA_mse":     dict(qg_steps=100, qg_lr=0.01, qg_dist="l2", qg_init="uniform",
                       qg_range=1.0, batch=128, fit_loss="mse"),  # MSE fit loss
    "vA_window":  dict(qg_steps=100, qg_lr=0.01, qg_dist="l2", qg_init="uniform",
                       qg_range=1.0, batch=128, window=20),
    # --- dual test: L1 vs MSE loss in the v18-like FAILING query-gen regime
    #     (coarse 30@0.1 + L1 distance + gaussian init + median). Identical
    #     except fit_loss, so any gap isolates the loss's role in the failure. ---
    "vF_l1":  dict(qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="gauss",
                   disagree="median_pair", batch=128, fit_loss="l1"),
    "vF_mse": dict(qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="gauss",
                   disagree="median_pair", batch=128, fit_loss="mse"),
    # --- leave-one-out from the FAILING vF_l1: restore ONE query-gen change
    #     back to the author's value; which restoration rescues L2? ---
    "vR_steps": dict(qg_steps=100, qg_lr=0.01, qg_dist="l1", qg_init="gauss",
                     disagree="median_pair", batch=128),   # restore 100 steps @ 0.01
    "vR_dist":  dict(qg_steps=30, qg_lr=0.1, qg_dist="l2", qg_init="gauss",
                     disagree="median_pair", batch=128),   # restore euclidean
    "vR_init":  dict(qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="uniform",
                     qg_range=1.0, disagree="median_pair", batch=128),  # restore uniform
    "vR_agg":   dict(qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="gauss",
                     disagree="mean_pair", batch=128),     # restore mean-pair
    # --- SHALLOW ablation on current code: leave-one-out from v18_lbfgs (revert
    #     one audit change) to test whether each actually helps on a shallow net.
    #     All keep batch=512 (v18 default) so only the named knob varies. ---
    "v18_mean":     dict(disagree="mean_pair", window=20, warmstart_iters=5,
                         fit_loss="mse", gate_kappa=3.0, lbfgs_polish=True),  # median->mean
    "v18_l1loss":   dict(disagree="median_pair", window=20, warmstart_iters=5,
                         fit_loss="l1", gate_kappa=3.0, lbfgs_polish=True),   # mse->l1
    "v18_nolbfgs":  dict(disagree="median_pair", window=20, warmstart_iters=5,
                         fit_loss="mse", gate_kappa=3.0),                     # drop lbfgs
    "v18_authkern": dict(disagree="median_pair", window=20, warmstart_iters=5,
                         fit_loss="mse", gate_kappa=3.0, lbfgs_polish=True,
                         qg_steps=100, qg_lr=0.01, qg_dist="l2",
                         qg_init="uniform", qg_range=1.0),                    # +author query kernel       # + windowing
    # --- single-knob decomposition of v18_authkern's shallow failure: restore
    #     ONE author query-gen knob at a time onto v18_lbfgs. v18_inituni is the
    #     decisive cell: deep only succeeds with uniform init (vR_init), so if
    #     shallow also tolerates it, one config wins both regimes. ---
    "v18_inituni":  dict(disagree="median_pair", window=20, warmstart_iters=5,
                         fit_loss="mse", gate_kappa=3.0, lbfgs_polish=True,
                         qg_init="uniform", qg_range=1.0),   # gauss->uniform init only
    "v18_authsteps": dict(disagree="median_pair", window=20, warmstart_iters=5,
                          fit_loss="mse", gate_kappa=3.0, lbfgs_polish=True,
                          qg_steps=100, qg_lr=0.01),         # fine query-opt (100@0.01) only
    "v18_authdist": dict(disagree="median_pair", window=20, warmstart_iters=5,
                         fit_loss="mse", gate_kappa=3.0, lbfgs_polish=True,
                         qg_dist="l2"),                      # euclidean disagreement only
    # --- isolate the remaining v18 extras (warmstart / fit-gating / batch).
    #     No window key: pass --window in the config (full window everywhere).
    #     NB warmstart queries are ALWAYS gaussian (method.py) regardless of
    #     qg_init, so inituni+warmstart mixes 5 gauss iters into a uniform run. ---
    "v18_nowarm":   dict(disagree="median_pair", warmstart_iters=0,
                         fit_loss="mse", gate_kappa=3.0, lbfgs_polish=True),
    "v18_nogate":   dict(disagree="median_pair", warmstart_iters=5,
                         fit_loss="mse", gate_kappa=0.0, lbfgs_polish=True),
    "v18_batch128": dict(disagree="median_pair", warmstart_iters=5,
                         fit_loss="mse", gate_kappa=3.0, lbfgs_polish=True,
                         batch=128),
    "v18_inituni_nowarm": dict(disagree="median_pair", warmstart_iters=0,
                               fit_loss="mse", gate_kappa=3.0, lbfgs_polish=True,
                               qg_init="uniform", qg_range=1.0),
    # --- deep decomposition of the inituni_nowarm failure: vR_init (proven
    #     deep success) plus exactly ONE of the remaining extras. Whichever
    #     cell fails names the depth-killer among batch512 / MSE / gating. ---
    "vRi_batch512": dict(qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="uniform",
                         qg_range=1.0, disagree="median_pair", batch=512),
    "vRi_mse":      dict(qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="uniform",
                         qg_range=1.0, disagree="median_pair", batch=128,
                         fit_loss="mse"),
    "vRi_gate":     dict(qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="uniform",
                         qg_range=1.0, disagree="median_pair", batch=128,
                         gate_kappa=3.0),
    # batch=512 with 4x epochs = identical total inner steps to batch128 x ep10;
    # success here proves the batch-512 deep failure is step count, not batch size
    "vRi_b512_ep40": dict(qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="uniform",
                          qg_range=1.0, disagree="median_pair", batch=512,
                          epochs=40),
    "vRi_warm":     dict(qg_steps=30, qg_lr=0.1, qg_dist="l1", qg_init="uniform",
                         qg_range=1.0, disagree="median_pair", batch=128,
                         warmstart_iters=5),   # warmstart on the succeeding deep config
    # ablation: the underfitting trap (expect failure) - validates C1
    "v0a_underfit": dict(epochs=2),
    # disagreement objective variants (Grok#1, GPT#1)
    "v1a_minpair": dict(disagree="min_pair"),
    "v1b_medianpair": dict(disagree="median_pair"),
    "v1c_variance": dict(disagree="variance"),
    # query-space variants (GPT#2, Grok#4)
    "v2_box": dict(query_box=1.5),
    "v3_div": dict(query_div_w=0.1),
    # dataset management (GPT#3, Grok#3)
    "v4_window": dict(window=20),
    # committee maintenance (Grok#2, GPT#5)
    "v5_maint": dict(maint_every=5, restart_worst=2),
    # last-mile float64 polish
    "v6_polish": dict(polish_f64=True),
    # random warm-start then committee (Grok#7, GPT#6)
    "v8_warmstart": dict(warmstart_iters=5),
    # adaptive tight-fit: stop inner epochs once D is fit (Kimi H1)
    "v9_fitdelta": dict(epochs=15, fit_delta=1e-3),
    # early stopping on App F signal (Kimi H8)
    "v10_earlystop": dict(stop_loss=1e-6, stop_patience=2),
    # combined best mechanisms
    "v7_combo": dict(disagree="min_pair", query_box=1.5, query_div_w=0.1,
                     window=20, maint_every=5, restart_worst=2),
    # ---- batch 2 (post-synthesis; Opus/GPT mechanisms) ----
    # squared fitting loss: residual-proportional gradients (Opus A2)
    "v11_mse": dict(fit_loss="mse"),
    # closed-form ridge LS solve of last layer every 5 iters + final (Opus A12)
    "v12_lastlayer": dict(lastlayer_every=5),
    # aligned population averaging, kappa=3 (Opus A9)
    "v13_popavg": dict(popavg_kappa=3.0),
    # float64 LBFGS squared-loss endgame (Opus A2+A3-lite)
    "v14_lbfgs": dict(lbfgs_polish=True),
    # fit-gated committee for query generation, kappa=3 (Opus T7)
    "v15_gate": dict(gate_kappa=3.0),
    # combo of orthogonal batch-1 winners
    "v16_combo2": dict(disagree="median_pair", window=20, warmstart_iters=5),
    # retuned early stop (v10's threshold never fired: final loss ~1e-4)
    "v10b_earlystop": dict(stop_loss=5e-4, stop_patience=2),
    # full stack: batch-1 winners + batch-2 mechanisms
    "v17_full": dict(disagree="median_pair", window=20, warmstart_iters=5,
                     fit_loss="mse", lastlayer_every=5, popavg_kappa=3.0,
                     lbfgs_polish=True, gate_kappa=3.0),
    # v17 attribution ablations (which ingredients are load-bearing?)
    "v17a_nolf": dict(disagree="median_pair", window=20, warmstart_iters=5,
                      fit_loss="mse", lastlayer_every=5, popavg_kappa=3.0,
                      gate_kappa=3.0),                      # no LBFGS
    "v17b_nopop": dict(disagree="median_pair", window=20, warmstart_iters=5,
                       fit_loss="mse", lastlayer_every=5,
                       lbfgs_polish=True, gate_kappa=3.0),  # no pop-avg
    "v17c_nomse": dict(disagree="median_pair", window=20, warmstart_iters=5,
                       lastlayer_every=5, popavg_kappa=3.0,
                       lbfgs_polish=True, gate_kappa=3.0),  # L1 fit loss
    "v17d_noll": dict(disagree="median_pair", window=20, warmstart_iters=5,
                      fit_loss="mse", popavg_kappa=3.0,
                      lbfgs_polish=True, gate_kappa=3.0),   # no last-layer
    # minimal stack: is popavg/lbfgs dead weight in v17d?
    "v18_min": dict(disagree="median_pair", window=20, warmstart_iters=5,
                    fit_loss="mse", gate_kappa=3.0),
    # v18_min + closed-form last-layer LS solve (test on wider nets: does the
    # solver help where the last layer dominates error, unlike 784x32x10?)
    "v18_ll": dict(disagree="median_pair", window=20, warmstart_iters=5,
                   fit_loss="mse", gate_kappa=3.0, lastlayer_every=5),
    # v18_min + float64 LBFGS squared-loss endgame
    "v18_lbfgs": dict(disagree="median_pair", window=20, warmstart_iters=5,
                      fit_loss="mse", gate_kappa=3.0, lbfgs_polish=True),
    # v18_min + both solvers
    "v18_ll_lbfgs": dict(disagree="median_pair", window=20, warmstart_iters=5,
                         fit_loss="mse", gate_kappa=3.0,
                         lastlayer_every=5, lbfgs_polish=True),
}


def run_cnn(args):
    """CNN extraction: population + disagreement + conv channel-aligned consensus."""
    device = args.device
    torch.manual_seed(args.teacher_seed)
    out_dim = args.out_dim
    if args.conv.strip().lower() == "lenet":       # LeNet-5 preset (MNIST 28x28)
        input_shape = (1, 28, 28)
        conv_cfgs = [(1, 6, 5, 1, 2, 2),           # C1+S2: ->6x28x28 ->pool 6x14x14
                     (6, 16, 5, 1, 0, 2),          # C3+S4: ->16x10x10 ->pool 16x5x5
                     (16, 120, 5, 1, 0, 0)]        # C5:    ->120x1x1
        fc_dims = (84,)                            # F6
        print("[lenet] using LeNet-5 preset (MNIST): conv "
              f"{conv_cfgs} fc {fc_dims}", flush=True)
    elif args.conv.strip().lower() == "alexnet":   # torchvision-exact geometry @224
        # EXACT torchvision AlexNet layout (same kernels/strides/pads, same
        # 55-27-13-13-13-6 feature maps, 9216->4096->4096->1000 head), with the
        # ONE pipeline-mandated substitution: 2x2 avg-pool for 3x3/2 maxpool
        # (output sizes match; maxpool's own kinks would break the linear-
        # pooling assumption behind the gauge + kink machinery). Teacher is
        # trained on CIFAR-100 bilinear-upsampled to 224 per batch, with the
        # 1000-way k-means split labels so every logit carries real signal.
        input_shape = (3, 224, 224)
        conv_cfgs = [(3, 64, 11, 4, 2, 2),         # ->64x55x55   ->pool 64x27x27
                     (64, 192, 5, 1, 2, 2),        # ->192x27x27  ->pool 192x13x13
                     (192, 384, 3, 1, 1, 0),       # ->384x13x13
                     (384, 256, 3, 1, 1, 0),       # ->256x13x13
                     (256, 256, 3, 1, 1, 2)]       # ->256x13x13  ->pool 256x6x6
        fc_dims = (tuple(int(x) for x in args.fc.split(","))
                   if args.fc.strip() else (4096, 4096))
        if out_dim == 10:
            out_dim = 1000
        print("[alexnet] torchvision-exact geometry @3x224x224 (avg-pool sub), "
              f"CIFAR-100 upsampled, 1000-way split labels: conv {conv_cfgs} "
              f"fc {fc_dims} out {out_dim}", flush=True)
    elif args.conv.strip().lower() == "alexnet32":  # economical CIFAR-scale variant
        input_shape = (3, 32, 32)
        conv_cfgs = [(3, 64, 5, 1, 2, 2),          # ->64x32x32  ->pool 64x16x16
                     (64, 192, 5, 1, 2, 2),        # ->192x16x16 ->pool 192x8x8
                     (192, 384, 3, 1, 1, 0),       # ->384x8x8
                     (384, 256, 3, 1, 1, 0),       # ->256x8x8
                     (256, 256, 3, 1, 1, 2)]       # ->256x8x8   ->pool 256x4x4
        fc_dims = (tuple(int(x) for x in args.fc.split(","))
                   if args.fc.strip() else (4096, 4096))
        if out_dim == 10:
            out_dim = 1000                         # 1000-way k-means split labels
        print("[alexnet32] AlexNet-style CIFAR-scale preset (avg-pool): conv "
              f"{conv_cfgs} fc {fc_dims} out {out_dim}", flush=True)
    else:
        input_shape = tuple(int(x) for x in args.input_shape.split(","))
        conv_cfgs = [tuple(int(v) for v in spec.split(","))
                     for spec in args.conv.split()]
        for c in conv_cfgs:
            if len(c) not in (4, 5, 6):
                raise SystemExit("each --conv spec must be in,out,k,s[,pad[,pool]]")
        fc_dims = tuple(int(x) for x in args.fc.split(",")) if args.fc.strip() else ()
    act = args.cnn_act
    print(f"[setup] CNN teacher input{input_shape} conv{conv_cfgs} fc{fc_dims} "
          f"out={out_dim} act={act} epochs={args.teacher_epochs} "
          f"device={device}", flush=True)
    teacher = make_teacher_cnn(input_shape, conv_cfgs, fc_dims, out_dim,
                               epochs=args.teacher_epochs, seed=args.teacher_seed,
                               device=device, verbose=True, act=act)

    overrides = dict(VARIANTS[args.variant])
    overrides["act"] = act
    if args.window is not None:
        overrides["window"] = args.window
    if args.qg_lr is not None:
        overrides["qg_lr"] = args.qg_lr
    if args.qg_chunk:
        overrides["qg_chunk"] = args.qg_chunk
    if args.qg_steps is not None:
        overrides["qg_steps"] = args.qg_steps
    if args.inverse:
        overrides["inverse"] = True
        for k in ("inv_lam", "inv_inner", "inv_lr", "inv_radius", "inv_agg",
                  "inv_reach_w", "inv_y_noise", "inv_step_max"):
            v = getattr(args, k)
            if v is not None:
                overrides[k] = v
        print("[inverse] query generation = inverse disagreement (targets y optimized, "
              "teacher queried at each member's inverse x_i(y; x0))", flush=True)
    if args.reverse_peel:
        overrides["reverse_peel"] = True
        if args.reverse_eps is not None: overrides["reverse_eps"] = args.reverse_eps
        if args.reverse_thresh is not None: overrides["reverse_thresh"] = args.reverse_thresh
        if args.reverse_force: overrides["reverse_force"] = True
        print("[reverse-peel] head-first peel: head consensus (column frame) -> exact kink-jump refine "
              "-> pin; then consensus-pin downwards", flush=True)
    if args.layer_consensus:
        overrides["layer_consensus"] = True
    if args.resume_pop:
        overrides["resume_pop"] = args.resume_pop
        overrides["warmstart_iters"] = 0
        overrides["target_mode"] = args.target_mode
        overrides["target_lam"] = args.target_lam
        overrides["target_frac"] = args.target_frac
        overrides["target_act_w"] = args.target_act_w
        overrides["target_reinit"] = args.target_reinit
        overrides["target_consensus"] = args.target_consensus
        overrides["target_value"] = args.target_value
        overrides["target_units"] = args.target_units
    if args.targetdeep:
        overrides["targetdeep"] = True
    if args.balance is not None:
        overrides["balance"] = args.balance
    if args.guessfree_after:
        overrides["guessfree_after"] = args.guessfree_after
        overrides["guessfree_lines"] = args.guessfree_lines
    if args.verify_cands:
        overrides["verify_cands"] = True
    if args.repel:
        overrides["repel"] = args.repel
    if args.frontier_lr:
        overrides["frontier_lr"] = args.frontier_lr
    if getattr(args, "reinit_lr", 0.0):
        overrides["reinit_lr"] = args.reinit_lr
    if getattr(args, "reroll_dups", False):
        overrides["reroll_dups"] = True
    if getattr(args, "mine_in_h", False):
        overrides["mine_in_h"] = True
    if getattr(args, "hot_reroll", 0.0):
        overrides["hot_reroll"] = args.hot_reroll
    if getattr(args, "reroll_stale", 0):
        overrides["reroll_stale"] = args.reroll_stale
    if getattr(args, "mine_h_range", 0.0):
        overrides["query_raw_range"] = args.mine_h_range
    if getattr(args, "query_map_init_only", False):
        overrides["query_map_init_only"] = True
    if getattr(args, "cluster_eps", 0.0):
        overrides["cluster_eps"] = args.cluster_eps
    if getattr(args, "cluster_eps_l1", 0.0):
        overrides["cluster_eps_l1"] = args.cluster_eps_l1
    if getattr(args, "cluster_quorum", 0.0):
        overrides["cluster_quorum"] = args.cluster_quorum
    if getattr(args, "residual_warm", False):
        overrides["residual_warm"] = True
    if getattr(args, "orient_check", False):
        overrides["orient_check"] = True
    if getattr(args, "masked_consensus", False):
        overrides["masked_consensus"] = True
    if getattr(args, "qg_init_data", False):
        _dd = ([int(v) for v in args.query_via_mlp_prefix.split(":")[0].split(",")] if getattr(args, "query_via_mlp_prefix", "")
               else (dims if "dims" in dir() else [int(torch.tensor(input_shape).prod()), out_dim]))   # CNN path: MNIST by (784, 10)
        (_xtr_d, _), (_xte_d, _) = load_data(_dd, device)
        overrides["qg_init"] = "data"; overrides["query_data"] = torch.cat([_xtr_d, _xte_d]).to(device); overrides["query_data_jitter"] = args.qg_data_jitter; overrides["query_data_pool"] = args.qg_data_pool
        print(f"[setup] --qg-init-data: {overrides['query_data'].shape[0]} natural inputs as mining base points (jitter {args.qg_data_jitter:g})", flush=True)
    if getattr(args, "query_map_abs", False):
        overrides["query_map"] = (lambda raw: raw.abs()); overrides["query_raw_dim"] = dims[0]
    if getattr(args, "h_push", 0):
        overrides["h_push_steps"] = args.h_push; overrides["h_push_lr"] = args.h_push_lr; overrides["h_push_radius"] = args.h_push_radius
        overrides["h_push_mode"] = args.h_push_mode
        if "_query_map" in dir() and _query_map is not None:
            overrides["h_push_fn"] = _query_map
        if args.h_push_mode == "oracle":
            overrides["h_push_oracle"] = teacher.layers[0]          # DIAGNOSTIC upper bound: teacher frontier units
        print(f"[setup] --h-push {args.h_push} mode {args.h_push_mode}: prefix fn set = {'h_push_fn' in overrides}", flush=True)
    if args.loc_refine_l1:
        overrides["loc_refine_l1"] = True
    if (getattr(args, "query_via_prefix", False) or getattr(args, "query_via_mlp_prefix", "")) and "_query_map" in dir() and _query_map is not None:
        overrides["query_map"] = _query_map
        overrides["query_raw_dim"] = 784 if args.query_via_prefix else int(args.query_via_mlp_prefix.split(":")[0].split(",")[0])
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.qg_tf32:
        overrides["qg_tf32"] = True
    if args.qg_chunk_gb is not None:
        overrides["qg_chunk_gb"] = args.qg_chunk_gb
    if args.qg_block_gb is not None:
        overrides["qg_block_gb"] = args.qg_block_gb
    if args.qg_host_gb is not None:
        overrides["qg_host_gb"] = args.qg_host_gb
    if args.eval_loss:
        overrides["eval_loss"] = True
    if args.cpu_gather:
        overrides["gpu_seg"] = False
    if args.disk is not None:
        overrides["disk"] = args.disk
    if args.disk_ram_gb is not None:
        overrides["disk_ram_gb"] = args.disk_ram_gb
    if args.disk_seg_gb is not None:
        overrides["disk_seg_gb"] = args.disk_seg_gb
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.balance_mode is not None:
        overrides["balance_mode"] = args.balance_mode
    if args.balance_target is not None:
        overrides["balance_target"] = args.balance_target
    if args.balance_rownorm is not None:
        overrides["balance_rownorm"] = args.balance_rownorm
    if args.balance_dirlr is not None:
        overrides["balance_dirlr"] = args.balance_dirlr
    if args.balance_dirlr_decay is not None:
        overrides["balance_dirlr_decay"] = args.balance_dirlr_decay
    if args.balance_dirlr_head is not None:
        overrides["balance_dirlr_head"] = bool(args.balance_dirlr_head)
    if args.balance_refresh is not None:
        overrides["balance_refresh"] = args.balance_refresh
    if args.no_stacked:
        overrides["stacked"] = False
    if args.warmstart is not None:
        overrides["warmstart_iters"] = args.warmstart
    _ai = (args.inverse or args.reverse_peel) if args.align_inverse is None else args.align_inverse
    if _ai:
        overrides["align_inverse"] = True
        print("[align-inverse] eps metric = activation-matched per-layer pairing "
              "(param_errors_actmatch)", flush=True)
    if args.verbose:
        overrides["log_every"] = 1
    if args.log_every is not None:
        overrides["log_every"] = args.log_every
    if args.freeze_reinit or args.peel or args.resume:  # peel: freeze consensus layers, reinit
        overrides["freeze_reinit"] = True
        overrides["freeze_thresh"] = args.freeze_thresh
        overrides["freeze_precision"] = args.freeze_precision
    if args.peelrefresh or args.peelrestart:   # CNN peelrestart == peel_refresh
        overrides["peel_refresh"] = True       # (cold reinit + t=0 clock restart)
    if args.fast_peel:
        # --fast-peel (CNN, any mode): peel the frontier layer as soon as the committee
        # FULLY agrees on it (quorum 100%), refine the consensus rows, freeze, continue.
        # Implies --peel. (Under --cheat it needs --cheat-pop > 1 for a committee.)
        overrides["freeze_reinit"] = True
        overrides["freeze_thresh"] = 1.0
        overrides["freeze_precision"] = 0.0
        overrides["fast_peel"] = True
        print("[peel] --fast-peel: refine + peel the frontier layer on FULL committee "
              "consensus (implies --peel, quorum 100%)", flush=True)
    if args.loc_refine:                        # CNN peel refiner -> kink_solve (forward-only,
        overrides["loc_refine"] = True         # conv-aware kink points), see _cnn_refine_layer
        print("[peel] --loc-refine: CNN refiner = kink_solve (conv units = channel x position, "
              "surface tracking); fp64 oracle", flush=True)
    # CNN --partial: the SAME semantics as the MLP -- every log_every iters refine the
    # frontier's SOLVABLE channels and pin them IN PLACE (grad-masked hooks), no
    # reinit/advance/restart; stragglers keep training. Distinct from --peel (the
    # eps-gated full-layer peel that advances) and --peelrestart (cold restart on a
    # full-layer peel). Routes to cfg.partial, exactly like the MLP path.
    if args.partial:
        overrides["partial"] = True
        overrides["peel_angle_gate"] = args.peel_angle_gate
    if args.fast_peel_partial:
        overrides["fast_peel_partial"] = True
        if args.peelrefresh or args.peelrestart:
            overrides["freeze_reinit"] = True
            overrides["freeze_thresh"] = 1.0
            overrides["freeze_precision"] = 0.0
        overrides["peel_angle_gate"] = args.peel_angle_gate
        print("[peel] --fast-peel-partial: per-neuron quorum consensus peel, pinned in place "
              "across the committee; layer advances when fully solved", flush=True)
    if args.retry:
        overrides["retry"] = args.retry
        print(f"[peel] --retry {args.retry}: reinit + retry when a budget ends with the "
              "frontier layer not fully peeled", flush=True)
    if args.restart_stuck:
        overrides["restart_stuck"] = True
        overrides["peel_angle_gate"] = args.peel_angle_gate
        if not args.partial:
            # restart-stuck restarts AROUND the partial-solved rows -- without
            # --partial nothing ever banks and the trigger can never arm
            overrides["partial"] = True
            print("[restart-stuck] implies --partial (the restart keeps "
                  "partial-solved channels pinned); enabling it", flush=True)
    if args.peeltry:                           # explicit --peeltry keeps the old path
        overrides["peel_try"] = args.peeltry
        overrides["peel_miss_abort"] = args.peel_miss_abort
        overrides["peel_advance_frac"] = args.peel_advance_frac
    if args.peel_warm:
        overrides["peel_warm"] = True
    if args.peel_stuck:
        overrides["peel_stuck"] = args.peel_stuck
    if args.peel_reroll:
        overrides["peel_reroll"] = True
    if args.layer_space_init is not None:
        overrides["layer_space_init"] = args.layer_space_init
    if args.layer_rand_init:
        overrides["layer_rand_init"] = True
    if args.layer_mine:
        overrides["layer_mine"] = True
    if args.hard:
        overrides["hard"] = True
        overrides["hard_loss"] = args.hard_loss
        overrides["hard_floor"] = args.hard_floor
    if args.cheat:
        overrides["cheat"] = True
        overrides["cheat_peel_max"] = args.cheat_peel_max
        overrides["cheat_peel_mean"] = args.cheat_peel_mean
        if args.cheat_solo:
            print("[cheat] --cheat-solo is MLP-only; ignored on the CNN path",
                  flush=True)
        if args.fast_peel:
            if args.cheat_pop > 1:
                overrides["fast_peel"] = True
                print("[cheat] --fast-peel: the frontier layer refines+peels on full "
                      "committee consensus (guesses = quorum means); needs --peel",
                      flush=True)
            else:
                print("[cheat] --fast-peel ignored (needs --cheat-pop > 1)", flush=True)
        if args.cheat_pop > 1:
            args.p = args.cheat_pop
            overrides["cheat_bb_ref"] = True
            print(f"[cheat] population p={args.p} (--cheat-pop); disagreement "
                  "runs on each (member, blackbox) pair -- no member-member "
                  "terms", flush=True)
        else:
            if args.p != 1:
                print(f"[cheat] forcing population p=1 (was {args.p}); "
                      "disagreement runs on the (student, blackbox) pair",
                      flush=True)
            args.p = 1
    if args.pop_save_every > 0:
        recon_dir = os.path.join(os.path.dirname(__file__), "recon")
        os.makedirs(recon_dir, exist_ok=True)
        sh = "x".join(map(str, input_shape))
        cs = "_".join("-".join(map(str, c)) for c in conv_cfgs)
        ctag = ("_cheat" if args.cheat else "") + (f"_{args.tag}" if args.tag else "")
        overrides["pop_save_every"] = args.pop_save_every
        overrides["pop_save_path"] = os.path.join(
            recon_dir, f"_pop__{args.variant}{ctag}_cnn__{sh}__{cs}__s{args.seed}.pt")
    cfg = Cfg(p=args.p, q=args.q, outer=args.outer, **overrides)
    if cfg.pop_save_path:
        print(f"[ckpt] snapshotting best member every {cfg.pop_save_every} iters "
              f"-> {cfg.pop_save_path}", flush=True)
    print(f"[run] {args.variant} CNN seed={args.seed} "
          f"params={count_params(teacher)} budget={cfg.outer * cfg.q} queries",
          flush=True)

    resume_state = None; resume_record = None
    if args.resume:
        rck = torch.load(args.resume, map_location=device, weights_only=True)
        resume_state = rck.get("refined_state", rck.get("state_dict", rck))
        print(f"[resume] loaded student from {args.resume}", flush=True)
        if getattr(args, "resume_record", False):
            if rck.get("partial_state") is not None and rck.get("partial_mask"):
                resume_record = (rck["partial_state"], rck["partial_mask"])
                print(f"[resume] --resume-record: using the snapshot's fp64 record ({ {l: int(m.sum()) for l, m in rck['partial_mask'].items()} } pinned rows)", flush=True)
            else:
                print("[resume] --resume-record requested but the snapshot has no record; re-solving instead", flush=True)

    best, log, final = reconstruct_cnn(teacher, input_shape, conv_cfgs, out_dim,
                                       cfg, device, seed=args.seed, fc_dims=fc_dims,
                                       resume_state=resume_state, resume_record=resume_record)

    tag = f"_{args.tag}" if args.tag else ""
    if args.hard:
        tag = "_hard" + tag
    if args.cheat:
        tag = "_cheat" + tag
    sh = "x".join(map(str, input_shape))
    cs = "_".join("-".join(map(str, c)) for c in conv_cfgs)
    fs = ("_fc" + "-".join(map(str, fc_dims))) if fc_dims else ""
    stub = f"{args.variant}_cnn{tag}__{sh}__{cs}{fs}__s{args.seed}"
    recon_dir = os.path.join(os.path.dirname(__file__), "recon")
    os.makedirs(recon_dir, exist_ok=True)
    # A resume run must NEVER clobber the student it resumed from: save to a distinct
    # "_resumed" file. And before overwriting ANY existing checkpoint, keep a .bak.
    out_name = (stub + "_resumed_final.pt") if args.resume else (stub + "_final.pt")
    out_path = os.path.join(recon_dir, out_name)
    if os.path.exists(out_path):
        os.replace(out_path, out_path + ".bak")
    torch.save({"input_shape": input_shape, "conv_cfgs": conv_cfgs,
                "fc_dims": fc_dims, "out_dim": out_dim, "act": act,
                "state_dict": {k: v.detach().cpu()
                               for k, v in best.state_dict().items()}, **final},
               out_path)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, stub + ".json"), "w") as f:
        json.dump({"variant": args.variant, "model": "cnn",
                   "input_shape": input_shape, "conv_cfgs": conv_cfgs,
                   "fc_dims": fc_dims, "out_dim": out_dim, "act": act,
                   "seed": args.seed, "log": log, **final}, f, indent=1)
    print(f"[done] {stub} | final_max_eps={final['final_max_eps']:.3e} "
          f"wall={final['wall_s']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="784,64,10")
    ap.add_argument("--cheat-peel", type=int, nargs="?", const=1, default=0, metavar="N",
                    help="DIAGNOSTIC: --arch is the full teacher; remove N leading "
                         "layers (default 1) and train on direct teacher-tail queries "
                         "from the start. Shortcut for --teacher-full/--teacher-drop.")
    ap.add_argument("--teacher-full", default="",
                    help="(MLP) make the blackbox the SUB-NET of a bigger trained "
                         "teacher: build/load the teacher for THIS full arch (e.g. "
                         "3072,200,200,200,100), drop --teacher-drop leading layers, "
                         "and query the remainder directly. The remaining dims must "
                         "equal --arch. Lets you test 'is L2 easier as a genuine "
                         "input layer' with the real pipeline, no L1, no inversion.")
    ap.add_argument("--teacher-drop", type=int, default=1,
                    help="with --teacher-full: number of leading layers to remove.")
    ap.add_argument("--teacher-seed", type=int, default=0)
    ap.add_argument("--teacher-epochs", type=int, default=25)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--outer", type=int, default=40)
    ap.add_argument("--q", type=int, default=1500)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--window", type=int, default=None,
                    help="override variant's sample window (in outer iters; "
                         "0 = keep all). Default: use the variant's value.")
    ap.add_argument("--combine", action="store_true",
                    help="the first outer iter the cluster consensus becomes "
                         "available, replace the worst committee member with "
                         "the consensus net.")
    ap.add_argument("--factor-tail", action="store_true",
                    help="square-factorize every hidden layer after the first "
                         "(W_l=A_l B_l, exact reparam) so the tail gets a better "
                         "optimizer geometry; W1 stays ordinary/interpretable.")
    ap.add_argument("--factor-inner", type=float, default=1.0,
                    help="inner-dim multiplier for --factor-tail: inner dim = "
                         "round(factor_inner*out). 1.0=square; >1 over-squares "
                         "(larger relaxation / stronger preconditioner, still exact).")
    ap.add_argument("--factor-lr", type=float, default=0.25,
                    help="factor Adam LR as a FRACTION of the fit LR (1e-3); "
                         "default 0.25 = eta/4. Sweep e.g. 1,0.5,0.25,0.125.")
    ap.add_argument("--factor-mode", choices=["residual", "square"],
                    default="residual",
                    help="residual: W_eff=W+U V^T (keeps baseline -G path, robust, "
                         "PREFERRED). square: W=A B (no -G path).")
    ap.add_argument("--factor-rank", type=int, default=256,
                    help="rank of the residual express lanes (mode=residual); "
                         "0 => full min(out,in). Pure relaxation knob.")
    ap.add_argument("--factor-layers", choices=["tail", "l1", "all"],
                    default="tail",
                    help="which weight layers to reparameterize: tail (all but "
                         "W1) | l1 (W1 only) | all.")
    ap.add_argument("--inverse", action="store_true",
                    help="INVERSE disagreement query generation: optimize target OUTPUTS y "
                         "and query the teacher at the inputs each member needs to reach y "
                         "from a shared reference x0 (gen_queries_inverse). Plain committee "
                         "path only (no cheat / peel frontier / ensemble-boost).")
    ap.add_argument("--inv-lam", type=float, default=None, help="inverse: proximity prior lam")
    ap.add_argument("--inv-inner", type=int, default=None, help="inverse: Gauss-Newton steps per member")
    ap.add_argument("--inv-lr", type=float, default=None, help="inverse: Adam lr on targets y")
    ap.add_argument("--inv-radius", type=float, default=None, help="inverse: trust region |y-y0|/std (0=off)")
    ap.add_argument("--inv-agg", default=None, choices=["var", "median_pair", "mean_pair", "min_pair"],
                    help="inverse: input-spread aggregator")
    ap.add_argument("--inv-reach-w", type=float, default=None, help="inverse: unreachable-target penalty weight")
    ap.add_argument("--inv-y-noise", type=float, default=None, help="inverse: initial target jitter (std units)")
    ap.add_argument("--inv-step-max", type=float, default=None, help="inverse: clip Gauss-Newton step norm")
    ap.add_argument("--reverse-peel", action="store_true",
                    help="top-down peel: track consensus on the HEAD (gauge-correct column frame), "
                         "refine it exactly (kink-jump refiner), pin it, then consensus-pin the "
                         "layers below one at a time (reverse_peel.py). Implies --align-inverse.")
    ap.add_argument("--reverse-eps", type=float, default=None)
    ap.add_argument("--reverse-thresh", type=float, default=None)
    ap.add_argument("--resume-pop", default="", help="load a population snapshot (recon/_pop__*.pt) as the starting committee; buffer starts empty; warm-start iterations are skipped")
    ap.add_argument("--target-mode", default="off", choices=["off", "control", "pin", "l1"],
                    help="hold certified weights ('pin' = direction snaps, 'l1' = L1 anchor) and aim --target-frac of the queries at the unsolved units; 'control' = resume only")
    ap.add_argument("--target-lam", type=float, default=1.0)
    ap.add_argument("--target-frac", type=float, default=1.0)
    ap.add_argument("--target-act-w", type=float, default=1.0, help="weight of the unsolved-units activation term in the targeted sampler")
    ap.add_argument("--target-reinit", action="store_true", help="re-draw the unsolved units' rows and outgoing columns in every member at start")
    ap.add_argument("--target-consensus", action="store_true", help="put every member on the consensus values of the certified entries at start (pin to the GUESS, not to each member's own value)")
    ap.add_argument("--target-value", default="seed", choices=["seed", "mean"], help="consensus value: the reference member's entry (seed) or the quorum mean")
    ap.add_argument("--target-units", default="", help="force the target set at start: '2:121;3:3,43,117' (hidden layer 1-based : unit indices in the reference frame)")
    ap.add_argument("--balance-mode", choices=["rownorm", "actnorm", "layer", "unit"], default=None, help="rownorm (default; the version that worked, run balancelayer3): canonicalise once to pre-activation RMS = target, then project every hidden (row,bias) back to its fixed norm after EVERY step (head free). actnorm: per step, equalise row norms within each hidden layer and float the layer scale so its pre-activation RMS = --balance-target (NEGATIVE at scale: head scale migrates, consensus 0).  canonicalise once to pre-activation RMS = target, then project every hidden row back to its fixed norm after EVERY step (no compensation; head free). layer: after each rebalance every hidden layer has the same pre-activation RMS (one scalar per layer, exact gauge move); unit: per-unit incoming == outgoing norm")
    ap.add_argument("--balance-dirlr", type=float, default=None, help="directional learning rate eta: per-layer Adam lr = eta*median row norm/sqrt(fan_in) (x base schedule), head included, so every layer's rows rotate by eta per step regardless of scale. Fixes the L1 jitter floor (3%%/step at pinned norm 2) and the frozen head (norm 35) under rownorm. Try 0.0025 (= what the unbalanced run's inflated W1 effectively had). 0/absent = off")
    ap.add_argument("--balance-dirlr-decay", type=float, default=None, help="multiply the directional step eta by this factor every outer iteration (eta_it = eta * decay^(it-1)). E.g. --balance-dirlr 0.03 --balance-dirlr-decay 0.85: 3%%/step at it 1 (what rownorm converged L1 under) down to 0.3%% by it 15. Default 1 (constant)")
    ap.add_argument("--balance-dirlr-head", type=int, default=None, help="1: apply the directional-lr rule to the head as well (run dirlr2 behaviour: head norm ran away 336->2071 because lr ~ norm). Default 0: head keeps the base lr")
    ap.add_argument("--balance-refresh", type=int, default=None, help="rownorm: every N outer iterations re-canonicalise the pinned norms once (function-preserving, head compensated) and re-freeze, undoing the depth-wise pre-activation decay (L5 0.75 -> 0.03) and the head growth. 1 = every iteration; absent/0 = off")
    ap.add_argument("--balance-rownorm", type=float, default=None, help="rownorm mode: fix every hidden (row,bias) norm to this value (e.g. 20). Adam steps are absolute, so this IS the directional step size: rel. change per step ~ lr*sqrt(fan_in)/norm. Default: derive norms from --balance-target")
    ap.add_argument("--balance-target", type=float, default=None, help="layer mode: the fixed pre-activation RMS every hidden layer is set to (default 1.0). 0 = drifting geometric mean: UNSAFE, the scale runs away")
    ap.add_argument("--batch", type=int, default=None, help="surrogate training minibatch (variant default; mergedbest512 = 512). Per-step cost has a ~2.3 ms floor on a 3090, so per-sample cost falls 5.3 us @512 -> 3.5 @2048 -> 2.9 @8192; fewer steps per epoch, so re-tune --lr")
    ap.add_argument("--lr", type=float, default=None, help="surrogate training Adam lr (default 1e-3)")
    ap.add_argument("--epochs", type=int, default=None, help="training epochs per outer iteration (default 10)")
    ap.add_argument("--qg-tf32", action="store_true", help="query generation only: TF32 matmuls in the disagreement search (~1.4x faster sampler on Ampere; same disagreement reached, different individual points). Training stays fp32")
    ap.add_argument("--qg-chunk-gb", type=float, default=None, help="query generation: GB budget of the automatic forward chunk (default 6; 1.5 was the old default)")
    ap.add_argument("--qg-host-gb", type=float, default=None, help="query generation: assemble the q x d query block on the host when it exceeds this many GB (default 8); needed when q x d x 4 bytes approaches the GPU memory")
    ap.add_argument("--qg-block-gb", type=float, default=None, help="query generation: GB of leaf+Adam state per outer block (default 2 = 40690 queries at d=3072)")
    ap.add_argument("--eval-loss", action="store_true", help="rank members by a full post-training pass over the (RAM-tier) buffer every iteration (old behaviour; ~15 s per 1.8M rows). Default: the free per-member mean training loss of the last epoch")
    ap.add_argument("--cpu-gather", action="store_true", help="training: old minibatch feeding (CPU gather + H2D per batch). Default copies each buffer segment to the GPU once and gathers there (step at its GPU floor)")
    ap.add_argument("--disk", type=str, default=None, help="two-tier replay ring: keep the newest --disk-ram-gb of the window in host RAM and stream the older blocks from this directory (flat X.bin/Y.bin, sequential prefetched reads; diskring.py). Lets --window x --q exceed host RAM at no step-time cost while the NVMe reads faster than the GPU consumes the disk tier")
    ap.add_argument("--disk-ram-gb", type=float, default=None, help="--disk: host-RAM budget of the RAM tier (default 180)")
    ap.add_argument("--disk-seg-gb", type=float, default=None, help="--disk: streamed segment size in GB (default 4; x4 staging buffers)")
    ap.add_argument("--balance", type=int, default=None, help="leaky-ReLU only: every N epochs rebalance each hidden unit's incoming vs outgoing weight norm (exact gauge move, Adam state transformed). Stops the W1 scale drift that trains deep layers at an inflated effective lr. 1 = every epoch; 0/absent = off")
    ap.add_argument("--warmstart", type=int, default=None, help="number of warm-start iterations with random Gaussian queries before the disagreement sampler starts (variant default 5; 0 disables)")
    ap.add_argument("--no-stacked", action="store_true", help="disable the stacked (batched-matmul) training fast path; train members one by one as before")
    ap.add_argument("--targetdeep", action="store_true", help="disagreement sampler compares every layer's activations (aligned, standardised) concatenated with the output")
    ap.add_argument("--layer-consensus", action="store_true",
                    help="print the isolated two-sided per-layer consensus (rows from below, cols from "
                         "above, units paired by activation) every log point; no pinning.")
    ap.add_argument("--reverse-force", action="store_true",
                    help="reverse peel: force pins into members whose units match nowhere (default off)")
    ap.add_argument("--align-inverse", action=argparse.BooleanOptionalAction, default=None,
                    help="eps metric: activation-matched per-layer unit pairing "
                         "(align.param_errors_actmatch) instead of the chained weight "
                         "alignment. Default ON for --inverse runs; --no-align-inverse "
                         "to force the chained metric there.")
    ap.add_argument("--qg-steps", type=int, default=None,
                    help="override the variant's query-optimization steps (0 = RANDOM queries, the passive-sampling control)")
    ap.add_argument("--qg-chunk", type=int, default=0,
                    help="chunk size for the query-generation member forward "
                         "(0=full batch). Set e.g. 8192 so wide --expand students "
                         "don't OOM during query optimization; gradient is exact.")
    ap.add_argument("--expand", type=float, default=1.0,
                    help="Expand-and-Cluster fit gate: widen every HIDDEN layer "
                         "by this factor (input/output fixed). >1 => students "
                         "overparameterized vs teacher; only fit loss is logged "
                         "(weight scoring/consensus need matched dims). Run "
                         "WITHOUT --fast/--combine/--save-recon.")
    ap.add_argument("--ensemble", type=int, default=0,
                    help="ensemble-merge scheme: train N same-architecture "
                         "members (sets population p=N, overriding --p and the "
                         "--cheat p=1 forcing) and every --ensemble-every outer "
                         "iters merge them into ONE net -- per hidden layer, "
                         "group-OMP-select the width-n subset of all pooled "
                         "member neurons that best linearly explains every "
                         "member's next-layer preactivations, rebuild the next "
                         "layer over the selected basis, recurse, ridge-solve "
                         "the output head on the query buffer. The merged net "
                         "replaces the worst member when it wins on loss. "
                         "Attacker-side only; peel-pinned rows keep their "
                         "slot+values.")
    ap.add_argument("--ensemble-every", type=int, default=5,
                    help="without --ensemble-boost: outer-iteration cadence "
                         "of the merge. With --ensemble-boost: the cascade is "
                         "RECOMBINED (compressed into one width-n net) once "
                         "it reaches this many STAGES (fresh-query gated; "
                         "forced at 2x). Default 5.")
    ap.add_argument("--ensemble-samples", type=int, default=200000,
                    help="query-sample size for the --ensemble merge fits "
                         "(default 200k; capped at the buffer size).")
    ap.add_argument("--query-box", type=float, default=None,
                    help="bound disagreement queries into [-box, box]^d via "
                         "tanh (cfg.query_box). Default: variant value (0 = "
                         "unbounded), EXCEPT under --ensemble-boost where it "
                         "defaults to 1.0 -- an unbounded query search chases "
                         "the cascade's off-manifold excursions and diverges.")
    ap.add_argument("--ensemble-boost", action="store_true",
                    help="residual-cascade boosting: every outer iteration "
                         "appends --ensemble N sequential residual stages "
                         "(each a fresh net trained epochs at lr on the unit-"
                         "renormalized residual of the cascade-so-far); the "
                         "model is the growing SUM. Compressed into one "
                         "width-n net past ~24 stages (fresh-query gated) and "
                         "once terminally (the extraction candidate). Under "
                         "--cheat population stays p=1. Incompatible with "
                         "--hard.")
    ap.add_argument("--solver-polish", action="store_true",
                    help="at the end of each outer iter, tighten every member "
                         "with the staged LBFGS recipe (MSE then MAE).")
    ap.add_argument("--solverwindow", type=int, default=None,
                    help="window (in outer iters) of most-recent queries every "
                         "query-solver fits on -- the in-loop polish, the --fast "
                         "endgame, and the lbfgs endgame. 0 = all queries "
                         "(default).")
    ap.add_argument("--pop-save-every", type=int, default=5,
                    help="checkpoint the committee population every k outer "
                         "iters to recon/_pop__VARIANT__ARCH__sSEED.pt "
                         "(atomic overwrite; crash recovery + mid-run "
                         "inspection). 0 disables.")
    ap.add_argument("--log-every", type=int, default=None,
                    help="log / consensus-check interval in outer iterations (default 5; "
                         "--verbose = 1). Applies to the per-layer eps lines, the forward "
                         "peel checks, the reverse-peel checks and the isolated consensus report.")
    ap.add_argument("--verbose", action="store_true",
                    help="print the per-iteration log block every iteration "
                         "(sets log_every=1) instead of every log_every iters.")
    ap.add_argument("--qg_lr", type=float, default=None,
                    help="override the query-generator Adam lr on top of the "
                         "variant (Cfg.qg_lr, e.g. 0.1)")
    ap.add_argument("--fast", action="store_true",
                    help="stop training the first iter a consensus forms, then "
                         "staged MSE->MAE solve it, and emit the solved "
                         "reconstruction (much fewer queries).")
    ap.add_argument("--stop-layer", type=int, default=None,
                    help="with --fast, stop as soon as this hidden layer (0-indexed) "
                         "reaches FULL consensus, instead of waiting for the whole "
                         "net. Use 0 to peel the first layer. Dumps the population + "
                         "the partial consensus (that layer solved).")
    ap.add_argument("--stop-outer", action="store_true",
                    help="with --fast, stop as soon as the OUTERMOST hidden layer "
                         "forms a full consensus -- deeper layers need not agree. "
                         "Shorthand for --stop-layer 0.")
    ap.add_argument("--consensus-threshold", type=float, default=None,
                    help="consensus TOLERANCE (cfg.cluster_eps, default 0.02): the "
                         "inf-norm distance within which two members' aligned "
                         "[w|b] neuron rows count as agreeing. Loosen it to let a "
                         "deep net's looser layers reach consensus. Sets the "
                         "printed consensus, --combine, and the --fast stop/dump "
                         "threshold consistently. The quorum (5/8) stays fixed.")
    ap.add_argument("--save-recon", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="save the pre-endgame reconstruction (student + full "
                         "population + query set + teacher) to recon/*.pt so "
                         "the solvers can be re-tried offline via polish.py. "
                         "On by default; pass --no-save-recon to disable.")
    ap.add_argument("--verify", action="store_true",
                    help="every log-iter, efficiently VERIFY each layer-1 neuron "
                         "that reached consensus (even a single one): probe the "
                         "teacher as a black box to test whether its kink "
                         "hyperplane is within (eps_offset, eps_angle) of the "
                         "truth. ~O(k) queries/neuron, independent of input dim. "
                         "Diagnostic queries are not counted in the budget.")
    ap.add_argument("--verify-refine", action="store_true",
                    help="with --verify, also REFINE within-eps neurons: offset to "
                         "machine precision (cheap) + true normal via the rank-1 "
                         "gradient jump (~2d queries/neuron).")
    ap.add_argument("--verify-max", type=int, default=0,
                    help="with --verify, cap how many consensus layer-1 neurons "
                         "are probed per log-iter (0 = all).")
    ap.add_argument("--verify-eps-offset", type=float, default=1e-2,
                    help="within-eps tolerance on plane offset (default 1e-2).")
    ap.add_argument("--verify-eps-angle", type=float, default=1.0,
                    help="within-eps tolerance on normal tilt in degrees (default 1.0).")
    ap.add_argument("--verify-k", type=int, default=16,
                    help="random perpendicular probes for the angle estimate "
                         "(higher = tighter estimate, linearly more queries).")
    ap.add_argument("--peel", action="store_true",
                    help="alias for --freeze-reinit (CNN + MLP): freeze consensus "
                         "layers and reinit the committee onto deeper layers.")
    ap.add_argument("--retry", type=int, nargs="?", const=5, default=0, metavar="N",
                    help="(any peel mode) if the budget ends with the frontier layer not "
                         "fully peeled, reinit and retry, up to N times (bare flag: 5). "
                         "--fast-peel-partial/--partial: keep solved rows pinned, reinit "
                         "only the unsolved rows + deeper layers; other peel modes: reinit "
                         "the entire frontier layer (+ deeper). Buffer flushed, iteration "
                         "clock restarted, fresh --outer budget each retry.")
    ap.add_argument("--fast-peel-partial", action="store_true",
                    help="(CNN, committee) PER-NEURON consensus peel: every log iter, "
                         "kink-refine the frontier layer's consensus channels from the "
                         "quorum-mean rows, inject each solved row into every member at "
                         "that member's own magnitude (in-place, grad-masked pin), and "
                         "advance to the next layer only once the whole layer is solved.")
    ap.add_argument("--partial", action="store_true",
                    help="(MLP + --cheat) every log_every iters, refine the "
                         "frontier's still-unsolved neurons REGARDLESS of the peel "
                         "threshold and in-place freeze (warm, scale-matched) "
                         "whatever solves. Locks in solvable neurons early while "
                         "stragglers keep training; never advances the whole layer "
                         "(the full --peel still does that). Runs every log_every "
                         "iters unconditionally -- early passes can be slow while "
                         "most neurons still abstain.")
    ap.add_argument("--freeze-reinit", action="store_true",
                    help="peel: once a hidden layer's consensus ratio reaches "
                         "--freeze-thresh, REINITIALIZE the committee sharing that "
                         "layer's consensus weights FROZEN (consensus neurons pinned "
                         "+ shared, stragglers stay trainable) and keep training on "
                         "the reused queries, collapsing the search onto the deeper "
                         "layers. Cascades layer by layer.")
    ap.add_argument("--freeze-thresh", type=float, default=1.0,
                    help="consensus ratio (n_cons/n_tot) at which --freeze-reinit "
                         "freezes a hidden layer (default 1.0 = full consensus).")
    ap.add_argument("--freeze-precision", type=float, default=0.0,
                    help="with --freeze-reinit, only freeze a layer once its cons_max "
                         "<= this (0 = no precision gate; freeze on count alone).")
    ap.add_argument("--layer-space-init", action=argparse.BooleanOptionalAction, default=None,
                    help="(CNN peel) seed disagreement queries in the frozen frontier's "
                         "INPUT space (even per-neuron boundary coverage via the exact "
                         "prefix) instead of image space. On by default when peeling; "
                         "--no-layer-space-init to disable.")
    ap.add_argument("--layer-mine", action="store_true",
                    help="(CNN peel) mine disagreement queries DIRECTLY in the "
                         "frontier layer's input space (member tails + the "
                         "blackbox tail under --cheat), then invert the "
                         "optimized activations to images through the exact "
                         "frozen prefix. Sees frontier directions the prefix "
                         "squashes from image space. Takes precedence over "
                         "--layer-rand-init and boundary seeding.")
    ap.add_argument("--layer-rand-init", action="store_true",
                    help="(CNN peel) initialize the disagreement-query optimizer "
                         "with images whose FRONTIER-INPUT activations are "
                         "randomly (isotropically) distributed: sample random "
                         "targets in the frozen frontier's input space and "
                         "approximately invert them through the exact prefix "
                         "(least squares), then run the normal query "
                         "optimization from that init. Replaces the kink-"
                         "targeted boundary seeding while peeling.")
    ap.add_argument("--resume", default=None,
                    help="(CNN peel) path to a saved recon/*_final.pt student. "
                         "Re-solve+freeze its exact layers (kink-refine, shallow->deep, "
                         "exact-or-abstain), spin up a FRESH committee onto the unsolved "
                         "layers, and continue for --outer more iters with new samples. "
                         "The blackbox teacher is the cached one (deterministic).")
    ap.add_argument("--extract-freeze", action="store_true",
                    help="every log-iter, EXACTLY extract each layer-1 consensus neuron "
                         "with the black-box exact-affine probe (~2d teacher queries). "
                         "Neurons that extract cleanly are pinned to their exact value "
                         "across the committee and frozen for good; failures are "
                         "skipped. Seeds the peel with float-precision neurons.")
    ap.add_argument("--extract-max-angle", type=float, default=5.0,
                    help="with --extract-freeze, reject a probe whose recovered normal "
                         "is more than this many degrees from the guess (default 5).")
    ap.add_argument("--extract-jump-ratio", type=float, default=50.0,
                    help="with --extract-freeze, reject a probe unless its gradient "
                         "jump is cleanly rank-1 (top/second singular value, default 50).")
    ap.add_argument("--extract-s", type=float, default=1e-3,
                    help="with --extract-freeze, kink side-offset. Must be smaller than "
                         "the distance to the nearest OTHER kink, or the two sides span "
                         "deeper regions and the jump stops being rank-1. Shrink on deep "
                         "nets (e.g. 1e-4) if probes reject with low jump_ratio.")
    ap.add_argument("--extract-r", type=float, default=1e-4,
                    help="with --extract-freeze, affine-fit stencil radius (shrink "
                         "together with --extract-s).")
    ap.add_argument("--extract-tries", type=int, default=8,
                    help="with --extract-freeze, retry each neuron from this many seeds "
                         "before giving up (default 8).")
    ap.add_argument("--extract-base-scale", type=float, default=1.0,
                    help="with --extract-freeze, probe seed norm (distance from the "
                         "guess plane's foot). Small keeps the guessed kink in-window; "
                         "raise if the region near the origin is degenerate.")
    ap.add_argument("--cheat", action="store_true",
                    help="oracle diagnostic: force the population to a SINGLE "
                         "student (p=1) and let the blackbox itself join "
                         "disagreement query generation as the second "
                         "committee member, so queries are optimized to "
                         "separate the student from the truth (as if p=2). "
                         "Needs white-box gradient access to the teacher -- "
                         "an upper bound on query quality, NOT an honest "
                         "attack. Output files get a _cheat tag.")
    ap.add_argument("--cheat-pop", type=int, default=1, metavar="P",
                    help="(--cheat) committee size in cheat mode (default 1 = "
                         "the usual forced single student). P>1 keeps P "
                         "members; the disagreement queries maximize each "
                         "member's disagreement with the BLACKBOX only (no "
                         "member-member terms -- they're independent shots at "
                         "the truth, not a mutually-repelled committee). The "
                         "peel gate/refine runs on the best member; frozen "
                         "rows are pinned into all members.")
    ap.add_argument("--cheat-solo", action="store_true",
                    help="(--cheat --cheat-pop P>1, MLP only) SOLO committee: "
                         "each iter every member runs its OWN q/P-query "
                         "disagreement search against the blackbox alone "
                         "(p=1-style targeting -- no median dilution across "
                         "members' different stragglers) and trains ONLY on "
                         "its own slice of the buffer; its periodic LSQ head "
                         "solve also uses only its own rows. P parallel "
                         "independent attacks at the SAME total oracle budget "
                         "(q/iter). Best-member ranking/logging stays on the "
                         "full buffer; consensus column and peel machinery "
                         "unchanged. Cheaper per iter than the shared search "
                         "(2 nets x q/P rows per qg step instead of P+1 x q).")
    ap.add_argument("--fast-peel", action="store_true",
                    help="(--cheat --cheat-pop P>1 + --peel, MLP only) peel "
                         "one layer at a time on COMMITTEE CONSENSUS: when "
                         "the frontier hidden layer reaches full quorum "
                         "consensus (--fast's trigger), kink-refine the "
                         "consensus rows (quorum means) to exact, freeze them "
                         "into every member and continue -- with --peelrestart "
                         "the committee cold-restarts onto the next layer. No "
                         "single member has to clear the eps gate (each keeps "
                         "its own private stragglers; the quorum means are "
                         "refiner-basin quality long before any member's "
                         "whole-layer max is). The eps-gate trigger stays "
                         "active too; whichever fires first wins. Unlike "
                         "--fast, the run does NOT stop.")
    ap.add_argument("--cheat-peel-max", type=float, default=1e-2,
                    help="with --cheat + --peel (CNN): consensus can't form at "
                         "p=1, so the frontier layer peels (kink-refine + "
                         "freeze) once its oracle eps vs the teacher satisfies "
                         "max <= this AND mean <= --cheat-peel-mean (the "
                         "printed 'eps/layer (best)' numbers).")
    ap.add_argument("--cheat-peel-mean", type=float, default=1e-3,
                    help="mean-eps threshold of the cheat-mode peel gate "
                         "(see --cheat-peel-max).")
    ap.add_argument("--peeltry", type=int, nargs="?", const=5, default=0,
                    metavar="K",
                    help="(CNN) every K iters (bare flag: every 5) ATTEMPT the "
                         "kink refiner on the frontier layer directly -- no "
                         "eps/consensus gate, the refiner IS the gate. Solved "
                         "channels are cached across attempts; an attempt "
                         "early-aborts after 2 abstains. The layer peels "
                         "(freeze+reinit; composes with --peelrefresh) only "
                         "when EVERY channel refines exactly and the rows are "
                         "duplicate-free. Kink probing needs real-valued "
                         "outputs (incompatible with --hard).")
    ap.add_argument("--peel-warm", action="store_true",
                    help="(CNN peel-try) WARM partial/full freeze: keep the "
                         "committee's current trained weights (deeper layers + "
                         "unsolved frontier rows) across a freeze instead of "
                         "cold-reinitializing them; only the newly solved "
                         "channels are pinned. Unsolved frontier rows are still "
                         "re-rolled for flip escape. Stops the partial-freeze "
                         "reboot from discarding tail training every time a "
                         "channel is banked.")
    ap.add_argument("--peel-cold", action="store_true",
                    help="(CNN) force COLD reinit on partial freeze even under "
                         "--partial (which defaults to warm). Re-randomizes the "
                         "unsolved frontier rows + deeper layers on every banked "
                         "channel -- the old behavior; usually slower to converge.")
    ap.add_argument("--peel-angle-gate", type=float, default=12.0,
                    help="(MLP peel-try) max angle (deg) recovered-vs-guess to "
                         "accept a kink refine. Loose by design: the sweep is "
                         "exact regardless of guess angle; this only guards "
                         "against locking a neighbour's kink (jump_ratio + "
                         "runaway do the real gating). A tight 2deg rejects "
                         "correct recoveries from ~3deg guesses. Lower only if "
                         "neurons are poorly separated.")
    ap.add_argument("--peel-miss-abort", type=int, default=50, metavar="N",
                    help="(peel-try) abort an attempt after N CONSECUTIVE "
                         "unsolved-neuron misses (resets on each solve). High "
                         "(default 50) lets one attempt sweep all currently-"
                         "solvable neurons instead of bailing after the first "
                         "couple of hard ones. Lower it to cap per-attempt cost "
                         "on very wide layers.")
    ap.add_argument("--guessfree-after", type=int, default=0, metavar="N",
                    help="(CNN --fast-peel-partial) when the frontier layer's solved count has not grown for N "
                         "outer iterations, recover its unsolved channels GUESS-FREE from black-box kinks "
                         "(random-line scan, prefix-Jacobian mapback; guessfree.py) and inject them into every "
                         "member so the normal consensus -> refine -> pin path accepts them. Unblocks stragglers "
                         "GD never learns (one-sided channels) and frontiers the committee cannot guess (C5). 0 = off")
    ap.add_argument("--guessfree-lines", type=int, default=8,
                    help="random lines per guess-free attempt (~160k queries each on LeNet)")
    ap.add_argument("--query-via-prefix", action="store_true",
                    help="(MLP arch = a deeper layer's tail, e.g. 400,120,84,10 = LeNet C5->head) every query is "
                         "parametrised as phi(x) through the exact LeNet prefix (conv1+conv2), disagreement optimised "
                         "over pixels x: the attack only ever queries the REACHABLE part of the layer's input space")
    ap.add_argument("--query-via-mlp-prefix", default="",
                    help="(MLP arch = the tail of a deeper MLP) 'FULLDIMS:K' e.g. '3072,200,200,200,200,200,100:2': every query is "
                         "parametrised as prefix(x) through the first K layers of the cached full teacher (seed --teacher-seed-full, "
                         "default 0), disagreement optimised over x -- the real peel's query distribution, with the prefix exact")
    ap.add_argument("--teacher-seed-full", type=int, default=0)
    ap.add_argument("--loc-refine-l1", action="store_true",
                    help="(MLP peel) use the kink_solve solver for layer 1 too (the legacy layer-1 refiner rejects "
                         "11-15%% of correct consensus candidates; kink_solve solved 200/200 on the same record)")
    ap.add_argument("--mine-in-h", action="store_true",
                    help="(MLP peel, leaky) mine queries in the frontier's input space and invert them EXACTLY through "
                         "the pinned prefix -> the via-prefix attack becomes the direct attack on the tail")
    ap.add_argument("--cluster-eps", type=float, default=0.0,
                    help="consensus tolerance override (Cfg.cluster_eps, default 0.02); looser at deep frontiers where the "
                         "reachable input distribution is anisotropic and rows agree to ~0.995 cos but not 0.02")
    ap.add_argument("--standardize-h", action="store_true",
                    help="with --query-via-mlp-prefix: standardise the prefix output per coordinate (tail teacher re-expressed)")
    ap.add_argument("--query-map-init-only", action="store_true",
                    help="ablation with --query-via-*: reachable INIT (prefix of random raw) but feature-space ascent")
    ap.add_argument("--h-push", type=int, default=0,
                    help="push raw queries along random frontier-input directions through the exact prefix for N Adam steps "
                         "before disagreement mining (deep prefixes: balances the frontier units without inversion)")
    ap.add_argument("--qg-init-data", action="store_true",
                    help="mining base points = natural inputs of the teacher's dataset (+ jitter) instead of Gaussian noise")
    ap.add_argument("--qg-data-jitter", type=float, default=0.05)
    ap.add_argument("--qg-data-pool", type=int, default=0, help="with --qg-init-data: draw N x q images, keep the q with the largest committee disagreement")
    ap.add_argument("--masked-consensus", action="store_true",
                    help="deep-layer consensus on input columns in consensus only (end-to-end consensus becomes visible)")
    ap.add_argument("--orient-check", action="store_true",
                    help="(MLP partial peel) global orientation certificate on pinned+candidate rows (orient.py); corrects flipped signs")
    ap.add_argument("--resume-record", action="store_true",
                    help="(CNN --resume) take the pinned rows from the snapshot's fp64 record instead of re-solving them from the member")
    ap.add_argument("--residual-warm", action="store_true",
                    help="(CNN --resume) warm committee from the resumed member; only unsolved frontier channels + their outgoing columns train")
    ap.add_argument("--cluster-quorum", type=float, default=0.0, help="consensus quorum as a fraction of members (default 0.625 = 5/8)")
    ap.add_argument("--cluster-eps-l1", type=float, default=0.0, help="consensus eps for layer 1 only (deeper layers: --cluster-eps)")
    ap.add_argument("--h-push-lr", type=float, default=0.1)
    ap.add_argument("--h-push-mode", type=str, default="dir", choices=["dir", "unit", "rare", "oracle"],
                    help="dir: random frontier-input direction; unit: random student frontier unit pushed to a random sign")
    ap.add_argument("--query-map-abs", action="store_true",
                    help="ablation: students' input = |raw| (nonnegative-orthant query distribution, no prefix)")
    ap.add_argument("--h-push-radius", type=float, default=0.0, help="|x| cap for --h-push (0 = 1.1 x median initial |x|)")
    ap.add_argument("--mine-h-range", type=float, default=0.0,
                    help="(with --mine-in-h / --query-via-*) uniform U[-r,r] init of the raw/h queries instead of N(0, 0.5^2); "
                         "the direct tail attack used the variant's uniform init with range 1 in the teacher's gauge")
    ap.add_argument("--reroll-stale", type=int, default=0, help="partial peel: every K iterations re-randomise the frontier's free non-candidate rows")
    ap.add_argument("--hot-reroll", type=float, default=0.0,
                    help="(MLP partial peel) multiply the Adam step of rerolled, not-yet-pinned rows by this factor (e.g. 10)")
    ap.add_argument("--reroll-dups", action="store_true",
                    help="(MLP partial peel) re-randomise free frontier rows that duplicate a pinned row in most members")
    ap.add_argument("--reinit-lr", type=float, default=0.0,
                    help="(freeze-reinit peel, --fast-peel) lr for the reinitialised deeper layers after a layer freezes "
                         "(default: --lr). Use 1e-3 when resuming a converged committee at --lr 1e-4")
    ap.add_argument("--frontier-lr", type=float, default=0.0,
                    help="(MLP peel) learning rate held for the partially pinned layer's free/rerolled rows, independent "
                         "of the lr schedule (e.g. 1e-3 while --lr 1e-4 for the rest). Rerolled rows at the decayed lr "
                         "never re-find missing units; at 1e-3 they do")
    ap.add_argument("--repel", type=float, default=0.0,
                    help="(MLP peel) penalty weight on cos^2 between free rows of a partially pinned layer and its pinned "
                         "rows (anti-duplicate: rerolled units stop re-converging to copies of pinned rows). Try 1.0")
    ap.add_argument("--verify-cands", action="store_true",
                    help="(CNN --fast-peel-partial) gate every frontier consensus candidate with the global kink test "
                         "(black box must have a kink wherever the candidate hyperplane crosses zero at fresh points) "
                         "before the refiner; rejects the ~35%% of C5 guesses that the solver would otherwise 'solve' "
                         "to a wrong (F6) kink surface. ~500 queries per candidate")
    ap.add_argument("--peel-advance-frac", type=float, default=1.0, metavar="F",
                    help="(peel-try) advance the frontier to the next layer once "
                         "fraction F of it is solved, instead of requiring 100%%. "
                         "F<1 (e.g. 0.98) unblocks deeper layers when a few "
                         "neurons are permanently unsolvable (508/512). CAVEAT: "
                         "deeper layers are then refined with an imperfect prefix, "
                         "so recovery is approximate in the dims tied to the "
                         "stragglers (guards still reject gross corruption).")
    ap.add_argument("--peel-reroll", action="store_true",
                    help="(peel-try warm freeze) re-randomize still-unsolved "
                         "frontier rows on every partial freeze (flip escape). "
                         "OFF by default: it resets progressing neurons to "
                         "random every bank so they never accumulate training. "
                         "Use --peel-stuck for the escape on a real plateau.")
    ap.add_argument("--peel-stuck", type=int, default=0, metavar="K",
                    help="(CNN peel-try + --peel-warm) after K consecutive "
                         "attempts with no new frontier channel solved, "
                         "COLD-reinit the unfrozen params (fresh signs for the "
                         "stuck rows + fresh tail; solved channels stay pinned) "
                         "to escape the warm plateau. 0 = never. Try ~5-10 to "
                         "shake loose the last hard channels warm can't reach.")
    ap.add_argument("--peel-clip", type=float, default=0.0, metavar="NORM",
                    help="(peel-try) once any layer is pinned, clip each member's "
                         "training grad-norm to NORM (0=off). Prevents the free "
                         "params (last unsolved neuron + deeper layers) from "
                         "overshooting once frozen L1 concentrates the adversarial "
                         "cheat queries onto them -- the width>=512 divergence. "
                         "Try ~1.0. No effect until the first pin, and none on the "
                         "no-peel path.")
    ap.add_argument("--peelrefresh", action="store_true",
                    help="(CNN peel) when a layer freezes+reinits, also EMPTY "
                         "the sample buffer and RESTART the outer-iteration "
                         "clock: fresh warmstart, full --outer budget and a "
                         "fresh lr schedule for the collapsed deeper search "
                         "(old samples were mined against the pre-peel "
                         "student). Total queries = sum over phases.")
    ap.add_argument("--peelrestart", action="store_true",
                    help="(MLP + --cheat --peel) after a hidden layer is FULLY "
                         "peeled (and a deeper one remains), FULL COLD restart of "
                         "the deeper search: re-randomize the unsolved+deeper "
                         "weights (frozen rows pinned), flush the sample buffer, "
                         "and restart the lr/iter schedule. (No restart on the "
                         "last hidden layer -- its successor is the linear output, "
                         "solved by closed-form LSQ.)")
    ap.add_argument("--normalized-peel", action="store_true",
                    help="MLP: after a prefix freezes, train a whitened peeled tail and "
                         "optimize bounded reachable disagreement queries; replaces --peel-direct.")
    ap.add_argument("--peel-direct", action="store_true",
                    help="(leaky-ReLU MLP peel) optimize disagreement in local reachable "
                         "hidden-input charts after a prefix freezes. Students skip the "
                         "prefix; bounded raw teacher queries realize the hidden inputs. "
                         "Charts change each round; this is not arbitrary global tail access.")
    ap.add_argument("--restart-stuck", action="store_true",
                    help="(MLP + --cheat --partial) also peel-restart when the "
                         "frontier CAN'T be fully solved: fires on stagnation "
                         "(>=50%% of the frontier solved AND no new neurons frozen "
                         "for 20 iters) or on the final iter. Keeps the solved rows "
                         "pinned, cold-reinits only the unsolved+deeper, flushes the "
                         "buffer, restarts the iter clock. Capped at 5 restarts.")
    ap.add_argument("--lastlayer-every", type=int, default=0, metavar="K",
                    help="every K iters, closed-form ridge-LSQ solve the LINEAR "
                         "output head (variable projection) instead of leaving it to "
                         "SGD. Keeps the head BOUNDED + optimal so it can't diverge "
                         "in the cheat game -- essential on deep peels where a "
                         "runaway head (L_out max_eps -> 1e2+) stalls the tail "
                         "layers' training. Try 5.")
    ap.add_argument("--hard", action="store_true",
                    help="hard-label regime: the blackbox returns ONLY the "
                         "argmax class index (e.g. 0/1/2), never the logit "
                         "vector. The committee trains with cross-entropy on "
                         "those labels; members are ranked/gated by mean "
                         "cross-entropy instead of MAE. All eps metrics "
                         "quotient the label-preserving output family "
                         "W->sW+1u^T, b->sb+beta*1 (unlearnable from argmax; "
                         "hidden layers stay exact). Output files get a "
                         "_hard tag. --fast and --combine work (their "
                         "consensus mechanics are teacher-free; the endgame/"
                         "inject polish becomes a low-LR xent fine-tune). "
                         "Incompatible with what genuinely needs real-valued "
                         "outputs: --verify, --extract-freeze, the CNN "
                         "peel/--resume (kink probing), --solver-polish, and "
                         "variants with lastlayer/lbfgs/f64 solvers.")
    ap.add_argument("--hard-loss", choices=["xent", "boundary"], default="xent",
                    help="--hard training objective. xent: cross-entropy on the labels "
                         "(default). boundary: zero on every correctly classified point; "
                         "a misclassified point costs the squared first-order distance "
                         "its member's boundary must move, (f_j-f_y)^2/||grad_x(f_y-f_j)||^2 "
                         "for the runner-up class j. Invariant under the unidentifiable "
                         "head family (no logit-scale inflation); members are ranked by "
                         "mean boundary-distance violation instead of mean xent, and the "
                         "endgame/inject polish uses the same loss. Needs a double backward.")
    ap.add_argument("--hard-floor", type=float, default=1e-6,
                    help="--hard-loss boundary: floor on ||grad_x(f_y-f_j)|| (default 1e-6).")
    ap.add_argument("--tanh", action="store_true",
                    help="tanh hidden activations in BOTH the teacher and "
                         "the committee (trains + caches a separate tanh "
                         "teacher; output files get a _tanh tag). Alignment "
                         "canonicalizes the odd-symmetry polarity "
                         "tanh(-z) = -tanh(z) (sign flip, nothing absorbed). "
                         "No kinks: --verify / peel refiners are unavailable.")
    ap.add_argument("--tanhsolver", action="store_true",
                    help="smooth-net (tanh/sigmoid) joint jet refiner "
                         "(tanh_solve.refine): after training, refine EVERY "
                         "parameter of the delivered net by Gauss-Newton on "
                         "black-box values + finite-difference Jacobians. "
                         "Applied to the consensus when a --fast trigger ends "
                         "the run, else to the trained best net. Needs "
                         "--tanh or --sigmoid; the guess must be inside the "
                         "basin (~1e-2 on every layer).")
    ap.add_argument("--tanhsolver-points", type=int, default=512,
                    help="jet query points (default 512; 4*d oracle queries "
                         "each with the full Jacobian).")
    ap.add_argument("--tanhsolver-dirs", type=int, default=0,
                    help="directional derivatives per point instead of the "
                         "full Jacobian (0 = full; m>0 = m random unit "
                         "directions, 4*m queries per point).")
    ap.add_argument("--tanhsolver-iters", type=int, default=40,
                    help="max accepted Gauss-Newton/LM steps (default 40).")
    ap.add_argument("--tanhsolver-scales", default="0.5,1,2",
                    help="Gaussian input scales cycled over the jet points.")
    ap.add_argument("--tanhsolver-cg", type=int, default=200,
                    help="CG iteration cap per Gauss-Newton step (default 200, "
                         "enough for tanh nets; deep SIGMOID nets are far more "
                         "ill-conditioned -- raise to ~1000 at ~5x the per-step "
                         "cost, see tanh_solve.py).")
    ap.add_argument("--square", action="store_true",
                    help="f(z) = z^2 hidden activations in BOTH the teacher and "
                         "the committee (trains + caches a separate square "
                         "teacher; output files get a _square tag). Alignment "
                         "canonicalizes the full per-neuron gauge of an even, "
                         "2-homogeneous activation: unit-norm rows with the "
                         "SQUARED norm pushed into the next column, plus the "
                         "sign of the row (nothing changes in the next layer). "
                         "No kinks: --verify / peel refiners are unavailable; "
                         "--tanhsolver (jet refiner) is supported.")
    ap.add_argument("--act", default="", choices=["elu", "gelu", "silu", "mish", "softplus", "softsign", "celu", "selu", "hardswish", "squareplus", "abs", "pow3", "pow5", "sine", "softmax", "prelu", "glu", "swiglu", "nonleakyrelu"],
                    help="hidden activation of BOTH the teacher and the committee. "
                         "Alignment canonicalizes each activation's own gauge (align.py): "
                         "leaky-ReLU scale gauge for nonleakyrelu (plain ReLU); permutation only for elu/gelu/silu/mish/softplus/celu/selu/"
                         "hardswish/squareplus; odd sign flip for softsign; sign + linear "
                         "scale for abs; sign + k-th-power scale for pow3/pow5; odd flip + "
                         "bias mod pi for sine; layer-level shifts for softmax; positive "
                         "scale + global slope flip (|a|<=1) for prelu (forces --no-stacked, "
                         "no --tanhsolver); value-row scale/sign for glu/swiglu (plain "
                         "committee + eps + --tanhsolver only). Output files get a _<act> "
                         "tag. No kinks: no --verify / kink refiners.")
    ap.add_argument("--sigmoid", action="store_true",
                    help="sigmoid hidden activations in BOTH the teacher and "
                         "the committee (trains + caches a separate sigmoid "
                         "teacher; output files get a _sigmoid tag). "
                         "Alignment canonicalizes sigmoid polarity "
                         "(sigma(-z)=1-sigma(z)) instead of ReLU scaling; "
                         "--verify is unsupported (kink probing assumes "
                         "piecewise-linear nets).")
    ap.add_argument("--conv", default="",
                    help="CNN mode: space-separated conv specs 'in,out,k,stride"
                         "[,pad[,pool]]' e.g. '1,6,3,2 6,8,3,2'; or a preset: "
                         "'lenet' (LeNet-5, MNIST) / 'alexnet' (torchvision-exact "
                         "geometry @224, avg-pool sub, CIFAR-100 upsampled, 1000-way "
                         "split labels) / 'alexnet32' (CIFAR-scale variant). "
                         "A Linear head is appended. "
                         "Uses the population+consensus pipeline with conv channel "
                         "alignment (no peel/rank-cert/fast/expand).")
    ap.add_argument("--fc", default="",
                    help="CNN hidden FC widths between conv stack and output head, "
                         "comma-separated e.g. '84' (LeNet's F6). Empty = none.")
    ap.add_argument("--input-shape", default="1,28,28",
                    help="CNN input shape C,H,W (default 1,28,28 = MNIST).")
    ap.add_argument("--out-dim", type=int, default=10, help="CNN output dim.")
    ap.add_argument("--cnn-act", default="relu",
                    choices=["relu", "leaky_relu", "tanh", "square", "elu", "gelu", "silu", "mish", "softplus", "softsign", "celu", "selu", "hardswish", "squareplus", "abs", "pow3", "pow5", "sine", "softmax", "prelu", "glu", "swiglu"],
                    help="CNN hidden activation (default relu). Use 'leaky_relu' "
                         "to avoid dying units; the kink refiner supports it.")
    ap.add_argument("--xspace-refine", action="store_true",
                    help="recover each peeled layer from its x-space kink normal "
                         "+ prefix-Jacobian mapback (the deep multi-region path) "
                         "instead of the h-space seal. DEPTH-FLAT: precision set "
                         "by the prefix's condition number, not the seal's ~100x/"
                         "layer amplification, so deep layers reach ~fp64 (e-12) "
                         "like L1 instead of degrading (L2~e-6, worse deeper). "
                         "Slower (per-kink Jacobian + multi-base x-space search); "
                         "pair with --f64. MLP peel only.")
    ap.add_argument("--loc-refine", action="store_true",
                    help="legacy tracked kink-point recovery for deep MLP layers; "
                         "first layer keeps the original refiner. Pair with --f64.")
    ap.add_argument("--design-refine", action="store_true",
                    help="use multiscale information-directed kink sampling for "
                         "every peeled MLP hidden layer, including the first. "
                         "Uses the trained/consensus guesses; implies --f64. "
                         "More oracle queries than the legacy refiner.")
    ap.add_argument("--f64", action="store_true",
                    help="do the LAYER EXTRACTION in float64 while training/"
                         "search stay fp32 (fast). The kink refiner already "
                         "computes in fp64; this also STORES each peeled "
                         "layer's exact weights in fp64 (instead of truncating "
                         "them back into the fp32 student) and seals the next "
                         "layer through that fp64 prefix, so deep peels don't "
                         "amplify a ~6e-8 fp32 prefix ~100x per layer. Training "
                         "is untouched (fp32). MLP peel only.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.cheat_peel:
        if args.conv or args.teacher_full:
            ap.error('--cheat-peel is MLP-only and replaces --teacher-full')
        full = args.arch.split(',')
        if not 1 <= args.cheat_peel < len(full)-1:
            ap.error('--cheat-peel N must leave at least one weight layer')
        args.teacher_full = args.arch
        args.teacher_drop = args.cheat_peel
        args.arch = ','.join(full[args.cheat_peel:])
        args.tag = (args.tag + f'_cheatpeel{args.cheat_peel}').lstrip('_')
        print(f"[cheat-peel] DIAGNOSTIC: direct teacher tail after removing {args.cheat_peel} "
              "layer(s) from the start; this is privileged access, not simulated peeling", flush=True)

    if args.normalized_peel:
        if args.conv or args.sigmoid or args.tanh or args.square or args.act or args.hard or args.peel_direct:
            ap.error('--normalized-peel requires a leaky-ReLU MLP; omit --peel-direct')
    if sum(bool(f) for f in (args.sigmoid, args.tanh, args.square, args.act)) > 1:
        ap.error("--sigmoid, --tanh, --square and --act are mutually exclusive")
    if args.tanhsolver:
        if not (args.sigmoid or args.tanh or args.square or args.act):
            ap.error("--tanhsolver refines smooth nets only; add --tanh, --sigmoid, --square or --act")
        if args.conv or args.hard:
            ap.error("--tanhsolver needs a real-valued MLP black box (no --conv/--hard)")
    if args.act in ("softmax", "prelu", "glu", "swiglu") and args.ensemble:
        ap.error(f"--ensemble (per-unit OMP merge) is not defined for --act {args.act}")
    if args.act in ("glu", "swiglu"):
        for on, name in [(args.layer_consensus, "--layer-consensus"), (args.reverse_peel, "--reverse-peel"),
                         (args.peel, "--peel"), (args.masked_consensus, "--masked-consensus"),
                         (args.fast_peel_partial, "--fast-peel-partial"), (args.partial, "--partial")]:
            if on:
                ap.error(f"{name} indexes hidden rows as units; not ported to gated units (--act {args.act}), "
                         "where a unit owns a value row AND a gate row. Supported: plain committee, consensus, "
                         "--align-inverse eps, --tanhsolver")
    if args.act == "prelu":
        if args.tanhsolver:
            ap.error("--tanhsolver does not support --act prelu (the slope is not a layer parameter)")
        if not args.no_stacked:
            print("[act] prelu: the stacked trainer only trains layer weights; forcing --no-stacked so the slope is learned", flush=True)
            args.no_stacked = True
    if args.design_refine:
        if args.conv or args.sigmoid or args.tanh or args.square or args.act or args.hard:
            ap.error('--design-refine requires a real-valued ReLU/leaky-ReLU MLP')
        if args.loc_refine or args.xspace_refine:
            ap.error('--design-refine selects its own refiner; omit --loc-refine and --xspace-refine')
    if args.hard_loss != "xent" and not args.hard:
        ap.error("--hard-loss needs --hard")
    if args.hard:
        for on, name in [(args.solver_polish, "--solver-polish"),
                         (args.verify, "--verify"),
                         (args.peeltry, "--peeltry"),
                         (args.extract_freeze, "--extract-freeze")]:
            if on:
                ap.error(f"--hard: {name} needs real-valued teacher outputs "
                         "(logits); the hard-label blackbox only returns class "
                         "indices")
        if args.conv and (args.peel or args.freeze_reinit or args.resume):
            ap.error("--hard: the CNN peel/--resume path refines layers with "
                     "the kink prober (real-valued outputs)")
    if args.conv:
        return run_cnn(args)
    smooth = args.sigmoid or args.tanh or args.square or bool(args.act)
    if smooth and args.verify:
        ap.error("--verify probes ReLU kink hyperplanes; incompatible "
                 "with --sigmoid / --tanh / --square / --act")

    dims = [int(x) for x in args.arch.split(",")]
    device = args.device
    torch.manual_seed(args.teacher_seed)

    act = ("sigmoid" if args.sigmoid else "tanh" if args.tanh
           else "square" if args.square else args.act if args.act else "leaky_relu")
    if args.teacher_full:
        # blackbox = a bigger trained teacher with its first `drop` layers removed;
        # queried DIRECTLY in the sub-net's input space (no L1, no inversion, no seal).
        full_dims = [int(x) for x in args.teacher_full.split(",")]
        drop = args.teacher_drop
        sub_dims = full_dims[drop:]
        if sub_dims != list(dims):
            raise SystemExit(f"--teacher-full {full_dims} minus {drop} leading "
                             f"layer(s) = {sub_dims}, must equal --arch {list(dims)}")
        print(f"[setup] blackbox = teacher{full_dims} with first {drop} layer(s) "
              f"removed -> sub-net {sub_dims}; act={act} device={device}", flush=True)
        full_teacher = make_teacher(full_dims, epochs=args.teacher_epochs,
                                    seed=args.teacher_seed, device=device,
                                    verbose=smooth, act=act)
        teacher = MLP(sub_dims, act=act).to(device)
        with torch.no_grad():
            for j in range(len(sub_dims) - 1):
                teacher.layers[j].weight.copy_(full_teacher.layers[drop + j].weight)
                teacher.layers[j].bias.copy_(full_teacher.layers[drop + j].bias)
        # eval points in the sub-net's natural input distribution: push gaussian x
        # through the dropped prefix (no dataset). Used only for the agree diagnostic.
        with torch.no_grad():
            h = torch.randn(2000, full_dims[0], device=device)
            for j in range(drop):
                h = full_teacher.act(full_teacher.layers[j](h))
            eval_pts = h
    else:
        print(f"[setup] teacher {dims} epochs={args.teacher_epochs} "
              f"act={act} device={device}", flush=True)
        teacher = make_teacher(dims, epochs=args.teacher_epochs,
                               seed=args.teacher_seed, device=device,
                               verbose=smooth, act=act)
        (_, _), (xte, _) = load_data(dims, device)
        if xte.shape[1] != dims[0]:
            # no dataset matches this input width (e.g. a truncated network queried at an inner
            # layer): evaluate on Gaussian points of the right width instead
            print(f"[setup] no dataset with {dims[0]} inputs; eval points = N(0,0.5^2)^{dims[0]}", flush=True)
            xte = torch.randn(2000, dims[0], device=device) * 0.5
        eval_pts = xte[:2000]
        _query_map = None
        if args.query_via_mlp_prefix:
            _fd, _K = args.query_via_mlp_prefix.split(":"); _fd = [int(v) for v in _fd.split(",")]; _K = int(_K)
            _full = make_teacher(_fd, epochs=args.teacher_epochs, seed=args.teacher_seed_full, device=device, verbose=False, act=act)
            for _p in _full.parameters():
                _p.requires_grad_(False)
            def _query_map_raw(raw, _net=_full, _K=_K):
                h = raw
                for i in range(_K):
                    h = _net.act(_net.layers[i](h))
                return h
            _query_map = _query_map_raw
            if getattr(args, "standardize_h", False):
                # --standardize-h: the students see z = (h - mu) / sigma with per-coordinate stats of the prefix output
                # under random inputs (attacker-legal: the prefix is known). The black box is the tail teacher, so fold
                # the affine map into ITS first layer: W' = W diag(sigma), b' = b + W mu  (recovered W' -> W = W' / sigma).
                # Under reachable queries the raw h7 coordinates have a 65x spread of std (0.11 .. 7.2) and 14/200 are
                # positive <5% of the time; the direct attack's inputs are isotropic. This removes the scale part.
                with torch.no_grad():
                    _hs = torch.cat([_query_map_raw(torch.rand(8192, _fd[0], device=device) * 2 - 1) for _ in range(4)])
                    _mu = _hs.mean(0); _sig = _hs.std(0).clamp_min(1e-3)
                    _W = teacher.layers[0].weight; _b = teacher.layers[0].bias
                    _b.add_(_W @ _mu); _W.mul_(_sig[None, :])
                def _query_map(raw, _f=_query_map_raw, _mu=_mu, _sig=_sig):
                    return (_f(raw) - _mu) / _sig
                print(f"[setup] --standardize-h: prefix output standardised per coordinate (sigma min/median/max "
                      f"{float(_sig.min()):.3f}/{float(_sig.median()):.3f}/{float(_sig.max()):.3f}); tail teacher's first layer re-expressed in z", flush=True)
            with torch.no_grad():
                eval_pts = _query_map(torch.rand(2000, _fd[0], device=device) * 2 - 1)
            overrides_extra = {"query_raw_dim": _fd[0]}
            print(f"[setup] --query-via-mlp-prefix: queries = first {_K} layers of the {_fd} teacher applied to x; disagreement optimised over x", flush=True)
        if args.query_via_prefix:
            _lenet = make_teacher_cnn((1, 28, 28), [(1, 6, 5, 1, 2, 2), (6, 16, 5, 1, 0, 2), (16, 120, 5, 1, 0, 0)], (84,), 10,
                                      epochs=25, seed=0, device=device, act="leaky_relu")
            for _p in _lenet.parameters():
                _p.requires_grad_(False)
            def _query_map(raw, _net=_lenet):
                x = raw.view(raw.shape[0], 1, 28, 28)
                for i in range(2):
                    x = torch.nn.functional.avg_pool2d(_net.act(_net.layers[i](x)), 2)
                return x.flatten(1)
            with torch.no_grad():
                eval_pts = _query_map(torch.randn(2000, 784, device=device) * 0.5)
            print("[setup] --query-via-prefix: queries = phi(x) through the exact LeNet conv1+conv2 (reachable set only); "
                  "disagreement optimised over pixels", flush=True)

    overrides = dict(VARIANTS[args.variant])
    if args.window is not None:
        overrides["window"] = args.window
    if args.expand != 1.0:
        overrides["expand"] = args.expand
    if args.qg_chunk:
        overrides["qg_chunk"] = args.qg_chunk
    if args.qg_steps is not None:
        overrides["qg_steps"] = args.qg_steps
    if args.inverse:
        overrides["inverse"] = True
        for k in ("inv_lam", "inv_inner", "inv_lr", "inv_radius", "inv_agg",
                  "inv_reach_w", "inv_y_noise", "inv_step_max"):
            v = getattr(args, k)
            if v is not None:
                overrides[k] = v
        print("[inverse] query generation = inverse disagreement (targets y optimized, "
              "teacher queried at each member's inverse x_i(y; x0))", flush=True)
    if args.reverse_peel:
        overrides["reverse_peel"] = True
        if args.reverse_eps is not None: overrides["reverse_eps"] = args.reverse_eps
        if args.reverse_thresh is not None: overrides["reverse_thresh"] = args.reverse_thresh
        if args.reverse_force: overrides["reverse_force"] = True
        print("[reverse-peel] head-first peel: head consensus (column frame) -> exact kink-jump refine "
              "-> pin; then consensus-pin downwards", flush=True)
    if args.layer_consensus:
        overrides["layer_consensus"] = True
    if args.resume_pop:
        overrides["resume_pop"] = args.resume_pop
        overrides["warmstart_iters"] = 0
        overrides["target_mode"] = args.target_mode
        overrides["target_lam"] = args.target_lam
        overrides["target_frac"] = args.target_frac
        overrides["target_act_w"] = args.target_act_w
        overrides["target_reinit"] = args.target_reinit
        overrides["target_consensus"] = args.target_consensus
        overrides["target_value"] = args.target_value
        overrides["target_units"] = args.target_units
    if args.targetdeep:
        overrides["targetdeep"] = True
    if args.balance is not None:
        overrides["balance"] = args.balance
    if args.guessfree_after:
        overrides["guessfree_after"] = args.guessfree_after
        overrides["guessfree_lines"] = args.guessfree_lines
    if args.verify_cands:
        overrides["verify_cands"] = True
    if args.repel:
        overrides["repel"] = args.repel
    if args.frontier_lr:
        overrides["frontier_lr"] = args.frontier_lr
    if getattr(args, "reinit_lr", 0.0):
        overrides["reinit_lr"] = args.reinit_lr
    if getattr(args, "reroll_dups", False):
        overrides["reroll_dups"] = True
    if getattr(args, "mine_in_h", False):
        overrides["mine_in_h"] = True
    if getattr(args, "hot_reroll", 0.0):
        overrides["hot_reroll"] = args.hot_reroll
    if getattr(args, "reroll_stale", 0):
        overrides["reroll_stale"] = args.reroll_stale
    if getattr(args, "mine_h_range", 0.0):
        overrides["query_raw_range"] = args.mine_h_range
    if getattr(args, "query_map_init_only", False):
        overrides["query_map_init_only"] = True
    if getattr(args, "cluster_eps", 0.0):
        overrides["cluster_eps"] = args.cluster_eps
    if getattr(args, "cluster_eps_l1", 0.0):
        overrides["cluster_eps_l1"] = args.cluster_eps_l1
    if getattr(args, "cluster_quorum", 0.0):
        overrides["cluster_quorum"] = args.cluster_quorum
    if getattr(args, "residual_warm", False):
        overrides["residual_warm"] = True
    if getattr(args, "orient_check", False):
        overrides["orient_check"] = True
    if getattr(args, "masked_consensus", False):
        overrides["masked_consensus"] = True
    if getattr(args, "qg_init_data", False):
        _dd = ([int(v) for v in args.query_via_mlp_prefix.split(":")[0].split(",")] if getattr(args, "query_via_mlp_prefix", "")
               else (dims if "dims" in dir() else [int(torch.tensor(input_shape).prod()), out_dim]))   # CNN path: MNIST by (784, 10)
        (_xtr_d, _), (_xte_d, _) = load_data(_dd, device)
        overrides["qg_init"] = "data"; overrides["query_data"] = torch.cat([_xtr_d, _xte_d]).to(device); overrides["query_data_jitter"] = args.qg_data_jitter; overrides["query_data_pool"] = args.qg_data_pool
        print(f"[setup] --qg-init-data: {overrides['query_data'].shape[0]} natural inputs as mining base points (jitter {args.qg_data_jitter:g})", flush=True)
    if getattr(args, "query_map_abs", False):
        overrides["query_map"] = (lambda raw: raw.abs()); overrides["query_raw_dim"] = dims[0]
    if getattr(args, "h_push", 0):
        overrides["h_push_steps"] = args.h_push; overrides["h_push_lr"] = args.h_push_lr; overrides["h_push_radius"] = args.h_push_radius
        overrides["h_push_mode"] = args.h_push_mode
        if "_query_map" in dir() and _query_map is not None:
            overrides["h_push_fn"] = _query_map
        if args.h_push_mode == "oracle":
            overrides["h_push_oracle"] = teacher.layers[0]          # DIAGNOSTIC upper bound: teacher frontier units
        print(f"[setup] --h-push {args.h_push} mode {args.h_push_mode}: prefix fn set = {'h_push_fn' in overrides}", flush=True)
    if args.loc_refine_l1:
        overrides["loc_refine_l1"] = True
    if (getattr(args, "query_via_prefix", False) or getattr(args, "query_via_mlp_prefix", "")) and "_query_map" in dir() and _query_map is not None:
        overrides["query_map"] = _query_map
        overrides["query_raw_dim"] = 784 if args.query_via_prefix else int(args.query_via_mlp_prefix.split(":")[0].split(",")[0])
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.qg_tf32:
        overrides["qg_tf32"] = True
    if args.qg_chunk_gb is not None:
        overrides["qg_chunk_gb"] = args.qg_chunk_gb
    if args.qg_block_gb is not None:
        overrides["qg_block_gb"] = args.qg_block_gb
    if args.qg_host_gb is not None:
        overrides["qg_host_gb"] = args.qg_host_gb
    if args.eval_loss:
        overrides["eval_loss"] = True
    if args.cpu_gather:
        overrides["gpu_seg"] = False
    if args.disk is not None:
        overrides["disk"] = args.disk
    if args.disk_ram_gb is not None:
        overrides["disk_ram_gb"] = args.disk_ram_gb
    if args.disk_seg_gb is not None:
        overrides["disk_seg_gb"] = args.disk_seg_gb
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.balance_mode is not None:
        overrides["balance_mode"] = args.balance_mode
    if args.balance_target is not None:
        overrides["balance_target"] = args.balance_target
    if args.balance_rownorm is not None:
        overrides["balance_rownorm"] = args.balance_rownorm
    if args.balance_dirlr is not None:
        overrides["balance_dirlr"] = args.balance_dirlr
    if args.balance_dirlr_decay is not None:
        overrides["balance_dirlr_decay"] = args.balance_dirlr_decay
    if args.balance_dirlr_head is not None:
        overrides["balance_dirlr_head"] = bool(args.balance_dirlr_head)
    if args.balance_refresh is not None:
        overrides["balance_refresh"] = args.balance_refresh
    if args.no_stacked:
        overrides["stacked"] = False
    if args.warmstart is not None:
        overrides["warmstart_iters"] = args.warmstart
    _ai = (args.inverse or args.reverse_peel) if args.align_inverse is None else args.align_inverse
    if _ai:
        overrides["align_inverse"] = True
        print("[align-inverse] eps metric = activation-matched per-layer pairing "
              "(param_errors_actmatch)", flush=True)
    elif args.expand != 1.0:
        overrides["qg_chunk"] = 8192   # auto: wide students OOM the full-batch
        print("[expand] auto --qg-chunk 8192 (query-gen would OOM at full batch; "
              "override with --qg-chunk).", flush=True)
    if args.factor_tail:
        overrides["factor_tail"] = True
        overrides["factor_inner"] = args.factor_inner
        overrides["factor_lr_mult"] = args.factor_lr
        overrides["factor_mode"] = args.factor_mode
        overrides["factor_rank"] = args.factor_rank
        overrides["factor_layers"] = args.factor_layers
    if args.combine:
        overrides["combine"] = True
    if args.verify:
        overrides["verify"] = True
        overrides["verify_refine"] = args.verify_refine
        overrides["verify_max"] = args.verify_max
        overrides["verify_eps_offset"] = args.verify_eps_offset
        overrides["verify_eps_angle"] = args.verify_eps_angle
        overrides["verify_k"] = args.verify_k
    if args.fast_peel:                         # implies --peel with a 100% quorum trigger
        overrides["freeze_reinit"] = True
        overrides["freeze_thresh"] = 1.0
        overrides["freeze_precision"] = 0.0
        overrides["fast_peel"] = True
    if args.freeze_reinit or args.peel:
        overrides["freeze_reinit"] = True
        overrides["freeze_thresh"] = args.freeze_thresh
        overrides["freeze_precision"] = args.freeze_precision
    if args.peelrestart:
        overrides["peel_restart"] = True
    if args.f64:
        overrides["f64"] = True
    if args.xspace_refine:
        overrides["xspace_refine"] = True
    if args.fast_peel_partial:                 # MLP: per-neuron consensus peel, pinned in place
        overrides["fast_peel_partial"] = True
        overrides["peel_angle_gate"] = args.peel_angle_gate
    if args.retry:
        overrides["retry"] = args.retry
        print(f"[peel] --retry {args.retry}: reinit + retry when a budget ends with the "
              "frontier layer not fully peeled", flush=True)
        print("[peel] --fast-peel-partial: per-neuron consensus peel, pinned in place "
              "across the committee; layer advances when fully solved", flush=True)
    if args.loc_refine:
        overrides["loc_refine"] = True
    if args.design_refine:
        overrides["design_refine"] = True
        overrides["f64"] = True
        print("[peel] --design-refine: information-directed sampling for every hidden layer; "
              "fp64 frozen prefix enabled", flush=True)
    if args.normalized_peel:
        overrides["normalized_peel"] = True
        print("[peel] --normalized-peel: whitened tail Adam + bounded reachable query search after prefix freeze",flush=True)
    if args.peel_direct:
        overrides["peel_seal"] = True
    if args.restart_stuck:
        overrides["restart_stuck"] = True
        if not args.partial:
            # restart-stuck restarts AROUND the partial-solved rows -- without
            # --partial nothing ever banks and the trigger can never arm
            overrides["partial"] = True
            overrides["peel_angle_gate"] = args.peel_angle_gate
            print("[restart-stuck] implies --partial (the restart keeps "
                  "partial-solved rows pinned); enabling it", flush=True)
    if args.lastlayer_every:
        overrides["lastlayer_every"] = args.lastlayer_every
    if args.peeltry:
        overrides["peel_try"] = args.peeltry
    if args.peel_warm:
        overrides["peel_warm"] = True
    if args.peel_stuck:
        overrides["peel_stuck"] = args.peel_stuck
    if args.peeltry:
        overrides["peel_angle_gate"] = args.peel_angle_gate
        overrides["peel_miss_abort"] = args.peel_miss_abort
        overrides["peel_advance_frac"] = args.peel_advance_frac
        overrides["peel_clip"] = args.peel_clip
    if args.peel_reroll:
        overrides["peel_reroll"] = True
    if args.extract_freeze:
        overrides["extract_freeze"] = True
        overrides["extract_max_angle"] = args.extract_max_angle
        overrides["extract_jump_ratio"] = args.extract_jump_ratio
        overrides["extract_s"] = args.extract_s
        overrides["extract_r"] = args.extract_r
        overrides["extract_tries"] = args.extract_tries
        overrides["extract_base_scale"] = args.extract_base_scale
    if args.solver_polish:
        overrides["solver_polish"] = True
    if args.solverwindow is not None:
        overrides["solverwindow"] = args.solverwindow
    if args.qg_lr is not None:
        overrides["qg_lr"] = args.qg_lr
    if args.query_box is not None:
        overrides["query_box"] = args.query_box
    if args.consensus_threshold is not None:
        overrides["cluster_eps"] = args.consensus_threshold
    if args.stop_layer is not None:
        overrides["stop_layer"] = args.stop_layer
    elif args.stop_outer:
        overrides["stop_layer"] = 0        # outermost hidden layer
    if args.verbose:
        overrides["log_every"] = 1        # print the per-iteration block every iter
    if args.log_every is not None:
        overrides["log_every"] = args.log_every
    if args.hard:
        overrides["hard"] = True
        overrides["hard_loss"] = args.hard_loss
        overrides["hard_floor"] = args.hard_floor
    if args.cheat:
        overrides["cheat"] = True
        overrides["cheat_peel_max"] = args.cheat_peel_max
        overrides["cheat_peel_mean"] = args.cheat_peel_mean
        if args.cheat_pop > 1:
            args.p = args.cheat_pop
            overrides["cheat_bb_ref"] = True
            if args.cheat_solo:
                overrides["cheat_solo"] = True
                print(f"[cheat] population p={args.p} (--cheat-pop --cheat-solo); "
                      "each member gets its OWN q/p disagreement queries vs the "
                      "blackbox and trains only on them", flush=True)
            else:
                print(f"[cheat] population p={args.p} (--cheat-pop); disagreement "
                      "runs on each (member, blackbox) pair -- no member-member "
                      "terms", flush=True)
        else:
            if args.cheat_solo:
                print("[cheat] --cheat-solo ignored (needs --cheat-pop > 1)",
                      flush=True)
            if args.p != 1:
                print(f"[cheat] forcing population p=1 (was {args.p}); "
                      "disagreement runs on the (student, blackbox) pair",
                      flush=True)
            args.p = 1
        if args.fast_peel:
            if args.cheat_pop > 1:
                overrides["fast_peel"] = True
                print("[cheat] --fast-peel: the frontier layer refines+peels "
                      "on full committee consensus (guesses = quorum means)",
                      flush=True)
            else:
                print("[cheat] --fast-peel ignored (needs --cheat-pop > 1)",
                      flush=True)
    if args.partial:
        overrides["partial"] = True
        overrides["peel_angle_gate"] = args.peel_angle_gate
    if args.ensemble or args.ensemble_boost:
        if args.ensemble_boost:
            # boost: --ensemble N = number of RESIDUAL STAGES to train (the
            # cascade length; 0 = until --outer runs out). Population is NOT
            # touched: under --cheat it stays p=1 -- ONE student per stage,
            # residual cascade, (cascade+student, blackbox) disagreement.
            overrides["boost_stages_max"] = args.ensemble
        else:
            if args.cheat:
                print(f"[ensemble] overriding cheat's p=1: population "
                      f"p={args.ensemble} (disagreement runs over all members "
                      "+ blackbox)", flush=True)
            args.p = args.ensemble
        overrides["ensemble_every"] = args.ensemble_every
        overrides["ensemble_samples"] = args.ensemble_samples
        overrides["ensemble_boost"] = args.ensemble_boost
        if args.ensemble_boost and args.query_box is None:
            overrides["query_box"] = 1.0
            print("[ensemble-boost] query search bounded to [-1,1]^d "
                  "(--query-box to override): unbounded disagreement chases "
                  "the cascade's off-manifold excursions and diverges",
                  flush=True)
    overrides["act"] = act
    tag = f"_{args.tag}" if args.tag else ""
    if args.sigmoid:
        tag = "_sigmoid" + tag
    if args.tanh:
        tag = "_tanh" + tag
    if args.square:
        tag = "_square" + tag
    if args.act:
        tag = f"_{args.act}" + tag
    if args.hard:
        tag = "_hard" + tag
    if args.cheat:
        tag = "_cheat" + tag
    arch_tag = "x".join(map(str, dims))
    recon_dir = os.path.join(os.path.dirname(__file__), "recon")
    fast_dump = os.path.join(
        recon_dir, f"_fast__{args.variant}{tag}__{arch_tag}__s{args.seed}.pt")
    if args.fast:
        os.makedirs(recon_dir, exist_ok=True)
        overrides["stop_on_consensus"] = True
        overrides["dump_path"] = fast_dump
    if args.pop_save_every > 0:
        os.makedirs(recon_dir, exist_ok=True)
        overrides["pop_save_every"] = args.pop_save_every
        overrides["pop_save_path"] = os.path.join(
            recon_dir, f"_pop__{args.variant}{tag}__{arch_tag}__s{args.seed}.pt")
    cfg = Cfg(p=args.p, q=args.q, outer=args.outer, **overrides)
    print(f"[run] {args.variant} seed={args.seed} params={count_params(teacher)} "
          f"budget={cfg.outer * cfg.q} queries"
          + (" [--fast]" if args.fast else ""), flush=True)
    t0 = time.time()
    save_recon = None
    if args.save_recon and not args.fast:
        os.makedirs(recon_dir, exist_ok=True)
        save_recon = os.path.join(
            recon_dir, f"{args.variant}{tag}__{arch_tag}__s{args.seed}.pt")
    best, log, final = reconstruct(teacher, dims, cfg, device, eval_pts,
                                   seed=args.seed, save_recon=save_recon)

    # --- phase summary: (name, mean_eps, max_eps, wall_s since start) after
    # the SGD/training phase, the MSE->MAE polish, and the tanhsolver ---
    phases = []

    def _phase(name, net):
        e = param_errors(net, teacher, hard=args.hard)
        phases.append((name, sum(e["mean_eps_per_matrix"]) / len(e["mean_eps_per_matrix"]),
                       e["max_eps"], round(time.time() - t0, 1)))

    _phase("SGD (trained best)", best)

    if args.fast and os.path.exists(fast_dump):
        # reconstruct stopped + dumped at the first consensus; build it and run
        # the staged MSE->MAE solve on the queries collected so far.
        ck = torch.load(fast_dump, map_location="cpu", weights_only=False)
        # PERSIST the population + consensus guess (lean: no X/Y) to a stable path
        # before the temp dump is cleaned up, so the exact/peel solver can load them.
        consensus_path = os.path.join(
            recon_dir, f"{args.variant}{tag}__{arch_tag}__s{args.seed}_consensus.pt")
        torch.save({"dims": ck["dims"], "act": ck["act"], "iter": ck["iter"],
                    "quorum_ratio": ck.get("quorum_ratio"),
                    "pop_states": ck["pop_states"],
                    "consensus_state": ck.get("consensus_state")}, consensus_path)
        print(f"[fast] saved population ({len(ck['pop_states'])} members) + "
              f"consensus guess -> {consensus_path}", flush=True)
        # queries stay on CPU (the full matrix can dwarf GPU memory at large
        # input dims); solver_polish_ streams them to the GPU in chunks.
        Xf, Yf = ck["X"], ck["Y"]
        n_total = len(Xf)
        # solver window (the single --solverwindow knob): restrict the solve to the
        # last cfg.solverwindow outer iters of queries (the most-recent tail,
        # Xf[-N*q:], since queries are appended chronologically). 0 = full set.
        if cfg.solverwindow and cfg.solverwindow > 0:
            keep = cfg.solverwindow * cfg.q
            if keep < n_total:
                Xf, Yf = Xf[-keep:], Yf[-keep:]
                print(f"[fast] solver window: last {cfg.solverwindow} iters "
                      f"= {len(Xf)}/{n_total} most-recent queries", flush=True)
        pop = []
        for s in ck["pop_states"]:
            m = MLP(dims, act=act).to(device); m.load_state_dict(s); pop.append(m)
        cons = build_consensus(pop, dims, eps=cfg.cluster_eps,
                               quorum_ratio=cfg.cluster_quorum)
        if cons is None and ck.get("consensus_state") is not None:
            # layer-stop: whole-net consensus is None, but the dump saved the
            # partial consensus (the stopped layer solved). Use it.
            cons = MLP(dims, act=act).to(device)
            cons.load_state_dict(ck["consensus_state"])
            print(f"[fast] whole-net consensus n/a; using saved partial consensus "
                  f"(stop_layer={ck.get('stop_layer')})", flush=True)
        if cons is None:
            print("[fast] no consensus formed; keeping trained best.", flush=True)
        else:
            cons = cons.to(device)
            print(f"[fast] consensus at iter {ck['iter']} ({len(Xf)} queries): "
                  f"max_eps {param_errors(cons, teacher, hard=args.hard)['max_eps']:.3e}"
                  " -> " + ("xent polish..." if args.hard
                            else "staged MSE->MAE solve..."), flush=True)
            _phase(f"SGD (consensus @ iter {ck['iter']})", cons)
            if args.hard:
                # hard labels: no regression solve exists; low-LR xent
                # fine-tune removes the consensus assembly artifact instead
                cons = polish_consensus(cons, Xf, Yf, lr=1e-4, steps=400,
                                        hard=True, hard_loss=args.hard_loss,
                                        hard_floor=args.hard_floor)
                _phase("xent polish", cons)
            else:
                solver_polish_(cons, Xf, Yf, mse_steps=40, mae_steps=40,
                               verbose=False, tag=" fast")
                _phase("MSE->MAE polish", cons)
            if (args.design_refine or args.loc_refine) and not args.hard:
                # --design-refine / --loc-refine under --fast: the whole-net
                # consensus is a ~1e-2 guess of every layer. Refine the hidden
                # layers one at a time with the kink solver (forward-only; each
                # refined layer is the exact prefix of the next), then solve the
                # linear output layer in closed form on the collected queries.
                import method as _m
                cons = cons.double()
                Lh = len(dims) - 2
                _q_ref = 0; _t_ref = time.time()
                for l in range(Lh):
                    Wr, br, rmask, _nq = _m._mlp_refine_layer(
                        teacher, cons, l, device, act, angle_gate=12.0,
                        loc=args.loc_refine, design=args.design_refine)
                    _q_ref += _nq
                    idx = rmask.nonzero(as_tuple=True)[0]
                    with torch.no_grad():
                        if len(idx):                      # scale-match unit [w|b] rows
                            Lf = cons.layers[l]
                            gw = Lf.weight.data[idx]; gb = Lf.bias.data[idx]
                            uw = Wr[idx].double(); ub = br[idx].double()
                            proj = (gw * uw).sum(1) + gb * ub
                            gn = (gw.pow(2).sum(1) + gb.pow(2)).sqrt()
                            cs = torch.where(proj > 1e-6 * gn, proj, gn).clamp_min(1e-8)
                            Lf.weight.data[idx] = cs[:, None] * uw
                            Lf.bias.data[idx] = cs * ub
                    _e = param_errors(cons, teacher)["max_eps_per_matrix"][2 * l]
                    print(f"[fast] refine L{l + 1}: {len(idx)}/{rmask.numel()} rows "
                          f"({_nq} queries) -> L{l + 1} max_eps {_e:.2e}", flush=True)
                    if args.design_refine and not bool(rmask.all()):
                        print("[fast] incomplete refined prefix; deferring deeper hidden layers", flush=True)
                        break
                with torch.no_grad():                     # output layer: closed-form LSQ
                    Hh = Xf.to(device).double()
                    for L_ in cons.layers[:-1]:
                        Hh = cons.act(L_(Hh))
                    A_ = torch.cat([Hh, torch.ones(len(Hh), 1, device=device, dtype=Hh.dtype)], 1)
                    sol = torch.linalg.lstsq(A_.cpu(), Yf.double()).solution.to(device)
                    cons.layers[-1].weight.copy_(sol[:-1].T); cons.layers[-1].bias.copy_(sol[-1])
                n_total += _q_ref
                print(f"[fast] refine done: {_q_ref} oracle queries, {time.time() - _t_ref:.0f}s; "
                      f"output layer solved in closed form on {len(Xf)} queries", flush=True)
            if args.tanhsolver:
                cons, _nq = _run_tanhsolver(args, cons, teacher, dims, act, device,
                                            pool=Xf)
                n_total += _nq
                _phase("tanhsolver", cons)
            errs = param_errors(cons, teacher, hard=args.hard)
            best = cons
            final = {
                "final_max_eps": errs["max_eps"],
                "final_mean_eps": sum(errs["mean_eps_per_matrix"]) /
                len(errs["mean_eps_per_matrix"]),
                "final_max_eps_per_matrix": errs["max_eps_per_matrix"],
                "final_agree": agreement(cons.float() if next(cons.parameters()).dtype == torch.float64 else cons, teacher, eval_pts),
                "queries": n_total, "stopped_iter": ck["iter"],
                "wall_s": round(time.time() - t0, 1),
            }
        os.remove(fast_dump)
        tanhsolver_done = args.tanhsolver and best is cons
    else:
        tanhsolver_done = False
    if args.tanhsolver and not tanhsolver_done:
        # no fast trigger (or no consensus formed): refine the trained best net
        best, _nq = _run_tanhsolver(args, best, teacher, dims, act, device)
        _phase("tanhsolver", best)
        errs = param_errors(best, teacher)
        final = {**final,
                 "final_max_eps": errs["max_eps"],
                 "final_mean_eps": sum(errs["mean_eps_per_matrix"]) /
                 len(errs["mean_eps_per_matrix"]),
                 "final_max_eps_per_matrix": errs["max_eps_per_matrix"],
                 "final_agree": agreement(best.float(), teacher, eval_pts),
                 "queries": final.get("queries", 0) + _nq,
                 "wall_s": round(time.time() - t0, 1)}
    os.makedirs(recon_dir, exist_ok=True)
    model_path = os.path.join(
        recon_dir, f"{args.variant}{tag}__{arch_tag}__s{args.seed}_final.pt")
    torch.save({"arch": dims, "act": act,
                "state_dict": {k: v.detach().cpu()
                               for k, v in best.state_dict().items()},
                **final}, model_path)
    print(f"[save] {model_path}", flush=True)
    out = {
        "variant": args.variant,
        "cfg": {k: (list(v) if isinstance(v, tuple) else (None if callable(v) else v))
                for k, v in cfg.__dict__.items()},
        "arch": dims,
        "n_params": count_params(teacher),
        "seed": args.seed,
        "teacher_seed": args.teacher_seed,
        "device": device,
        "log": log,
        "phases": [{"phase": n, "mean_eps": me, "max_eps": mx, "wall_s": ws}
                   for n, me, mx, ws in phases],
        **final,
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(
        RESULTS, f"{args.variant}{tag}__{arch_tag}__s{args.seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[done] {path} | max_eps={final['final_max_eps']:.3e} "
          f"agree={final['final_agree']:.4f} wall={final['wall_s']}s",
          flush=True)
    w = max(len(n) for n, *_ in phases)
    print(f"\n[phases] {'phase':<{w}}   {'mean_eps':>10} {'max_eps':>10} {'wall_s':>9}")
    for name, me, mx, ws in phases:
        print(f"[phases] {name:<{w}}   {me:>10.3e} {mx:>10.3e} {ws:>9.1f}")
    print(flush=True)


def _run_tanhsolver(args, net, teacher, dims, act, device, pool=None):
    """--tanhsolver: joint jet refinement of `net` against the sealed teacher
    forward (fp64 oracle, query-counted). Returns (net_fp64, n_queries)."""
    import tanh_solve as ts
    teacher64 = teacher.clone().double().eval()
    bb = ts.Oracle(lambda x: teacher64(x))              # forward queries only
    scales = tuple(float(s) for s in args.tanhsolver_scales.split(","))
    X = ts.sample_points(args.tanhsolver_points, dims[0], scales=scales,
                         pool=pool, seed=args.seed, device=device)
    e0 = param_errors(net, teacher)
    print(f"[tanhsolver] guess max_eps {e0['max_eps']:.3e} -> jet refine on "
          f"{len(X)} points ({'full Jacobian' if not args.tanhsolver_dirs else str(args.tanhsolver_dirs) + ' dirs'})...",
          flush=True)
    net64, info = ts.refine(bb, net, X, act, dirs=args.tanhsolver_dirs,
                            iters=args.tanhsolver_iters, cg_iters=args.tanhsolver_cg,
                            seed=args.seed, score=lambda n: param_errors(n, teacher))
    print(f"[tanhsolver] done: {info['iters']} LM steps, loss {info['loss_init']:.2e} -> "
          f"{info['loss_final']:.2e}, {info['queries']} oracle queries, {info['wall_s']}s",
          flush=True)
    return net64, info["queries"]


if __name__ == "__main__":
    main()
