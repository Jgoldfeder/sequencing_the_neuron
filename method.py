"""Reconstruction algorithm (paper Appendix B, Algorithms 1 & 2) with variant
hooks for benchmarking methodological improvements.

Baseline (paper-faithful):
  - Committee Disagreement Sampling: learnable input tensor optimized by
    gradient ascent on normalized pairwise L1 output disagreement (App. C).
  - Population of p surrogates trained on ALL accumulated samples each outer
    iteration; LR step-decay; return lowest-loss member.

Variant hooks:
  disagree      : 'mean_pair' (paper) | 'min_pair' | 'median_pair' | 'variance'
  query_box     : bound queries into [-box, box]^d via tanh parameterization
  query_div_w   : weight of inter-query repulsion penalty (query diversity)
  window        : keep only the last `window` outer-iterations of samples
                  (0 = keep all, paper behavior)
  maint_every   : every k iters, align population to best member, try soup,
                  restart worst members (0 = never)
  restart_worst : number of worst members restarted during maintenance
  polish_f64    : final float64 polish phase on CPU
"""
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict, replace as dc_replace

import torch
import torch.nn.functional as F

from align import (scale_normalize_, greedy_perm, permute_layer_, match_layer_,
                   param_errors, align_clone_to, layer_eps_split,
                   cnn_canonicalize_, cnn_align_to_, cnn_param_errors)
from nets import MLP, ConvNet
from factored import resolve_layers, FactoredTail, AdditiveResidual


@dataclass
class Cfg:
    # population / budget
    p: int = 8                 # population size
    q: int = 1500              # new queries per outer iteration
    outer: int = 40            # outer iterations
    factor_tail: bool = False  # square-factorize every hidden layer AFTER the
                               # first as W_l=A_l B_l (exact reparam, no extra
                               # function class); W1 stays ordinary. Optimizer-
                               # only change -- members remain plain MLPs.
    factor_inner: float = 1.0  # inner-dim multiplier for factor_tail: inner dim
                               # k = round(factor_inner*out). 1.0=square (minimal);
                               # >1 over-squares (bigger relaxation, still exact).
    factor_lr_mult: float = 0.25  # factor Adam LR as a FRACTION of the fit lr
                               # (do NOT inherit it: effective step on W=AB scales
                               # with factor norms/dims). 0.25 = eta/4.
    factor_mode: str = "residual"  # 'residual' (W_eff=W+U V^T, keeps baseline -G
                               # path, robust) | 'square' (W=A B, no -G path).
    factor_rank: int = 256     # rank of the residual express lanes (mode=residual);
                               # 0 => full min(out,in). Pure relaxation knob.
    factor_layers: str = "tail"  # which weight layers to reparameterize:
                               # 'tail' (all but W1) | 'l1' (W1 only) | 'all'.
    qg_chunk: int = 0          # chunk size for the query-gen member forward (0 =
                               # full batch). Set (e.g. 8192) so WIDE students in
                               # the fit gate don't OOM; gradient is exact.
    expand: float = 1.0        # Expand-and-Cluster fit gate: widen every HIDDEN
                               # layer by this factor (input/output dims fixed).
                               # >1 => students are overparameterized vs teacher;
                               # weight-vs-teacher scoring/consensus is skipped
                               # (dims mismatch) and only the fit loss is logged.
    act: str = "leaky_relu"  # hidden activation of teacher AND committee
                             # ('leaky_relu' | 'sigmoid' | 'tanh'); sigmoid/
                             # tanh have no scaling isomorphism -- alignment
                             # canonicalizes their polarity isomorphism
                             # (sigma(-z)=1-sigma(z), tanh(-z)=-tanh(z))
                             # instead (align.sign_canonicalize_)
    epochs: int = 10         # training epochs per outer iteration
                           # (VALIDATED: must be high enough to fit D tightly;
                           #  epochs=2 stalls completely - the underfit trap)
    lr: float = 1e-3           # surrogate training lr (Adam)
    lr_sched: tuple = (0.6, 0.85)  # decay lr /10 at these outer-iter fracs
    batch: int = 512
    # query generation
    qg_steps: int = 30         # gradient steps optimizing the query tensor
    qg_lr: float = 0.1
    qg_sched: tuple = (0.5, 0.8)
    qg_init_std: float = 0.5
    qg_dist: str = "l1"        # disagreement distance: 'l1' | 'l2' (euclidean,
                               # matching the original get_adv's torch.cdist)
    qg_init: str = "gauss"     # query init: 'gauss' N(0,qg_init_std^2) |
                               # 'uniform' U[-qg_range, qg_range] (original get_adv)
    qg_range: float = 1.0      # bound for uniform query init
    # variant switches
    disagree: str = "mean_pair"
    query_box: float = 0.0     # 0 = unbounded (paper); else tanh box bound
    query_div_w: float = 0.0
    window: int = 0
    maint_every: int = 0
    restart_worst: int = 0
    polish_f64: bool = False
    polish_epochs: int = 3
    warmstart_iters: int = 0   # first k iters use random Gaussian queries
    fit_delta: float = 0.0     # >0: stop inner epochs early when batch L1 < delta
    stop_loss: float = 0.0     # >0: early-stop outer when best_loss < stop_loss
    stop_agree: float = 0.0    # >0: ...AND population weight dispersion < this
    stop_patience: int = 3
    # batch-2 hypotheses
    fit_loss: str = "l1"       # 'l1' (paper) | 'mse' (Opus A2: residual-proportional grads)
    hard: bool = False         # hard-label regime: the blackbox returns ONLY the
                               # argmax class index (0..C-1), never the logit
                               # vector. Y holds long class indices; the committee
                               # trains with cross-entropy and members are
                               # ranked/gated by xent_on. Regression-only
                               # machinery (last-layer/LBFGS/f64 solvers,
                               # solver_polish, combine's consensus polish, kink
                               # probing) needs real-valued outputs -> rejected.
    lastlayer_every: int = 0   # >0: closed-form ridge LS solve of last layer every k iters (Opus A12)
    lastlayer_sample: int = 20000  # rows subsampled for the periodic last-layer solve
                               # (>> any head's feature dim; keeps --lastlayer-every 1
                               # nearly free vs forwarding the whole buffer). 0 = all.
    popavg_kappa: float = 0.0  # >0: aligned averaging of members within kappa x best loss (Opus A9)
    lbfgs_polish: bool = False # float64 LBFGS squared-loss endgame (Opus A2+A3-lite)
    gate_kappa: float = 0.0    # >0: exclude members with loss > kappa x best from disagreement (Opus T7)
    cheat: bool = False        # ORACLE DIAGNOSTIC, not an honest attack: the
                               # blackbox itself joins gen_queries as an extra
                               # frozen member, so disagreement queries are
                               # optimized to separate the student(s) from the
                               # TRUTH (needs white-box gradient access to the
                               # teacher). The CLI forces p=1 -> disagreement
                               # runs on the (student, blackbox) pair as if p=2.
    cheat_bb_ref: bool = False  # cheat + p>1 (--cheat-pop): the qg disagreement
                               # scores each member ONLY against the blackbox
                               # (no member-member pairs); the peel gate/refine
                               # still runs on the best member, frozen rows are
                               # pinned into every member.
    cheat_solo: bool = False   # cheat + p>1 (--cheat-solo): SOLO committee --
                               # each iter every member runs its OWN q/p-query
                               # disagreement search against the blackbox alone
                               # (p=1-style targeting, no median dilution) and
                               # trains ONLY on its own slice of the buffer
                               # (ownership is positional: within each q-row
                               # iteration block, member i owns rows
                               # [i*(q//p), (i+1)*(q//p)); the last member also
                               # takes the remainder). Total oracle budget per
                               # iter stays q. Best-member ranking stays on the
                               # FULL buffer (comparable losses); the peel
                               # gate/refine/freeze machinery is unchanged.
                               # MLP path only.
    xspace_refine: bool = False  # --xspace-refine: recover each peeled layer from
                               # its x-space kink normal + prefix-Jacobian mapback
                               # (the deep multi-region path) instead of the h-space
                               # seal. DEPTH-FLAT: precision is set by the prefix's
                               # condition number, not the seal's ~100x/layer
                               # amplification -- so deep layers reach ~fp64 (e-12)
                               # like L1 instead of degrading (L2 ~e-6, worse deeper).
                               # Slower: per-kink Jacobian + multi-base x-space search.
    loc_refine: bool = False   # --loc-refine: legacy tracked kink-point recovery
                               # for deep layers; first layer uses the original path.
    design_refine: bool = False  # --design-refine: multiscale information-directed
                               # kink sampling for ALL hidden layers, including L0.
                               # Requires fp64 frozen prefixes; acceptance checks
                               # are heuristic, not a machine-accuracy certificate.
    f64: bool = False          # --f64: do the layer EXTRACTION in float64 while
                               # training/search stay fp32. The refine net that
                               # holds the exactly-peeled prefix is float64, so
                               # each peel is stored at fp64 (not truncated to
                               # the fp32 student) and the next layer's seal
                               # inverts through an fp64-accurate prefix. Keeps
                               # deep peels ~1e-10+ instead of the fp32 seal
                               # amplifying ~6e-8 by ~100x per layer. MLP peel.
    fast_peel: bool = False    # cheat + p>1 (--fast-peel): ALSO fire the
                               # refine+peel when the FRONTIER hidden layer
                               # reaches FULL committee consensus (--fast's
                               # trigger), using the consensus rows (quorum
                               # means) as the refiner guesses. Peels layer by
                               # layer on committee agreement even though no
                               # single member clears the eps gate; the eps
                               # trigger stays active, whichever fires first.
                               # Unlike --fast the run does NOT stop. MLP only.
    cheat_peel_max: float = 1e-2   # cheat + freeze_reinit (CNN): consensus can't
    cheat_peel_mean: float = 1e-3  # form at p=1, so instead peel the frontier
                               # layer once its ORACLE eps vs the teacher (the
                               # printed "eps/layer (best)" numbers) satisfies
                               # max<=cheat_peel_max AND weight-mean<=
                               # cheat_peel_mean. Also flip-safe: a layer with
                               # sign-flipped channels (eps ~0.3) can't pass.
    fast_peel_partial: bool = False  # --fast-peel-partial (CNN): every log iter, kink-
                               # refine the frontier's CONSENSUS channels from the
                               # quorum-mean rows, inject each solved row into EVERY
                               # member at that member's own magnitude and pin it; the
                               # frontier advances only once the whole layer is solved
    partial: bool = False      # cheat + --partial: every log_every iters, refine
                               # the frontier's still-unsolved neurons REGARDLESS
                               # of the peel threshold, and in-place freeze (warm,
                               # scale-matched) whatever solves -- accumulating the
                               # solved subset without peeling/advancing the whole
                               # layer. Complements the full cheat-peel: stragglers
                               # keep training while solved neurons lock in early.
    ensemble_every: int = 0    # >0: every k outer iters MERGE the population into
                               # ONE same-architecture net: per hidden layer, pool
                               # every member's neurons and group-OMP-select the
                               # width-n subset whose activations best linearly
                               # explain EVERY member's next-layer preactivations
                               # (downstream sufficiency, not geometric clustering),
                               # express the next layer over the selected basis,
                               # recurse; output layer ridge-solved on the query
                               # buffer. Attacker-side only (queries + member
                               # weights; never the oracle). The merged net
                               # replaces the worst member when it wins on loss.
                               # Peel-pinned rows keep their slot + exact values;
                               # only unsolved slots are selected. CLI: --ensemble.
    ensemble_samples: int = 200_000  # query-sample size for the merge fits
    ensemble_block: int = 16   # OMP block size (candidates added per LS refit)
    ensemble_boost: bool = False  # residual-cascade boosting (Judah's protocol):
                               # every outer iteration appends a ROUND of
                               # boost_stages_max sequential residual stages --
                               # each a FRESH net trained cfg.epochs at cfg.lr
                               # on the unit-renormalized residual of the
                               # cascade-so-far; the attacker model is the
                               # growing SUM. Disagreement queries run between
                               # (cascade + member) and the blackbox. The
                               # cascade is compressed back into one width-n
                               # net when it exceeds ~24 stages (fresh-query
                               # gated) and once terminally (the extraction
                               # candidate). Needs real-valued Y -> incompatible
                               # with hard. Under --cheat the population stays
                               # p=1 (a single tracking member).
    boost_stages_max: int = 0  # ensemble-boost: MAX residual stages per outer
                               # iteration (--ensemble N under boost; 0 -> 1).
                               # Stages are added until the buffer residual
                               # hits boost_tol ("train to zero, THEN new
                               # data") or this cap.
    boost_tol: float = 1e-2    # ensemble-boost: per-iteration target -- keep
                               # adding stages until cascade train L1 on the
                               # current buffer falls below this
    peel_try: int = 0          # CNN: every k iters, ATTEMPT the kink refiner on
                               # the frontier layer directly (replaces the eps/
                               # consensus peel gate). Solved channels cached in
                               # exact_net/exact_mask across attempts; an attempt
                               # early-aborts after 2 fresh abstains (rotating
                               # start). The layer peels only when EVERY channel
                               # is exact. Needs real-valued outputs (kink
                               # probing) -> incompatible with hard.
    peel_warm: bool = False    # CNN peel-try: on a partial/full freeze, keep the
                               # committee's CURRENT trained weights (deeper layers
                               # + unsolved frontier rows) instead of cold-reinit-
                               # ializing them -- only the newly solved channels are
                               # pinned. Preserves tail training across peels (the
                               # partial-freeze reboot otherwise discards it every
                               # time a channel is banked). Unsolved frontier rows
                               # are still re-rolled (flip escape); deeper layers
                               # stay warm.
    peel_advance_frac: float = 1.0  # peel-try: advance the frontier to the next
                               # layer once THIS fraction of it is solved, instead
                               # of requiring 100%. <1.0 unblocks deeper layers
                               # when the frontier has a few untrainable neurons
                               # (e.g. 508/512): set 0.98 and the peel moves on to
                               # L2/L3/... instead of stalling forever on the last
                               # few. CAVEAT: a deeper layer is then refined with a
                               # prefix that still has those unsolved neurons, so
                               # its recovery is EXACT only in the input dims whose
                               # upstream neurons are solved; the dims tied to the
                               # stragglers are approximate. The refiner's guards
                               # (jump_ratio/angle/runaway) still reject gross
                               # corruption, so most banked deep neurons are clean.
    peel_miss_abort: int = 50  # peel-try: abort an attempt after this many
                               # CONSECUTIVE unsolved-neuron misses (resets on
                               # each solve). Old default was 2 TOTAL misses,
                               # which bailed after the first couple of hard
                               # neurons and banked ~5/attempt even when hundreds
                               # were solvable. High + consecutive lets an
                               # attempt sweep all currently-solvable neurons
                               # (fast triage rejects make this cheap) while
                               # still bailing on a genuinely exhausted layer.
    peel_reroll: bool = False  # peel-try warm freeze: on a PARTIAL freeze, also
                               # re-randomize the still-unsolved frontier rows
                               # (fresh signs, flip escape). Default OFF: that
                               # reset every bank puts progressing neurons back
                               # to random so they never accumulate training
                               # (treadmill). Keep them WARM; use --peel-stuck
                               # for the escape on a genuine plateau instead.
    peel_angle_gate: float = 12.0  # peel-try (MLP): MAX angle (deg) recovered-vs-
                               # guess to ACCEPT a kink refine. NOT a precision
                               # bound (the sweep is exact); only guards against
                               # locking a neighbour's kink, so it can be loose
                               # given inter-neuron separation (jump_ratio +
                               # runaway 0.3 do the real gating). A tight 2deg
                               # here wrongly rejected CORRECT recoveries from
                               # imperfect (~3deg) guesses -> 0 solved.
    peel_clip: float = 0.0     # peel-try: once ANY layer is pinned/frozen, clip the
                               # per-member training grad-norm to this value (0=off).
                               # Freezing most of a layer makes the adversarial
                               # (cheat) disagreement queries concentrate all their
                               # pressure on the few still-free params (the last
                               # unsolved neuron + deeper layers); at full LR they
                               # overshoot and the net DIVERGES (seen at width>=512
                               # where the head is still oscillating at pin time).
                               # Capping the step keeps the free params stable
                               # without touching the no-peel path.
    peel_stuck: int = 0        # CNN peel-try (with peel_warm): after this many
                               # consecutive attempts with NO new channel solved
                               # on the frontier, COLD-reinit the unfrozen params
                               # (fresh random signs for the stuck rows + fresh
                               # tail; solved channels stay pinned) to escape the
                               # warm plateau. 0 = never (pure warm). The hard
                               # tail channels that warm can't reach get fresh
                               # draws this way.
    peel_refresh: bool = False  # CNN peel: after a layer freezes+reinits, EMPTY
                               # the sample buffer and RESTART the outer-iteration
                               # clock (fresh warmstart, full budget, fresh lr
                               # schedule for the collapsed deeper search) -- the
                               # old samples were mined against the pre-peel
                               # student and are stale for the new committee.
    peel_restart: bool = False  # cheat-peel (MLP): after a hidden layer is FULLY
                               # peeled (and a deeper one remains), do a FULL COLD
                               # restart of the deeper search: (1) re-randomize the
                               # unsolved + deeper weights (frozen rows pinned),
                               # (2) flush the sample buffer, (3) restart the lr/
                               # iter schedule. Not for the last hidden layer (its
                               # only successor is the linear output, LSQ-solved).
    peel_seal: bool = False    # cheat-peel (MLP) --peel-direct: once a leading prefix
                               # is exactly frozen, generate the disagreement queries
                               # in the FRONTIER layer's input space and map them back
                               # through the exact prefix inverse -> directly targets
                               # the next layer (== querying it with the prefix
                               # removed; forward+gradient identical). Fixes the
                               # x-space straggler stall.
    restart_stuck: bool = False  # --partial: also peel-restart when the frontier is
                               # NOT fully solvable -- either (a) STAGNATION (>= frac
                               # of the frontier solved AND no new neurons frozen for
                               # `window` iters) or (b) the final iter -- keep the
                               # partial-frozen rows pinned, cold-reinit only the
                               # unsolved + deeper, flush buffer, restart the clock.
    restart_stuck_frac: float = 0.5   # (a): min fraction of the frontier solved
    restart_stuck_window: int = 20    # (a): iters of zero growth that = stagnation
    restart_stuck_max: int = 5        # safety cap on stuck-restarts (each adds budget)
    retry: int = 0                    # --retry N: in a peel mode, when the budget ends with the
                                      # frontier layer not fully peeled, reinit and retry (N times):
                                      # partial modes keep solved rows pinned and reinit only the
                                      # unsolved rows + deeper; full-layer modes reinit the whole
                                      # frontier layer (+ deeper). Buffer flushed, clock restarted.
    # cluster-consensus diagnostic
    cluster_quorum: float = 0.625  # quorum as a ratio of the committee (5/8);
                                  # a unit needs >= ceil(ratio*p) members to
                                  # agree, else the whole consensus is n/a.
                                  # Matches the --fast dump trigger so the
                                  # printed `cluster` column (and --combine)
                                  # key off the same consensus --fast stops on.
                                  # HARDCODED default -- not exposed on the CLI.
    cluster_eps: float = 0.02     # consensus TOLERANCE: two members' aligned
                                  # neurons cluster only if their [w|b] rows are
                                  # within this inf-norm distance. Loosen it
                                  # (--consensus-threshold) to let a deep net's
                                  # looser layers reach consensus at all.
    cluster_gate: float = 10.0    # NOTE: no longer applied to the logged/combine
                                  # consensus (that path is now ungated, to match
                                  # the --fast trigger). Only the dead diagnostic
                                  # wrapper cluster_consensus() still references it.
    combine: bool = False         # the FIRST outer iter the consensus becomes
                                  # available (n/a -> answer), replace the worst
                                  # committee member with the (polished)
                                  # consensus net
    solver_polish: bool = False   # at the end of each outer iter, tighten each
                                  # member with the staged LBFGS recipe
                                  # (MSE then MAE) -> faster committee agreement
    solverwindow: int = 0         # every query-solver (in-loop polish, --fast
                                  # endgame, lbfgs endgame) fits on the last
                                  # solverwindow outer iters of queries; 0 = all
    verbose: bool = False         # print per-member polish detail (loss
                                  # before->after, #evals, time)
    dump_at_iter: int = 0         # >0: torch.save population+queries at this
    dump_path: str = ""           # outer iter (real schedule intact) and stop
    stop_on_consensus: bool = False  # dump+stop the FIRST log-iter a consensus
                                     # (ungated 5/8) forms, instead of a fixed iter
    stop_layer: int = -1             # -1: --fast stops on WHOLE-NET consensus (all
                                     # layers). >=0: stop as soon as this single
                                     # hidden layer reaches FULL consensus (every
                                     # neuron), for peeling -- don't wait on deeper,
                                     # harder layers. Dumps the partial consensus.
    combine_polish_lr: float = 1e-4   # low-LR polish of the consensus before
    combine_polish_steps: int = 200   # injecting: removes the averaging /
                                      # assembly artifact (loss ~200x lower) so
                                      # the injected member wins loss-selection,
                                      # without drifting its recovered params
    # cryptanalytic layer-1 verifier (black-box critical-point probe)
    verify: bool = False          # every log-iter, efficiently VERIFY each layer-1
                                  # consensus neuron against the teacher's true kink
                                  # hyperplane: is it within (eps_offset, eps_angle)?
                                  # ~O(k) queries/neuron, independent of input dim.
                                  # Diagnostic queries; NOT counted in the budget.
    verify_refine: bool = False   # also REFINE within-eps neurons: offset to machine
                                  # precision (cheap) + normal via rank-1 jump (~2d
                                  # queries/neuron). Off by default (the costly part).
    verify_max: int = 0           # cap neurons probed per log-iter (0 = all)
    verify_eps_offset: float = 1e-2   # within-eps tolerance: plane offset
    verify_eps_angle: float = 1.0     # within-eps tolerance: normal tilt (degrees)
    verify_k: int = 16            # random perpendicular probes for the angle estimate
    # logging
    log_every: int = 5
    eval_pts: int = 2000
    pop_save_every: int = 0       # >0: snapshot the population every k iters
    pop_save_path: str = ""       # (inspect mid-run; overwrites, no queries)
    # freeze-reinit peel: once a hidden layer reaches consensus, REINITIALIZE the
    # committee so every member SHARES that layer's consensus weights FROZEN
    # (consensus neurons pinned + shared, stragglers stay fresh/trainable), then keep
    # training on the reused queries -- collapsing the search onto the deeper layers.
    freeze_reinit: bool = False
    freeze_thresh: float = 0.9    # freeze a hidden layer once n_cons/n_tot >= this
    freeze_precision: float = 0.0  # only freeze if that layer's cons_max <= this
    layer_space_init: bool = True  # peel: seed disagreement queries in the FROZEN
                                   # frontier's INPUT space (even per-neuron boundary
                                   # coverage via the exact prefix) instead of image
                                   # space -- fixes deep-wide-layer starvation
                                   # (<=0 disables the precision gate; count only)
    spatial_boundary_seed: bool = False  # extend layer_space_init to SPATIAL conv
                                   # frontiers: seed each query on a targeted
                                   # (channel, spatial-position) kink and invert
                                   # through the exact prefix, round-robin over ALL
                                   # channels -> even per-channel coverage where the
                                   # dense hyperplane seed would otherwise bail
                                   # (DEPRECATED: no CLI surface -- empirically bad
                                   # on the LeNet cheat runs; use layer_rand_init)
    layer_mine: bool = False   # peel: run the DISAGREEMENT optimization directly
                               # in the frontier's INPUT space (tails of the
                               # members + blackbox under cheat), then invert
                               # the optimized activations through the exact
                               # frozen prefix to images. Beats image-space
                               # mining when the prefix squashes some frontier
                               # directions (their disagreement gradient is
                               # invisible from image space). Queries are used
                               # DIRECTLY (image-space opt would undo it).
    layer_rand_init: bool = False  # peel: INITIALIZE the disagreement-query
                                   # optimizer with images whose frontier-INPUT
                                   # activations are randomly (isotropically)
                                   # distributed -- random targets in the frozen
                                   # frontier's input space, approximately
                                   # inverted through the exact prefix (least
                                   # squares); the normal disagreement opt then
                                   # runs from that init. Takes precedence over
                                   # layer_space_init's direct boundary seeds.
    # extract-freeze: every log-iter, EXACTLY extract each layer-1 consensus neuron
    # with the black-box exact-affine probe (~2d teacher queries); the ones that
    # extract cleanly are pinned to their exact value across the committee and frozen.
    extract_freeze: bool = False
    extract_max_angle: float = 5.0   # reject a probe whose normal is >this from the guess
    extract_jump_ratio: float = 50.0  # reject unless the gradient jump is cleanly rank-1
    extract_s: float = 1e-3          # kink side-offset: must be < distance to the nearest
                                     # OTHER kink, else the two sides span deeper regions
                                     # and the jump stops being rank-1 (deep nets: shrink)
    extract_r: float = 1e-4          # affine-fit stencil radius (shrink with extract_s)
    extract_tries: int = 8           # retry each neuron from this many seeds before giving up
    extract_base_scale: float = 1.0  # seed norm: probe seeds sit ~this far from the guess
                                     # plane's foot (small -> guessed kink lands in-window)


# ---------------------------------------------------------------- queries --
def _normalize(v):
    return v / v.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)


def disagreement(outs, mode, dist="l1", ref_last=False):
    """outs: (p, q, out_dim) frozen committee outputs. Return loss to
    MINIMIZE (negative disagreement). `dist`: 'l1' (reimpl) | 'l2' (euclidean,
    matching the original get_adv's torch.cdist). ref_last: the LAST entry is
    a REFERENCE (cheat blackbox) -- score every other member against it ONLY
    (no member-member pairs; a cheat committee needs each member pushed toward
    its own blackbox mismatches, not mutual repulsion)."""
    f = _normalize(outs)
    if ref_last:
        diff = f[:-1] - f[-1:]                       # (p-1, q, out)
        if mode == "variance":
            return -((diff ** 2).sum(-1)).mean()
        if dist == "l2":
            pairs = (diff ** 2).sum(-1).clamp_min(1e-24).sqrt()
        else:
            pairs = diff.abs().sum(-1)               # (p-1, q)
        if mode == "mean_pair":
            per_sample = pairs.mean(dim=0)
        elif mode == "min_pair":
            per_sample = pairs.min(dim=0).values
        elif mode == "median_pair":
            per_sample = pairs.median(dim=0).values
        else:
            raise ValueError(mode)
        return -per_sample.mean()
    if mode == "variance":
        mean = f.mean(dim=0, keepdim=True)
        var = ((f - mean) ** 2).sum(-1).mean(dim=0)  # per sample
        return -var.mean()
    # pairwise distance matrix per sample: (p, p, q)
    diff = f.unsqueeze(1) - f.unsqueeze(0)
    if dist == "l2":
        D = (diff ** 2).sum(-1).clamp_min(1e-24).sqrt()
    else:
        D = diff.abs().sum(-1)
    p = f.shape[0]
    i, j = torch.triu_indices(p, p, offset=1, device=f.device)
    pairs = D[i, j, :]  # (n_pairs, q)
    if mode == "mean_pair":
        per_sample = pairs.mean(dim=0)
    elif mode == "min_pair":
        per_sample = pairs.min(dim=0).values
    elif mode == "median_pair":
        per_sample = pairs.median(dim=0).values
    else:
        raise ValueError(mode)
    return -per_sample.mean()


def _repulsion(Z, n_pairs=65536):
    """Mean squared cosine similarity over random query pairs (to minimize)."""
    q = Z.shape[0]
    flat = Z.reshape(q, -1).float()
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    i = torch.randint(0, q, (n_pairs,), device=Z.device)
    j = torch.randint(0, q, (n_pairs,), device=Z.device)
    cos = (flat[i] * flat[j]).sum(-1)
    return (cos ** 2).mean()


def gen_queries(pop, cfg, input_dim, device, gen, losses=None, init_override=None,
                ref_last=False):
    """Paper Algorithm 2 (+ box / diversity / fit-gating variants).
    init_override: optional (q, input_dim) seed to start the disagreement search from
    (e.g. layer-space boundary seeds) instead of the gaussian/uniform init.
    ref_last: pop's LAST entry is the cheat blackbox; disagreement scores each
    member against it only (see disagreement())."""
    members = pop
    if cfg.gate_kappa > 0 and losses is not None and not ref_last:
        bl = min(losses)
        gated = [m for m, l in zip(pop, losses)
                 if l <= bl * cfg.gate_kappa]
        if len(gated) >= 2:
            members = gated
    q = cfg.q
    if cfg.query_box > 0:                       # optimize `raw`, feature = box*tanh
        leaf = torch.randn(q, input_dim, generator=gen, device=device)
        box = cfg.query_box
        T = lambda x: box * torch.tanh(x)       # noqa: E731
    elif cfg.qg_init == "uniform":
        leaf = (torch.rand(q, input_dim, generator=gen, device=device)
                * 2 - 1) * cfg.qg_range
        T = lambda x: x                         # noqa: E731
    else:
        leaf = torch.randn(q, input_dim, generator=gen,
                           device=device) * cfg.qg_init_std
        T = lambda x: x                         # noqa: E731
    if init_override is not None:               # layer-space boundary seed (feature space)
        leaf = init_override.detach().to(device).clone()
        T = lambda x: x                         # noqa: E731
    leaf.requires_grad_(True)
    opt = torch.optim.Adam([leaf], lr=cfg.qg_lr)
    sched = {int(s * cfg.qg_steps) for s in cfg.qg_sched}
    # memory: chunk the (per-query) member forward so wide students don't hold all
    # q activations at once. Disagreement is per-query -> gradient accumulation is
    # EXACT (same optimization as full batch). 0 => full batch (unchanged).
    chunk = cfg.qg_chunk if (cfg.qg_chunk and cfg.qg_chunk > 0) else q
    for step in range(cfg.qg_steps):
        if step in sched:
            for g in opt.param_groups:
                g["lr"] /= 10
        opt.zero_grad()
        if chunk >= q:
            I = T(leaf)
            outs = torch.stack([net(I) for net in members])  # (p', q, out)
            loss = disagreement(outs, cfg.disagree, cfg.qg_dist,
                                ref_last=ref_last)
            if cfg.query_div_w > 0:
                loss = loss + cfg.query_div_w * _repulsion(I)
            loss.backward()
        else:
            for c0 in range(0, q, chunk):
                sl = slice(c0, min(c0 + chunk, q))
                Ic = T(leaf[sl])
                outs = torch.stack([net(Ic) for net in members])
                d = disagreement(outs, cfg.disagree, cfg.qg_dist,
                                 ref_last=ref_last)  # mean over chunk
                (d * ((sl.stop - sl.start) / q)).backward()        # -> exact q-mean grad
            if cfg.query_div_w > 0:                # cross-query term (no member fwd)
                (cfg.query_div_w * _repulsion(T(leaf))).backward()
        opt.step()
    with torch.no_grad():
        return T(leaf).detach()


def _solo_pools(nrows, q, p):
    """cheat_solo row ownership. The buffer always holds WHOLE q-row iteration
    blocks (append q/iter, window trim at multiples of q), and each block is the
    concatenation of the members' query chunks in member order -- so ownership
    is positional: within a block, member i owns [i*cs, i*cs+cs) (cs = q//p;
    the last member also takes the q - cs*p remainder). Returns one LongTensor
    of row indices per member (CPU). None if nrows isn't block-aligned."""
    if nrows == 0 or nrows % q != 0:
        return None
    cs = q // p
    base = torch.arange(nrows // q, dtype=torch.long) * q
    pools = []
    for i in range(p):
        w = cs + (q - cs * p if i == p - 1 else 0)
        off = torch.arange(i * cs, i * cs + w, dtype=torch.long)
        pools.append((base[:, None] + off[None, :]).reshape(-1))
    return pools


# ------------------------------------------------------------- maintenance --
@torch.no_grad()
def solve_last_layer_(net, X, Y, ridge=1e-6, chunk=8192, sample=0):
    """Closed-form ridge least-squares solve of the final layer given the
    penultimate features (Opus A12: the last layer is a convex quadratic;
    variable projection instead of first-order SGD). `sample`>0: fit on a random
    subsample of that many buffer rows -- a few thousand fully determine a small
    head, so --lastlayer-every 1 costs one small forward pass instead of running the
    whole (~millions-of-rows) buffer through the net every iter. 0 = use all rows."""
    if sample and len(X) > sample:
        idx = torch.randperm(len(X))[:sample]
        X = X[idx.to(X.device)]; Y = Y[idx.to(Y.device)]
    dev = net.layers[-1].weight.device
    # stream the normal-equations accumulation (A=HaᵀHa, B=HaᵀY) per chunk on the
    # GPU in float64: never materializes the full feature buffer (no OOM on a huge
    # buffer) and keeps the heavy GEMM off the CPU, so a subsampled solve is a couple
    # of small matmuls -- cheap enough for --lastlayer-every 1.
    A = B = None
    for i in range(0, len(X), chunk):
        h = X[i:i + chunk].to(dev)
        for l in net.layers[:-1]:
            h = net.act(l(h))
        ha = torch.cat([h.double(),
                        torch.ones(len(h), 1, dtype=torch.float64, device=dev)], 1)
        yb = Y[i:i + chunk].to(dev).double()
        if A is None:
            A = ha.new_zeros(ha.shape[1], ha.shape[1])
            B = ha.new_zeros(ha.shape[1], yb.shape[1])
        A += ha.T @ ha
        B += ha.T @ yb
    A += ridge * A.diag().mean().clamp_min(1e-12) * torch.eye(
        A.shape[0], dtype=torch.float64, device=dev)
    W = torch.linalg.solve(A, B)
    last = net.layers[-1]
    last.weight.copy_(W[:-1].T.float().to(last.weight.device))
    last.bias.copy_(W[-1].float().to(last.bias.device))


@torch.no_grad()
def aligned_pop_average(pop, losses, kappa):
    """Aligned coordinate-wise average of members within kappa x best loss
    (Opus A9). Members are aligned into the best member's frame with
    function-preserving transforms; returns (avg_net, k)."""
    bi = min(range(len(pop)), key=lambda i: losses[i])
    thresh = losses[bi] * kappa
    idxs = [i for i in range(len(pop)) if losses[i] <= thresh]
    if len(idxs) < 2:
        return None, len(idxs)
    anchor = pop[bi].clone()
    aligned = [anchor]
    for i in idxs:
        if i == bi:
            continue
        m = pop[i].clone()
        align_member_to(anchor, m)
        aligned.append(m)
    return soup_of(aligned), len(idxs)


@torch.no_grad()
def align_member_to(best, member):
    """Function-preserving alignment of member into best's frame (in-place
    on member)."""
    scale_normalize_(best)  # scale-norm is function preserving
    scale_normalize_(member)
    for l in range(len(best.layers) - 1):
        match_layer_(best, member, l)      # sign-aware for sigmoid/tanh


@torch.no_grad()
def soup_of(members):
    avg = members[0].clone()
    for avg_p, *ps in zip(avg.parameters(),
                          *[m.parameters() for m in members]):
        avg_p.copy_(torch.stack(list(ps)).mean(0))
    return avg


@torch.no_grad()
def _largest_eps_ball(feats, eps):
    """Indices of the largest set of rows all within `eps` (inf-norm) of some
    anchor row -- the tight-agreement cluster at ONE aligned neuron position."""
    D = torch.cdist(feats.float(), feats.float(), p=float("inf"))
    best = []
    for a in range(feats.shape[0]):
        ball = (D[a] < eps).nonzero(as_tuple=True)[0].tolist()
        if len(ball) > len(best):
            best = ball
    return best


@torch.no_grad()
def _align_members(pop, idxs, ref_idx):
    """Scale-normalize + permutation-align every member[idxs] into the reference
    member's frame. align_clone_to propagates each layer's permutation into the
    next (permute_layer_), so after this a neuron's incoming row at any depth is
    directly comparable across members. Returns (aligned clones, normalized ref)."""
    ref = pop[ref_idx].clone(); scale_normalize_(ref)
    aligned = [align_clone_to(pop[i], ref)[1] for i in idxs]
    return aligned, ref


@torch.no_grad()
def _consensus_from_ref(pop, dims, idxs, ref_idx, eps, quorum):
    """Align members[idxs] into member[ref_idx]'s frame (align_clone_to propagates
    each layer's permutation into the next), then take per-neuron quorum consensus.
    Returns (net_or_None, per_layer) where per_layer[l] = (n_total, n_consensus)
    for hidden layer l, and net is None unless EVERY hidden neuron reached quorum."""
    aligned, ref = _align_members(pop, idxs, ref_idx)
    net = ref.clone()
    L = len(net.layers)
    per_layer, full = [], True
    for l in range(L):                              # every weight matrix
        hidden = l < L - 1
        H, nc = net.layers[l].weight.shape[0], 0
        for k in range(H):
            feats = torch.stack([torch.cat([a.layers[l].weight[k],
                                            a.layers[l].bias[k:k + 1]])
                                 for a in aligned])
            S = _largest_eps_ball(feats, eps)
            if len(S) >= quorum:
                if hidden:
                    nc += 1
                net.layers[l].weight.data[k] = torch.stack(
                    [aligned[s].layers[l].weight[k] for s in S]).mean(0)
                net.layers[l].bias.data[k] = torch.stack(
                    [aligned[s].layers[l].bias[k] for s in S]).mean(0)
            elif hidden:
                full = False                        # a hidden unit lacks quorum
        if hidden:
            per_layer.append((H, nc))
    return (net if full else None), per_layer


@torch.no_grad()
def _build_consensus_deep(pop, dims, losses, eps, quorum_ratio, gate_kappa):
    """build_consensus for nets with >1 hidden layer. Mutual-NN clustering doesn't
    generalize cleanly across layers (a layer's permutation reindexes the next
    layer's input columns), so instead align every gated member into one member's
    frame, then take per-neuron quorum consensus layer by layer. Returns the
    consensus net, or None unless EVERY hidden neuron reaches quorum (same
    all-or-nothing contract as the single-layer path). Robust to a bad reference:
    with losses known it aligns to the best member; without, it tries each member
    as the frame and returns the first that yields full consensus."""
    import math
    P = len(pop)
    quorum = math.ceil(quorum_ratio * P)
    idxs = list(range(P))
    if losses is not None and gate_kappa > 0:
        bl = min(losses)
        idxs = [i for i in range(P) if losses[i] <= gate_kappa * bl]
    if len(idxs) < quorum:
        return None
    refs = [min(idxs, key=lambda i: losses[i])] if losses is not None else idxs
    for ref_idx in refs:
        net, _ = _consensus_from_ref(pop, dims, idxs, ref_idx, eps, quorum)
        if net is not None:
            return net
    return None


@torch.no_grad()
def _consensus_layer_eps_deep(pop, teacher, dims, idxs, ref_idx, eps, quorum):
    """Per-hidden-layer parameter error (max/mean over incoming weights + bias) on
    ONLY the consensus neurons. Rebuilds the quorum-consensus net in member[ref_idx]'s
    frame while tracking which neurons reached quorum, aligns it to the teacher
    (propagating each layer's permutation, and the consensus mask along with it), then
    scores just the consensus neurons. Returns one dict per hidden layer:
    {cons_max, cons_mean} (both None if that layer has no consensus neuron)."""
    aligned, ref = _align_members(pop, idxs, ref_idx)
    cnet = ref.clone()
    L = len(cnet.layers)
    dev = cnet.layers[0].weight.device
    masks = []
    for l in range(L - 1):                              # hidden layers only
        H = cnet.layers[l].weight.shape[0]
        mask = torch.zeros(H, dtype=torch.bool, device=dev)
        for k in range(H):
            feats = torch.stack([torch.cat([a.layers[l].weight[k],
                                            a.layers[l].bias[k:k + 1]])
                                 for a in aligned])
            S = _largest_eps_ball(feats, eps)
            if len(S) >= quorum:
                cnet.layers[l].weight.data[k] = torch.stack(
                    [aligned[s].layers[l].weight[k] for s in S]).mean(0)
                cnet.layers[l].bias.data[k] = torch.stack(
                    [aligned[s].layers[l].bias[k] for s in S]).mean(0)
                mask[k] = True
        masks.append(mask)
    t = teacher.clone(); scale_normalize_(t)           # align consensus net -> teacher
    r = cnet.clone();    scale_normalize_(r)
    out = []
    for l in range(L - 1):
        perm = match_layer_(t, r, l)                   # sign-aware; propagates into l+1
        m = masks[l][torch.tensor(perm, device=dev)]   # carry consensus mask along
        werr = (r.layers[l].weight - t.layers[l].weight).abs()
        berr = (r.layers[l].bias - t.layers[l].bias).abs()
        nmax = torch.maximum(werr.max(1).values, berr)
        nmean = (werr.sum(1) + berr) / (werr.shape[1] + 1)
        if bool(m.any()):
            out.append({"cons_max": float(nmax[m].max()),
                        "cons_mean": float(nmean[m].mean())})
        else:
            out.append({"cons_max": None, "cons_mean": None})
    return out


@torch.no_grad()
def _consensus_stats_deep(pop, teacher, dims, eps, quorum_ratio, hard=False):
    """consensus_neuron_stats for >1 hidden layer: per-hidden-layer quorum counts
    (teacher-free; the reference giving the most agreement is used) plus the
    consensus net's error vs the teacher for scoring."""
    import math
    P = len(pop)
    quorum = math.ceil(quorum_ratio * P)
    idxs = list(range(P))
    best_pl, best_tot, best_ref = [], -1, idxs[0]
    for ref_idx in idxs:                            # pick the most-agreeing frame
        _, per_layer = _consensus_from_ref(pop, dims, idxs, ref_idx, eps, quorum)
        tot = sum(nc for _, nc in per_layer)
        if tot > best_tot:
            best_tot, best_pl, best_ref = tot, per_layer, ref_idx
    layers = [{"n_cons": nc, "n_tot": H} for H, nc in best_pl]
    try:                                            # per-layer eps on consensus neurons
        for d, e in zip(layers, _consensus_layer_eps_deep(
                pop, teacher, dims, idxs, best_ref, eps, quorum)):
            d.update(e)
    except Exception:
        pass
    cnet = build_consensus(pop, dims, quorum_ratio=quorum_ratio, eps=eps)
    if cnet is not None:
        pe = param_errors(cnet, teacher, hard=hard)
        max_eps = pe["max_eps"]
        mean_eps = sum(pe["mean_eps_per_matrix"]) / len(pe["mean_eps_per_matrix"])
    else:
        max_eps = mean_eps = None
    return {"n_consensus": sum(x["n_cons"] for x in layers),
            "n_total": sum(x["n_tot"] for x in layers),
            "max_eps": max_eps, "mean_eps": mean_eps, "layers": layers}


@torch.no_grad()
def _partial_consensus(pop, dims, eps, quorum_ratio, ref_idx=None):
    """Consensus net + per-hidden-layer boolean masks (which neurons reached quorum),
    built in the most-agreeing member's frame (or member `ref_idx`'s frame if given).
    Consensus rows hold the quorum mean; non-consensus (straggler) rows keep that
    reference member's values. Used by the freeze-reinit peel to know what to pin
    and which rows to leave trainable."""
    import math
    P = len(pop)
    quorum = math.ceil(quorum_ratio * P)
    idxs = list(range(P))
    best_ref, best_tot = idxs[0], -1
    for ref_idx in (idxs if ref_idx is None else [ref_idx]):   # most-agreeing frame
        _, per_layer = _consensus_from_ref(pop, dims, idxs, ref_idx, eps, quorum)
        tot = sum(nc for _, nc in per_layer)
        if tot > best_tot:
            best_tot, best_ref = tot, ref_idx
    aligned, ref = _align_members(pop, idxs, best_ref)
    cnet = ref.clone()
    L = len(cnet.layers)
    dev = cnet.layers[0].weight.device
    masks = []
    for l in range(L - 1):                              # hidden layers
        H = cnet.layers[l].weight.shape[0]
        mask = torch.zeros(H, dtype=torch.bool, device=dev)
        for k in range(H):
            feats = torch.stack([torch.cat([a.layers[l].weight[k],
                                            a.layers[l].bias[k:k + 1]])
                                 for a in aligned])
            S = _largest_eps_ball(feats, eps)
            if len(S) >= quorum:
                cnet.layers[l].weight.data[k] = torch.stack(
                    [aligned[s].layers[l].weight[k] for s in S]).mean(0)
                cnet.layers[l].bias.data[k] = torch.stack(
                    [aligned[s].layers[l].bias[k] for s in S]).mean(0)
                mask[k] = True
        masks.append(mask)
    return cnet, masks


def _frozen_fp64_view(net, frozen):
    """Return a net for MEASUREMENT/DELIVERY that carries the fp64-recovered frozen
    rows at full precision. The student `net` is fp32, so the frozen rows pinned into
    it are truncated to fp32 (~6e-8) even when the refiner recovered them to fp64
    (~1e-12); this rebuilds an fp64 clone and overlays the frozen dict's fp64 values,
    so param_errors / the saved model see the true precision. No-op if `frozen` holds
    no fp64 rows (fp32 training path unaffected)."""
    if not frozen or not any(v[0].dtype == torch.float64 for v in frozen.values()):
        return net
    m = net.clone().double()
    with torch.no_grad():
        for l, (W, b, mask) in frozen.items():
            dev = m.layers[l].weight.device
            mm = mask.to(dev)
            m.layers[l].weight.data[mm] = W.to(dev).double()[mm]
            m.layers[l].bias.data[mm] = b.to(dev).double()[mm]
    return m


class _FrontierNet(torch.nn.Module):
    """View of `base` that forwards from layer `frontier` (skipping the frozen
    prefix). Shares base's Parameters, so gradients/opt.step act on `base`.
    --peel-direct uses this so the students are refined against the frontier's
    OWN input h (normal-norm, fp32) -- the huge-norm x = x_of_h(h) roundtrip
    (fp32-noisy at depth >=2, which stalled the deep search) never touches the
    students; only the fp64 teacher query does."""
    def __init__(self, base, frontier):
        super().__init__()
        self.base = base
        self.frontier = frontier

    def forward(self, h):
        a = h
        L = self.base.layers
        for i in range(self.frontier, len(L)):
            a = L[i](a)
            if i < len(L) - 1:
                a = self.base.act(a)
        return a


def _install_freeze(net, frozen):
    """Pin each frozen layer's consensus rows to the shared values and mask their
    gradients so they never move. frozen: {layer: (W, b, row_mask)}. Straggler rows
    (mask False) are left at the member's own fresh init and stay trainable."""
    for l, (W, b, mask) in frozen.items():
        lay = net.layers[l]
        dev = lay.weight.device
        m = mask.to(dev)
        with torch.no_grad():
            # cast to member dtype: frozen may be fp64 (--f64 extraction) while the
            # member is fp32; masked index_put requires matching dtypes
            lay.weight[m] = W.to(dev).to(lay.weight.dtype)[m]
            lay.bias[m] = b.to(dev).to(lay.bias.dtype)[m]
        wkeep = (~m).to(lay.weight.dtype)[:, None]      # 0 on frozen rows, 1 elsewhere
        bkeep = (~m).to(lay.bias.dtype)
        lay.weight.register_hook(lambda g, k=wkeep: g * k)
        lay.bias.register_hook(lambda g, k=bkeep: g * k)


def _reinit_frozen_population(dims, device, P, frozen, act="leaky_relu"):
    """Fresh committee of P members; every layer in `frozen` has its consensus rows
    pinned+frozen and shared across all members (stragglers + deeper layers are fresh
    random, so the search diversifies onto exactly what's unsolved)."""
    pop = []
    for _ in range(P):
        m = MLP(dims, act=act).to(device)
        _install_freeze(m, frozen)
        pop.append(m)
    return pop


@torch.no_grad()
def _angle_between(a, b):
    import math
    a = a / a.norm().clamp_min(1e-30)
    b = b / b.norm().clamp_min(1e-30)
    return math.degrees(math.acos(float((a @ b).abs().clamp(0.0, 1.0))))


def _install_mask_hook(member, mask):
    """Gradient hooks that zero the frozen rows of layer 0 so pinned neurons never
    move. `mask` is a LIVE bool tensor (updated in place as more neurons freeze)."""
    member.layers[0].weight.register_hook(
        lambda g, m=mask: g * (~m).to(g.dtype).unsqueeze(1))
    member.layers[0].bias.register_hook(
        lambda g, m=mask: g * (~m).to(g.dtype))


@torch.no_grad()
def _apply_exact_freeze(pop, opts, freeze_masks, hooked, dims, device, exact,
                        thresh=0.98):
    """Pin every exactly-extracted neuron IN PLACE in each committee member: find the
    member's matching neuron (max cosine to the exact direction), overwrite that row
    with the exact value scaled to the member's OWN norm (so the next layer stays
    valid), and freeze the row. NO reinit -- deeper layers untouched. Idempotent and
    self-healing: re-hooks + re-pins members replaced by combine / dead-reinit."""
    for i, (member, opt) in enumerate(zip(pop, opts)):
        if hooked[i] is not member:                      # new/replaced member -> (re)hook
            freeze_masks[i] = torch.zeros(dims[1], dtype=torch.bool, device=device)
            _install_mask_hook(member, freeze_masks[i])
            hooked[i] = member
        mask = freeze_masks[i]
        W, B = member.layers[0].weight, member.layers[0].bias
        for w_ex, b_ex in exact:
            norms = W.norm(dim=1).clamp_min(1e-30)
            cos = (W @ w_ex.to(W.dtype)) / norms         # signed (LeakyReLU: keep orientation)
            cos = cos.masked_fill(mask, -2.0)            # ignore already-frozen rows
            j = int(cos.argmax())
            if float(cos[j]) < thresh:
                continue                                 # this member lacks it (unfrozen)
            a = float(norms[j])                          # preserve the member's own scale
            W[j] = a * w_ex.to(W.dtype)
            B[j] = a * float(b_ex)
            mask[j] = True
            for p in (W, B):                             # zero Adam momentum on that row
                st = opt.state.get(p)
                if st:
                    if "exp_avg" in st:
                        st["exp_avg"][j] = 0
                    if "exp_avg_sq" in st:
                        st["exp_avg_sq"][j] = 0


def _extract_freeze_round(pop, opts, freeze_masks, hooked, dims, cfg, device,
                          teacher, exact, seed):
    """Probe each layer-1 CONSENSUS neuron with the exact-affine extractor (fresh,
    black-box, ~2d queries). Neurons that come out clean (guarded: clean rank-1 jump +
    within-angle of the guess) are pinned to their exact value IN PLACE across the
    committee and frozen; failures are skipped. Returns a stats dict."""
    from verify_layer1 import extract_neuron_exact, _Oracle, _double_teacher
    cnet, masks = _partial_consensus(pop, dims, cfg.cluster_eps, cfg.cluster_quorum)
    td = _double_teacher(teacher)
    gen = torch.Generator(device=device).manual_seed(seed)
    probed = rejected = 0
    found = []                                           # (jump_ratio, angle) per new neuron
    reasons = {}                                         # reject reason -> count
    rej_jr = []                                          # jump ratios of rejected probes
    for k in range(dims[1]):
        if not bool(masks[0][k]):
            continue                                     # not a consensus neuron
        wk = cnet.layers[0].weight[k].double()
        bk = float(cnet.layers[0].bias[k])
        if any(_angle_between(wk, ew) < 1.0 for ew, _ in exact):
            continue                                     # already solved this neuron
        probed += 1
        res = {"ok": False, "reason": "no_try"}
        for _try in range(cfg.extract_tries):            # retry from several seeds
            base = torch.randn(dims[0], device=device, dtype=torch.float64, generator=gen)
            try:
                res = extract_neuron_exact(_Oracle(td), wk, bk, base, gen=gen,
                                           s=cfg.extract_s, r=cfg.extract_r,
                                           base_scale=cfg.extract_base_scale,
                                           max_angle_deg=cfg.extract_max_angle,
                                           min_jump_ratio=cfg.extract_jump_ratio)
            except Exception as e:
                res = {"ok": False, "reason": "err:" + type(e).__name__}
            if res.get("ok"):
                break
        if not res.get("ok"):                            # (1) failed to extract -> continue
            why = res.get("reason", "error")
            reasons[why] = reasons.get(why, 0) + 1
            if res.get("jump_ratio") is not None:
                rej_jr.append(res["jump_ratio"])
            rejected += 1
            continue
        w_ex = res["w"].to(device).double()
        if any(_angle_between(w_ex, ew) < 1.0 for ew, _ in exact):
            reasons["duplicate"] = reasons.get("duplicate", 0) + 1
            rejected += 1
            continue                                     # duplicate of one we already have
        exact.append((w_ex, float(res["b"])))            # (2) extracted exactly
        found.append((res["jump_ratio"], res["angle_deg"]))
    if exact:                                            # pin+freeze all exact neurons in place
        _apply_exact_freeze(pop, opts, freeze_masks, hooked, dims, device, exact)
    rjstr = (f" | jump_ratio min/med/max {min(rej_jr):.1f}/"
             f"{sorted(rej_jr)[len(rej_jr) // 2]:.1f}/{max(rej_jr):.1f}"
             if rej_jr else "")
    return {"probed": probed, "extracted": len(found), "rejected": rejected,
            "total": len(exact), "found": found, "reasons": reasons, "rjstr": rjstr}


def build_consensus(pop, dims, losses=None, eps=0.02,
                    quorum_ratio=0.75, gate_kappa=10.0):
    """Teacher-free tight-cluster alignment across the committee, then per-unit
    consensus. Returns the consensus NET (reconstructed from committee
    agreement instead of the single best member), or None (n/a) if the
    committee doesn't back it strongly enough. Fully teacher-free.
    Handles any number of hidden layers (>1 hidden layer -> _build_consensus_deep).

    Robustness (prefer n/a over a shaky consensus):
      - loss-gate: only members within `gate_kappa` x the best training loss
        vote, so a collapsed/correlated majority of stuck members can't form a
        spurious cluster (needs `losses`; without it, all members vote).
      - quorum: a unit is only trusted if >= ceil(quorum_ratio * p) members
        agree tightly (default 0.75 => 6 of 8). If fewer well-fit members than
        the quorum exist, or any unit falls short, the whole thing is n/a.

    Method: scale-normalize gated members, connect units that are mutual
    nearest neighbours AND within `eps` (max-element distance), union-find into
    clusters (one per teacher unit), average each cluster. Stragglers are loners
    below quorum; collapsed members are gated out entirely."""
    import math
    if len(pop) < 2 or len(dims) < 3:
        return None
    if len(dims) > 3:                                # >1 hidden layer
        return _build_consensus_deep(pop, dims, losses, eps, quorum_ratio, gate_kappa)
    P, H = len(pop), dims[1]
    quorum = math.ceil(quorum_ratio * P)          # ratio of the FULL committee

    idxs = list(range(P))
    if losses is not None and gate_kappa > 0:
        bl = min(losses)
        idxs = [i for i in range(P) if losses[i] <= gate_kappa * bl]
    if len(idxs) < quorum:
        return None  # too few well-fit members to reach quorum -> n/a

    mem = [pop[i].clone() for i in idxs]
    for r in mem:
        scale_normalize_(r)
    g = len(mem)
    F = [torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1)
         for r in mem]
    parent = list(range(g * H))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(g):
        for b in range(a + 1, g):
            D = torch.cdist(F[a].float(), F[b].float(), p=float("inf"))
            nn_ab, nn_ba = D.argmin(1), D.argmin(0)
            for i in range(H):
                j = int(nn_ab[i])
                if int(nn_ba[j]) == i and D[i, j] < eps:
                    ra, rb = find(a * H + i), find(b * H + j)
                    if ra != rb:
                        parent[ra] = rb

    from collections import defaultdict
    clusters = defaultdict(list)
    for node in range(g * H):
        clusters[find(node)].append(node)
    good = sorted((v for v in clusters.values() if len(v) >= quorum),
                  key=len, reverse=True)[:H]
    if len(good) < H:
        return None  # some unit lacks a quorum -> n/a (prefer n/a over shaky)

    net = mem[0].clone()
    for k, comp in enumerate(good):
        rows = torch.stack([mem[n // H].layers[0].weight[n % H] for n in comp])
        bias = torch.stack([mem[n // H].layers[0].bias[n % H] for n in comp])
        outs = torch.stack([mem[n // H].layers[1].weight[:, n % H]
                            for n in comp])
        net.layers[0].weight.data[k] = rows.mean(0)
        net.layers[0].bias.data[k] = bias.mean(0)
        net.layers[1].weight.data[:, k] = outs.mean(0)
    net.layers[1].bias.data = mem[0].layers[1].bias.clone()
    return net


@torch.no_grad()
def consensus_neuron_stats(pop, teacher, dims, eps=0.02, quorum_ratio=0.625,
                           hard=False):
    """Per-neuron consensus diagnostic (single hidden layer, ungated -- matches
    the logged consensus). Instead of the all-or-nothing full consensus, report
    how many hidden neurons reached a quorum consensus across the committee, and
    the max/mean parameter error on JUST those neurons (each averaged over its
    cluster, then aligned to the teacher). Teacher used only for scoring.
    hard=True: quotient the label-preserving head family before ANY head
    comparison (columns vs teacher, output-unit clustering, out eps) -- center
    every head and scale-fit each member to the centered teacher head, so the
    stats measure the identifiable content in teacher scale (comparable to
    param_errors' hard L2) instead of the xent scale drift.
    Returns {n_consensus, n_total, max_eps, mean_eps} or None."""
    import math
    from collections import defaultdict
    if len(pop) < 2 or len(dims) < 3:
        return None
    if len(dims) > 3:                                # >1 hidden layer
        return _consensus_stats_deep(pop, teacher, dims, eps, quorum_ratio,
                                     hard=hard)
    P, H = len(pop), dims[1]
    quorum = math.ceil(quorum_ratio * P)
    mem = [m.clone() for m in pop]
    for r in mem:
        scale_normalize_(r)
    if hard:
        th = teacher.clone(); scale_normalize_(th)
        tW = th.layers[1].weight; tb = th.layers[1].bias
        v = torch.cat([(tW - tW.mean(0, keepdim=True)).flatten(),
                       tb - tb.mean()])
        vn = v.norm()
        for r in mem:
            # members are NOT yet permutation-aligned to the teacher here, so
            # an inner-product LSQ scale would pair mismatched columns (~0);
            # the Frobenius-norm ratio is permutation-invariant and s>0.
            W, b = r.layers[1].weight, r.layers[1].bias
            W.sub_(W.mean(0, keepdim=True)); b.sub_(b.mean())
            u = torch.cat([W.flatten(), b])
            s = float(vn / u.norm().clamp_min(1e-30))
            W.mul_(s); b.mul_(s)
    F = [torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1) for r in mem]
    parent = list(range(P * H))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a in range(P):
        for b in range(a + 1, P):
            D = torch.cdist(F[a].float(), F[b].float(), p=float("inf"))
            ab, ba = D.argmin(1), D.argmin(0)
            for i in range(H):
                j = int(ab[i])
                if int(ba[j]) == i and D[i, j] < eps:
                    ra, rb = find(a * H + i), find(b * H + j)
                    if ra != rb:
                        parent[ra] = rb
    clusters = defaultdict(list)
    for node in range(P * H):
        clusters[find(node)].append(node)
    good = [v for v in clusters.values() if len(v) >= quorum]
    if not good:
        return {"n_consensus": 0, "n_total": H, "max_eps": None, "mean_eps": None}

    rows = torch.stack([torch.stack([mem[n // H].layers[0].weight[n % H]
                                     for n in c]).mean(0) for c in good])
    bias = torch.stack([torch.stack([mem[n // H].layers[0].bias[n % H]
                                     for n in c]).mean(0) for c in good])
    outs = torch.stack([torch.stack([mem[n // H].layers[1].weight[:, n % H]
                                     for n in c]).mean(0) for c in good])
    # align consensus neurons to teacher (scale-normalized); greedy L1 match
    t = teacher.clone(); scale_normalize_(t)
    if hard:                                # score heads in the centered frame
        t.layers[1].weight.sub_(t.layers[1].weight.mean(0, keepdim=True))
        t.layers[1].bias.sub_(t.layers[1].bias.mean())
    TF = torch.cat([t.layers[0].weight, t.layers[0].bias[:, None]], 1)
    CF = torch.cat([rows, bias[:, None]], 1)
    Dm = torch.cdist(CF.float(), TF.float(), p=1)
    l0d, l1d = [], []                       # L0 = input weights+bias, L1 = output col
    match_ti = [0] * len(good)              # teacher hidden neuron matched to cluster ci
    for _ in range(len(good)):
        flat = int(Dm.argmin()); ci, ti = flat // H, flat % H
        l0d.append((rows[ci] - t.layers[0].weight[ti]).abs())
        l0d.append((bias[ci] - t.layers[0].bias[ti]).abs().reshape(1))
        l1d.append((outs[ci] - t.layers[1].weight[:, ti]).abs())
        match_ti[ci] = ti
        Dm[ci, :] = float("inf"); Dm[:, ti] = float("inf")
    L0 = torch.cat(l0d); L1 = torch.cat(l1d); alld = torch.cat([L0, L1])
    out = {"n_consensus": len(good), "n_total": H,
           "max_eps": alld.max().item(), "mean_eps": alld.mean().item(),
           "l0_max": L0.max().item(), "l0_mean": L0.mean().item(),
           "l1_max": L1.max().item(), "l1_mean": L1.mean().item()}

    # --- output-unit consensus: for each of the O output units, do the members
    #     agree on its bias AND its weights to the consensus hidden neurons? ---
    O = dims[2]
    bias_out = torch.stack([r.layers[1].bias for r in mem])          # (P, O)
    agree = (bias_out.max(0).values - bias_out.min(0).values) < eps  # (O,)
    for c in good:
        w = torch.stack([mem[n // H].layers[1].weight[:, n % H] for n in c])  # (|c|,O)
        agree = agree & ((w.max(0).values - w.min(0).values) < eps)
    n_out = int(agree.sum())
    out["n_out_consensus"] = n_out
    out["out_total"] = O
    if n_out > 0:
        ti_idx = torch.tensor(match_ti, device=outs.device)
        bias_err = (bias_out.mean(0) - t.layers[1].bias).abs()       # (O,)
        w_err = (outs.T - t.layers[1].weight[:, ti_idx]).abs()       # (O, K)
        od = torch.cat([bias_err[agree].reshape(-1), w_err[agree].reshape(-1)])
        out["out_max"] = od.max().item()
        out["out_mean"] = od.mean().item()
    else:
        out["out_max"] = None
        out["out_mean"] = None
    return out


@torch.no_grad()
def consensus_layer1_neurons(pop, dims, eps=0.02, quorum_ratio=0.625):
    """Layer-1 consensus neurons, teacher-free, for ANY depth. Layer 1's incoming
    rows live in input space and don't depend on any prior layer's permutation, so
    the same mutual-NN eps-ball clustering used by the single-hidden-layer
    consensus applies unchanged to the first weight matrix of a deep net. Returns a
    list of (idx, w, b): the consensus index, the cluster-mean incoming row, and
    bias -- one entry per layer-1 neuron that reached quorum. (w, b) are in the
    scale-normalized frame (unit-norm row), i.e. exactly a hypothesized hyperplane
    {x : w.x + b = 0} ready to hand to verify_layer1."""
    from collections import defaultdict
    P = len(pop)
    if P < 2 or len(dims) < 3:
        return []
    H = dims[1]
    quorum = math.ceil(quorum_ratio * P)
    mem = [m.clone() for m in pop]
    for r in mem:
        scale_normalize_(r)
    F = [torch.cat([r.layers[0].weight, r.layers[0].bias[:, None]], 1) for r in mem]
    parent = list(range(P * H))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a in range(P):
        for b in range(a + 1, P):
            D = torch.cdist(F[a].float(), F[b].float(), p=float("inf"))
            ab, ba = D.argmin(1), D.argmin(0)
            for i in range(H):
                j = int(ab[i])
                if int(ba[j]) == i and D[i, j] < eps:
                    ra, rb = find(a * H + i), find(b * H + j)
                    if ra != rb:
                        parent[ra] = rb
    clusters = defaultdict(list)
    for node in range(P * H):
        clusters[find(node)].append(node)
    good = [c for c in clusters.values() if len(c) >= quorum]
    out = []
    for ci, c in enumerate(good):
        w = torch.stack([mem[n // H].layers[0].weight[n % H] for n in c]).mean(0)
        b = torch.stack([mem[n // H].layers[0].bias[n % H] for n in c]).mean(0)
        out.append((ci, w.detach(), b.detach()))
    return out


@torch.no_grad()
def cluster_consensus(pop, teacher, dims, losses=None, eps=0.02,
                      quorum_ratio=0.75, gate_kappa=10.0):
    """Diagnostic wrapper around build_consensus: returns the consensus net's
    max parameter error (teacher used only to score), or None (n/a)."""
    net = build_consensus(pop, dims, losses=losses, eps=eps,
                          quorum_ratio=quorum_ratio, gate_kappa=gate_kappa)
    return None if net is None else param_errors(net, teacher)["max_eps"]


def solver_polish_(net, X, Y, mse_steps=15, mae_steps=15,
                   verbose=False, tag="", bs=8192):
    """Tighten a single net's fit in-place with the staged LBFGS recipe:
    MSE (descend into the basin) then MAE (constant gradient finishes the flat
    directions MSE's vanishing gradient abandons). Modifies net.parameters()
    in place so the caller's optimizer stays valid. GPU float32.

    X/Y may live on CPU: the full query matrix can exceed GPU memory at large
    input dims, so the closure streams `bs`-row chunks to the net's device and
    accumulates the gradient across all chunks into .grad before each LBFGS
    step -- identical to one full-batch backward, just bounded in peak memory."""
    dev = next(net.parameters()).device
    dt = next(net.parameters()).dtype     # follow the net's precision (fp32/fp64)
    denom = X.shape[0] * Y.shape[1]
    for kind, steps in (("mse", mse_steps), ("mae", mae_steps)):
        opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=20,
                                history_size=20, line_search_fn="strong_wolfe")
        n_eval = [0]

        def closure():
            n_eval[0] += 1
            opt.zero_grad()
            total = 0.0
            for i in range(0, X.shape[0], bs):
                xb, yb = X[i:i + bs].to(dev, dt), Y[i:i + bs].to(dev, dt)
                r = net(xb) - yb
                loss = (r ** 2).sum() if kind == "mse" else r.abs().sum()
                (loss / denom).backward()
                total += loss.item() / denom
            return total
        if verbose:
            t0 = time.time()
            l0 = l1_on([net], X, Y)[0]
        for _ in range(steps):
            opt.step(closure)
        if verbose:
            print(f"    [polish]{tag} {kind}: L1 {l0:.3e}->"
                  f"{l1_on([net], X, Y)[0]:.3e}  {n_eval[0]} evals  "
                  f"{time.time() - t0:.1f}s", flush=True)
    return net


def polish_consensus(net, X, Y, lr=1e-4, steps=200, max_samples=30000,
                     hard=False):
    """Gentle low-LR polish of a consensus net on query MSE (hard=True:
    cross-entropy on hard labels). The consensus has
    good parameters but a small function-assembly artifact from averaging
    scale-normalized members (=> high loss); a low-LR pass removes it (loss
    drops ~100-200x) while the recovered parameters hold (max_eps stable). A
    higher LR lowers loss faster but starts pulling weakly-identified units back
    toward their query-loss-optimal (wrong) values, so keep it low. Returns a
    fresh polished clone; does not mutate `net`."""
    net = net.clone()
    dev = next(net.parameters()).device
    Xs = (X[-max_samples:] if len(X) > max_samples else X).to(dev)
    Ys = (Y[-max_samples:] if len(Y) > max_samples else Y).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        # hard=True: Y holds class indices -- same objective the members train
        # on. The consensus/combine mechanics are teacher-free either way.
        loss = (F.cross_entropy(net(Xs), Ys) if hard
                else ((net(Xs) - Ys) ** 2).mean())
        loss.backward()
        opt.step()
    return net


# -------------------------------------------------------------- evaluation --
@torch.no_grad()
def batched(net, X, bs=4096):
    return torch.cat([net(X[i:i + bs]) for i in range(0, len(X), bs)])


@torch.no_grad()
def l1_on(pop, X, Y, bs=4096):
    losses = []
    for net in pop:
        dev = next(net.parameters()).device
        dt = next(net.parameters()).dtype
        tot, n = 0.0, 0
        for i in range(0, len(X), bs):
            xb, yb = X[i:i + bs].to(dev, dt), Y[i:i + bs].to(dev, dt)
            tot += (net(xb) - yb).abs().sum().item()
            n += yb.numel()
        losses.append(tot / n)
    return losses


@torch.no_grad()
def mse_on(pop, X, Y, bs=4096):
    """Per-output-element MSE of each member on (X, Y). Mirrors l1_on (which is
    the per-element MAE) so the two share denominators and are comparable."""
    losses = []
    for net in pop:
        dev = next(net.parameters()).device
        dt = next(net.parameters()).dtype
        tot, n = 0.0, 0
        for i in range(0, len(X), bs):
            xb, yb = X[i:i + bs].to(dev, dt), Y[i:i + bs].to(dev, dt)
            tot += ((net(xb) - yb) ** 2).sum().item()
            n += yb.numel()
        losses.append(tot / n)
    return losses


@torch.no_grad()
def xent_on(pop, X, Y, bs=4096):
    """Per-sample mean cross-entropy of each member on (X, Y) where Y holds HARD
    class indices (cfg.hard). The hard-label counterpart of l1_on: used for
    member ranking/gating when residual losses don't apply."""
    losses = []
    for net in pop:
        dev = next(net.parameters()).device
        dt = next(net.parameters()).dtype
        tot, n = 0.0, 0
        for i in range(0, len(X), bs):
            xb, yb = X[i:i + bs].to(dev, dt), Y[i:i + bs].to(dev)
            tot += F.cross_entropy(net(xb), yb, reduction="sum").item()
            n += len(yb)
        losses.append(tot / n)
    return losses


@torch.no_grad()
def agreement(net, teacher, X, bs=4096):
    pa = batched(net, X, bs).argmax(1)
    ta = batched(teacher, X, bs).argmax(1)
    return (pa == ta).float().mean().item()


# ------------------------------------------------------------ main routine --
def _omp_select(G, C, forced, n_pick, block=16, coh=0.95):
    """Greedy group-OMP column selection in the Gram domain. G [K,K] = D^T D,
    C [K,T] = D^T Z (fp64; D = candidate activations, Z = pooled targets).
    `forced` columns (const + pinned rows) are always in the basis and don't
    count toward n_pick. Picks n_pick further columns by residual explanatory
    power over ALL targets jointly, least-squares-refitting after every block
    of `block` picks. Near-duplicates of a same-block pick are deferred via
    the coherence gate (they get rescored next refit). Returns pick list."""
    dev = G.device
    Gs, Cs = G.float(), C.float()
    gdiag = Gs.diagonal().clamp_min(1e-30)
    lam0 = 1e-8 * float(gdiag.mean())
    picked = []
    while len(picked) < n_pick:
        S = torch.tensor(forced + picked, device=dev, dtype=torch.long)
        lam, L = lam0, None
        for _ in range(4):                       # ridge escalation on failure
            try:
                L = torch.linalg.cholesky(
                    Gs[S][:, S].double()
                    + lam * torch.eye(len(S), device=dev, dtype=torch.float64))
                break
            except Exception:
                lam *= 100
        if L is None:
            break
        L = L.float()
        A = torch.cholesky_solve(Cs[S], L)                 # [s,T] refit coeffs
        GkS = Gs[:, S]                                     # [K,s]
        resC = Cs - GkS @ A                                # residual correlations
        B = torch.cholesky_solve(GkS.T.contiguous(), L)    # [s,K]
        dres = (gdiag - (GkS.T * B).sum(0)).clamp_min(0)   # residual col energy
        score = (resC ** 2).sum(1) / dres.clamp_min(1e-12 * float(gdiag.mean()))
        score[S] = float("-inf")
        score[dres < 1e-7 * gdiag] = float("-inf")         # spanned duplicates
        take = []
        want = min(block, n_pick - len(picked))
        for j in torch.argsort(score, descending=True).tolist():
            if len(take) >= want or not torch.isfinite(score[j]) or score[j] <= 0:
                break
            if all(abs(float(Gs[j, i])) <
                   coh * float((gdiag[j] * gdiag[i]).sqrt()) for i in take):
                take.append(j)
        if not take:
            break                                          # dictionary exhausted
        picked += take
    return picked


def finetune_(net, X, Y, device, epochs, batch, lr, gen, tol=0.0):
    """Anneal-to-tolerance polish of a merged net on (X, Y): plateau-triggered
    lr decay down to 1e-6, stop early once train L1 < tol. The pooled-neuron
    rank bound caps what SELECTION can preserve (linear combos of existing
    neurons), but the net's own capacity is not rank-bounded -- distilling to
    tolerance re-represents the buffer with fresh neurons. Frozen-row hooks
    (if installed) keep pinned rows in place."""
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                           lr=lr)
    n = len(X)
    cur, best, stall = lr, float("inf"), 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen, device=device)
        el, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch].cpu()
            xb, yb = X[idx].to(device), Y[idx].to(device)
            loss = (net(xb) - yb).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            el += loss.item()
            nb += 1
        el /= max(nb, 1)
        if tol > 0 and el < tol:
            break
        if el < best * 0.997:
            best, stall = el, 0
            continue
        stall += 1
        if stall >= 8:
            if cur <= 1e-6 * 1.01:
                break
            cur *= 0.3
            stall = 0
            for g in opt.param_groups:
                g["lr"] = cur
    return net


@torch.no_grad()
def merge_ensemble(pop, dims, act_name, X, Y, device, frozen=None,
                   partial_exact=None, partial_mask=None, samples=200_000,
                   chunk=16384, block=16, hard=False, weights=None):
    """Merge a same-architecture population into ONE net of those dims.

    weights: optional per-member scalars multiplying that member's next-layer
    preactivation TARGETS -- selection then prioritizes members by their
    contribution to the ensemble sum (e.g. cascade stage scales) instead of
    treating all members equally. Leaky-ReLU is positively homogeneous, so the
    scaled reconstructions stay exact up to downstream LS gauge; ignored for
    sigmoid/tanh (not homogeneous).

    Per hidden layer h (frontier down): dictionary = every member's layer-h
    neuron activations (verbatim at the frontier, where all members share the
    pinned exact prefix; reconstructed over the merged basis above it), targets
    = every member's layer-(h+1) preactivations on a query sample. Group-OMP
    selects the width-n subset that best linearly explains all targets jointly;
    the least-squares refit coefficients ARE the next layer's rows over the
    selected basis. The output layer is ridge-solved against the blackbox
    labels Y (member-averaged under --hard). Attacker-side only: queries +
    member weights, never the oracle. Peel-pinned rows (frozen/partial) keep
    their slot and exact values -- selection only fills unsolved slots, so the
    merged net stays consistent with the peel bookkeeping.

    Returns (net, info) -- net is None (info = reason) when there is nothing
    to select."""
    t0 = time.time()
    M, Lh = len(pop), len(dims) - 2

    def _act(z):
        if act_name == "sigmoid":
            return torch.sigmoid(z)
        if act_name == "tanh":
            return torch.tanh(z)
        return F.leaky_relu(z, 0.01)

    # --- pinned rows {layer: (W, b, mask)} merged from peel frozen + partial ---
    pin = {}
    for l in range(Lh):
        m = torch.zeros(dims[l + 1], dtype=torch.bool)
        W = torch.zeros(dims[l + 1], dims[l])
        b = torch.zeros(dims[l + 1])
        if partial_exact is not None and partial_mask and l in partial_mask:
            pmk = partial_mask[l].cpu()
            W[pmk] = partial_exact.layers[l].weight.data[pmk].float().cpu()
            b[pmk] = partial_exact.layers[l].bias.data[pmk].float().cpu()
            m |= pmk
        if frozen and l in frozen:
            fm = frozen[l][2].cpu()
            W[fm] = frozen[l][0].float().cpu()[fm]
            b[fm] = frozen[l][1].float().cpu()[fm]
            m |= fm
        if bool(m.any()):
            pin[l] = (W, b, m)
    fl = next((l for l in range(Lh)
               if l not in pin or not bool(pin[l][2].all())), Lh)
    if fl == Lh:
        return None, "all hidden layers fully pinned; nothing to select"

    n_take = min(samples, len(X))
    ridx = torch.randperm(len(X))[:n_take]
    # held-out slice: never touches the fits; selects the head ridge and is
    # returned (val_idx) so the caller can gate injection out-of-sample --
    # the collinear-basis head explosion wins in-sample and loses here.
    n_val = max(n_take // 5, 1) if not hard else 0
    vidx, fidx = ridx[:n_val], ridx[n_val:]
    Xs = X[fidx].float()
    Ys = None if hard else Y[fidx].float()
    Xv = X[vidx].float() if n_val else None
    Yv = Y[vidx].float() if n_val else None
    n_take = len(fidx)

    net = MLP(dims, act=act_name).to(device)
    for l in range(fl):                    # fully pinned prefix: copy verbatim
        W, b, _ = pin[l]
        net.layers[l].weight.data.copy_(W.to(device))
        net.layers[l].bias.data.copy_(b.to(device))

    info = {"levels": [], "frontier": fl + 1}
    graft = None          # [1+n_prev, M*n_out] fp32: [const;g_prev] -> member z_h
    last = None           # (S basis -> merged-row map, G, C, Cy) of final level
    for h in range(fl, Lh):
        n_out, n_next = dims[h + 1], dims[h + 2]
        pm = pin.get(h)
        pmask = pm[2] if pm else torch.zeros(n_out, dtype=torch.bool)
        pidx = pmask.nonzero(as_tuple=True)[0]
        free_rows = (~pmask).nonzero(as_tuple=True)[0].tolist()
        free_slots = list(free_rows)
        cand_mask = (~pmask).repeat(M)             # member-major candidate cols
        nP = len(pidx)
        K = 1 + nP + M * len(free_rows)
        T = M * n_next
        pWd = pm[0].to(device)[pidx.to(pm[0].device)].to(device) if nP else None
        pbd = pm[1].to(device)[pidx.to(pm[1].device)].to(device) if nP else None
        G = torch.zeros(K, K, dtype=torch.float64, device=device)
        C = torch.zeros(K, T, dtype=torch.float64, device=device)
        Cy = (torch.zeros(K, dims[-1], dtype=torch.float64, device=device)
              if (h == Lh - 1 and Ys is not None) else None)
        zz = 0.0
        wts = (weights if (weights is not None
                          and act_name in ("relu", "leaky_relu"))
               else [1.0] * M)
        for i0 in range(0, n_take, chunk):
            xb = Xs[i0:i0 + chunk].to(device)
            Zt, Hown = [], []
            for mi, mem in enumerate(pop):         # member forwards: own h_h + z_{h+1}
                hm = xb
                for j in range(h + 1):
                    hm = _act(mem.layers[j](hm))
                Hown.append(hm)
                Zt.append(mem.layers[h + 1](hm) * wts[mi])
            Zt = torch.cat(Zt, 1)
            g = xb                                 # merged prefix below h
            for j in range(h):
                g = _act(net.layers[j](g))
            cols = [torch.ones(len(xb), 1, device=device)]
            if nP:
                cols.append(_act(g @ pWd.T + pbd))
            if h == fl:
                # members share the exact pinned prefix -> rows verbatim
                for hm in Hown:
                    cols.append(hm[:, ~pmask])
            else:
                u = torch.cat([torch.ones(len(xb), 1, device=device), g], 1)
                cols.append(_act(u @ graft)[:, cand_mask])
            D = torch.cat(cols, 1)
            G += (D.T @ D).double()
            C += (D.T @ Zt).double()
            zz += float((Zt.double() ** 2).sum())
            if Cy is not None:
                Cy += (D.T @ Ys[i0:i0 + chunk].to(device)).double()
        forced = list(range(1 + nP))
        picked = _omp_select(G, C, forced, len(free_slots), block=block)
        # --- build merged layer h ---
        Wl, bl = net.layers[h].weight.data, net.layers[h].bias.data
        if pm:
            msk = pmask.to(device)
            Wl[msk] = pm[0].to(device)[msk]
            bl[msk] = pm[1].to(device)[msk]
        for slot, j in zip(free_slots, picked):
            mi, r = divmod(j - 1 - nP, len(free_rows))
            r = free_rows[r]
            if h == fl:
                Wl[slot] = pop[mi].layers[h].weight.data[r]
                bl[slot] = pop[mi].layers[h].bias.data[r]
            else:
                a = graft[:, mi * n_out + r]
                bl[slot] = a[0]
                Wl[slot] = a[1:]
        # leftover slots (dictionary exhausted) keep their fresh random init
        # --- final LS refit on the chosen basis = grafts for the next level ---
        # NB ridge is deliberately stronger than the OMP-internal one: the pooled
        # basis is highly collinear (members duplicate each other's features) and
        # a near-zero ridge produces huge canceling coefficients that overfit the
        # buffer and explode off-distribution (seen as max_eps blowup on L_out).
        S = forced + picked
        St = torch.tensor(S, device=device, dtype=torch.long)
        lam = 1e-6 * float(G.diagonal().mean())
        A_S = torch.linalg.solve(
            G[St][:, St] + lam * torch.eye(len(St), device=device,
                                           dtype=torch.float64), C[St])
        # basis entry -> merged row of layer h (-1 = const)
        row_of = ([-1] + pidx.tolist()
                  + [free_slots[i] for i in range(len(picked))])
        graft_next = torch.zeros(1 + n_out, T, device=device)
        for k, row in enumerate(row_of):
            graft_next[0 if row < 0 else 1 + row] = A_S[k].float()
        expl = 1.0 - max(0.0, zz - float((A_S * C[St]).sum())) / max(zz, 1e-30)
        info["levels"].append({"layer": h + 1, "picked": len(picked),
                               "slots": len(free_slots),
                               "explained": round(expl, 6)})
        graft = graft_next
        last = (row_of, St, G, C, Cy, lam)

    # --- output layer over the final selected basis ---
    row_of, St, G, C, Cy, lam = last
    eye = torch.eye(len(St), device=device, dtype=torch.float64)
    GS = G[St][:, St]

    def _scatter_head(Wout):
        Wo = net.layers[Lh].weight.data
        Wo.zero_()
        for k, row in enumerate(row_of):
            if row < 0:
                net.layers[Lh].bias.data.copy_(Wout[k].float())
            else:
                Wo[:, row] = Wout[k].float()

    if Cy is not None:
        # ridge grid for the head, selected on the HELD-OUT slice: the pooled
        # basis is highly collinear and a near-zero ridge interpolates the
        # buffer with huge canceling weights that detonate off-distribution
        # (runaway seen as L_out max_eps 1e3->1e6 across merge cycles).
        gv = None
        if Xv is not None and len(Xv):
            gvs = []
            for i0 in range(0, len(Xv), chunk):
                gg = Xv[i0:i0 + chunk].to(device)
                for j in range(Lh):
                    gg = _act(net.layers[j](gg))
                gvs.append(gg)
            gv = torch.cat(gvs)
        dm = float(G.diagonal().mean())
        best = None                          # (val_loss, Wout, scale)
        for s in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            Wtry = torch.linalg.solve(GS + (dm * s) * eye, Cy[St])
            if gv is None:
                best = (None, Wtry, s)
                break
            Wfull = torch.zeros(dims[-1], dims[Lh], device=device)
            bias = torch.zeros(dims[-1], device=device)
            for k, row in enumerate(row_of):
                if row < 0:
                    bias = Wtry[k].float()
                else:
                    Wfull[:, row] = Wtry[k].float()
            vl = float((gv @ Wfull.T + bias - Yv.to(device)).abs().mean())
            if best is None or vl < best[0]:
                best = (vl, Wtry, s)
        _scatter_head(best[1])
        info["val_loss"] = best[0]
        info["head_ridge"] = best[2]
    else:                                    # hard: average the member heads
        Wout = torch.linalg.solve(GS + lam * eye, C[St])
        _scatter_head(Wout.view(len(St), M, dims[-1]).mean(1))
    info["val_idx"] = vidx.cpu() if n_val else None
    info["wall_s"] = round(time.time() - t0, 1)
    return net, info


def reconstruct(teacher, dims, cfg: Cfg, device, teacher_eval_pts, seed=0,
                save_recon=None, input_transform=None):
    # input_transform: optional callable applied to every generated query batch
    # BEFORE it hits the teacher / students / storage. Used by the peel pipeline
    # to run the population in a SEALED subnet's activation space -- queries are
    # sampled unbounded (gaussian) then squashed into the valid activation range
    # (e.g. sigmoid(.)->(0,1)) so x_of_h can invert them. None => unchanged.
    if cfg.design_refine:
        if cfg.hard or cfg.act not in ("relu", "leaky_relu"):
            raise ValueError("design_refine requires real-valued outputs and a ReLU/leaky-ReLU MLP")
        cfg.f64 = True
    if cfg.hard:
        bad = [k for k, on in [("solver_polish", cfg.solver_polish),
                               ("lastlayer_every", cfg.lastlayer_every),
                               ("lbfgs_polish", cfg.lbfgs_polish),
                               ("polish_f64", cfg.polish_f64),
                               ("verify", cfg.verify),
                               ("peel_try", cfg.peel_try),
                               ("ensemble_boost", cfg.ensemble_boost),
                               ("extract_freeze", cfg.extract_freeze)] if on]
        if bad:
            raise ValueError(f"cfg.hard: {bad} need real-valued teacher outputs "
                             "(regression solve / kink probing); disable them "
                             "in the hard-label regime")
    if cfg.peel_try and cfg.expand != 1.0:
        raise ValueError("cfg.peel_try needs student dims == teacher dims "
                         "(the refiner writes exact rows); incompatible with expand")
    if cfg.peel_try and cfg.factor_tail:
        raise ValueError("cfg.peel_try reinits the committee with plain Adam; "
                         "incompatible with factor_tail (stale factored opts)")
    loss_on = xent_on if cfg.hard else l1_on   # member ranking/gating metric
    gen = torch.Generator(device=device).manual_seed(seed)
    # Expand-and-Cluster fit gate: widen hidden layers (input/output fixed).
    student_dims = ([dims[0]]
                    + [max(1, int(round(dims[i] * cfg.expand)))
                       for i in range(1, len(dims) - 1)]
                    + [dims[-1]])
    expanded = student_dims != list(dims)
    if expanded:
        print(f"[expand] student dims {student_dims} vs teacher {list(dims)} "
              f"(x{cfg.expand}); weight scoring/consensus disabled, loss only.",
              flush=True)
    pop = [MLP(student_dims, act=cfg.act).to(device) for _ in range(cfg.p)]
    flayers = (resolve_layers(cfg.factor_layers, len(pop[0].layers))
               if cfg.factor_tail else None)

    def _mk_fac(n, base_lr):
        if cfg.factor_mode == "residual":
            return AdditiveResidual(n, flayers, base_lr,
                                    base_lr * cfg.factor_lr_mult, cfg.factor_rank)
        return FactoredTail(n, flayers, base_lr * cfg.factor_lr_mult,
                            cfg.factor_inner)

    if cfg.factor_tail:
        # reparameterized layers' weights handled by _mk_fac; the ordinary Adam
        # keeps every bias + the weights of the non-reparameterized layers.
        opts, facs = [], []
        for n in pop:
            ordinary = []
            for l in range(len(n.layers)):
                ordinary.append(n.layers[l].bias)
                if l not in flayers:
                    ordinary.append(n.layers[l].weight)
            opts.append(torch.optim.Adam(ordinary, lr=cfg.lr))
            facs.append(_mk_fac(n, cfg.lr))
    else:
        opts = [torch.optim.Adam(n.parameters(), lr=cfg.lr) for n in pop]
        facs = [None] * cfg.p
    # query buffer lives on CPU/host RAM; minibatches are streamed to `device`
    # during training so the full accumulated query set never sits on the GPU.
    X = torch.empty(0, dims[0])
    Y = (torch.empty(0, dtype=torch.long) if cfg.hard
         else torch.empty(0, dims[-1]))
    decay_at = {int(s * cfg.outer) for s in cfg.lr_sched}
    log = []
    t0 = time.time()
    best = None
    stop_hits = 0
    combined_done = False
    peel_refinement_queries = 0
    frozen = {}                       # freeze-reinit peel: {layer: (W, b, row_mask)}
    exact_neurons = []                # extract-freeze: exactly-probed L1 neurons (w_unit,b)
    freeze_masks = [None] * cfg.p     # extract-freeze: per-member frozen-row masks
    hooked = [None] * cfg.p           # member objects whose freeze hooks are installed
    exact_net = None                  # peel-try: persistent source-of-truth for refined rows
    exact_mask = {}                   # peel-try: {layer: bool[out]} which neurons are kink-exact
    peel_try_n = 0                    # peel-try attempt counter (rotates start neuron)
    peel_stall = 0                    # consecutive no-progress peel-try attempts
    peel_stuck_best = {}              # {frontier: best solved count seen}
    peel_hooked = set()               # {(member_idx, layer)}: live-mask grad hook installed
    peel_live = {}                    # {(member_idx, layer): live bool mask (updated in place)}
    out_solved = False                # cheat-peel: linear output layer LSQ-solved once
    partial_exact = None              # --partial: persistent guess-scale exact rows
    partial_mask = {}                 # --partial: {layer: bool[out]} partial-frozen neurons
    partial_hooked = set()            # --partial: {(id(member), layer)} live-mask hook installed
    partial_live = {}                 # --partial: {(id(member), layer): live bool mask}
    stuck_pf = None                   # --restart-stuck: frontier layer being tracked
    stuck_last_count = 0              # --restart-stuck: last-seen solved count on stuck_pf
    stuck_last_growth = 0             # --restart-stuck: global t at last solved-count increase
    stuck_restarts = 0                # --restart-stuck: number of stuck-restarts fired so far
    retries_done = 0                  # --retry: retries fired so far
    boost_stages = []                 # ensemble-boost: FROZEN cascade [(net, s)];
                                      # the attacker model is the SUM -- the
                                      # validated zero-loss recipe. NO per-stage
                                      # compression; merge happens terminally.
    boost_scale = 1.0                 # current stage residual scale (renorm)
    boost_P = torch.empty(0, dims[-1])    # cached cascade predictions on X (CPU)

    def _cascade(x):
        out = boost_stages[0][0](x) * boost_stages[0][1]
        for _n, _s in boost_stages[1:]:
            out = out + _n(x) * _s
        return out

    def _cascade_l1(Xb, Yb, bs=8192):
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(Xb), bs):
                xb = Xb[i:i + bs].to(device)
                tot += ((_cascade(xb) - Yb[i:i + bs].to(device))
                        .abs().mean().item() * len(xb))
        return tot / max(len(Xb), 1)
    Yfit = Y                          # member training target (Y, or residual)
    cheat_bb = None
    if cfg.cheat:
        # cheat: the blackbox joins gen_queries as a frozen extra member.
        # Gradients must flow THROUGH the teacher to the query leaf, but its
        # params stay inert (requires_grad off: no .grad accumulation).
        for prm in teacher.parameters():
            prm.requires_grad_(False)
        cheat_bb = (teacher if input_transform is None
                    else (lambda x: teacher(input_transform(x))))

    t = -1
    probe_frontier = None       # --peel-direct: layer the students forward FROM
    _teacher64 = None           # lazily-built fp64 teacher for the sealed query
    budget_end = cfg.outer      # --peelrestart extends this by +cfg.outer per peel,
                                # so each fully-peeled layer earns a FRESH full search
    peel_base = 0               # global t at the last restart -> the DISPLAYED iter
                                # counter (it_disp) restarts at 1 after every peel
    while True:
        t += 1
        if t >= budget_end:
            break
        it_disp = t + 1 - peel_base    # shown iter (restarts on peel); queries stay
                                       # cumulative via the global t below
        # --- query the blackbox ---
        # --peel-direct (cfg.peel_seal): once a leading prefix is EXACTLY frozen,
        # generate the disagreement queries in the FRONTIER layer's input space (h)
        # and map them back through the exact prefix inverse x_of_h. teacher(x_of_h(h))
        # is provably identical (forward + gradient) to querying the sub-net with the
        # prefix removed -- i.e. this DIRECTLY targets the next layer, which solves
        # cleanly where x-space training stalls a few stragglers.
        # --peel-direct (cfg.peel_seal): once a leading prefix is EXACTLY frozen,
        # target the FRONTIER layer directly. STUDENTS forward from the frontier
        # on its OWN input h (normal-norm, fp32) via _FrontierNet -- so the huge-
        # norm x = x_of_h(h) roundtrip (fp32-noisy at depth >=2, which stalled the
        # deep search) never touches them. Only the TEACHER is queried through the
        # seal, in fp64: BB(h) = teacher64(x_of_h(h)). Buffer holds h (frontier
        # input); flushed when the frontier advances.
        pf = None
        if cfg.peel_seal and frozen and not cfg.ensemble_boost:
            nf = 0
            while nf in frozen and bool(frozen[nf][2].all()):
                nf += 1
            if 0 < nf < len(dims) - 1:          # frontier is a hidden layer
                pf = nf
        _qdim = dims[pf] if pf is not None else dims[0]
        if pf != probe_frontier:                # frontier changed -> flush h-buffer
            X = torch.empty(0, _qdim)
            Y = (torch.empty(0, dtype=torch.long) if cfg.hard
                 else torch.empty(0, dims[-1]))
            probe_frontier = pf
        _bb = cheat_bb
        if pf is not None:
            import copy as _copy
            if _teacher64 is None:
                _teacher64 = _copy.deepcopy(teacher).double().to(device).eval()
            _xoh64 = _mlp_prefix_inverse(frozen, pf, cfg.act, device)  # fp64 seal

            def _bb(h, _t=_teacher64, _x=_xoh64):  # honest fp64 black-box query
                return _t(_x(h.double())).to(h.dtype)
        qg_members = ([_FrontierNet(m, pf) for m in pop] if pf is not None
                      else list(pop))
        solo = (cfg.cheat and cfg.cheat_solo and cfg.p > 1
                and not cfg.ensemble_boost)
        if t < cfg.warmstart_iters:
            I = torch.randn(cfg.q, _qdim, generator=gen,
                            device=device) * cfg.qg_init_std
        elif solo:
            # SOLO committee: member i runs its OWN q/p-query disagreement
            # search against the blackbox alone (the p=1 objective, undiluted
            # by the other members' error profiles). Chunks concatenate in
            # member order so buffer ownership stays positional (_solo_pools).
            _cs = cfg.q // cfg.p
            chunks = []
            for mi, m in enumerate(qg_members):
                qi = _cs + (cfg.q - _cs * cfg.p if mi == cfg.p - 1 else 0)
                chunks.append(gen_queries([m, _bb], dc_replace(cfg, q=qi),
                                          _qdim, device, gen))
            I = torch.cat(chunks)
        else:
            losses_qg = (loss_on(pop, X, Y)
                         if (cfg.gate_kappa > 0 and t > 0 and not cfg.cheat
                             and not cfg.ensemble_boost)
                         else None)
            if cfg.ensemble_boost and boost_stages:
                # members predict RESIDUALS; disagreement must run between
                # comparable predictors of Y -> wrap each as cascade + member
                qg_pop = [(lambda x, m=m, s=boost_scale:
                           _cascade(x) + s * m(x)) for m in pop]
            else:
                qg_pop = list(qg_members)
            if cfg.cheat:
                qg_pop = qg_pop + [_bb]
            I = gen_queries(qg_pop, cfg, _qdim, device, gen,
                            losses=losses_qg,
                            ref_last=cfg.cheat and cfg.cheat_bb_ref)
        if pf is not None:
            with torch.no_grad():
                T = _bb(I)                      # honest sealed query, h stays in buffer
        else:
            if input_transform is not None:
                I = input_transform(I)      # squash into the sealed activation domain
            with torch.no_grad():
                T = teacher(I)
        with torch.no_grad():
            if cfg.hard:                    # blackbox reveals ONLY the class index
                T = T.argmax(1)
        X = torch.cat([X, I.cpu()])
        Y = torch.cat([Y, T.cpu()])
        if cfg.window > 0:
            keep = cfg.window * cfg.q
            X, Y = X[-keep:], Y[-keep:]
            if cfg.ensemble_boost:
                boost_P = boost_P[-keep:]

        # --- ensemble-boost: append a fresh ROUND of sequential residual
        #     stages (Judah's protocol: train a net cfg.epochs on the current
        #     residual, then the next on what remains, ... N per iteration,
        #     each unit-renormalized; the cascade sum is the model) ---
        if cfg.ensemble_boost and len(X) > 0:
            with torch.no_grad():          # extend cached preds to new rows
                if len(boost_P) < len(X):
                    xn = X[len(boost_P):]
                    if boost_stages:
                        pr = []
                        for i0 in range(0, len(xn), 8192):
                            pr.append(_cascade(xn[i0:i0 + 8192].to(device)).cpu())
                        boost_P = torch.cat([boost_P, torch.cat(pr)])
                    else:
                        boost_P = torch.cat(
                            [boost_P, torch.zeros(len(xn), dims[-1])])
            c0 = float((Y - boost_P).abs().mean())
            nmax = max(cfg.boost_stages_max, 1)
            c1, k_used, ns = c0, 0, len(X)
            while c1 > cfg.boost_tol and k_used < nmax:
                # one stage, trained TO CONVERGENCE: plateau-annealed lr
                # (the only regime that ever reached ~zero), fresh net,
                # unit-renormalized target
                R = Y - boost_P
                s = max(float(R.abs().mean()), 1e-12)
                T = R / s
                f = MLP(student_dims, act=cfg.act).to(device)
                opt_s = torch.optim.Adam(f.parameters(), lr=cfg.lr)
                lr_s, best_s, stall = cfg.lr, float("inf"), 0
                for ep in range(200):                  # per-stage epoch cap
                    perm = torch.randperm(ns, generator=gen, device=device)
                    el, nb = 0.0, 0
                    for i in range(0, ns, cfg.batch):
                        idx = perm[i:i + cfg.batch].cpu()
                        xb, tb = X[idx].to(device), T[idx].to(device)
                        sl = (f(xb) - tb).abs().mean()
                        opt_s.zero_grad()
                        sl.backward()
                        opt_s.step()
                        el += sl.item()
                        nb += 1
                    el /= max(nb, 1)
                    if el < best_s * 0.997:
                        best_s, stall = el, 0
                        continue
                    stall += 1
                    if stall >= 8:
                        if lr_s <= 1e-6 * 1.01:
                            break                      # stage converged
                        lr_s *= 0.3
                        stall = 0
                        for g in opt_s.param_groups:
                            g["lr"] = lr_s
                for prm in f.parameters():
                    prm.requires_grad_(False)
                boost_stages.append((f, s))
                with torch.no_grad():
                    for i0 in range(0, ns, 8192):
                        boost_P[i0:i0 + 8192] += s * f(
                            X[i0:i0 + 8192].to(device)).cpu()
                c1 = float((Y - boost_P).abs().mean())
                k_used += 1
            boost_scale = max(c1, 1e-12)
            print(f"  [ensemble-boost] iter {t + 1}: +{k_used} stages "
                  f"({len(boost_stages)} total) | cascade train {c0:.3e} -> "
                  f"{c1:.3e} | tol {cfg.boost_tol:g} "
                  f"{'HIT' if c1 <= cfg.boost_tol else 'NOT reached'}",
                  flush=True)

        # member target: the cascade's current (renormalized) residual
        if cfg.ensemble_boost and boost_stages:
            Yfit = (Y - boost_P) / boost_scale
        else:
            Yfit = Y

        # --- lr schedule (ensemble-boost: the per-member plateau annealer owns
        #     the member lrs; the global step-decay would fight it) ---
        if t in decay_at and not cfg.ensemble_boost:
            for o in opts:
                for g in o.param_groups:
                    g["lr"] /= 10
            for fac in facs:
                if fac is not None:
                    fac.decay(0.1)

        # --- train population on D ---
        n = len(X)
        solo_pools = _solo_pools(n, cfg.q, cfg.p) if solo else None

        def _sgd_step(net, opt, fac, xb, yb):
            if fac is not None:
                net.zero_grad(set_to_none=True)
            else:
                opt.zero_grad()
            if cfg.hard:
                loss = F.cross_entropy(net(xb), yb)
            elif cfg.fit_loss == "mse":
                loss = ((net(xb) - yb) ** 2).mean()
            else:
                loss = (net(xb) - yb).abs().mean()
            loss.backward()
            if cfg.peel_clip > 0 and frozen:   # peel active: cap free-param step
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.peel_clip)
            opt.step()
            if fac is not None:
                fac.step(net)          # factored update of the tail weights
            return loss.item()

        for ep in range(cfg.epochs):
            ep_loss = 0.0
            nb = 0
            # peel-direct: train the students forward-from-frontier on the h-buffer
            # (qg_members are _FrontierNet views sharing each member's params, so
            # opt.step still updates the base member).
            if solo_pools is not None:
                # SOLO: each member trains only on its OWN rows of the buffer
                for mi, (net, opt, fac) in enumerate(zip(qg_members, opts, facs)):
                    pool = solo_pools[mi]
                    perm = pool[torch.randperm(len(pool), generator=gen,
                                               device=device).cpu()]
                    for i in range(0, len(perm), cfg.batch):
                        idx = perm[i:i + cfg.batch]
                        xb = X[idx].to(device)
                        yb = Yfit[idx].to(device)
                        ep_loss += _sgd_step(net, opt, fac, xb, yb)
                        nb += 1
            else:
                perm = torch.randperm(n, generator=gen, device=device)
                for i in range(0, n, cfg.batch):
                    idx = perm[i:i + cfg.batch].cpu()
                    xb, yb = X[idx].to(device), Yfit[idx].to(device)
                    bl = 0.0
                    for net, opt, fac in zip(qg_members, opts, facs):
                        bl += _sgd_step(net, opt, fac, xb, yb)
                    ep_loss += bl / cfg.p
                    nb += 1
            if cfg.fit_delta > 0 and ep_loss / max(nb, 1) < cfg.fit_delta:
                break

        # --- per-iteration solver polish: tighten every member with the
        #     staged MSE->MAE LBFGS recipe on the recent solverwindow of
        #     queries (in place, so the Adam optimizers stay valid) ---
        if cfg.solver_polish:
            keep = cfg.solverwindow * cfg.q
            Xp, Yp = ((X[-keep:], Yfit[-keep:]) if keep and len(X) > keep
                      else (X, Yfit))
            if cfg.verbose:
                print(f"  [polish] it {t + 1}: {cfg.p} members on {len(Xp)} "
                      f"queries", flush=True)
            for mi, net in enumerate(pop):
                solver_polish_(net, Xp, Yp, verbose=cfg.verbose,
                               tag=f" m{mi}")

        # --- closed-form last-layer solve (variable projection) ---
        if cfg.lastlayer_every and (t + 1) % cfg.lastlayer_every == 0:
            if solo_pools is not None:
                # SOLO: each member's head is solved on its OWN rows (subsample
                # BEFORE gathering so the big fancy-index copy stays small)
                for net, pool in zip(pop, solo_pools):
                    if cfg.lastlayer_sample and len(pool) > cfg.lastlayer_sample:
                        pool = pool[torch.randperm(len(pool))[:cfg.lastlayer_sample]]
                    solve_last_layer_(net, X[pool], Yfit[pool])
            else:
                for net in pop:
                    solve_last_layer_(net, X, Yfit, sample=cfg.lastlayer_sample)

        # --- early stopping (App F signals, actually wired in) ---
        if cfg.stop_loss > 0 and (t + 1) % cfg.log_every == 0:
            losses_now = loss_on(qg_members, X, Yfit)
            bl = min(losses_now)
            disp = 0.0
            if cfg.stop_agree > 0:
                bi = min(range(cfg.p), key=lambda i: losses_now[i])
                anchor = pop[bi].clone()
                disp = max(
                    (anchor.layers[0].weight - m.layers[0].weight)
                    .abs().max().item()
                    for m in pop if m is not pop[bi])
            if bl < cfg.stop_loss and (cfg.stop_agree <= 0 or
                                       disp < cfg.stop_agree):
                stop_hits += 1
                if stop_hits >= cfg.stop_patience:
                    best = pop[min(range(cfg.p), key=lambda i: losses_now[i])]
                    print(f"  [early-stop] iter {t + 1}: loss {bl:.2e} "
                          f"disp {disp:.2e}", flush=True)
                    break
            else:
                stop_hits = 0

        # --- committee maintenance (alignment/soup/restart) ---
        souped = None
        if cfg.maint_every and (t + 1) % cfg.maint_every == 0 and t >= 5:
            losses = loss_on(qg_members, X, Yfit)
            order = sorted(range(cfg.p), key=lambda i: losses[i])
            best_now = pop[order[0]].clone()
            for i in order[1:]:
                align_member_to(best_now, pop[i])
            cand = soup_of([best_now] + [pop[i] for i in order[1:]])
            closs, wloss = loss_on([cand], X, Yfit)[0], losses[order[-1]]
            if closs < wloss:
                pop[order[-1]] = cand.to(device)
                opts[order[-1]] = torch.optim.Adam(
                    pop[order[-1]].parameters(),
                    lr=opts[order[-1]].param_groups[0]["lr"])
            for k in range(min(cfg.restart_worst, cfg.p - 1)):
                idx = order[-1 - k]
                pop[idx] = MLP(dims, act=cfg.act).to(device)
                opts[idx] = torch.optim.Adam(
                    pop[idx].parameters(),
                    lr=opts[idx].param_groups[0]["lr"])
            souped = closs

        # --- ensemble-boost: CASCADE COMPRESSION. Rounds add stages every
        #     iteration, so the sum must periodically be compressed back into
        #     one width-n net (the original "...and merges") or forwards/query
        #     -gen become unbounded. Threshold = --ensemble-every (STAGE count
        #     under boost, not iterations). Gated on fresh neutral queries:
        #     accept a mild (<=10%) compression cost; past 2x the threshold
        #     accept regardless (tractability) with a warning. ---
        _cmp_at = max(cfg.ensemble_every, 2)
        if (cfg.ensemble_boost and len(boost_stages) >= _cmp_at
                and len(X) > 0):
            try:
                merged, minfo = merge_ensemble(
                    [n_ for n_, _ in boost_stages], student_dims, cfg.act,
                    X, Y, device, frozen=frozen, partial_exact=partial_exact,
                    partial_mask=partial_mask, samples=cfg.ensemble_samples,
                    block=cfg.ensemble_block, hard=cfg.hard,
                    weights=[s_ for _, s_ in boost_stages])
                if merged is not None:
                    # post-merge DISTILL-TO-TOLERANCE on the memorized set:
                    # selection is rank-bounded (pooled L1 rank95 >> 512 even
                    # on-buffer) but the net's own capacity is not -- anneal
                    # until the buffer zeros are re-held (or floor)
                    if frozen:
                        _install_freeze(merged, frozen)
                    finetune_(merged, X, Y, device, 200, cfg.batch,
                              cfg.lr, gen, tol=cfg.boost_tol)
                    ng = min(8192, cfg.q)
                    Ig = torch.randn(ng, dims[0], generator=gen,
                                     device=device) * cfg.qg_init_std
                    if input_transform is not None:
                        Ig = input_transform(Ig)
                    with torch.no_grad():
                        Ygt = teacher(Ig)
                    Xt, Yt = Ig.cpu(), Ygt.cpu()
                    cl = _cascade_l1(Xt, Yt)
                    ml = l1_on([merged], Xt, Yt)[0]
                    # buffer = the memorized set the cascade was zeroed on:
                    # how much of that interpolation survives compression?
                    bl_c = float((Y - boost_P).abs().mean())
                    bl_m = l1_on([merged], X, Y)[0]
                    force_c = len(boost_stages) >= 2 * _cmp_at
                    if ml <= cl * 1.1 or force_c:
                        for prm in merged.parameters():
                            prm.requires_grad_(False)
                        kold = len(boost_stages)
                        boost_stages = [(merged, 1.0)]
                        with torch.no_grad():   # refresh cached predictions
                            pr = []
                            for i0 in range(0, len(X), 8192):
                                pr.append(merged(
                                    X[i0:i0 + 8192].to(device)).cpu())
                            boost_P = torch.cat(pr)
                        boost_scale = max(
                            float((Y - boost_P).abs().mean()), 1e-12)
                        print(f"  [ensemble-boost] iter {t + 1}: COMPRESSED "
                              f"{kold} stages -> 1 width-{student_dims[1]} "
                              f"net | fresh loss {cl:.3e} -> {ml:.3e} | "
                              f"buffer(memorized) {bl_c:.3e} -> {bl_m:.3e}"
                              + (" [FORCED]" if force_c and ml > cl * 1.1
                                 else ""), flush=True)
                    else:
                        print(f"  [ensemble-boost] iter {t + 1}: compression "
                              f"deferred (fresh {cl:.3e} vs merged {ml:.3e} | "
                              f"buffer(memorized) {bl_c:.3e} -> {bl_m:.3e})",
                              flush=True)
            except Exception as e:
                print(f"  [ensemble-boost] iter {t + 1}: compression failed "
                      f"({e})", flush=True)

        # --- ensemble merge: pool every member's neurons, select the width-n
        #     shared basis per hidden layer by downstream explanatory power
        #     (group-OMP), rebuild deeper layers over it, solve the output
        #     head; inject the merged net over the worst member if it wins ---
        if (cfg.ensemble_every and cfg.p > 1 and len(X) > 0
                and not cfg.ensemble_boost
                and (t + 1) % cfg.ensemble_every == 0):
            try:
                merged, minfo = merge_ensemble(
                    pop, student_dims, cfg.act, X, Y, device,
                    frozen=frozen, partial_exact=partial_exact,
                    partial_mask=partial_mask, samples=cfg.ensemble_samples,
                    block=cfg.ensemble_block, hard=cfg.hard)
                if merged is None:
                    print(f"  [ensemble] iter {t + 1}: skipped ({minfo})",
                          flush=True)
                else:
                    vix = minfo.get("val_idx")
                    Xg, Yg = ((X[vix], Y[vix])
                              if vix is not None and len(vix) else (X, Y))
                    mloss = loss_on([merged], Xg, Yg)[0]
                    losses_e = loss_on(pop, Xg, Yg)
                    wi = max(range(cfg.p), key=lambda i: losses_e[i])
                    lv = " ".join(
                        f"L{x['layer']}[{x['picked']}/{x['slots']} "
                        f"expl {x['explained']:.4f}]" for x in minfo["levels"])
                    eps_str = ""
                    if not expanded:
                        me = param_errors(merged, teacher, hard=cfg.hard)
                        eps_str = (f" | max_eps {me['max_eps']:.2e} mean_eps "
                                   f"{sum(me['mean_eps_per_matrix']) / len(me['mean_eps_per_matrix']):.2e}")
                    if mloss < losses_e[wi]:
                        cur_lr = opts[wi].param_groups[0]["lr"]
                        if frozen:                 # keep peel invariants on the
                            _install_freeze(merged, frozen)   # injected member
                        for l, pmk in (partial_mask or {}).items():
                            if partial_exact is None or not bool(pmk.any()):
                                continue
                            key = (id(merged), l)
                            live = pmk.clone().to(device)
                            partial_live[key] = live
                            merged.layers[l].weight.register_hook(
                                lambda g, k=live: g * (~k).to(g.dtype).unsqueeze(1))
                            merged.layers[l].bias.register_hook(
                                lambda g, k=live: g * (~k).to(g.dtype))
                            partial_hooked.add(key)
                        pop[wi] = merged
                        if cfg.factor_tail:
                            ordinary = []
                            for l in range(len(merged.layers)):
                                ordinary.append(merged.layers[l].bias)
                                if l not in flayers:
                                    ordinary.append(merged.layers[l].weight)
                            opts[wi] = torch.optim.Adam(ordinary, lr=cur_lr)
                            facs[wi] = _mk_fac(merged, cur_lr)
                        else:
                            opts[wi] = torch.optim.Adam(
                                [p for p in merged.parameters()
                                 if p.requires_grad], lr=cur_lr)
                        verdict = (f"-> replaced worst member {wi} "
                                   f"(loss {losses_e[wi]:.2e})")
                    else:
                        verdict = (f"kept out (worst member "
                                   f"{losses_e[wi]:.2e} is better)")
                    print(f"  [ensemble] iter {t + 1}: merged {cfg.p} members "
                          f"in {minfo['wall_s']}s | {lv} | loss {mloss:.2e}"
                          f"{eps_str} {verdict}", flush=True)
            except Exception as e:
                print(f"  [ensemble] iter {t + 1}: failed ({e})", flush=True)

        # --- logging (consensus + combine run FIRST so this iter's max_eps
        #     reflects any injected member) ---
        if (t + 1) % cfg.log_every == 0 or t == budget_end - 1:
            losses = loss_on(qg_members, X, Yfit)
            if expanded:                       # fit gate: loss only (dims mismatch)
                bi = min(range(cfg.p), key=lambda i: losses[i])
                best = pop[bi]
                if cfg.ensemble_boost and boost_stages:
                    pass          # boost: expanded fit-gate keeps member best;
                                  # the cascade is scored via base_loss below
                wall = round(time.time() - t0, 1)
                med = sorted(losses)[cfg.p // 2]
                log.append({"iter": t + 1, "queries": (t + 1) * cfg.q,
                            "best_loss": losses[bi], "med_loss": med,
                            "worst_loss": max(losses), "wall_s": wall})
                print(f"  it {t + 1:3d} | q {(t + 1) * cfg.q:7d} | loss best "
                      f"{losses[bi]:.3e} med {med:.3e} worst {max(losses):.3e} | "
                      f"{wall}s  [EXPAND x{cfg.expand}]", flush=True)
                continue
            # ungated 5/8 — identical to the --fast dump trigger, so the printed
            # `cluster` column and --combine act on the same consensus --fast does
            cnet = build_consensus(pop, dims, eps=cfg.cluster_eps,
                                   quorum_ratio=cfg.cluster_quorum)
            cc = (param_errors(cnet, teacher, hard=cfg.hard)["max_eps"]
                  if cnet is not None else None)
            cstats = consensus_neuron_stats(pop, teacher, dims,
                                            eps=cfg.cluster_eps,
                                            quorum_ratio=cfg.cluster_quorum,
                                            hard=cfg.hard)
            combined_now = None
            if cfg.combine and not combined_done and cnet is not None:
                polished = polish_consensus(
                    cnet, X, Y, lr=cfg.combine_polish_lr,
                    steps=cfg.combine_polish_steps, hard=cfg.hard)
                wi = max(range(cfg.p), key=lambda i: losses[i])
                cur_lr = opts[wi].param_groups[0]["lr"]
                pop[wi] = polished.to(device)
                if cfg.factor_tail:
                    ordinary = []
                    for l in range(len(pop[wi].layers)):
                        ordinary.append(pop[wi].layers[l].bias)
                        if l not in flayers:
                            ordinary.append(pop[wi].layers[l].weight)
                    opts[wi] = torch.optim.Adam(ordinary, lr=cur_lr)
                    facs[wi] = _mk_fac(pop[wi], cur_lr)
                else:
                    opts[wi] = torch.optim.Adam(pop[wi].parameters(), lr=cur_lr)
                combined_done = True
                combined_now = t + 1
                pe = param_errors(polished, teacher, hard=cfg.hard)["max_eps"]
                pl = loss_on([polished], X, Y)[0]
                print(f"  [combine] iter {t + 1}: replaced worst member {wi} "
                      f"(loss {losses[wi]:.2e}) with polished consensus "
                      f"(max_eps {cc:.2e}->{pe:.2e}, loss {pl:.2e})",
                      flush=True)
                losses = loss_on(qg_members, X, Yfit)  # reflect the injected member
            bi = min(range(cfg.p), key=lambda i: losses[i])
            best = pop[bi]
            base_loss = None
            if cfg.ensemble_boost and boost_stages:
                # boost: `loss` reports the CASCADE (the attacker model);
                # eps columns describe the best member (a residual net --
                # weight-space eps vs teacher is meaningful only terminally)
                ncap = min(len(X), 100000)
                base_loss = float(
                    (Y[-ncap:] - boost_P[-ncap:]).abs().mean())
            errs = param_errors(_frozen_fp64_view(best, frozen), teacher,
                                hard=cfg.hard)
            rec = {
                "iter": t + 1,
                "queries": (t + 1) * cfg.q,
                "peel_refinement_queries": peel_refinement_queries,
                "best_loss": base_loss if base_loss is not None else losses[bi],
                **({"stage_best_loss": losses[bi] * boost_scale}
                   if base_loss is not None else {}),
                "med_loss": sorted(losses)[cfg.p // 2],
                "worst_loss": max(losses),
                "max_eps": errs["max_eps"],
                "mean_eps": sum(errs["mean_eps_per_matrix"]) /
                len(errs["mean_eps_per_matrix"]),
                "agree": agreement(best, teacher, teacher_eval_pts),
                "wall_s": round(time.time() - t0, 1),
                "cluster_max_eps": cc,
            }
            if cstats is not None:
                rec["n_consensus"] = cstats["n_consensus"]
                rec["n_total"] = cstats["n_total"]
                rec["consensus_max_eps"] = cstats["max_eps"]
                rec["consensus_mean_eps"] = cstats["mean_eps"]
                for k in ("l0_max", "l0_mean", "l1_max", "l1_mean",
                          "n_out_consensus", "out_total", "out_max", "out_mean"):
                    if k in cstats:
                        rec["consensus_" + k] = cstats[k]
            if souped is not None:
                rec["soup_loss"] = souped
            if combined_now is not None:
                rec["combined_iter"] = combined_now
            # --- per-layer eps breakdown (added logging) ---
            _nmat = len(dims) - 1
            _pmax = errs.get("max_eps_per_matrix")
            _pmean = errs.get("mean_eps_per_matrix")
            if _pmax:                       # per matrix: max over weight+bias; mean of weight
                rec["eps_per_layer_max"] = [max(_pmax[2 * i], _pmax[2 * i + 1])
                                            for i in range(_nmat)]
                rec["eps_per_layer_mean"] = [_pmean[2 * i] for i in range(_nmat)]
            if cstats is not None and "layers" in cstats:
                rec["consensus_eps_per_layer"] = cstats["layers"]
            log.append(rec)
            cc_str = f"{cc:.2e}" if cc is not None else "  n/a  "
            if cstats is not None and "layers" in cstats:      # >1 hidden layer
                parts = " ".join(f"L{li+1}:{x['n_cons']}/{x['n_tot']}"
                                 for li, x in enumerate(cstats["layers"]))
                e = (f"max {cstats['max_eps']:.2e} mean {cstats['mean_eps']:.2e}"
                     if cstats.get("max_eps") is not None else "n/a")
                cons_str = (f"{cstats['n_consensus']}/{cstats['n_total']} "
                            f"[{parts}] {e}")
            elif cstats is not None and cstats["max_eps"] is not None:
                cons_str = (f"{cstats['n_consensus']}/{cstats['n_total']} "
                            f"L0[max {cstats['l0_max']:.2e} "
                            f"mean {cstats['l0_mean']:.2e}] "
                            f"L1[max {cstats['l1_max']:.2e} "
                            f"mean {cstats['l1_mean']:.2e}]")
                om = (f"max {cstats['out_max']:.2e} mean {cstats['out_mean']:.2e}"
                      if cstats.get("out_max") is not None else "n/a")
                cons_str += (f" | out {cstats['n_out_consensus']}/"
                             f"{cstats['out_total']} [{om}]")
            else:
                cons_str = f"0/{dims[1]}"
            _ls = (f"{base_loss:.2e} (stage {losses[bi] * boost_scale:.2e})"
                   if base_loss is not None else f"{losses[bi]:.2e}")
            print(f"  it {it_disp:3d} | q {(t + 1) * cfg.q:6d} | "
                  f"loss {_ls} | max_eps {errs['max_eps']:.2e} | "
                  f"mean_eps {rec['mean_eps']:.2e} | "
                  f"cluster {cc_str} | cons {cons_str} | "
                  f"agree {rec['agree']:.4f} | {rec['wall_s']}s", flush=True)
            if _pmax:                                   # per-layer eps: ALL neurons
                print("        eps/layer (all):  " + "  ".join(
                    f"L{i+1}[max {max(_pmax[2*i], _pmax[2*i+1]):.2e} "
                    f"mean {_pmean[2*i]:.2e}]" for i in range(_nmat)), flush=True)
                # PARTIALLY-frozen layers: split the eps into frozen vs unsolved,
                # so it's clear the frozen part is exact and where the error lives.
                _solved = {}
                for l in range(len(dims) - 2):
                    m = None
                    if l in frozen: m = frozen[l][2].clone().to(device)
                    if l in partial_mask:
                        pm = partial_mask[l].to(device)
                        m = pm.clone() if m is None else (m | pm)
                    if m is not None and 0 < int(m.sum()) < m.numel():   # partial only
                        _solved[l] = m
                if _solved:
                    spl = layer_eps_split(best, teacher, _solved)
                    print("        eps/layer (frozen|unsolved): " + "  ".join(
                        f"L{l+1}[{spl[l]['n']}f: max {spl[l]['fz'][0]:.2e} "
                        f"mean {spl[l]['fz'][1]:.2e} | {m.numel()-spl[l]['n']}u: "
                        f"max {spl[l]['uf'][0]:.2e} mean {spl[l]['uf'][1]:.2e}]"
                        for l, m in ((l, _solved[l]) for l in sorted(spl))), flush=True)
            if cstats is not None and "layers" in cstats:   # per-layer eps: CONSENSUS only
                def _clyr(i, x):
                    if x.get("cons_max") is not None:
                        return (f"L{i+1}[max {x['cons_max']:.2e} mean "
                                f"{x['cons_mean']:.2e} ({x['n_cons']}/{x['n_tot']})]")
                    return f"L{i+1}[n/a ({x['n_cons']}/{x['n_tot']})]"
                print("        eps/layer (cons, training copies): " + "  ".join(
                    _clyr(i, x) for i, x in enumerate(cstats["layers"])), flush=True)

            # --- --partial: opportunistically refine + in-place freeze the
            #     frontier's SOLVABLE neurons every log_every iters, regardless of
            #     the whole-layer peel threshold. Stragglers keep training; solved
            #     neurons lock in early (warm, scale-matched, momentum-zeroed).
            #     Never peels/advances the layer -- that stays with the full peel. ---
            if (cfg.partial and cfg.cheat and (t + 1) % cfg.log_every == 0
                    and best is not None and cfg.act in ("relu", "leaky_relu")):
                import copy as _cp
                Lh = len(dims) - 2
                # combined solved = the full peel's frozen rows OR partial's own
                # pinned rows. A layer stays a candidate while ANY neuron is unsolved
                # -- so partial KEEPS re-attempting stragglers in a layer the full
                # peel only PARTIALLY froze; it never stops just because the layer
                # entered `frozen`.
                def _psolved(l):
                    m = partial_mask.get(l, torch.zeros(dims[l + 1], dtype=torch.bool,
                                                        device=device)).clone()
                    if l in frozen:
                        m = m | frozen[l][2].to(device)
                    return m
                pf = next((l for l in range(Lh)
                           if int(_psolved(l).sum()) < dims[l + 1]), None)
                if pf is not None:
                    Cout = best.layers[pf].weight.shape[0]
                    if partial_exact is None:
                        partial_exact = _cp.deepcopy(best).to(device)
                    if pf not in partial_mask:
                        partial_mask[pf] = torch.zeros(Cout, dtype=torch.bool, device=device)
                    wdt = partial_exact.layers[pf].weight.dtype
                    # refresh the EXACT prefix (layers < pf) + pf's already-frozen
                    # rows from the peel's `frozen` source of truth, so the sealed
                    # refine sees the exact earlier layers (best's prefix is pinned
                    # to those anyway, but partial_exact was deep-copied earlier).
                    for l in range(pf + 1):
                        if l in frozen:
                            fm = frozen[l][2].to(device)
                            partial_exact.layers[l].weight.data[fm] = frozen[l][0].to(device)[fm].to(partial_exact.layers[l].weight.dtype)
                            partial_exact.layers[l].bias.data[fm] = frozen[l][1].to(device)[fm].to(partial_exact.layers[l].bias.dtype)
                    unsolved = (~_psolved(pf)).nonzero(as_tuple=True)[0].tolist()
                    for c in unsolved:                       # seed unsolved <- current guess
                        partial_exact.layers[pf].weight.data[c] = best.layers[pf].weight.data[c].to(wdt)
                        partial_exact.layers[pf].bias.data[c] = best.layers[pf].bias.data[c].to(wdt)
                    _pt0 = time.time()
                    Wr, br, rmask, _nq = _mlp_refine_layer(
                        teacher, partial_exact, pf, device, cfg.act,
                        only_channels=unsolved,
                        angle_gate=getattr(cfg, "peel_angle_gate", 12.0),
                        bases=teacher_eval_pts, light_only=True,
                        design=cfg.design_refine)
                    peel_refinement_queries += _nq
                    if log:
                        log[-1]["peel_refinement_queries"] = peel_refinement_queries
                    newly = [c for c in unsolved if bool(rmask[c])]
                    if newly:
                        idx = torch.tensor(newly, device=device)
                        gwf = partial_exact.layers[pf].weight.data[idx].double()   # = guess
                        gbf = partial_exact.layers[pf].bias.data[idx].double()
                        uwf = Wr[idx].double(); ubf = br[idx].double()
                        # guess-scale; an ANTI-ALIGNED guess (proj<=0) would clamp
                        # to ~0 and pin a ZERO row -- frozen wrong forever. Fall
                        # back to the guess magnitude (peel-try's guard).
                        proj = (gwf * uwf).sum(1) + gbf * ubf
                        gn = (gwf.pow(2).sum(1) + gbf.pow(2)).sqrt()
                        cs = torch.where(proj > 1e-6 * gn, proj, gn).clamp_min(1e-8)
                        partial_exact.layers[pf].weight.data[idx] = (cs[:, None] * uwf).to(wdt)
                        partial_exact.layers[pf].bias.data[idx] = (cs * ubf).to(wdt)
                        partial_mask[pf][idx] = True
                        for m, opt in zip(pop, opts):        # in-place pin (warm, hook-once)
                            key = (id(m), pf)
                            if key not in partial_hooked:
                                live = torch.zeros(m.layers[pf].weight.shape[0],
                                                   dtype=torch.bool, device=device)
                                partial_live[key] = live
                                m.layers[pf].weight.register_hook(
                                    lambda g, k=live: g * (~k).to(g.dtype).unsqueeze(1))
                                m.layers[pf].bias.register_hook(
                                    lambda g, k=live: g * (~k).to(g.dtype))
                                partial_hooked.add(key)
                            live = partial_live[key]
                            with torch.no_grad():
                                m.layers[pf].weight[idx] = partial_exact.layers[pf].weight.data[idx].to(m.layers[pf].weight.dtype)
                                m.layers[pf].bias[idx] = partial_exact.layers[pf].bias.data[idx].to(m.layers[pf].bias.dtype)
                            live[idx] = True
                            for p in (m.layers[pf].weight, m.layers[pf].bias):
                                st = opt.state.get(p)
                                if st:
                                    if "exp_avg" in st: st["exp_avg"][idx] = 0
                                    if "exp_avg_sq" in st: st["exp_avg_sq"][idx] = 0
                    print(f"  [partial] L{pf + 1}: +{len(newly)} frozen in place "
                          f"({int(_psolved(pf).sum())}/{Cout} total)  |  "
                          f"{time.time() - _pt0:.1f}s, {_nq} queries", flush=True)

            # --- fast-peel-partial (MLP): PER-NEURON consensus peel. Every log iter:
            #     frontier = first hidden layer not fully solved; align every member to
            #     the best member's frame up to the frontier (permutation only, function-
            #     preserving) so row indices agree across the committee; consensus in
            #     that fixed frame; kink-refine the frontier's consensus rows not yet
            #     solved (quorum means as guesses, exact prefix from partial_exact);
            #     inject each solved row into EVERY member at THAT member's own
            #     magnitude and pin it (grad-masked). The frontier advances only once
            #     the whole layer is solved.
            if (cfg.fast_peel_partial and (t + 1) % cfg.log_every == 0
                    and cfg.p > 1 and best is not None
                    and cfg.act in ("relu", "leaky_relu")):
                import copy as _cp
                Lh = len(dims) - 2
                def _psolvedF(l):
                    mq = partial_mask.get(l, torch.zeros(dims[l + 1], dtype=torch.bool,
                                                         device=device)).clone()
                    if l in frozen:
                        mq = mq | frozen[l][2].to(device)
                    return mq
                pf = next((l for l in range(Lh) if int(_psolvedF(l).sum()) < dims[l + 1]), None)
                if pf is not None:
                    _pt0 = time.time()
                    bi_f = next(i for i, m_ in enumerate(pop) if m_ is best)
                    ref_norm = best.clone(); scale_normalize_(ref_norm)
                    for i_, (m_, opt_) in enumerate(zip(pop, opts)):     # (0) common frame
                        if i_ != bi_f:
                            _mlp_apply_align_(m_, ref_norm, pf, opt=opt_, live_masks=partial_live)
                    cnet, masks = _partial_consensus(pop, dims, cfg.cluster_eps,
                                                     cfg.cluster_quorum, ref_idx=bi_f)
                    Cout = dims[pf + 1]
                    if partial_exact is None:
                        partial_exact = _cp.deepcopy(cnet).to(device).double()   # fp64 record of solved rows
                    elif partial_exact.layers[0].weight.dtype != torch.float64:
                        partial_exact = partial_exact.double()
                    if pf not in partial_mask:
                        partial_mask[pf] = torch.zeros(Cout, dtype=torch.bool, device=device)
                    wdt = partial_exact.layers[pf].weight.dtype
                    for l in range(pf + 1):                              # exact prefix rows from
                        if l in frozen:                                  # `frozen`, keeping our fp64 rows
                            fm = frozen[l][2].to(device)
                            if l in partial_mask:
                                fm = fm & ~partial_mask[l].to(device)
                            partial_exact.layers[l].weight.data[fm] = frozen[l][0].to(device)[fm].to(partial_exact.layers[l].weight.dtype)
                            partial_exact.layers[l].bias.data[fm] = frozen[l][1].to(device)[fm].to(partial_exact.layers[l].bias.dtype)
                    cand = (masks[pf].to(device) & ~_psolvedF(pf)).nonzero(as_tuple=True)[0].tolist()
                    newly = []; _nq = 0
                    if cand:
                        for c in cand:                                   # guesses <- quorum means
                            partial_exact.layers[pf].weight.data[c] = cnet.layers[pf].weight.data[c].to(wdt)
                            partial_exact.layers[pf].bias.data[c] = cnet.layers[pf].bias.data[c].to(wdt)
                        Wr, br, rmask, _nq = _mlp_refine_layer(
                            teacher, partial_exact, pf, device, cfg.act,
                            only_channels=cand,
                            angle_gate=getattr(cfg, "peel_angle_gate", 12.0),
                            xspace=cfg.xspace_refine, loc=cfg.loc_refine,
                            design=cfg.design_refine)
                        peel_refinement_queries += _nq
                        newly = [c for c in cand if Wr is not None and bool(rmask[c])]
                        if newly:
                            idx = torch.tensor(newly, device=device)
                            uwf = Wr[idx].double(); ubf = br[idx].double()
                            def _inject(layer, dt_):                     # own-magnitude injection
                                gw = layer.weight.data[idx].double(); gb = layer.bias.data[idx].double()
                                proj = (gw * uwf).sum(1) + gb * ubf
                                gn = (gw.pow(2).sum(1) + gb.pow(2)).sqrt()
                                cs = torch.where(proj > 1e-6 * gn, proj, gn).clamp_min(1e-8)
                                layer.weight.data[idx] = (cs[:, None] * uwf).to(dt_)
                                layer.bias.data[idx] = (cs * ubf).to(dt_)
                            _inject(partial_exact.layers[pf], wdt)
                            partial_mask[pf][idx] = True
                            for m_, opt_ in zip(pop, opts):
                                key = (id(m_), pf)
                                if key not in partial_hooked:
                                    live = torch.zeros(Cout, dtype=torch.bool, device=device)
                                    partial_live[key] = live
                                    m_.layers[pf].weight.register_hook(
                                        lambda g, k=live: g * (~k).to(g.dtype).unsqueeze(1))
                                    m_.layers[pf].bias.register_hook(
                                        lambda g, k=live: g * (~k).to(g.dtype))
                                    partial_hooked.add(key)
                                live = partial_live[key]
                                with torch.no_grad():
                                    _inject(m_.layers[pf], m_.layers[pf].weight.dtype)
                                live[idx] = True
                                for p in (m_.layers[pf].weight, m_.layers[pf].bias):
                                    st = opt_.state.get(p)
                                    if st:
                                        if "exp_avg" in st: st["exp_avg"][idx] = 0
                                        if "exp_avg_sq" in st: st["exp_avg_sq"][idx] = 0
                    ns = int(_psolvedF(pf).sum())
                    print(f"  [fast-peel-partial] L{pf + 1}: {len(cand)} consensus candidates, "
                          f"+{len(newly)} solved & pinned in all {len(pop)} members "
                          f"({ns}/{Cout} total)  |  {_nq} queries, {time.time() - _pt0:.1f}s", flush=True)
                    if newly:                                # accuracy of the STORED fp64 rows
                        try:
                            sm = {l: _psolvedF(l) for l in range(Lh) if bool(_psolvedF(l).any())}
                            sp_ = layer_eps_split(partial_exact, teacher, sm)
                            rep = "  ".join(f"L{l + 1}[max {sp_[l]['fz'][0]:.2e} mean {sp_[l]['fz'][1]:.2e} "
                                            f"({sp_[l]['n']}/{dims[l + 1]})]" for l in sorted(sp_))
                            print(f"  [stored fp64] solved rows vs teacher: {rep}", flush=True)
                        except Exception as e:
                            print(f"  [stored fp64] report skipped ({e})", flush=True)
                    if ns == Cout and pf + 1 < Lh:
                        print(f"  [fast-peel-partial] L{pf + 1} complete -> frontier advances to L{pf + 2}",
                              flush=True)
                    elif ns < Cout:
                        try:
                            uns = (~_psolvedF(pf)).nonzero(as_tuple=True)[0].tolist()
                            rows = []
                            for m_ in pop:
                                cc = m_.clone(); scale_normalize_(cc)
                                for l_ in range(pf + 1):
                                    match_layer_(ref_norm, cc, l_)
                                rows.append(torch.cat([cc.layers[pf].weight, cc.layers[pf].bias[:, None]], 1))
                            R_ = torch.stack(rows)
                            agree = ((R_ - R_[bi_f][None]).abs().amax(2) <= cfg.cluster_eps).sum(0)
                            print("  [agreement] L%d unsolved: " % (pf + 1)
                                  + "  ".join(f"ch{c}: {int(agree[c])}/{len(pop)} members within eps" for c in uns),
                                  flush=True)
                        except Exception as e:
                            print(f"  [agreement] report skipped ({e})", flush=True)

            # --- restart-stuck: --partial can peel-restart even when the frontier
            #     can't be FULLY solved. Fires on (a) STAGNATION -- >= frac of the
            #     frontier solved AND no new neurons frozen for `window` iters -- or
            #     (b) the FINAL iter. Keeps the solved rows pinned, cold-reinits only
            #     the unsolved + deeper, flushes the buffer, restarts the iter clock
            #     (fresh full budget). Capped at restart_stuck_max (each adds budget). ---
            if (cfg.restart_stuck and cfg.partial and cfg.cheat
                    and partial_exact is not None
                    and ((t + 1) % cfg.log_every == 0 or t == budget_end - 1)
                    and stuck_restarts < cfg.restart_stuck_max):
                Lh = len(dims) - 2
                def _psolved2(l):
                    m = partial_mask.get(l, torch.zeros(dims[l + 1], dtype=torch.bool,
                                                        device=device)).clone()
                    if l in frozen:
                        m = m | frozen[l][2].to(device)
                    return m
                sf = next((l for l in range(Lh)
                           if int(_psolved2(l).sum()) < dims[l + 1]), None)
                if sf is not None:
                    cur = int(_psolved2(sf).sum()); ncout = dims[sf + 1]
                    # (re)start the stagnation clock when the frontier moves or grows
                    if stuck_pf != sf:
                        stuck_pf = sf; stuck_last_count = cur; stuck_last_growth = t
                    elif cur > stuck_last_count:
                        stuck_last_count = cur; stuck_last_growth = t
                    stagnant = (cur >= cfg.restart_stuck_frac * ncout
                                and t - stuck_last_growth >= cfg.restart_stuck_window)
                    final = (t == budget_end - 1)
                    if stagnant or final:
                        why = "stagnation" if stagnant else "final iter"
                        # frozen dict = every solved row on the frontier and below,
                        # from partial_exact (the guess-scale exact source of truth)
                        pfz = {}
                        for l in range(sf + 1):
                            m = _psolved2(l)
                            if bool(m.any()):
                                pfz[l] = (
                                    partial_exact.layers[l].weight.detach().clone(),
                                    partial_exact.layers[l].bias.detach().clone(),
                                    m.clone())
                        pop = _reinit_frozen_population(dims, device, cfg.p, pfz,
                                                        act=cfg.act)
                        opts = [torch.optim.Adam(
                            [p for p in n.parameters() if p.requires_grad],
                            lr=cfg.lr) for n in pop]
                        X = torch.empty(0, dims[0])
                        Y = (torch.empty(0, dtype=torch.long) if cfg.hard
                             else torch.empty(0, dims[-1]))
                        budget_end = t + 1 + cfg.outer
                        peel_base = t + 1
                        decay_at = {t + 1 + int(s * cfg.outer) for s in cfg.lr_sched}
                        partial_hooked.clear(); partial_live.clear()
                        combined_done = False
                        stuck_restarts += 1
                        stuck_last_growth = t; stuck_last_count = cur
                        print(f"  [restart-stuck] L{sf + 1} {cur}/{ncout} solved, "
                              f"{why}: kept solved rows pinned, cold-reinit "
                              f"unsolved+deeper, flushed buffer, RESTARTED iter clock "
                              f"(it->1, fresh {cfg.outer}-iter budget, runs to "
                              f"t={budget_end}) [{stuck_restarts}/"
                              f"{cfg.restart_stuck_max}]", flush=True)
                        continue

            # --- --retry: in a peel mode, when the budget ends with the frontier layer
            #     not fully peeled, reinit and go again. Partial modes: keep every solved
            #     row pinned, reinit only the unsolved rows + deeper layers. Full-layer
            #     modes: reinit the ENTIRE frontier layer (+ deeper). Buffer flushed,
            #     iteration clock restarted. Capped at cfg.retry retries.
            _peel_mode = (cfg.freeze_reinit or cfg.fast_peel or cfg.fast_peel_partial
                          or cfg.partial or bool(cfg.peel_try))
            if cfg.retry > 0 and _peel_mode and t == budget_end - 1 and retries_done < cfg.retry:
                _partial_mode = bool(cfg.fast_peel_partial or cfg.partial)
                LhR = len(dims) - 2
                def _solvedR(l):
                    n_ = dims[l + 1]
                    mr = torch.zeros(n_, dtype=torch.bool, device=device)
                    if l in partial_mask: mr = mr | partial_mask[l].to(device)
                    if l in frozen: mr = mr | frozen[l][2].to(device)
                    if l in exact_mask: mr = mr | exact_mask[l].to(device)
                    return mr
                rf = next((l for l in range(LhR) if int(_solvedR(l).sum()) < dims[l + 1]), None)
                if rf is not None:
                    cur = int(_solvedR(rf).sum()); ncout_r = dims[rf + 1]
                    pfz = {}
                    for l in range(rf + 1):
                        if l == rf and not _partial_mode:
                            break                          # whole frontier layer restarts
                        mr = _solvedR(l)
                        if not bool(mr.any()):
                            continue
                        Wp = pop[0].layers[l].weight.data.double().clone(); bp = pop[0].layers[l].bias.data.double().clone()
                        fz = frozen[l][2].to(device) if l in frozen else torch.zeros_like(mr)
                        if l in frozen:
                            Wp[fz] = frozen[l][0].to(device)[fz].double(); bp[fz] = frozen[l][1].to(device)[fz].double()
                        for src, msk in ((partial_exact, partial_mask.get(l)), (exact_net, exact_mask.get(l))):
                            if src is not None and msk is not None:
                                mm = msk.to(device) & ~fz
                                Wp[mm] = src.layers[l].weight.data[mm].double(); bp[mm] = src.layers[l].bias.data[mm].double()
                        pfz[l] = (Wp, bp, mr.clone())
                    if not _partial_mode:                  # frontier's partial rows are dropped
                        frozen.pop(rf, None); partial_mask.pop(rf, None); exact_mask.pop(rf, None)
                    pop = _reinit_frozen_population(dims, device, cfg.p, pfz, act=cfg.act)
                    opts = [torch.optim.Adam([p for p in n.parameters() if p.requires_grad],
                                             lr=cfg.lr) for n in pop]
                    X = torch.empty(0, dims[0])
                    Y = (torch.empty(0, dtype=torch.long) if cfg.hard else torch.empty(0, dims[-1]))
                    budget_end = t + 1 + cfg.outer
                    peel_base = t + 1
                    decay_at = {t + 1 + int(s * cfg.outer) for s in cfg.lr_sched}
                    combined_done = False
                    partial_hooked.clear(); partial_live.clear()
                    retries_done += 1
                    print(f"  [retry] L{rf + 1} {cur}/{ncout_r} peeled at end of budget: "
                          + ("kept solved rows pinned, reinit unsolved + deeper" if _partial_mode
                             else "reinit ENTIRE frontier layer + deeper")
                          + ", flushed buffer, RESTARTED iter clock " + f"(fresh {cfg.outer}-iter budget, runs to t={budget_end})"
                          + f" [{retries_done}/{cfg.retry}]", flush=True)
                    continue

            # --- freeze-reinit peel: once the frontier hidden layer reaches
            #     consensus, pin it (+ everything above it) and reinit the committee
            #     so the search collapses onto the deeper, still-unsolved layers ---
            if (cfg.freeze_reinit and cfg.cheat
                    and (t + 1) % cfg.log_every == 0):
                # cheat p=1: consensus can never form (nothing to cluster
                # against), so gate on the ORACLE per-layer eps of the single
                # member (same knobs as the CNN cheat-peel), then refine the
                # gated layers EXACTLY with the kink solver -- _cnn_refine_layer
                # handles FC frontiers generically, an MLP just needs the
                # ConvNet duck-type attrs -- and freeze SOLVED neurons only.
                Lh = len(dims) - 2
                # a layer is DONE only when EVERY neuron is solved -- a partially
                # frozen layer (stragglers left training) stays the frontier so we
                # keep re-attempting its unsolved neurons as they converge.
                def _done(l):
                    return l in frozen and bool(frozen[l][2].all())
                frontier = next((l for l in range(Lh) if not _done(l)), None)
                _pm = errs.get("max_eps_per_matrix")
                if frontier is not None and _pm is not None:
                    _pe = errs["mean_eps_per_matrix"]
                    lmax = max(_pm[2 * frontier], _pm[2 * frontier + 1])
                    lmean = _pe[2 * frontier]
                    fire_eps = (lmax <= cfg.cheat_peel_max
                                and lmean <= cfg.cheat_peel_mean)
                    cons_guess = None
                    if cfg.fast_peel and not fire_eps and cfg.p > 1:
                        # --fast-peel: the frontier layer peels as soon as the
                        # COMMITTEE fully agrees on it (the --fast consensus
                        # trigger) even though no single member clears the eps
                        # gate -- each member keeps its own private stragglers
                        # while the quorum means are already refiner-basin
                        # quality. The consensus rows become the guesses.
                        _cn, _cm = _partial_consensus(pop, dims,
                                                      cfg.cluster_eps,
                                                      cfg.cluster_quorum)
                        if (_cn is not None and frontier < len(_cm)
                                and bool(_cm[frontier].all())):
                            cons_guess = _cn
                    if fire_eps or cons_guess is not None:
                        # EXTEND: if the next hidden layer is ALREADY within the
                        # threshold too (refining the shallower ones barely moves
                        # its eps), peel it in the SAME pass instead of waiting a
                        # whole iteration -- and keep going while they qualify.
                        # (eps trigger only: a consensus fire says nothing about
                        # deeper layers, so it peels just the frontier.)
                        last = frontier
                        while fire_eps and last + 1 < Lh and (
                                max(_pm[2 * (last + 1)], _pm[2 * (last + 1) + 1])
                                <= cfg.cheat_peel_max
                                and _pe[2 * (last + 1)] <= cfg.cheat_peel_mean):
                            last += 1
                        if fire_eps:
                            print(f"  [cheat-peel] L{frontier + 1} eps max {lmax:.2e} "
                                  f"mean {lmean:.2e} within ({cfg.cheat_peel_max:g}, "
                                  f"{cfg.cheat_peel_mean:g}) -> refine+peel"
                                  + (f" (extending through L{last + 1})"
                                     if last > frontier else ""), flush=True)
                        else:
                            _cpm = param_errors(cons_guess, teacher,
                                                hard=cfg.hard)["max_eps_per_matrix"]
                            _cmax = max(_cpm[2 * frontier], _cpm[2 * frontier + 1])
                            print(f"  [fast-peel] L{frontier + 1} full committee "
                                  f"consensus ({dims[frontier + 1]}/"
                                  f"{dims[frontier + 1]} at quorum; CONSENSUS-GUESS "
                                  f"eps max {_cmax:.2e} (best single member "
                                  f"{lmax:.2e})) -> refine+peel from consensus "
                                  "guesses", flush=True)
                        try:
                            # MLP refine via _mlp_refine_layer with REAL-INPUT
                            # bases (the refine_v18min recipe) -- NOT the CNN
                            # refiner, whose idim>2048 "verify-first" path is
                            # conv-only and abstains on a dense MLP layer (0/512).
                            # --f64: refine + STORE the peeled prefix in fp64
                            # (training/members stay fp32). The refiner already
                            # computes in fp64; an fp64 `exact` stops the recovered
                            # weights being truncated back to fp32, and the frozen
                            # prefix restored below (+ the seal that inverts it for
                            # deeper layers) is then fp64-accurate.
                            exact = best.clone().to(device)
                            if cfg.f64:
                                exact = exact.double()
                            if cons_guess is not None:
                                # frontier guess = the committee's quorum means.
                                # GAUGE-MATCH: _partial_consensus unit-normalizes
                                # every prefix layer (scale_normalize_), but the
                                # frozen prefix in `exact` is pinned at GUESS-SCALE.
                                # So the consensus frontier weight's input columns
                                # are calibrated for a UNIT preceding layer, while
                                # the refiner's seal feeds guess-scale activations
                                # (h = diag(c)*h_unit, c = ||frozen prefix rows||).
                                # Left uncorrected, a "tight" 6e-3 consensus guess
                                # is column-mis-scaled in the refiner's frame and it
                                # locks a NEIGHBOUR kink -> frozen-wrong neurons.
                                # Divide each column j by c_j to land the guess in
                                # exact's gauge (bias is unaffected by input scale).
                                _wc = exact.layers[frontier].weight
                                cw = (cons_guess.layers[frontier].weight.detach()
                                      .to(device=_wc.device, dtype=_wc.dtype))
                                cb = (cons_guess.layers[frontier].bias.detach()
                                      .to(device=_wc.device, dtype=_wc.dtype))
                                if frontier > 0:
                                    c = (exact.layers[frontier - 1].weight.detach()
                                         .norm(dim=1).clamp_min(1e-12)
                                         .to(cw.dtype))
                                    cw = cw / c[None, :]
                                _wc.data.copy_(cw)
                                exact.layers[frontier].bias.data.copy_(cb)
                            _bpool = teacher_eval_pts.to(device).double()
                            fmasks = {}
                            _t_ref = time.time(); _q_ref = 0
                            for l in range(last + 1):
                                nt = exact.layers[l].weight.shape[0]
                                wdt = exact.layers[l].weight.dtype
                                # restore already-solved exact rows + refine ONLY
                                # the still-unsolved ones (so re-fires don't redo
                                # the whole layer -- and earlier stragglers get
                                # another shot every pass).
                                prior = (frozen[l][2].clone().to(device) if l in frozen
                                         else torch.zeros(nt, dtype=torch.bool, device=device))
                                if l in frozen:
                                    exact.layers[l].weight.data[prior] = frozen[l][0].to(device)[prior].to(wdt)
                                    exact.layers[l].bias.data[prior] = frozen[l][1].to(device)[prior].to(wdt)
                                todo = (~prior).nonzero(as_tuple=True)[0].tolist()
                                if not todo:
                                    fmasks[l] = prior
                                    continue
                                if cfg.design_refine and any(not bool(fmasks[k].all()) for k in range(l)):
                                    fmasks[l] = prior
                                    print(f"  [design-refine] L{l + 1}: deferred until the earlier prefix is complete",
                                          flush=True)
                                    continue
                                Wr, br, rmask, _nq = _mlp_refine_layer(
                                    teacher, exact, l, device, cfg.act,
                                    only_channels=todo,
                                    angle_gate=getattr(cfg, "peel_angle_gate", 12.0),
                                    bases=_bpool, xspace=cfg.xspace_refine,
                                    loc=cfg.loc_refine, design=cfg.design_refine)
                                _q_ref += _nq
                                peel_refinement_queries += _nq
                                if log:
                                    log[-1]["peel_refinement_queries"] = peel_refinement_queries
                                if Wr is not None:
                                    idx = rmask.nonzero(as_tuple=True)[0]     # newly solved
                                    if len(idx):
                                        # SCALE-MATCH: refiner returns UNIT [w|b];
                                        # rescale each to the guess's ReLU-gauge
                                        # scale (proj of guess onto the unit dir)
                                        # so the warm-kept downstream reads L at the
                                        # magnitude it trained for. Same hyperplane.
                                        gwf = exact.layers[l].weight.data[idx].double()
                                        gbf = exact.layers[l].bias.data[idx].double()
                                        uwf = Wr[idx].double(); ubf = br[idx].double()
                                        # anti-aligned guess -> clamp would pin a
                                        # ZERO row; fall back to guess magnitude
                                        proj = (gwf * uwf).sum(1) + gbf * ubf
                                        gn = (gwf.pow(2).sum(1) + gbf.pow(2)).sqrt()
                                        cs = torch.where(proj > 1e-6 * gn,
                                                         proj, gn).clamp_min(1e-8)
                                        exact.layers[l].weight.data[idx] = (cs[:, None] * uwf).to(wdt)
                                        exact.layers[l].bias.data[idx] = (cs * ubf).to(wdt)
                                    fmasks[l] = prior | rmask.to(device)
                                else:
                                    fmasks[l] = prior
                                print(f"  [cheat-peel] L{l + 1}: "
                                      f"{int(fmasks[l].sum())}/{nt} exact "
                                      f"(+{int((fmasks[l] & ~prior).sum())} new)",
                                      flush=True)
                            print(f"  [cheat-peel] refine: {time.time() - _t_ref:.1f}s"
                                  f"  |  {_q_ref} oracle queries", flush=True)
                            frozen = {l: (exact.layers[l].weight.detach().clone(),
                                          exact.layers[l].bias.detach().clone(),
                                          fmasks[l])
                                      for l in range(last + 1)}
                            if (cfg.peel_restart and bool(frozen[last][2].all())
                                    and last < Lh - 1):
                                # --peelrestart: FULL cold restart of the deeper
                                # search. (1) re-randomize unsolved + deeper weights
                                # (frozen rows pinned), (2) flush the sample buffer
                                # (stale -- mined against the pre-peel student),
                                # (3) restart the lr/iter schedule from here.
                                pop = _reinit_frozen_population(dims, device, cfg.p,
                                                                frozen, act=cfg.act)
                                X = torch.empty(0, dims[0])
                                Y = (torch.empty(0, dtype=torch.long) if cfg.hard
                                     else torch.empty(0, dims[-1]))
                                budget_end = t + 1 + cfg.outer   # fresh FULL budget
                                peel_base = t + 1                # restart shown iter
                                decay_at = {t + 1 + int(s * cfg.outer)
                                            for s in cfg.lr_sched}
                                print(f"  [cheat-peel] froze L1..L{last + 1} "
                                      f"(exact); --peelrestart: cold-reinit "
                                      f"unsolved+deeper, flushed buffer, RESTARTED "
                                      f"iter clock (it->1, fresh {cfg.outer}-iter "
                                      f"budget, runs to t={budget_end})", flush=True)
                            else:
                                # WARM (default): keep each member's TRAINED
                                # downstream and only pin the frozen rows -- cold-
                                # reiniting threw the trained L2 away (L2 max 0.06
                                # -> 0.63). Warm + scale-match keeps it put.
                                pop = _mlp_warm_reinit_population(pop, dims, cfg.act,
                                                                  device, frozen)
                                print(f"  [cheat-peel] froze L1..L{last + 1} "
                                      f"(recovered rows only, guess-scale; "
                                      f"L{last + 1}: {int(frozen[last][2].sum())}/{frozen[last][2].numel()}); kept trained "
                                      f"downstream (warm)", flush=True)
                            opts = [torch.optim.Adam(
                                [p for p in n.parameters() if p.requires_grad],
                                lr=cfg.lr) for n in pop]
                            combined_done = False
                            # IMMEDIATE post-freeze eps, BEFORE any further training
                            # (measured on `exact`: refined L1..frontier + guess
                            # rows + current downstream). Lets a bad freeze be told
                            # apart from a straggler that only diverges LATER: if a
                            # layer's max is already high here, the freeze did it;
                            # if it's low here but high at the next log, training did.
                            _iee = param_errors(exact, teacher, hard=cfg.hard)
                            _im = _iee["max_eps_per_matrix"]; _imn = _iee["mean_eps_per_matrix"]
                            _isp = layer_eps_split(exact, teacher,
                                     {l: fmasks[l] for l in fmasks
                                      if 0 < int(fmasks[l].sum()) < fmasks[l].numel()})
                            def _fz(i):
                                s = (f"L{i+1}[max {max(_im[2*i], _im[2*i+1]):.2e} "
                                     f"mean {_imn[2*i]:.2e}]")
                                if i in _isp:
                                    s += (f"(f:{_isp[i]['fz'][0]:.1e} "
                                          f"u:{_isp[i]['uf'][0]:.1e})")
                                return s
                            print("  [cheat-peel] post-freeze eps: "
                                  + "  ".join(_fz(i) for i in range(len(dims) - 1)),
                                  flush=True)
                        except Exception as e:
                            import traceback
                            print(f"  [cheat-peel] skipped ({e})\n"
                                  + traceback.format_exc(), flush=True)
                # ALL hidden layers peeled -> the LINEAR output layer is just a
                # closed-form ridge LSQ on the now-exact penultimate features.
                # Solve it once (machine precision, no training) and we're done.
                if (not out_solved and len(X) > 0
                        and all(l in frozen and bool(frozen[l][2].all())
                                for l in range(len(dims) - 2))):
                    try:
                        with torch.no_grad():
                            _xs, _ys = X[-30000:], Y[-30000:]
                            for _m in pop:                 # tiny ridge: features are
                                solve_last_layer_(_m, _xs, _ys, ridge=1e-10)  # exact
                        out_solved = True
                        _oe = param_errors(pop[0], teacher, hard=cfg.hard)
                        print("  [cheat-peel] all hidden peeled -> output layer "
                              f"solved (closed-form LSQ) -> max_eps "
                              f"{_oe['max_eps']:.2e}", flush=True)
                    except Exception as e:
                        print(f"  [cheat-peel] output solve skipped ({e})", flush=True)
            elif cfg.freeze_reinit and cstats is not None and "layers" in cstats:
                Lh = len(dims) - 2                          # number of hidden layers
                frontier = next((l for l in range(Lh) if l not in frozen), None)
                if frontier is not None:
                    lx = cstats["layers"][frontier]
                    ratio = lx["n_cons"] / max(lx["n_tot"], 1)
                    cm = lx.get("cons_max")
                    prec_ok = (cfg.freeze_precision <= 0 or
                               (cm is not None and cm <= cfg.freeze_precision))
                    if ratio >= cfg.freeze_thresh and prec_ok:
                        try:
                            cnet, masks = _partial_consensus(
                                pop, dims, cfg.cluster_eps, cfg.cluster_quorum)
                            cnet = cnet.double()                 # fp64 record for the refined rows
                            # REFINE the consensus frontier to the kink solver's
                            # precision BEFORE freezing (non-cheat peel; the cheat
                            # branch above does the same). Refined rows come back
                            # as unit [w|b]: scale-match them to the consensus
                            # row's gauge so the downstream keeps its magnitudes.
                            # rows already solved by --fast-peel-partial count as solved:
                            # copy them from the fp64 record and refine only the rest
                            _pre_solved = torch.zeros(cnet.layers[frontier].weight.shape[0],
                                                      dtype=torch.bool, device=device)
                            if partial_exact is not None and frontier in partial_mask:
                                _pre_solved = partial_mask[frontier].to(device).clone()
                                if bool(_pre_solved.any()):
                                    wdt_ = cnet.layers[frontier].weight.dtype
                                    cnet.layers[frontier].weight.data[_pre_solved] = partial_exact.layers[frontier].weight.data[_pre_solved].to(wdt_)
                                    cnet.layers[frontier].bias.data[_pre_solved] = partial_exact.layers[frontier].bias.data[_pre_solved].to(wdt_)
                                    masks[frontier] = masks[frontier] | _pre_solved
                            if (cfg.design_refine or cfg.loc_refine) and not bool(_pre_solved.all()):
                                _t_ref = time.time()
                                Wr, br, rmask, _nq = _mlp_refine_layer(
                                    teacher, cnet, frontier, device, cfg.act,
                                    only_channels=(~_pre_solved).nonzero(as_tuple=True)[0].tolist(),
                                    angle_gate=getattr(cfg, "peel_angle_gate", 12.0),
                                    xspace=cfg.xspace_refine, loc=cfg.loc_refine,
                                    design=cfg.design_refine)
                                peel_refinement_queries += _nq
                                if Wr is not None:
                                    idx = rmask.nonzero(as_tuple=True)[0]
                                    if len(idx):
                                        Lf = cnet.layers[frontier]; wdt = Lf.weight.dtype
                                        gwf = Lf.weight.data[idx].double(); gbf = Lf.bias.data[idx].double()
                                        uwf = Wr[idx].double(); ubf = br[idx].double()
                                        proj = (gwf * uwf).sum(1) + gbf * ubf
                                        gn = (gwf.pow(2).sum(1) + gbf.pow(2)).sqrt()
                                        cs = torch.where(proj > 1e-6 * gn, proj, gn).clamp_min(1e-8)
                                        Lf.weight.data[idx] = (cs[:, None] * uwf).to(wdt)
                                        Lf.bias.data[idx] = (cs * ubf).to(wdt)
                                        masks[frontier] = masks[frontier] | rmask.to(masks[frontier].device)
                                print(f"  [peel-refine] L{frontier + 1}: {int(rmask.sum())}/{lx['n_tot']} "
                                      f"rows refined ({_nq} oracle queries, {time.time() - _t_ref:.1f}s); "
                                      f"unrefined rows frozen at consensus", flush=True)
                                try:
                                    _re = param_errors(cnet, teacher, hard=cfg.hard)
                                    print(f"  [peel-refine] L{frontier + 1} eps after refine: "
                                          f"max {_re['max_eps_per_matrix'][2 * frontier]:.2e}", flush=True)
                                except Exception:
                                    pass
                            frozen = {l: (cnet.layers[l].weight.detach().clone(),
                                          cnet.layers[l].bias.detach().clone(),
                                          masks[l].clone())
                                      for l in range(frontier + 1)}
                            if cfg.fast_peel_partial:
                                # deeper layers trained warm on the pinned prefix: keep them
                                pop = _mlp_warm_reinit_population(pop, dims, cfg.act, device, frozen)
                            else:
                                pop = _reinit_frozen_population(dims, device, cfg.p, frozen,
                                                                act=cfg.act)
                            opts = [torch.optim.Adam(
                                [p for p in n.parameters() if p.requires_grad],
                                lr=cfg.lr) for n in pop]
                            combined_done = False          # allow a fresh combine
                            print(f"  [freeze-reinit] froze L1..L{frontier + 1} "
                                  f"({int(masks[frontier].sum())}/{lx['n_tot']} "
                                  f"consensus in L{frontier + 1}, cons_max "
                                  f"{cm if cm is None else f'{cm:.2e}'}); reinit "
                                  f"committee onto deeper layers", flush=True)
                            if cfg.peel_restart and frontier < Lh - 1:
                                # --peelrestart (same as the cheat branch): flush the
                                # sample buffer (mined against the pre-peel student)
                                # and restart the iter/lr clock with a fresh FULL
                                # budget for the deeper layers.
                                X = torch.empty(0, dims[0])
                                Y = (torch.empty(0, dtype=torch.long) if cfg.hard
                                     else torch.empty(0, dims[-1]))
                                budget_end = t + 1 + cfg.outer
                                peel_base = t + 1
                                decay_at = {t + 1 + int(s * cfg.outer)
                                            for s in cfg.lr_sched}
                                print(f"  [freeze-reinit] --peelrestart: flushed buffer, "
                                      f"RESTARTED iter clock (it->1, fresh {cfg.outer}-iter "
                                      f"budget, runs to t={budget_end})", flush=True)
                        except Exception as e:
                            print(f"  [freeze-reinit] skipped ({e})", flush=True)

            # --- extract-freeze: exactly extract each layer-1 consensus neuron
            #     (black-box ~2d probe) and pin+freeze the ones that come out clean ---
            if cfg.extract_freeze and cstats is not None:
                try:
                    est = _extract_freeze_round(
                        pop, opts, freeze_masks, hooked, dims, cfg, device,
                        teacher, exact_neurons, seed + t)
                    for jr, ang in est["found"]:
                        print(f"  [extract-freeze] FOUND a neuron -> pinned EXACT in "
                              f"place (jump_ratio {jr:.0f}, angle {ang:.1e} deg)",
                              flush=True)
                    rj = ", ".join(f"{k}:{v}" for k, v in est.get("reasons", {}).items())
                    print(f"  [extract-freeze] probed {est['probed']} consensus neurons"
                          f" | +{est['extracted']} exact ({est['total']}/{dims[1]} frozen)"
                          f" | {est['rejected']} rejected"
                          + (f" [{rj}]" if rj else "") + est.get("rjstr", ""),
                          flush=True)
                except Exception as e:
                    print(f"  [extract-freeze] skipped ({e})", flush=True)

            # --- cryptanalytic layer-1 verifier: whenever ANY layer-1 neuron
            #     has consensus (even one), probe it against the teacher's true
            #     kink hyperplane to verify + refine (black-box, teacher queries
            #     only -- not counted in the query budget) ---
            if cfg.verify:
                try:
                    from verify_layer1 import verify_layer1, format_summary
                    l1 = consensus_layer1_neurons(
                        pop, dims, quorum_ratio=cfg.cluster_quorum)
                    if l1:
                        res = verify_layer1(
                            teacher, [(w, b) for _, w, b in l1], X, device,
                            eps_offset=cfg.verify_eps_offset,
                            eps_angle=cfg.verify_eps_angle, k=cfg.verify_k,
                            refine=cfg.verify_refine,
                            max_neurons=cfg.verify_max, seed=seed + t)
                        rec["verify"] = res["summary"]
                        print(f"  [verify] L1 {format_summary(res['summary'])}",
                              flush=True)
                    else:
                        print("  [verify] L1 no consensus neurons yet", flush=True)
                except Exception as e:  # never let a diagnostic kill a run
                    print(f"  [verify] skipped ({type(e).__name__}: {e})",
                          flush=True)

        # --- peel-try (MLP, cfg.peel_try=k): every k iters ASK the refiner on
        #     the frontier hidden layer directly (no consensus gate -- works at
        #     p=1/cheat). Solved neurons accumulate in exact_net/exact_mask and
        #     are pinned across the committee; WARM (--peel-warm) keeps deeper
        #     layers trained, COLD reinits them. --peel-stuck cold-reinits to
        #     escape a plateau. Frontier advances only when the whole layer is
        #     exact + duplicate-free. Mirrors the CNN peel-try. ---
        if cfg.peel_try and (t + 1) % cfg.peel_try == 0 and not expanded:
            import copy as _copy
            frontier = _mlp_frontier_layer(frozen, len(pop[0].layers),
                                           cfg.peel_advance_frac)
            if frontier is not None:
                bi_pt = (0 if cfg.p == 1 else
                         min(range(cfg.p), key=lambda i: loss_on(pop, X, Y)[i]))
                guess = pop[bi_pt].clone()
                Cout = guess.layers[frontier].weight.shape[0]
                if exact_net is None:
                    exact_net = _copy.deepcopy(guess).to(device)
                if frontier not in exact_mask:
                    exact_mask[frontier] = torch.zeros(Cout, dtype=torch.bool,
                                                       device=device)
                wdt = exact_net.layers[frontier].weight.dtype
                ui = (~exact_mask[frontier]).nonzero(as_tuple=True)[0].tolist()
                for c in ui:                    # unsolved guesses <- current best
                    exact_net.layers[frontier].weight.data[c] = \
                        guess.layers[frontier].weight.data[c].to(wdt)
                    exact_net.layers[frontier].bias.data[c] = \
                        guess.layers[frontier].bias.data[c].to(wdt)
                start = peel_try_n % max(len(ui), 1)
                peel_try_n += 1
                n0 = int(exact_mask[frontier].sum())
                miss = 0
                _pt_t = time.time(); _pt_q = 0
                order = ui[start:] + ui[:start]
                for pi, c in enumerate(order):
                    Wr, br, rmask, _nq = _mlp_refine_layer(
                        teacher, exact_net, frontier, device, cfg.act,
                        only_channels=[c], angle_gate=cfg.peel_angle_gate,
                        xspace=cfg.xspace_refine, loc=cfg.loc_refine, design=cfg.design_refine)
                    _pt_q += _nq
                    peel_refinement_queries += _nq
                    if log:
                        log[-1]["peel_refinement_queries"] = peel_refinement_queries
                    if Wr is not None and bool(rmask[c]):
                        exact_net.layers[frontier].weight.data[c] = Wr[c].to(wdt)
                        exact_net.layers[frontier].bias.data[c] = br[c].to(wdt)
                        exact_mask[frontier][c] = True
                        miss = 0                          # consecutive: reset on solve
                    else:
                        miss += 1
                        if miss >= cfg.peel_miss_abort:
                            break
                    if (pi + 1) % 128 == 0:               # heartbeat: attempt is long
                        print(f"      [peel-try] L{frontier + 1} probing "
                              f"{pi + 1}/{len(order)}... {int(exact_mask[frontier].sum())} "
                              f"solved so far", flush=True)
                ns = int(exact_mask[frontier].sum())
                print(f"  [peel-try] L{frontier + 1}: {ns}/{Cout} exact"
                      + (" (aborted attempt)" if miss >= cfg.peel_miss_abort
                         else "")
                      + f"  |  {time.time() - _pt_t:.1f}s, {_pt_q} oracle queries",
                      flush=True)

                def _pin(lyr):
                    """Pin the SOLVED rows of layer lyr in the EXISTING member(s):
                    overwrite with the exact value + a persistent live-mask grad
                    hook (installed once per member/layer -> no pileup) + zero the
                    frozen rows' Adam momentum so they can't drift. NO reinit, so
                    the still-training rows keep their optimizer momentum (no
                    disruption). Records frozen[lyr] for the frontier finder."""
                    frozen[lyr] = (
                        exact_net.layers[lyr].weight.detach().clone(),
                        exact_net.layers[lyr].bias.detach().clone(),
                        exact_mask[lyr].clone())
                    for m, opt in zip(pop, opts):
                        key = (id(m), lyr)
                        if key not in peel_hooked:
                            live = torch.zeros(m.layers[lyr].weight.shape[0],
                                               dtype=torch.bool, device=device)
                            peel_live[key] = live
                            m.layers[lyr].weight.register_hook(
                                lambda g, k=live: g * (~k).to(g.dtype).unsqueeze(1))
                            m.layers[lyr].bias.register_hook(
                                lambda g, k=live: g * (~k).to(g.dtype))
                            peel_hooked.add(key)
                        live = peel_live[key]
                        idx = (exact_mask[lyr] & ~live).nonzero(as_tuple=True)[0]
                        if len(idx) == 0:
                            continue
                        wd = m.layers[lyr].weight.dtype
                        with torch.no_grad():
                            # overwrite with the EXACT direction, but in the ReLU
                            # scale gauge closest to the member's current guess:
                            # c = proj(guess onto unit exact dir) (>0, sign already
                            # aligned). Keeps the neuron's MAGNITUDE ~ the guess so
                            # the (trained) downstream barely needs to re-adapt --
                            # only the <=12deg direction change remains.
                            uw = exact_net.layers[lyr].weight[idx].to(wd)   # unit dir (weight)
                            ub = exact_net.layers[lyr].bias[idx].to(wd)     # unit dir (bias)
                            gw = m.layers[lyr].weight[idx]                  # current guess
                            gb = m.layers[lyr].bias[idx]
                            proj = (gw * uw).sum(1) + gb * ub               # (k,) optimal scale
                            gnorm = (gw.pow(2).sum(1) + gb.pow(2)).sqrt()   # (k,) guess magnitude
                            # proj minimizes ||c*u - guess||, but a sign-misaligned
                            # guess (proj<=0) would collapse the neuron toward 0; fall
                            # back to the guess magnitude (u is already sign-aligned).
                            c = torch.where(proj > 1e-6 * gnorm, proj, gnorm).clamp_min(1e-8)
                            m.layers[lyr].weight[idx] = c[:, None] * uw
                            m.layers[lyr].bias[idx] = c * ub
                            # keep exact_net (the deeper-layer refinement PREFIX) in
                            # the SAME gauge as the member, else L2 would be refined
                            # against a unit-scale L1 while the member uses guess-
                            # scale L1 -> gauge mismatch. (p=1/cheat: one member.)
                            ed = exact_net.layers[lyr].weight.dtype
                            exact_net.layers[lyr].weight.data[idx] = (c[:, None] * uw).to(ed)
                            exact_net.layers[lyr].bias.data[idx] = (c * ub).to(ed)
                        live[idx] = True
                        for p in (m.layers[lyr].weight, m.layers[lyr].bias):
                            st = opt.state.get(p)
                            if st:                        # zero Adam momentum -> stays put
                                if "exp_avg" in st: st["exp_avg"][idx] = 0
                                if "exp_avg_sq" in st: st["exp_avg_sq"][idx] = 0

                def _reroll(lyr, rows):
                    """Un-pin + re-randomize `rows` of layer lyr in place (fresh
                    direction + reset momentum); everything else keeps training."""
                    if not rows:
                        return
                    for m, opt in zip(pop, opts):
                        live = peel_live.get((id(m), lyr))
                        fresh = MLP(dims, act=cfg.act).to(device)
                        with torch.no_grad():
                            for c in rows:
                                if live is not None:
                                    live[c] = False
                                m.layers[lyr].weight[c] = fresh.layers[lyr].weight[c].to(m.layers[lyr].weight.dtype)
                                m.layers[lyr].bias[c] = fresh.layers[lyr].bias[c].to(m.layers[lyr].bias.dtype)
                        for p in (m.layers[lyr].weight, m.layers[lyr].bias):
                            st = opt.state.get(p)
                            if st:
                                for c in rows:
                                    if "exp_avg" in st: st["exp_avg"][c] = 0
                                    if "exp_avg_sq" in st: st["exp_avg_sq"][c] = 0

                if ns == Cout:
                    D = torch.cat([exact_net.layers[frontier].weight,
                                   exact_net.layers[frontier].bias[:, None]], 1).float()
                    # compare DIRECTIONS: rows are now pinned at the guess scale
                    # (not unit), so a raw-distance test both misses real collapses
                    # at different magnitudes AND false-triggers on a near-zero row
                    # (its diagonal 2*||D_i|| drops below the threshold).
                    D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    dd = torch.cdist(D, D, p=2) + 2 * torch.eye(Cout, device=D.device)
                    df = torch.cdist(D, -D, p=2)
                    M = torch.minimum(dd, df)
                    if float(M.min()) < 1e-3:
                        # COLLAPSE: >=2 neurons refined to the SAME teacher kink.
                        # Un-solve + reroll one twin of each pair (in place) so
                        # training splits the collapse toward the missing neuron.
                        dup = set()
                        for a, b in (M < 1e-3).nonzero().tolist():
                            if a < b and a not in dup and b not in dup:
                                dup.add(b)
                        for c in dup:
                            exact_mask[frontier][c] = False
                        _reroll(frontier, list(dup))
                        frozen[frontier] = (
                            exact_net.layers[frontier].weight.detach().clone(),
                            exact_net.layers[frontier].bias.detach().clone(),
                            exact_mask[frontier].clone())
                        peel_stuck_best[frontier] = ns - len(dup)
                        print(f"  [peel-try] collapse: {len(dup)} duplicate neuron(s) "
                              f"un-solved + rerolled to split (L{frontier + 1} now "
                              f"{int(exact_mask[frontier].sum())}/{Cout})", flush=True)
                    else:
                        _pin(frontier)
                        combined_done = False
                        print(f"  [peel-try] L{frontier + 1} fully exact -> pinned "
                              f"in place (momentum preserved); frontier advances",
                              flush=True)
                elif ns > n0:
                    _pin(frontier)
                    print(f"  [peel-try] pinned {ns - n0} newly solved in place "
                          f"(L{frontier + 1} at {ns}/{Cout})", flush=True)
                if ns > peel_stuck_best.get(frontier, -1):
                    peel_stuck_best[frontier] = ns
                    peel_stall = 0
                elif cfg.peel_stuck > 0 and ns < Cout:
                    peel_stall += 1
                    if peel_stall >= cfg.peel_stuck:
                        # escape: reroll the still-unsolved frontier rows in place
                        # (fresh signs) -- solved rows stay pinned, tail keeps its
                        # momentum (no cold reinit).
                        unsolved = (~exact_mask[frontier]).nonzero(
                            as_tuple=True)[0].tolist()
                        _reroll(frontier, unsolved)
                        peel_stall = 0
                        print(f"  [peel-try] stuck at {ns}/{Cout} -> rerolled "
                              f"{len(unsolved)} unsolved rows to escape", flush=True)

        # periodic population snapshot (inspect mid-run; no queries, overwrites)
        if (cfg.pop_save_every and cfg.pop_save_path
                and (t + 1) % cfg.pop_save_every == 0):
            # tmp + atomic rename: a crash mid-write can't corrupt the
            # previous checkpoint
            tmp = cfg.pop_save_path + ".tmp"
            torch.save({
                "dims": dims, "iter": t + 1, "act": cfg.act,
                "pop_states": [{k: v.detach().cpu() for k, v in
                                m.state_dict().items()} for m in pop],
                "teacher_state": {k: v.detach().cpu() for k, v in
                                  teacher.state_dict().items()},
                **({"frozen": {l: tuple(v.detach().cpu() for v in entry)
                               for l, entry in frozen.items()},
                    "peel_refinement_queries": peel_refinement_queries}
                   if cfg.design_refine else {}),
            }, tmp)
            os.replace(tmp, cfg.pop_save_path)

        # dump population + queries at a target iter (or first consensus), stop
        # the consensus that triggers the --fast stop (computed once at the
        # configurable threshold, then SAVED so the endgame/peel can reuse it).
        # cfg.stop_layer >= 0: fire on ONE hidden layer's full consensus (peeling);
        # else the whole-net consensus (all layers).
        _cons_hit = None; _stop_masks = None
        if cfg.stop_on_consensus and (t + 1) % cfg.log_every == 0:
            if cfg.stop_layer >= 0:
                _cnet_p, _stop_masks = _partial_consensus(
                    pop, dims, cfg.cluster_eps, cfg.cluster_quorum)
                if (cfg.stop_layer < len(_stop_masks)
                        and bool(_stop_masks[cfg.stop_layer].all())):
                    _cons_hit = _cnet_p          # partial consensus: this layer solved
            else:
                _cons_hit = build_consensus(pop, dims, eps=cfg.cluster_eps,
                                            quorum_ratio=cfg.cluster_quorum)
        _hit = ((cfg.dump_at_iter and (t + 1) == cfg.dump_at_iter) or
                (_cons_hit is not None))
        if cfg.dump_path and _hit:
            torch.save({
                "dims": dims, "iter": t + 1, "act": cfg.act,
                "pop_states": [{k: v.detach().cpu() for k, v in
                                m.state_dict().items()} for m in pop],
                # the CONSENSUS GUESS at the stop (whole-net, or the partial net whose
                # stop_layer is fully solved). None only if dumped via dump_at_iter.
                "consensus_state": ({k: v.detach().cpu() for k, v in
                                     _cons_hit.state_dict().items()}
                                    if _cons_hit is not None else None),
                # per-hidden-layer boolean consensus masks (layer-stop path only)
                "consensus_masks": ([m.detach().cpu() for m in _stop_masks]
                                    if _stop_masks is not None else None),
                "stop_layer": cfg.stop_layer,
                "quorum_ratio": cfg.cluster_quorum,
                "teacher_state": {k: v.detach().cpu() for k, v in
                                  teacher.state_dict().items()},
                "X": X.detach().cpu(), "Y": Y.detach().cpu(),
            }, cfg.dump_path)
            _why = (f"L{cfg.stop_layer + 1} full consensus" if cfg.stop_layer >= 0
                    and _cons_hit is not None else
                    ("whole-net consensus" if _cons_hit is not None else "fixed iter"))
            print(f"  [dump] iter {t + 1} ({_why}): population ({cfg.p} members) + "
                  f"consensus guess{'' if _cons_hit is not None else ' (none)'} + "
                  f"{len(X)} queries -> {cfg.dump_path}", flush=True)
            break

    # --- expand fit gate: no weight-vs-teacher scoring possible (dims differ) ---
    if expanded:
        losses = loss_on(pop, X, Y)
        bi = min(range(cfg.p), key=lambda i: losses[i])
        best = pop[bi]
        final = {
            "final_max_eps": float("nan"), "final_mean_eps": float("nan"),
            "final_max_eps_per_matrix": [],
            "final_agree": agreement(best, teacher, teacher_eval_pts),
            "best_loss": losses[bi], "queries": cfg.outer * cfg.q,
            "wall_s": round(time.time() - t0, 1),
            "expand": cfg.expand, "student_dims": student_dims,
        }
        return best, log, final

    # --- final selection / averaging / polish ---
    if cfg.ensemble_boost and boost_stages:
        # TERMINAL merge: compress the frozen cascade (+ current members)
        # into ONE width-n net -- the in-class extraction candidate. This is
        # the only compression in the boost path; the cascade sum was the
        # working model throughout.
        try:
            merged, minfo = merge_ensemble(
                [n_ for n_, _ in boost_stages] + list(pop), student_dims,
                cfg.act, X, Y, device, frozen=frozen,
                partial_exact=partial_exact, partial_mask=partial_mask,
                samples=cfg.ensemble_samples, block=cfg.ensemble_block,
                hard=cfg.hard,
                weights=([s_ for _, s_ in boost_stages]
                         + [boost_scale] * len(pop)))
            ncap = min(len(X), 200000)
            casc = _cascade_l1(X[-ncap:], Y[-ncap:])
            if merged is not None:
                if frozen:
                    _install_freeze(merged, frozen)
                finetune_(merged, X, Y, device, 200, cfg.batch,
                          cfg.lr, gen, tol=cfg.boost_tol)
                ml = l1_on([merged], X, Y)[0]
                print(f"  [ensemble-boost] terminal merge: cascade "
                      f"({len(boost_stages)} stages, loss {casc:.3e}) -> "
                      f"merged width-{student_dims[1]} (loss {ml:.3e})",
                      flush=True)
                best = merged
        except Exception as e:
            print(f"  [ensemble-boost] terminal merge failed ({e})", flush=True)
    if best is None:
        losses = loss_on(pop, X, Y)
        best = pop[min(range(cfg.p), key=lambda i: losses[i])]
    queries_used = (t + 1) * cfg.q
    extra = {}

    # --- checkpoint the raw reconstruction BEFORE the endgame solvers, so the
    #     LBFGS / last-layer polish can be re-tried offline via polish.py
    #     without re-running the whole query loop. ---
    if save_recon is not None:
        torch.save({
            "dims": dims,
            "best_state": {k: v.detach().cpu()
                           for k, v in best.state_dict().items()},
            "pop_states": [{k: v.detach().cpu()
                            for k, v in m.state_dict().items()} for m in pop],
            "teacher_state": {k: v.detach().cpu()
                              for k, v in teacher.state_dict().items()},
            "X": X.detach().cpu(),
            "Y": Y.detach().cpu(),
            "cfg": asdict(cfg),
            "seed": seed,
            "queries": queries_used,
            **({"frozen": {l: tuple(v.detach().cpu() for v in entry)
                           for l, entry in frozen.items()},
                "peel_refinement_queries": peel_refinement_queries,
                "refined_state": {k: v.detach().cpu() for k, v in
                                  _frozen_fp64_view(best, frozen).state_dict().items()}}
               if cfg.design_refine else {}),
            "boost_stages": [({k: v.detach().cpu()
                               for k, v in n_.state_dict().items()}, s_)
                             for n_, s_ in boost_stages],
            "pre_endgame_max_eps": param_errors(best, teacher,
                                                hard=cfg.hard)["max_eps"],
        }, save_recon)
        print(f"  [save] reconstruction checkpoint -> {save_recon}", flush=True)

    if cfg.lastlayer_every:
        solve_last_layer_(best, X, Y)
    if cfg.popavg_kappa > 0 and not (cfg.ensemble_boost and boost_stages):
        losses = loss_on(pop, X, Y)
        bi = min(range(cfg.p), key=lambda i: losses[i])
        avg, k = aligned_pop_average(pop, losses, cfg.popavg_kappa)
        if avg is not None:
            avg_loss = loss_on([avg], X, Y)[0]
            extra["popavg"] = {
                "k": k,
                "avg_loss": avg_loss,
                "best_loss": losses[bi],
                "avg_max_eps": param_errors(avg, teacher,
                                            hard=cfg.hard)["max_eps"],
                "best_max_eps": param_errors(pop[bi], teacher,
                                             hard=cfg.hard)["max_eps"],
            }
            # observable selection rule: lower training loss wins
            best = avg if avg_loss <= losses[bi] else pop[bi]
        else:
            best = pop[bi]
            extra["popavg"] = {"k": k}
    if cfg.polish_f64:
        best = polish_f64(best, X, Y, cfg, gen)
    if cfg.lbfgs_polish:
        best = polish_lbfgs(best, X, Y, cfg)

    if cfg.design_refine:
        best = _frozen_fp64_view(best, frozen)
    errs = param_errors(_frozen_fp64_view(best, frozen), teacher, hard=cfg.hard)
    if cfg.design_refine:
        best_param = next(best.parameters())
        teacher_param = next(teacher.parameters())
        final_agree = (batched(best, teacher_eval_pts.to(best_param.device, best_param.dtype)).argmax(1)
                       == batched(teacher, teacher_eval_pts.to(teacher_param.device, teacher_param.dtype)).argmax(1)).float().mean().item()
    else:
        final_agree = agreement(best, teacher, teacher_eval_pts)
    # buffer-loss metrics forward the query set D; under peel-direct D is the
    # frontier's h-space, so forward `best` from that frontier too (weight-eps
    # and agree above use x-space eval / weights and stay on the full net).
    best_bufwd = (_FrontierNet(best, probe_frontier)
                  if probe_frontier is not None else best)
    final = {
        "final_max_eps": errs["max_eps"],
        "final_mean_eps": sum(errs["mean_eps_per_matrix"]) /
        len(errs["mean_eps_per_matrix"]),
        "final_max_eps_per_matrix": errs["max_eps_per_matrix"],
        "final_agree": final_agree,
        # loss of the selected reconstruction on the accumulated query set D
        **({"final_xent": xent_on([best_bufwd], X, Y)[0]} if cfg.hard
           else {"final_mae": l1_on([best_bufwd], X, Y)[0],
                 "final_mse": mse_on([best_bufwd], X, Y)[0]}),
        "queries": queries_used,
        "peel_refinement_queries": peel_refinement_queries,
        "wall_s": round(time.time() - t0, 1),
        **extra,
    }
    return best, log, final


def polish_f64(best, X, Y, cfg, gen):
    """Last-mile refinement in float64 on CPU (MPS lacks float64)."""
    dev = next(best.parameters()).device
    net = best.clone().cpu().double()
    Xc, Yc = X.detach().cpu().double(), Y.detach().cpu().double()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    n = len(Xc)
    for ep in range(cfg.polish_epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            loss = (net(Xc[idx]) - Yc[idx]).abs().mean()
            loss.backward()
            opt.step()
    return net.float().to(dev)


def polish_lbfgs(best, X, Y, cfg, max_samples=12000, steps=60):
    """float64 LBFGS endgame with squared loss (Opus A2 + A3-lite):
    residual-proportional gradients + curvature, at float64 precision."""
    dev = next(best.parameters()).device
    net = best.clone().cpu().double()
    # solver window: restrict to the last cfg.solverwindow outer iters of queries
    # (the most-recent tail), then cap to max_samples for float64 tractability.
    # solverwindow=0 keeps the original behavior (random subsample of all).
    Xw, Yw = X, Y
    if cfg.solverwindow and cfg.solverwindow > 0:
        keep = cfg.solverwindow * cfg.q
        if keep < len(X):
            Xw, Yw = X[-keep:], Y[-keep:]
    n = len(Xw)
    if n > max_samples:
        idx = torch.randperm(n)[:max_samples]
        Xs, Ys = Xw[idx].cpu().double(), Yw[idx].cpu().double()
    else:
        Xs, Ys = Xw.cpu().double(), Yw.cpu().double()
    opt = torch.optim.LBFGS(net.parameters(), lr=0.5, max_iter=20,
                            history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((net(Xs) - Ys) ** 2).mean()
        loss.backward()
        return loss

    for _ in range(steps):
        opt.step(closure)
    return net.float().to(dev)


# ============================= CNN extraction ============================== #
@torch.no_grad()
def _cnn_consensus(pop, ref_idx, eps, quorum):
    """Per-unit QUORUM consensus for CNNs -- the conv analog of
    _consensus_from_ref (NOT a blind average). Canonicalize + channel/neuron-align
    every member into ref's frame; then for each out-channel (conv) / neuron (FC)
    take the largest eps-ball (inf-norm agreement cluster) across members and
    average ONLY the agreeing members if the cluster meets `quorum`. Units without
    quorum keep the reference value. Returns (consensus ConvNet, per_layer) with
    per_layer[l] = (n_total, n_consensus) for each hidden layer."""
    ref = pop[ref_idx].clone(); cnn_canonicalize_(ref)
    aligned = []
    for m in pop:
        c = m.clone(); cnn_canonicalize_(c); cnn_align_to_(c, ref)
        aligned.append(c)
    net = ref.clone()
    L = len(net.layers)
    per_layer, masks = [], []
    for l in range(L):
        hidden = l < L - 1
        H, nc = net.layers[l].weight.shape[0], 0
        mask = torch.zeros(H, dtype=torch.bool, device=net.layers[l].weight.device)
        for k in range(H):
            feats = torch.stack([
                torch.cat([a.layers[l].weight[k].reshape(-1),
                           a.layers[l].bias[k:k + 1]]) for a in aligned])
            S = _largest_eps_ball(feats, eps)
            if len(S) >= quorum:
                mask[k] = True
                if hidden:
                    nc += 1
                net.layers[l].weight.data[k] = torch.stack(
                    [aligned[s].layers[l].weight[k] for s in S]).mean(0)
                net.layers[l].bias.data[k] = torch.stack(
                    [aligned[s].layers[l].bias[k] for s in S]).mean(0)
        masks.append(mask)
        if hidden:
            per_layer.append((H, nc))
    return net, per_layer, masks


@torch.no_grad()
def _cnn_consensus_eps(cons, masks, teacher):
    """Per-layer eps of the consensus vs teacher, measured ONLY over the units
    that actually reached quorum (mapped through the teacher alignment). Returns
    per_layer list of dicts {n_cons, n_tot, max, mean} with max/mean = None when
    no unit in that layer reached consensus (-> reported as n/a)."""
    t = teacher.clone(); cnn_canonicalize_(t)
    r = cons.clone(); cnn_canonicalize_(r)
    perms = cnn_align_to_(r, t)                     # perms for layers 0..L-2
    out = []
    for l in range(len(r.layers)):
        m = masks[l]
        if l < len(perms):                          # map mask into aligned frame
            m = m[perms[l].to(m.device)]
        idxs = m.nonzero(as_tuple=True)[0]
        H = r.layers[l].weight.shape[0]
        if len(idxs) == 0:
            out.append({"n_cons": 0, "n_tot": H, "max": None, "mean": None})
            continue
        dw = (r.layers[l].weight[idxs] - t.layers[l].weight[idxs]).abs()
        db = (r.layers[l].bias[idxs] - t.layers[l].bias[idxs]).abs()
        out.append({"n_cons": int(len(idxs)), "n_tot": H,
                    "max": max(dw.max().item(), db.max().item()),
                    "mean": dw.mean().item()})
    return out


def _cnn_install_freeze(net, frozen):
    """Conv analog of _install_freeze: pin each frozen layer's consensus out-channels
    (conv) / neurons (fc) to the shared consensus values and mask their gradients so
    they never move. frozen: {layer: (W, b, mask)} with mask a bool over out-channels.
    Non-consensus channels keep the member's fresh init and stay trainable."""
    for l, (W, b, mask) in frozen.items():
        lay = net.layers[l]
        dev = lay.weight.device
        m = mask.to(dev)
        with torch.no_grad():
            # cast to member dtype: frozen may be fp64 (--f64 extraction) while the
            # member is fp32; masked index_put requires matching dtypes
            lay.weight[m] = W.to(dev).to(lay.weight.dtype)[m]
            lay.bias[m] = b.to(dev).to(lay.bias.dtype)[m]
        shp = (-1,) + (1,) * (lay.weight.dim() - 1)     # (out,1,1,1) conv | (out,1) fc
        wkeep = (~m).to(lay.weight.dtype).reshape(shp)
        bkeep = (~m).to(lay.bias.dtype)
        lay.weight.register_hook(lambda g, k=wkeep: g * k)
        lay.bias.register_hook(lambda g, k=bkeep: g * k)


def _cnn_reinit_frozen_population(input_shape, conv_cfgs, fc_dims, out_dim, act,
                                  device, P, frozen):
    """Fresh committee of P ConvNets; every layer in `frozen` has its consensus
    out-channels pinned + gradient-masked and SHARED across all members (stragglers
    and deeper layers are fresh). Collapses the search onto the unsolved layers."""
    pop = []
    for _ in range(P):
        m = ConvNet(input_shape, conv_cfgs, fc_dims, out_dim, act).to(device)
        _cnn_install_freeze(m, frozen)
        pop.append(m)
    return pop


def _cnn_warm_reinit_population(pop, input_shape, conv_cfgs, fc_dims, out_dim,
                                act, device, frozen, reroll_frontier=None):
    """WARM analog of _cnn_reinit_frozen_population: keep every member's CURRENT
    trained weights -- deeper layers and (by default) unsolved frontier rows stay
    warm across a partial freeze, so the tail's training is NOT discarded -- and
    only (re)pin + gradient-mask the frozen channels to their shared exact values.
    Fresh module objects (avoids freeze-hook pileup) loaded from each member's own
    state_dict. reroll_frontier=(layer, unsolved_mask): additionally re-randomize
    just those frontier rows (fresh signs to escape flip minima) while everything
    else stays warm."""
    new_pop = []
    for src in pop:
        m = ConvNet(input_shape, conv_cfgs, fc_dims, out_dim, act).to(device)
        m.load_state_dict(src.state_dict())          # warm: copy current weights
        if reroll_frontier is not None:
            l, um = reroll_frontier
            idx = um.to(device).nonzero(as_tuple=True)[0]
            if len(idx):
                fresh = ConvNet(input_shape, conv_cfgs, fc_dims, out_dim,
                                act).to(device)
                with torch.no_grad():               # unsolved frontier rows only
                    m.layers[l].weight[idx] = fresh.layers[l].weight[idx]
                    m.layers[l].bias[idx] = fresh.layers[l].bias[idx]
        _cnn_install_freeze(m, frozen)               # pin frozen channels, mask grads
        new_pop.append(m)
    return new_pop


def _cnn_frontier_layer(frozen, nlay, advance_frac=1.0):
    """First hidden layer not yet `advance_frac`-solved. advance_frac=1.0: never
    advance until every channel is solved (default). <1.0: advance once that
    fraction is solved so stuck channels don't block deeper layers."""
    for l in range(nlay):
        fz = frozen.get(l)
        frac = fz[2].float().mean().item() if fz is not None else 0.0
        if frac < advance_frac:
            return l
    return None

def _cnn_prefix_input(pre, img_flat, frontier, rf):
    """Features feeding layer `frontier` of ConvNet `pre` (whose layers 0..frontier-1
    are the EXACT/refined earlier layers), as a differentiable function of the flat
    input. Conv frontier -> the receptive-field patch at rf=(r0,c0,k) flattened; FC
    frontier -> the full flattened input to that FC layer."""
    x = img_flat.view(img_flat.shape[0], *pre.input_shape)
    for i in range(pre.n_conv):
        if i == frontier:
            r0, c0, k = rf
            return x[:, :, r0:r0 + k, c0:c0 + k].reshape(x.shape[0], -1)
        x = pre.act(pre.layers[i](x))
        if pre.pools[i] > 0:
            x = F.avg_pool2d(x, pre.pools[i])
    x = torch.flatten(x, 1)
    for j in range(pre.n_conv, len(pre.layers) - 1):
        if j == frontier:
            return x
        x = pre.act(pre.layers[j](x))
    return x


@torch.no_grad()
def _cnn_safe_r(pre, x0, uh, frontier):
    """Radius of the current activation region along +/-uh from x0: how far we can
    travel before ANY earlier-layer (< frontier) ReLU flips. Confining the kink
    search to this radius means only target-layer (and deeper) kinks appear in the
    window -- no upstream contamination -- and the window auto-scales to the region
    (fixing the post-pooling small-gradient window problem). Returns inf if no
    earlier layers (frontier 0)."""
    eps = 1e-4
    xa = x0.view(1, *pre.input_shape)
    xb = (x0 + eps * uh).view(1, *pre.input_shape)
    safe = float("inf")
    for i in range(min(frontier, pre.n_conv)):
        pa, pb = pre.layers[i](xa), pre.layers[i](xb)
        dp = (pb - pa) / eps                          # directional deriv of the conv preact
        t = torch.where(dp.abs() > 1e-12, -(pa / dp), torch.full_like(pa, 1e30)).abs()
        safe = min(safe, t.min().item())
        xa, xb = pre.act(pa), pre.act(pb)
        if pre.pools[i] > 0:
            xa, xb = F.avg_pool2d(xa, pre.pools[i]), F.avg_pool2d(xb, pre.pools[i])
    if frontier >= pre.n_conv:
        xa, xb = torch.flatten(xa, 1), torch.flatten(xb, 1)
        for j in range(pre.n_conv, frontier):
            pa, pb = pre.layers[j](xa), pre.layers[j](xb)
            dp = (pb - pa) / eps
            t = torch.where(dp.abs() > 1e-12, -(pa / dp), torch.full_like(pa, 1e30)).abs()
            safe = min(safe, t.min().item())
            xa, xb = pre.act(pa), pre.act(pb)
    return safe


@torch.no_grad()
@torch.no_grad()
def _cnn_apply_perms_(net, perms, upto, opt=None, live_masks=None):
    """Permute hidden layers 0..upto of a RAW ConvNet member in place with the
    per-layer permutations `perms` (from cnn_align_to_ on its canonical clone):
    new position i holds old unit idx[i]; the next layer's input channels follow.
    Function-preserving (channel relabeling). Adam state of touched parameters
    is zeroed; live (pin) masks in `live_masks` {(id(net), l): mask} follow."""
    L = net.layers
    for i in range(upto + 1):
        idx = perms[i]
        if bool((idx == torch.arange(len(idx), device=idx.device)).all()):
            continue
        W, b = L[i].weight, L[i].bias
        W.copy_(W[idx]); b.copy_(b[idx])
        nxt = L[i + 1]
        conv = W.dim() == 4
        if conv and nxt.weight.dim() == 4:
            nxt.weight.copy_(nxt.weight[:, idx])
        elif conv and nxt.weight.dim() == 2:
            k = W.shape[0]; mm, n = nxt.weight.shape; sec = n // k
            nxt.weight.copy_(nxt.weight.view(mm, k, sec)[:, idx, :].reshape(mm, n))
        else:
            nxt.weight.copy_(nxt.weight[:, idx])
        if opt is not None:
            for p in (W, b, nxt.weight):
                st = opt.state.get(p)
                if st:
                    for k_ in ("exp_avg", "exp_avg_sq"):
                        if k_ in st:
                            st[k_].zero_()
        if live_masks is not None and (id(net), i) in live_masks:
            lm = live_masks[(id(net), i)]
            lm.copy_(lm[idx])


def _scale_match_rows(layer, idx, uW, ub):
    """Inject the EXACT refined hyperplane into rows `idx` of `layer`, but at the
    GUESS's magnitude -- the network-isomorphism-preserving injection. The kink
    refiner recovers only the hyperplane DIRECTION (uW, ub are unit-[w|b] per row);
    the magnitude is the ReLU/leaky scale gauge (scale a neuron by s>0, the next
    layer's incoming weights by 1/s -> same function). So project the current guess
    [w|b] onto the refined unit [w|b] and rescale to THAT magnitude: same exact
    hyperplane, guess magnitude, downstream layer left undisturbed (unit injection
    would break a warm/trained downstream, cf. the MLP L2 0.06->0.63 regression).
    Handles Conv2d (4D weight, per-output-channel gauge) and Linear (2D). `idx` is a
    LongTensor of row/channel indices; the layer's rows must currently hold the guess."""
    W = layer.weight.data; b = layer.bias.data
    k = int(idx.numel())
    if k == 0:
        return
    uW = uW.to(W.device); ub = ub.to(b.device)
    gw = W[idx].reshape(k, -1).double()               # guess weight per row (flat)
    gb = b[idx].double()                              # guess bias
    uw = uW[idx].reshape(k, -1).double()              # refined unit weight (flat)
    ubv = ub[idx].double()                            # refined unit bias
    # magnitude = guess projected onto the refined unit direction (least-squares
    # optimal: closest point on the refined line to the guess). If the refined dir is
    # ANTI-aligned with the guess (proj<=0 -- a solve from the NEGATED guess), the
    # projection would pin a ~ZERO (dead) row; fall back to the guess's full norm so
    # the neuron keeps a sensible positive magnitude. Mirrors the MLP partial guard.
    proj = (gw * uw).sum(1) + gb * ubv                # guess . refined-unit
    gn = (gw.pow(2).sum(1) + gb.pow(2)).sqrt()        # guess magnitude
    cs = torch.where(proj > 1e-6 * gn, proj, gn).clamp_min(1e-8)
    shp = [k] + [1] * (W.dim() - 1)                   # broadcast over conv/fc dims
    W[idx] = (cs.view(shp) * uW[idx].double()).to(W.dtype)
    b[idx] = (cs * ub[idx].double()).to(b.dtype)


_CNN_KINK = False   # set by reconstruct_cnn from cfg.loc_refine: route the CNN peel
                    # refiner to kink_solve (forward-only kink points, conv-aware)


def _cnn_refine_layer(teacher, cons, frontier, input_shape, device, act,
                      only_channels=None):
    if _CNN_KINK and act in ("relu", "leaky_relu"):
        import kink_solve
        _g = torch.Generator(device=device).manual_seed(1234 + frontier)
        W, b, mask, _nq = kink_solve.recover_layer(teacher, cons, frontier, device,
                                                   only_channels=only_channels, gen=_g,
                                                   sampling="track")
        return W, b, mask
    """Exactly refine the frontier layer's neurons with the VALIDATED kink refiner
    (verify_layer1), for ANY layer. `only_channels`: refine just these channel
    indices (for retrying previously-missed ones); default = all. Returns
    (W_ref, b_ref, refined_mask) with refined_mask[c]=True where the channel was
    exactly recovered (others keep the guess). General recipe (proven):
      * probe in image space; the neuron's image-space normal is J_phi^T w (J_phi =
        the known frozen-earlier-layers Jacobian, via autograd), so no closed-form
        mapback is needed to LOCATE it;
      * refine_neuron finds the kink and identifies THIS neuron by normal-angle
        (immune to nearer other/deeper kinks) -> a robust kink POINT;
      * collect points ACROSS activation regions, map each to h = prefix(image), and
        SVD-fit w.h+b=0 in the layer's INPUT space -> exact weights (aggregating
        regions handles per-region under-excitation).
    `cons` carries the exact earlier layers (frozen + shared in the population) as the
    prefix and the consensus frontier as the per-neuron guess. float64, exact-or-
    abstain (angle-gated, no oracle). Returns (W_ref, b_ref, n_ok). relu only."""
    import copy
    from verify_layer1 import (_Oracle, refine_neuron, _find_kinks,
                               _angle_estimate)
    if act not in ("relu", "leaky_relu"):
        return None, None, 0
    pre = copy.deepcopy(cons).double().to(device).eval()
    orc = _Oracle(copy.deepcopy(teacher).double().to(device).eval())
    idim = int(input_shape[0] * input_shape[1] * input_shape[2])
    # Large-input regime (e.g. 3x224x224): VERIFY-FIRST. The guess predicts
    # the kink's location (t~0 after Newton projection) and normal (uh) -- so
    # check that prediction with a narrow scan + an O(k) angle probe, and pay
    # for the exact bisect+sweep only when it holds. A bad guess then fails in
    # a handful of cheap probes instead of a full discovery budget. Small
    # inputs keep the validated LeNet-scale flow unchanged.
    big = idim > 2048          # CIFAR-dim (3072) and up; LeNet/MNIST (784) keep
    n_probe_cap = 40 if big else None                  # the validated slow path
    window_cap = 0.2 if big else 1.0
    VF_FAILS = 6                # big: consecutive failed predictions -> abstain
    is_conv = frontier < pre.n_conv
    rf = None
    if is_conv:                                       # interior receptive-field location
        C, H, W = input_shape
        for i in range(frontier):
            _, _, k, s, pad, pool = pre.conv_cfgs[i]
            H = (H - k + 2 * pad) // s + 1; W = (W - k + 2 * pad) // s + 1
            if pool > 0:
                H = (H - pool) // pool + 1; W = (W - pool) // pool + 1
        _, _, k, s, pad, _ = pre.conv_cfgs[frontier]
        oH = (H - k + 2 * pad) // s + 1
        loc = next((o for o in range(oH) if 0 <= o * s - pad and o * s - pad + k <= H), None)
        if loc is None:
            return None, None, 0
        rf = (loc * s - pad, loc * s - pad, k)
    Wl = pre.layers[frontier].weight                 # (Cout,Cin,k,k) or (out,in)
    bl = pre.layers[frontier].bias
    gw = Wl.reshape(Wl.shape[0], -1)                  # (Cout, Din) per-neuron guess
    Cout, Din = gw.shape
    W_ref = Wl.detach().clone(); b_ref = bl.detach().clone()

    def phi(imgb):
        return _cnn_prefix_input(pre, imgb, frontier, rf)

    def phi_vec(z):
        return phi(z.unsqueeze(0)).squeeze(0)

    refined = torch.zeros(Cout, dtype=torch.bool, device=device)
    chans = range(Cout) if only_channels is None else list(only_channels)

    sup = None
    if is_conv and frontier == 0:
        # layer-1 conv frontier: the neuron's image-space normal is exactly the
        # filter scattered at the receptive field -> sweep only its support
        # (k*k*C coords instead of the full image dim; exact, not approximate).
        C_in, H_in, W_in = input_shape
        r0, c0, k = rf
        ii = torch.arange(k, device=device)
        sup = (torch.arange(C_in, device=device)[:, None, None] * (H_in * W_in)
               + (r0 + ii)[None, :, None] * W_in
               + (c0 + ii)[None, None, :]).reshape(-1)

    def _solve(wg, bg):
        """One exact-or-abstain attempt from guess (wg, bg) -> (w, b) or None.
        Fresh gen per attempt keeps attempts reproducible across retries."""
        gen = torch.Generator(device=device).manual_seed(0)
        Cblocks, hs = [], []                          # per-kink normal constraints + points
        fails = 0                                     # big: consecutive prediction failures
        for _ in range(n_probe_cap or max(40, Din)):  # a FEW kinks, not Din points
            base = torch.randn(idim, generator=gen, device=device, dtype=torch.float64) * 0.5
            # ITERATED Newton projection onto the guessed plane (phi is piecewise-linear
            # through the frozen layers, so one step overshoots across their kinks).
            xcur, uh, nu, b_img = base, None, None, None
            for _ in range(6):
                xg = xcur.detach().clone().requires_grad_(True)
                s0 = (phi_vec(xg) * wg).sum() + bg
                u = torch.autograd.grad(s0, xg)[0]; nu = u.norm()
                if nu < 1e-12:
                    break
                uh = (u / nu).detach()
                b_img = (s0 - u @ xg).reshape(()).detach()
                if abs(float(s0)) < 1e-9:
                    break
                xcur = (xg - (s0 / nu ** 2) * u).detach()
            if uh is None or nu < 1e-12:
                continue
            with torch.no_grad():
                x0 = xcur
                sr = _cnn_safe_r(pre, x0, uh, frontier)        # activation-region radius
                window = min(0.9 * sr, window_cap)
                if window < 1e-4:
                    continue
                if big:
                    # VERIFY the prediction first: kink expected at t~0 with
                    # normal ~uh. Scan narrow-to-wide; at EACH level angle-test
                    # every candidate (dense landscapes put foreign kinks in
                    # the narrow levels -- they must not stop the escalation);
                    # only an angle-matching kink earns the bisect + sweep.
                    lvls = [w for w in (1e-3, 1e-2) if w < window] + [window]
                    picked = None
                    for wv in lvls:
                        for tc in _find_kinks(orc, x0, uh, wv, 81,
                                              kink_tol=1e-9, n_cand=3):
                            ang = _angle_estimate(orc, x0 + tc * uh, uh,
                                                  12, 1e-3, 1e-4, gen)
                            if ang <= 10.0:
                                picked = wv
                                break
                        if picked is not None:
                            break
                    if picked is None:
                        fails += 1
                        if fails >= VF_FAILS:
                            break                     # guess is bad: abstain fast
                        continue
                    out = refine_neuron(orc, (uh * nu).detach(), b_img, x0,
                                        window=picked, n_scan=81, n_cand=3,
                                        gen=gen, support=sup)
                else:
                    out = refine_neuron(orc, (uh * nu).detach(), b_img, x0,
                                        window=window, n_scan=81, n_cand=15,
                                        gen=gen, support=sup)
                if out is None or out.get("angle_deg") is None or out["angle_deg"] > 2.0:
                    if big:
                        fails += 1
                        if fails >= VF_FAILS:
                            break
                    continue
                # jump_ratio gate: a dirty fd normal (the sweep crossed a
                # NEIGHBOURING kink -- jump_ratio ~1e1 instead of ~1e9) can pass
                # the 2-deg angle gate, and ONE such constraint destroys the
                # null line below (the null_bad abstains) -> reject the kink.
                if out.get("jump_ratio", 0.0) < 1e6:
                    if big:
                        fails += 1
                        if fails >= VF_FAILS:
                            break
                    continue
                fails = 0                             # verified accept: reset
                xs = x0 + out["offset"] * uh
                n_k = out["w_refined"].to(device).double()         # image-space normal (unit)
                h_k = phi_vec(xs)
            # A_k = d phi / d image at the kink (Din x image_dim); the neuron's true
            # image-normal is A_k^T W, parallel to n_k -> constraint (I - n n^T) A_k^T W = 0.
            A_k = torch.autograd.functional.jacobian(phi_vec, xs.detach())
            with torch.no_grad():
                Cblocks.append(A_k.t() - torch.outer(n_k, A_k @ n_k))   # (image_dim, Din)
                hs.append(h_k)
            if len(Cblocks) >= 3:                     # stop once the null space is a clean line
                S = torch.linalg.svdvals(torch.cat(Cblocks, 0))
                if S[-1] < 1e-5 * S[-2]:
                    break
        if len(Cblocks) < 3:
            return None                               # abstain
        C = torch.cat(Cblocks, 0)
        _, S, Vh = torch.linalg.svd(C, full_matrices=False)
        tol = max(1e-9, 1e-4 * S[0].item())            # fd-limited: near-zero sing. vals
        null_dim = int((S < tol).sum().item())         # 1 for conv; >1 if FC input is
        if null_dim == 0 or null_dim > Din // 2:       # rank-deficient (unreachable dirs)
            return None                                 # no/too-large null -> abstain
        N = Vh[-null_dim:]                              # null-space basis (null_dim, Din)
        # 1-dim null -> exact W (the null direction). Multi-dim -> project the GUESS onto
        # the null: pins the reachable part, keeps the guess on functionally-irrelevant
        # (unreachable) directions. Reduces to exact recovery when null_dim == 1.
        W = (N @ wg) @ N
        W = W / W.norm().clamp_min(1e-12)
        b = -torch.stack([W @ h for h in hs]).mean()
        wb = torch.cat([W, b.reshape(1)])
        wb = wb / wb.norm().clamp_min(1e-12)           # unit-L2 [W,b] == cnn_canonicalize_
        w, b = wb[:-1], wb[-1]
        if (w * wg).sum() + b * bg < 0:
            w, b = -w, -b
        # runaway check (no oracle): a real refine barely moves the (canonicalized)
        # guess; a wrong-neuron null lock jumps far -> abstain instead of being
        # confidently wrong. Uses only the guess, keeping exact-or-abstain.
        sg = torch.cat([wg, bg.reshape(1)]).norm().clamp_min(1e-12)
        if max((w - wg / sg).abs().max().item(), (b - bg / sg).abs().item()) > 0.3:
            return None
        return w, b

    for c in chans:
        # Try the guess as-is, then NEGATED: a sign-flipped guess (flip local
        # minimum of the committee) has the SAME kink plane, so the refiner can
        # still lock it. NB the solved sign is inherited from the guess used,
        # not inferred -- negated-orientation solves are sign-suspect.
        res = _solve(gw[c], bl[c].reshape(()))
        if res is None:
            res = _solve(-gw[c], (-bl[c]).reshape(()))
            if res is not None:
                print(f"      [refine] ch {c} solved from NEGATED guess "
                      f"(sign inherited -> suspect)", flush=True)
        if res is None:
            continue
        w, b = res
        W_ref[c] = w.view(Wl.shape[1:]).to(W_ref.dtype); b_ref[c] = b.to(b_ref.dtype)
        refined[c] = True
    return (W_ref.to(pre.layers[frontier].weight.dtype),
            b_ref.to(pre.layers[frontier].bias.dtype), refined)


# ----------------------------------------------------------- MLP peel-try --
def _mlp_frontier_layer(frozen, nlay, advance_frac=1.0):
    """First HIDDEN layer (< nlay-1, the head is never a frontier) not yet
    `advance_frac`-solved. advance_frac=1.0: never advance until EVERY neuron is
    solved (default). <1.0: advance once that fraction is solved, so a few stuck
    neurons don't block deeper layers (peel_advance_frac). Mirrors
    _cnn_frontier_layer."""
    for l in range(nlay - 1):
        fz = frozen.get(l)
        frac = fz[2].float().mean().item() if fz is not None else 0.0
        if frac < advance_frac:
            return l
    return None


def _mlp_prefix_input(pre, x, frontier):
    """Flat input feeding layers[frontier] of MLP `pre` (layers 0..frontier-1
    are the exact/refined prefix); differentiable in x. frontier 0 -> x."""
    h = x
    for i in range(frontier):
        h = pre.act(pre.layers[i](h))
    return h


def _mlp_prefix_inverse(frozen, n, act, device):
    """x_of_h: map the input of layer `n` back to the raw network input by inverting
    the exactly-frozen layers 0..n-1. IMPORTANT: each frozen layer is UNIT-normalized
    per neuron before inverting, so `h` is the NORMALIZED activation (uniform scale)
    rather than the frozen layer's arbitrary GUESS magnitude. Inverting the guess-
    magnitude layer distorts the query space by the per-neuron magnitude (some neurons
    compressed -> under-excited -> stragglers); the unit gauge removes that. W_i^+ =
    W_iᵀ(W_iW_iᵀ)⁻¹, act⁻¹ exact for leaky_relu. fp64 (act⁻¹ /0.01 amplifies negatives)."""
    ns = 0.01
    # fp64: leaky_relu^-1 divides by 0.01 (x100), so in fp32 the x_of_h Jacobian loses
    # ~14% of the (tiny, ~1e-5) disagreement gradient's DIRECTION -> the 30-step query
    # optimizer walks to different queries and the frontier straggler never resolves.
    # Computing x_of_h + its Jacobian in fp64 restores the gradient (rel err ~1e-13), so
    # peel-direct queries == direct. Caller casts the fp64 output back to the model dtype.
    def act_inv(a):
        return a if act == "relu" else torch.where(a >= 0, a, a / ns)
    W = [frozen[i][0].to(device).double() for i in range(n)]
    b = [frozen[i][1].to(device).double() for i in range(n)]
    pinv = [W[i].t() @ torch.linalg.inv(W[i] @ W[i].t()) for i in range(n)]
    def x_of_h(h):
        a = h.double()
        for i in range(n - 1, -1, -1):
            a = (act_inv(a) - b[i]) @ pinv[i].t()
        return a
    return x_of_h


@torch.no_grad()
def _mlp_safe_r(pre, x0, uh, frontier):
    """Radius along +/-uh from x0 before any prefix (< frontier) LeakyReLU flips,
    so the kink search sees only target-and-deeper kinks. inf if frontier 0."""
    eps = 1e-4
    xa, xb = x0.unsqueeze(0), (x0 + eps * uh).unsqueeze(0)
    safe = float("inf")
    for i in range(frontier):
        pa, pb = pre.layers[i](xa), pre.layers[i](xb)
        dp = (pb - pa) / eps
        t = torch.where(dp.abs() > 1e-12, -(pa / dp),
                        torch.full_like(pa, 1e30)).abs()
        safe = min(safe, t.min().item())
        xa, xb = pre.act(pa), pre.act(pb)
    return safe


@torch.no_grad()
def _mlp_prefix_jac(pre, x, frontier):
    """EXACT Jacobian d phi/dx (Din_frontier x idim) of the piecewise-linear
    prefix (layers < frontier) at the single point x, WITHOUT autograd: the
    product of the layer matrices and the activation masks. Needed so the
    refiner can run in forked CPU workers (autograd after fork is unsafe once
    the parent has run CUDA backward)."""
    slope = float(getattr(pre.act, "negative_slope", 0.0))       # relu -> 0
    a = x.reshape(1, -1)
    J = None
    for i in range(frontier):
        W = pre.layers[i].weight
        z = pre.layers[i](a)
        D = torch.where(z[0] >= 0, torch.ones_like(z[0]), torch.full_like(z[0], slope))
        J = D[:, None] * W if J is None else D[:, None] * (W @ J)
        a = pre.act(z)
    if J is None:
        J = torch.eye(x.numel(), device=x.device, dtype=x.dtype)
    return J


@torch.no_grad()
def _mlp_apply_align_(member, ref_norm, upto, opt=None, live_masks=None):
    """Permute hidden layers 0..upto of a RAW MLP member in place into the frame of
    `ref_norm` (a scale-normalized reference): the permutation is computed on a
    scale-normalized clone (match_layer_), then applied to the raw member with
    permute_layer_ (function-preserving; the next layer's columns follow). Adam
    state of touched parameters is zeroed; live (pin) masks follow."""
    c = member.clone(); scale_normalize_(c)
    for l in range(upto + 1):
        perm = match_layer_(ref_norm, c, l)
        pt = torch.as_tensor(perm, device=member.layers[l].weight.device)
        if bool((pt == torch.arange(len(pt), device=pt.device)).all()):
            continue
        permute_layer_(member, l, perm)
        if opt is not None:
            for p in (member.layers[l].weight, member.layers[l].bias, member.layers[l + 1].weight):
                st = opt.state.get(p)
                if st:
                    for k_ in ("exp_avg", "exp_avg_sq"):
                        if k_ in st:
                            st[k_].zero_()
        if live_masks is not None and (id(member), l) in live_masks:
            lm = live_masks[(id(member), l)]
            lm.copy_(lm[pt])


def _mlp_refine_layer(teacher, cons, frontier, device, act, only_channels=None,
                      angle_gate=12.0, bases=None, light_only=False, xspace=False,
                      loc=False, deep_tries=None, design=False):
    # deep_tries: cap on the deep multi-region path's base points per channel
    # (default max(40, Din)); kink_solve passes a small cap so a hard channel
    # abstains in ~1s and goes to its own fallback instead of burning 15s here.
    # angle_gate: MAX angle (deg) between the swept-recovered normal and the GUESS
    # to ACCEPT. This is NOT a precision bound -- the sweep recovers the exact
    # normal regardless of guess angle; it only guards against find_kinks locking
    # a NEIGHBOUR's kink. Given typical inter-neuron separation (>>10deg), a loose
    # gate (12deg) plus the jump_ratio (rank-1 cleanliness) and runaway (0.3)
    # checks accepts correct recoveries from imperfect guesses that a tight 2deg
    # gate would wrongly reject. (Measured: at 3.3deg guesses, 22.8x separation,
    # the 2deg gate accepted 0/48 CORRECT machine-precision recoveries.)
    """MLP counterpart of _cnn_refine_layer: exactly refine layer `frontier`'s
    neurons with the validated kink refiner (verify_layer1), exact-or-abstain,
    for ANY hidden layer. `cons` carries the exact earlier layers (frozen +
    shared) as the prefix and the frontier guess per neuron. float64, leaky/relu
    only. Returns (W_ref, b_ref, refined_mask)."""
    import copy
    from verify_layer1 import _Oracle, refine_neuron
    if act not in ("relu", "leaky_relu"):
        return None, None, torch.zeros(cons.layers[frontier].weight.shape[0],
                                       dtype=torch.bool, device=device), 0
    # Designed sampling includes the first layer: its accuracy sets the floor
    # for every later recovered prefix. Legacy --loc-refine remains deep-only.
    if design or (loc and frontier > 0):
        import kink_solve
        _g = torch.Generator(device=device).manual_seed(1234 + frontier)
        kwargs = {}
        if design:
            din = cons.layers[frontier].weight.shape[1]
            kwargs = dict(need=max(400, 2*din), pool=max(768, 2*din),
                          failure_dir="peel/design_failures")
            print(f"  [design-refine] L{frontier + 1}: information-directed kink sampling",
                  flush=True)
        return kink_solve.recover_layer(teacher, cons, frontier, device,
                                        only_channels=only_channels, gen=_g,
                                        angle_gate=angle_gate,
                                        sampling="design" if design else "track", **kwargs)
    pre = copy.deepcopy(cons).double().to(device).eval()
    orc = _Oracle(copy.deepcopy(teacher).double().to(device).eval())
    idim = pre.layers[0].weight.shape[1]
    Wl = pre.layers[frontier].weight                 # (out, in_frontier)
    bl = pre.layers[frontier].bias
    gw = Wl.detach().clone()
    Cout, Din = gw.shape
    W_ref = Wl.detach().clone(); b_ref = bl.detach().clone()

    # EFFECTIVE oracle + bases. frontier 0: refine in INPUT space (teacher, real
    # inputs). frontier>0: the prefix (layers < frontier) is EXACT here, so SEAL
    # the teacher at the frontier's input -- x_of_h(h) inverts the earlier layers
    # so BBh(h)=teacher(x_of_h(h)) behaves like a layer-1 net whose input is h --
    # and refine in h-space with the IDENTICAL fast method. This makes L2/L3...
    # as fast as L1 (~0.2s/neuron) instead of the old deep multi-region path.
    # leaky_relu is fully invertible; relu uses the active-region convention.
    _ns = 0.01
    def _act_inv(a):
        return a if act == "relu" else torch.where(a >= 0, a, a / _ns)
    if frontier == 0:
        ref_orc = orc
        ref_bases = bases.to(device).double() if bases is not None else None
    else:
        _pinv = [pre.layers[i].weight.t() @ torch.linalg.inv(
                     pre.layers[i].weight @ pre.layers[i].weight.t())
                 for i in range(frontier)]                 # W_i^+ : (d_i, d_{i+1})

        def _x_of_h(h):
            a = h
            for i in range(frontier - 1, -1, -1):
                a = (_act_inv(a) - pre.layers[i].bias) @ _pinv[i].t()
            return a

        class _SealOrc:
            def __init__(s): s.n = 0
            @torch.no_grad()
            def __call__(s, H): s.n += int(H.shape[0]); return orc(_x_of_h(H))
        ref_orc = _SealOrc()
        ref_bases = (_mlp_prefix_input(pre, bases.to(device).double(), frontier)
                     if bases is not None else None)

    def phi_vec(z):
        return _mlp_prefix_input(pre, z.unsqueeze(0), frontier).squeeze(0)

    def _solve(wg, bg, cidx=0):
        gen = torch.Generator(device=device).manual_seed(0)

        def _accept(out):
            if (out is None or out.get("angle_deg") is None
                    or out["angle_deg"] > angle_gate
                    or out.get("jump_ratio", 0.0) < 1e6):
                return None
            w = out["w_refined"].to(device).double()
            b = torch.as_tensor(out["b_refined"], device=device,
                                dtype=torch.float64)
            wb = torch.cat([w, b.reshape(1)])
            wb = wb / wb.norm().clamp_min(1e-12)
            w, b = wb[:-1], wb[-1]
            if (w * wg).sum() + b * bg < 0:
                w, b = -w, -b
            sg = torch.cat([wg, bg.reshape(1)]).norm().clamp_min(1e-12)
            if max((w - wg / sg).abs().max().item(),
                   (b - bg / sg).abs().item()) > 0.3:
                return None
            return w, b

        if frontier != 0 and not xspace:
            # SEALED light path: the prefix is EXACT here, so refine in h-space
            # with real h-bases + plain refine_neuron on the sealed oracle --
            # h-space is just another input space, so L2/L3... refine as fast as
            # L1 (~0.2s/neuron) instead of the old deep multi-region path.
            # (--xspace-refine SKIPS this to force the depth-flat Jacobian path.)
            if ref_bases is not None and len(ref_bases) > 0:
                for trial in range(5):
                    base = ref_bases[(cidx + 997 * trial) % len(ref_bases)].to(device).double()
                    res = _accept(refine_neuron(ref_orc, wg.detach(), bg.detach(),
                                                base, gen=gen,
                                                triage_deg=angle_gate + 5.0))
                    if res is not None:
                        return res
            if light_only:
                return None          # --partial: fast, no robust fallback
            # else fall through to the DEEP multi-region path below (robust: fits
            # the true normal from several regions, accepts a looser drift, so it
            # catches the 12-27deg stragglers the single-base sealed pass misses).

        if frontier == 0:
            # LIGHT (the refine_v18min recipe): REAL-INPUT bases + PLAIN
            # refine_neuron (n_cand=3 defaults). A within-eps guess (peel fires
            # at L1 eps<1e-2) solves on the first clean base -> exact normal to
            # ~1e-9 at ~180ms/neuron. Real inputs land the base in the neuron's
            # active region, so no multi-config discovery search is needed;
            # tiny random bases (the heavy path below) sit in crowded/degenerate
            # regions, which is the ONLY reason that search existed. Measured:
            # 64/64 at a 6.1e-3 guess, plain, vs 0/64 from a zeros base.
            if bases is not None and len(bases) > 0:
                nb = len(bases)
                for trial in range(5):
                    base = bases[(cidx + 997 * trial) % nb].to(device).double()
                    res = _accept(refine_neuron(orc, wg.detach(), bg.detach(),
                                                base, gen=gen,
                                                triage_deg=angle_gate + 5.0))
                    if res is not None:
                        return res
            if light_only:                    # --partial: skip the heavy search
                return None                   # (grab easy neurons fast, defer hard)
            # INPUT-SPACE layer: the neuron's kink plane is wg.x+bg=0 directly,
            # so refine_neuron returns the EXACT input normal + offset in one
            # O(d) call -- no Newton loop, no per-kink Jacobian (which at
            # frontier 0 is the identity, built via d backward passes = the
            # 512-neuron freeze). A good guess solves on the FIRST base; a bad
            # guess (still >2deg from truth) should fail fast, not burn retries.
            # MULTI-BASE + ADAPTIVE-WINDOW + SKETCH localization. A single narrow
            # base misses the true kink: at a FAR base the offset t*=a(dw.x+db)
            # exceeds the window; at a NEAR base many neighbour kinks crowd the
            # foot. So try varied (base_scale, window) -- the kink lands in range
            # at SOME config -- and let the sketch (triage_deg = _angle_estimate,
            # the cos(s,s_hat) normal fingerprint) pick THIS neuron's kink out of
            # the neighbour crowd; jump_ratio + runaway confirm. Recovers the
            # exact normal regardless, once the right kink is localized. The guess
            # is a POINTER (identification), not a value being refined. Measured:
            # 387/512 vs ~160 for the single narrow base, on identical guesses.
            # CHEAP-FIRST order: narrow window solves ~63% on config 1 and exits
            # immediately; only the crowded/hard neurons escalate to wide windows
            # (same 5 configs -> identical coverage, far less average work).
            # HEAVY fallback: tiny random bases + multi-config sketch. Only runs
            # when the light real-base pass fails (no bases / crowded kink).
            for bscale, win in ((0.02, 0.15), (0.02, 0.5), (0.1, 1.0),
                                (0.3, 1.5), (0.5, 2.0)):
                base = torch.randn(idim, generator=gen, device=device,
                                   dtype=torch.float64) * bscale
                res = _accept(refine_neuron(orc, wg.detach(), bg.detach(), base,
                                            window=win, n_scan=81, n_cand=15,
                                            gen=gen, triage_deg=angle_gate + 5.0))
                if res is not None:
                    return res
            return None
        Cblocks, hs = [], []
        for _ in range(deep_tries or max(40, Din)):
            base = torch.randn(idim, generator=gen, device=device,
                               dtype=torch.float64) * 0.5
            xcur, uh, nu, b_img = base, None, None, None
            for _ in range(6):                        # iterated Newton onto guess plane
                xg = xcur.detach().clone()
                s0 = (phi_vec(xg) * wg).sum() + bg
                u = _mlp_prefix_jac(pre, xg, frontier).t() @ wg; nu = u.norm()   # = grad, no autograd
                if nu < 1e-12:
                    break
                uh = (u / nu).detach()
                b_img = (s0 - u @ xg).reshape(()).detach()
                if abs(float(s0)) < 1e-9:
                    break
                xcur = (xg - (s0 / nu ** 2) * u).detach()
            if uh is None or nu < 1e-12:
                continue
            with torch.no_grad():
                x0 = xcur
                sr = _mlp_safe_r(pre, x0, uh, frontier)
                window = min(0.9 * sr, 1.0)
                if window < 1e-4:
                    continue
                out = refine_neuron(orc, (uh * nu).detach(), b_img, x0,
                                    window=window, n_scan=81, n_cand=15, gen=gen,
                                    triage_deg=angle_gate + 5.0)
                if (out is None or out.get("angle_deg") is None
                        or out["angle_deg"] > angle_gate):
                    continue
                if out.get("jump_ratio", 0.0) < 1e6:
                    continue
                xs = x0 + out["offset"] * uh
                n_k = out["w_refined"].to(device).double()
                h_k = phi_vec(xs)
            A_k = _mlp_prefix_jac(pre, xs.detach(), frontier)          # exact, no autograd
            # FAST single-kink solve. In-region phi is EXACTLY affine, so the kink
            # normal is exactly n_k = A_k^T w. Whenever A_k has full column rank the
            # weight is determined by ONE clean kink: w = (A_k^T)^+ n_k -- no multi-
            # region null-space needed. leaky_relu is ALWAYS full rank (no dead
            # units); relu takes this too when the landed region activates all Din
            # prefix units (a dead region is rank-deficient -> fall through below).
            with torch.no_grad():
                sA = torch.linalg.svdvals(A_k)                 # A_k: (Din, idim)
                if A_k.shape[1] >= A_k.shape[0] and sA[-1] > 1e-9 * sA[0]:
                    w1 = (torch.linalg.pinv(A_k.t()) @ n_k.reshape(-1, 1)).reshape(-1)
                    resid = ((A_k.t() @ w1 - n_k).norm()
                             / n_k.norm().clamp_min(1e-30))
                    if resid < 1e-6:                            # n_k truly in range(A_k^T)
                        b1 = -(w1 @ h_k)                        # kink: w.h + b = 0
                        wb1 = torch.cat([w1, b1.reshape(1)])
                        wb1 = wb1 / wb1.norm().clamp_min(1e-12)
                        w1, b1 = wb1[:-1], wb1[-1]
                        if (w1 * wg).sum() + b1 * bg < 0:
                            w1, b1 = -w1, -b1
                        sg = torch.cat([wg, bg.reshape(1)]).norm().clamp_min(1e-12)
                        if max((w1 - wg / sg).abs().max().item(),
                               (b1 - bg / sg).abs().item()) <= 0.3:
                            return w1, b1                       # exact from 1 kink
                Cblocks.append(A_k.t() - torch.outer(n_k, A_k @ n_k))
                hs.append(h_k)
            if len(Cblocks) >= 3:
                S = torch.linalg.svdvals(torch.cat(Cblocks, 0))
                if S[-1] < 1e-5 * S[-2]:
                    break
        if len(Cblocks) < 3:
            return None
        C = torch.cat(Cblocks, 0)
        _, S, Vh = torch.linalg.svd(C, full_matrices=False)
        tol = max(1e-9, 1e-4 * S[0].item())
        null_dim = int((S < tol).sum().item())
        if null_dim == 0 or null_dim > Din // 2:
            return None
        N = Vh[-null_dim:]
        W = (N @ wg) @ N
        W = W / W.norm().clamp_min(1e-12)
        b = -torch.stack([W @ h for h in hs]).mean()
        wb = torch.cat([W, b.reshape(1)])
        wb = wb / wb.norm().clamp_min(1e-12)
        w, b = wb[:-1], wb[-1]
        if (w * wg).sum() + b * bg < 0:
            w, b = -w, -b
        sg = torch.cat([wg, bg.reshape(1)]).norm().clamp_min(1e-12)
        if max((w - wg / sg).abs().max().item(),
               (b - bg / sg).abs().item()) > 0.3:
            return None
        return w, b

    refined = torch.zeros(Cout, dtype=torch.bool, device=device)
    for c in (range(Cout) if only_channels is None else list(only_channels)):
        res = _solve(gw[c], bl[c].reshape(()), c)
        if res is None and not light_only:            # retry from the NEGATED guess
            res = _solve(-gw[c], (-bl[c]).reshape(()), c)
        if res is None:
            continue
        w, b = res
        W_ref[c] = w.to(W_ref.dtype); b_ref[c] = b.to(b_ref.dtype)
        refined[c] = True
    return (W_ref.to(cons.layers[frontier].weight.dtype),
            b_ref.to(cons.layers[frontier].bias.dtype), refined, orc.n)


def _mlp_warm_reinit_population(pop, dims, act, device, frozen,
                                reroll_frontier=None):
    """WARM analog of _reinit_frozen_population: keep every member's current
    trained weights (deeper layers + unsolved frontier rows) across a freeze,
    re-pinning only the frozen rows. reroll_frontier=(l, unsolved_mask):
    additionally re-randomize just those frontier rows (fresh signs, flip
    escape) while everything else stays warm."""
    new_pop = []
    for src in pop:
        m = MLP(dims, act=act).to(device)
        m.load_state_dict(src.state_dict())
        if reroll_frontier is not None:
            l, um = reroll_frontier
            idx = um.to(device).nonzero(as_tuple=True)[0]
            if len(idx):
                fresh = MLP(dims, act=act).to(device)
                with torch.no_grad():
                    m.layers[l].weight[idx] = fresh.layers[l].weight[idx]
                    m.layers[l].bias[idx] = fresh.layers[l].bias[idx]
        _install_freeze(m, frozen)
        new_pop.append(m)
    return new_pop


def _cnn_frontier_input(ref, x, frontier):
    """Flattened INPUT to layers[frontier], replicating ConvNet.forward (act+pool)
    up to but NOT applying layer `frontier`. `ref` supplies the frozen (recovered)
    prefix; differentiable in x. Attacker-side only -- no teacher."""
    n_conv = ref.n_conv
    h = x.view(x.shape[0], *ref.input_shape)
    for i in range(n_conv):
        if i == frontier:
            return h.reshape(h.shape[0], -1)
        h = ref.act(ref.layers[i](h))
        if ref.pools[i] > 0:
            h = F.avg_pool2d(h, ref.pools[i])
    h = torch.flatten(h, 1)
    for j in range(n_conv, len(ref.layers)):
        if j == frontier:
            return h
        h = ref.act(ref.layers[j](h))
    return h


@torch.no_grad()
def _cnn_solve_last_layer_(net, X, Y, ridge=1e-6, chunk=8192):
    """CNN analog of solve_last_layer_: closed-form ridge LSQ of the linear HEAD from
    the EXACT penultimate features (every conv+fc hidden layer frozen). Forwards flat
    queries through ConvNet's prefix via _cnn_frontier_input (head = layers[-1])."""
    dev = net.layers[-1].weight.device
    head = len(net.layers) - 1
    Hs = []
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk].to(dev)
        Hs.append(_cnn_frontier_input(net, xb, head).detach().cpu())
    H = torch.cat(Hs).double()
    Yd = Y.detach().cpu().double()
    Ha = torch.cat([H, torch.ones(len(H), 1, dtype=torch.float64)], dim=1)
    A = Ha.T @ Ha
    A = A + ridge * A.diag().mean().clamp_min(1e-12) * torch.eye(
        A.shape[0], dtype=torch.float64)
    W = torch.linalg.solve(A, Ha.T @ Yd)
    last = net.layers[-1]
    last.weight.copy_(W[:-1].T.float().to(dev))
    last.bias.copy_(W[-1].float().to(dev))


def _cnn_shape_at(ref, frontier):
    """(C, H, W) input shape of conv layer `frontier` (must be < n_conv)."""
    cur = ref.input_shape
    for i, (ic, oc, k, s, pad, pool) in enumerate(ref.conv_cfgs):
        if i == frontier:
            return cur
        cur = ConvNet._out_shape(cur, oc, k, s, pad)
        if pool > 0:
            cur = (cur[0], (cur[1] - pool) // pool + 1, (cur[2] - pool) // pool + 1)
    return cur


def _cnn_tail_forward(ref, h, frontier):
    """Forward from layer `frontier` (inclusive) to the logits, taking the
    FLATTENED frontier input `h` (as produced by _cnn_frontier_input)."""
    if frontier < ref.n_conv:
        C, H, W = _cnn_shape_at(ref, frontier)
        x = h.view(-1, C, H, W)
        for i in range(frontier, ref.n_conv):
            x = ref.act(ref.layers[i](x))
            if ref.pools[i] > 0:
                x = F.avg_pool2d(x, ref.pools[i])
        x = torch.flatten(x, 1)
    else:
        x = h
    for j in range(max(frontier, ref.n_conv), len(ref.layers) - 1):
        x = ref.act(ref.layers[j](x))
    return ref.layers[-1](x)


def _cnn_layer_mine(pop, teacher, frontier, cfg, device, gen, invert_steps=200):
    """Layer-space disagreement mining: optimize the disagreement objective
    DIRECTLY in the frontier layer's input space (tails of the members, plus
    the blackbox tail under cheat -- the frozen prefix equals the teacher's, so
    the teacher's tail on this space is exactly its behavior on induced
    images), then INVERT the optimized activations through the exact frozen
    prefix by least squares in image space. Rationale: from image space, the
    disagreement gradient along a frontier direction is scaled by the prefix's
    gain in that direction -- weakly-excited directions are invisible to the
    image-space miner no matter how large their tail disagreement. Mining at
    the frontier sees them at full strength; the inversion then pays the
    reachability cost once (achieved activations = projection onto the
    reachable set). Returns (q, in_dim) images, used directly as the query
    batch."""
    ref = pop[0]                                   # frozen prefix is shared
    in_dim = int(ref.input_shape[0] * ref.input_shape[1] * ref.input_shape[2])
    if cfg.qg_init == "uniform":
        X0 = (torch.rand(cfg.q, in_dim, generator=gen, device=device)
              * 2 - 1) * cfg.qg_range
    else:
        X0 = torch.randn(cfg.q, in_dim, generator=gen,
                         device=device) * cfg.qg_init_std
    with torch.no_grad():
        H0 = _cnn_frontier_input(ref, X0, frontier)   # on-manifold init
    tails = [lambda h, m=m: _cnn_tail_forward(m, h, frontier) for m in pop]
    if teacher is not None:
        tails.append(lambda h: _cnn_tail_forward(teacher, h, frontier))
    if len(tails) < 2:
        return X0                                   # no pair to disagree
    # 1) disagreement ascent in frontier-input space (full-strength gradients)
    Hl = H0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([Hl], lr=cfg.qg_lr)
    sched = {int(s * cfg.qg_steps) for s in cfg.qg_sched}
    for step in range(cfg.qg_steps):
        if step in sched:
            for grp in opt.param_groups:
                grp["lr"] /= 10
        opt.zero_grad()
        outs = torch.stack([f(Hl) for f in tails])
        disagreement(outs, cfg.disagree, cfg.qg_dist).backward()
        opt.step()
    Ht = Hl.detach()
    # 2) invert through the exact prefix: images inducing (the reachable
    #    projection of) the mined activations
    X = X0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([X], lr=0.05)
    for step in range(invert_steps):
        if step in (invert_steps // 2, int(invert_steps * 0.8)):
            for grp in opt.param_groups:
                grp["lr"] /= 10
        opt.zero_grad()
        ((_cnn_frontier_input(ref, X, frontier) - Ht) ** 2).mean().backward()
        opt.step()
    return X.detach()


def _cnn_layer_rand_init(pop, frontier, cfg, device, gen, probe_n=2048):
    """Random-LAYER-space init for the disagreement-query optimizer: draw
    isotropic random targets in the frontier layer's INPUT space (scale matched
    to the pushforward of random images) and approximately invert them through
    the shared frozen prefix by least squares in image space. Unreachable
    target components project out; the returned images' frontier-input
    distribution is spread over the reachable set instead of the collapsed
    pushforward of image-space noise. Attacker-side only (frozen prefix, no
    teacher). Returns (q, in_dim) images to use as gen_queries init_override."""
    ref = pop[0]                                   # frozen prefix is shared
    in_dim = int(ref.input_shape[0] * ref.input_shape[1] * ref.input_shape[2])

    def rand_imgs(n):
        if cfg.qg_init == "uniform":
            return (torch.rand(n, in_dim, generator=gen, device=device)
                    * 2 - 1) * cfg.qg_range
        return torch.randn(n, in_dim, generator=gen,
                           device=device) * cfg.qg_init_std

    with torch.no_grad():
        hp = _cnn_frontier_input(ref, rand_imgs(probe_n), frontier)
        mu = hp.mean(0)
        sd = hp.std().clamp_min(1e-6)              # scalar -> isotropic target
    tgt = mu + sd * torch.randn(cfg.q, mu.shape[0], generator=gen, device=device)
    x = rand_imgs(cfg.q).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=cfg.qg_lr)
    for _ in range(cfg.qg_steps):
        opt.zero_grad()
        ((_cnn_frontier_input(ref, x, frontier) - tgt) ** 2).mean().backward()
        opt.step()
    return x.detach()


def _spatial_boundary_normals(ref, pop, frontier, input_shape, D, Cout, q, P,
                              device, gen):
    """Per-query input-space kink normals for a SPATIAL conv frontier. Each query is
    assigned a (member, out-channel, output-position) triple, round-robin so every
    channel is covered evenly; its normal is that member's k x k filter for the
    channel, scattered into the position's receptive field of the flat frontier input
    (zeros elsewhere), and its bias is the channel bias. So <normal, prefix(x)> + bias
    is exactly that channel's preactivation at that position -- driving it to 0 lands
    the query on the channel's kink. Returns (Wsel (q, D), bsel (q,)); (None, None) if
    the map has no fully-interior output position. Attacker-side only (uses only the
    committee's own recovered weights/prefix)."""
    Cin, _, k, s, pad, _ = ref.conv_cfgs[frontier]
    # frontier input spatial size, replicating ConvNet.forward through the prefix
    C, H, W = input_shape
    for i in range(frontier):
        _, _, ki, si, padi, pooli = ref.conv_cfgs[i]
        H = (H - ki + 2 * padi) // si + 1; W = (W - ki + 2 * padi) // si + 1
        if pooli > 0:
            H = (H - pooli) // pooli + 1; W = (W - pooli) // pooli + 1
    Hin, Win = H, W
    if Cin * Hin * Win != D:                                   # geometry sanity
        return None, None
    Hout = (Hin - k + 2 * pad) // s + 1
    Wout = (Win - k + 2 * pad) // s + 1
    # fully-interior output positions only (receptive field entirely inside the map,
    # no padding zeros in the filter support -> a clean, complete input hyperplane)
    r0s = [o * s - pad for o in range(Hout) if 0 <= o * s - pad <= Hin - k]
    c0s = [o * s - pad for o in range(Wout) if 0 <= o * s - pad <= Win - k]
    if not r0s or not c0s:
        return None, None
    r0g = torch.tensor(r0s, device=device).repeat_interleave(len(c0s))   # (npos,)
    c0g = torch.tensor(c0s, device=device).repeat(len(r0s))              # (npos,)
    npos = r0g.numel()

    idx = torch.arange(q, device=device)
    chan = idx % Cout                                         # even per-channel coverage
    member = (idx // Cout) % P
    pos = (idx // (Cout * P)) % npos                          # spread over positions
    r0 = r0g[pos]; c0 = c0g[pos]                              # (q,) receptive-field origin

    Wall = torch.stack([m.layers[frontier].weight for m in pop])   # (P,Cout,Cin,k,k)
    ball = torch.stack([m.layers[frontier].bias for m in pop])     # (P,Cout)
    filt = Wall[member, chan]                                 # (q,Cin,k,k) target filter
    bsel = ball[member, chan]                                 # (q,)

    ar = torch.arange(k, device=device)
    row = r0[:, None, None] + ar[None, :, None]              # (q,k,1) input rows
    col = c0[:, None, None] + ar[None, None, :]              # (q,1,k) input cols
    sp = (row * Win + col).reshape(q, k * k)                 # (q,k*k) flat spatial idx
    normal = torch.zeros(q, Cin, Hin * Win, device=device, dtype=Wall.dtype)
    normal.scatter_(2, sp[:, None, :].expand(q, Cin, k * k),
                    filt.reshape(q, Cin, k * k))             # place filter, zeros else
    return normal.reshape(q, D), bsel


def _cnn_frontier_boundary_queries(pop, frontier, input_shape, cfg, device, gen,
                                   invert_steps=40, invert_lr=0.1):
    """Layer-space disagreement SEED (honest: attacker-side only). Spread query inputs
    so EVERY frontier neuron gets boundary coverage, instead of image-space gaussian
    which -- through the fixed prefix -- over-probes loud neurons and starves the rest.

    (1) reachable frontier input  h0 = prefix(x0);
    (2) project h0 onto each (member, neuron) kink hyperplane in the frontier's INPUT
        space, round-robin over ALL neurons -> even per-neuron coverage;
    (3) invert through the EXACT frozen prefix to realize those targets as raw inputs.

    Uses only the committee's own recovered prefix + weights; the teacher is never
    queried here (the seeds are labelled by the teacher afterwards, as usual)."""
    in_dim = int(input_shape[0] * input_shape[1] * input_shape[2])
    q, P, ref = cfg.q, len(pop), pop[0]
    x0 = torch.randn(q, in_dim, generator=gen, device=device) * cfg.qg_init_std
    n_invert, n_lr = invert_steps, invert_lr
    with torch.no_grad():                                      # build the target (detached)
        h0 = _cnn_frontier_input(ref, x0, frontier)           # (q, D) reachable
        D = h0.shape[1]
        Wf = ref.layers[frontier].weight
        Cout = Wf.shape[0]
        if Wf[0].numel() != D:        # spatial-conv frontier: no single input hyperplane
            if not cfg.spatial_boundary_seed:
                return None           # legacy: signal caller to use normal gen_queries
            # SPATIAL frontier: a channel c has a kink at EVERY output position p; the
            # (c, p) preactivation IS linear in the flat input (its filter scattered into
            # p's receptive field). Round-robin over ALL channels (and members/positions)
            # so every channel gets even boundary coverage, then invert through the exact
            # prefix exactly as the dense branch does.
            Wsel, bsel = _spatial_boundary_normals(
                ref, pop, frontier, input_shape, D, Cout, q, P, device, gen)
            if Wsel is None:          # no fully-interior output position (tiny map)
                return None
            # the sparse (150/1176-support) normal inverts slower than the dense one;
            # give it a bigger budget so queries land tightly on the kink (early-stops).
            n_invert, n_lr = max(invert_steps, 250), max(invert_lr, 0.2)
        else:
            idx = torch.arange(q, device=device)
            neuron = idx % Cout           # round-robin over ALL neurons -> even coverage
            member = (idx // Cout) % P
            Wall = torch.stack([m.layers[frontier].weight.reshape(Cout, -1) for m in pop])  # (P,Cout,D)
            ball = torch.stack([m.layers[frontier].bias for m in pop])                      # (P,Cout)
            Wsel = Wall[member, neuron]                            # (q, D) each query's target normal
            bsel = ball[member, neuron]                            # (q,)
    # drive each query's TARGETED neuron preactivation -> 0 through the exact prefix.
    # prefix(x) is always on the reachable (post-ReLU) manifold, so this lands a real
    # input on that neuron's kink -- no off-manifold projection to chase.
    x = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=n_lr)
    for _ in range(n_invert):
        opt.zero_grad()
        h = _cnn_frontier_input(ref, x, frontier)              # (q, D) reachable, differentiable
        preact = (h * Wsel).sum(1) + bsel                      # targeted neuron's preact
        mse = (preact ** 2).mean()
        mse.backward()
        opt.step()
        if mse.item() < 1e-12:                                  # already on the kink -> stop
            break
    return x.detach()


def reconstruct_cnn(teacher, input_shape, conv_cfgs, out_dim, cfg, device,
                    seed=0, fc_dims=(), resume_state=None):
    """CNN counterpart of reconstruct(): population + disagreement queries (reused
    gen_queries) + channel-aligned consensus. Supports the freeze-reinit PEEL
    (cfg.freeze_reinit): once the frontier hidden layer reaches consensus, pin it
    (+ everything above) across a reinitialized committee and keep going on the
    deeper layers. No rank-cert/fast/combine (those are MLP-only). A conv-layer
    analytic refiner (the FC verify_layer1 refiner does NOT apply to conv) would
    slot in just before freezing; for now we freeze the consensus average, gated by
    cfg.freeze_precision. Queries are flat (in_dim); ConvNet reshapes internally."""
    global _CNN_KINK
    _CNN_KINK = bool(getattr(cfg, 'loc_refine', False))

    if cfg.hard and (cfg.freeze_reinit or cfg.peel_try or resume_state is not None):
        raise ValueError("cfg.hard: the CNN peel/peel-try/resume path refines "
                         "layers with the kink prober (real-valued teacher "
                         "outputs); incompatible with the hard-label regime")
    loss_on = xent_on if cfg.hard else l1_on
    gen = torch.Generator(device=device).manual_seed(seed)
    in_dim = int(input_shape[0] * input_shape[1] * input_shape[2])
    pop = [ConvNet(input_shape, conv_cfgs, fc_dims, out_dim, cfg.act).to(device)
           for _ in range(cfg.p)]
    opts = [torch.optim.Adam(n.parameters(), lr=cfg.lr) for n in pop]
    X = torch.empty(0, in_dim)
    Y = (torch.empty(0, dtype=torch.long) if cfg.hard
         else torch.empty(0, out_dim))
    decay_at = {int(s * cfg.outer) for s in cfg.lr_sched}
    log = []
    t0 = time.time()
    best = None
    frozen = {}                       # freeze-reinit peel: {layer: (W, b, mask)}
    exact_net = None                  # persistent source-of-truth for refined layers
    exact_mask = {}                   # {layer: bool[Cout]} which channels are kink-EXACT
    out_solved = False                # cheat-peel: linear head LSQ-solved once
    partial_exact = None              # --partial: persistent guess-scale exact channels
    partial_mask = {}                 # --partial: {layer: bool[Cout]} partial-frozen
    partial_hooked = set()            # --partial: {(id(member), layer)} hook installed
    partial_live = {}                 # --partial: {(id(member), layer): live bool mask}
    stuck_pf = None; stuck_last_count = 0; stuck_last_growth = 0; stuck_restarts = 0
    if cfg.cheat:
        # cheat: the blackbox joins gen_queries as a frozen extra member (the
        # teacher already takes the same flat queries the students do).
        for prm in teacher.parameters():
            prm.requires_grad_(False)

    if resume_state is not None:
        # RESUME: reload a saved student, re-establish the exactly-solved layers by
        # re-running the kink refiner shallow->deep (exact-or-abstain, no oracle),
        # freeze the contiguous solved prefix, and spin up a FRESH committee onto the
        # unsolved layers. New queries follow naturally (each outer iter samples fresh).
        import copy as _copy
        rnet = ConvNet(input_shape, conv_cfgs, fc_dims, out_dim, cfg.act).to(device)
        rnet.load_state_dict(resume_state)
        cnn_canonicalize_(rnet)                       # into the canonical working frame
        exact_net = _copy.deepcopy(rnet).to(device)
        nlay = len(rnet.layers)
        kmax = -1
        for l in range(nlay - 1):                     # hidden layers only (exclude head)
            Coutl = exact_net.layers[l].weight.shape[0]
            exact_mask[l] = torch.zeros(Coutl, dtype=torch.bool, device=device)
            wdt = exact_net.layers[l].weight.dtype

            def _apply(rmask, Wr, br):                 # write solved channels into exact_net
                if Wr is None:
                    return
                idx = rmask.to(device).nonzero(as_tuple=True)[0]
                if len(idx):
                    _scale_match_rows(exact_net.layers[l], idx, Wr, br)  # guess magnitude
                    exact_mask[l][idx] = True

            # cheap probe first: if the shallow prefix has run out (this is the
            # frontier), the first few channels abstain and we skip the rest.
            probe = list(range(min(8, Coutl)))
            Wr, br, rmask = _cnn_refine_layer(
                teacher, exact_net, l, input_shape, device, cfg.act, only_channels=probe)
            _apply(rmask, Wr, br)
            probe_frac = (exact_mask[l][:len(probe)].float().mean().item())
            # refine the rest UNLESS the probe is dead (this is the frontier); 0.5
            # cleanly separates a solved/partial layer (~all solve) from the frontier (~0).
            if probe_frac >= 0.5 and Coutl > len(probe):
                rest = list(range(len(probe), Coutl))
                Wr, br, rmask = _cnn_refine_layer(
                    teacher, exact_net, l, input_shape, device, cfg.act, only_channels=rest)
                _apply(rmask, Wr, br)
            frac = exact_mask[l].float().mean().item()
            print(f"  [resume] L{l+1}: re-solved {int(exact_mask[l].sum())}/{Coutl}"
                  f" ({frac:.0%})", flush=True)
            if frac >= cfg.freeze_thresh:             # contiguous solved prefix -> freeze
                kmax = l
            else:
                break                                 # first under-threshold layer = frontier
        frozen = {l: (exact_net.layers[l].weight.detach().clone(),
                      exact_net.layers[l].bias.detach().clone(),
                      exact_mask[l].clone())
                  for l in range(kmax + 1)}
        if frozen:
            pop = _cnn_reinit_frozen_population(
                input_shape, conv_cfgs, fc_dims, out_dim, cfg.act, device, cfg.p, frozen)
            opts = [torch.optim.Adam(
                [p for p in n.parameters() if p.requires_grad], lr=cfg.lr) for n in pop]
            print(f"  [resume] froze L1..L{kmax+1} (solved channels only); fresh "
                  f"committee on L{kmax+2}.. — starting with new samples", flush=True)
        else:
            print("  [resume] no layer met freeze-thresh; starting fresh", flush=True)

    t = 0
    qtot = 0                          # cumulative queries (t restarts under peel_refresh)
    refresh_now = False
    peel_try_n = 0                    # peel-try attempt counter (rotates start channel)
    peel_stall = 0                    # consecutive no-progress peel-try attempts
    peel_stuck_best = {}              # {frontier: best solved count seen}
    stuck_restarts = 0                # restart-stuck: fired count (adds budget each)
    retries_done = 0                  # --retry: retries fired so far
    stuck_pf = -1                     # restart-stuck: frontier the clock watches
    stuck_last_count = -1             # restart-stuck: solved count at last growth
    stuck_last_growth = 0             # restart-stuck: iter of last frontier growth
    while t < cfg.outer:
        if t < cfg.warmstart_iters:
            I = torch.randn(cfg.q, in_dim, generator=gen,
                            device=device) * cfg.qg_init_std
        else:
            seed = None
            init = None
            fr = (_cnn_frontier_layer(frozen, len(pop[0].layers) - 1)
                  if frozen else None)
            if cfg.layer_mine and fr is not None and fr > 0:
                # disagreement mined AT the frontier input, inverted to images;
                # used directly (image-space opt would re-squash quiet directions)
                seed = _cnn_layer_mine(pop, teacher if cfg.cheat else None,
                                       fr, cfg, device, gen)
            elif cfg.layer_rand_init and fr is not None and fr > 0:
                # random-LAYER-space init: the disagreement optimizer starts from
                # images whose frontier-input activations are isotropically
                # spread (approximate inversion of random layer targets), then
                # optimizes normally.
                init = _cnn_layer_rand_init(pop, fr, cfg, device, gen)
            elif cfg.layer_space_init and fr is not None and fr > 0:
                seed = _cnn_frontier_boundary_queries(
                    pop, fr, input_shape, cfg, device, gen)
            # boundary seeds (seed) are used DIRECTLY (already on members' kinks,
            # evenly across neurons; re-running the aggregate optimizer would
            # re-concentrate on the loud neurons). layer_rand_init (init) is an
            # INIT: the normal disagreement search runs from it. Neither applies
            # (None) -> normal image-space disagreement search.
            qg_pop = pop + [teacher] if cfg.cheat else pop
            I = seed if seed is not None else gen_queries(
                qg_pop, cfg, in_dim, device, gen, init_override=init,
                ref_last=cfg.cheat and cfg.cheat_bb_ref)
        with torch.no_grad():
            T = teacher(I)
            if cfg.hard:                    # blackbox reveals ONLY the class index
                T = T.argmax(1)
        X = torch.cat([X, I.cpu()]); Y = torch.cat([Y, T.cpu()])
        qtot += cfg.q
        if cfg.window > 0:
            keep = cfg.window * cfg.q
            X, Y = X[-keep:], Y[-keep:]
        if t in decay_at:
            for o in opts:
                for g in o.param_groups:
                    g["lr"] /= 10
        n = len(X)
        for ep in range(cfg.epochs):
            perm = torch.randperm(n, generator=gen, device=device)
            for i in range(0, n, cfg.batch):
                idx = perm[i:i + cfg.batch].cpu()
                xb, yb = X[idx].to(device), Y[idx].to(device)
                for net, opt in zip(pop, opts):
                    opt.zero_grad()
                    if cfg.hard:
                        loss = F.cross_entropy(net(xb), yb)
                    elif cfg.fit_loss == "mse":
                        loss = ((net(xb) - yb) ** 2).mean()
                    else:
                        loss = (net(xb) - yb).abs().mean()
                    loss.backward()
                    opt.step()
        if (cfg.pop_save_every and cfg.pop_save_path
                and (t + 1) % cfg.pop_save_every == 0):
            # mid-run snapshot of the best member (inspect per-neuron eps offline;
            # atomic overwrite, no queries stored). CNN counterpart of the MLP hook.
            lv = loss_on(pop, X, Y)
            bidx = min(range(cfg.p), key=lambda i: lv[i])
            tmp = cfg.pop_save_path + ".tmp"
            torch.save({"input_shape": input_shape, "conv_cfgs": conv_cfgs,
                        "fc_dims": fc_dims, "out_dim": out_dim, "act": cfg.act,
                        "iter": t + 1,
                        "state_dict": {k: v.detach().cpu()
                                       for k, v in pop[bidx].state_dict().items()}},
                       tmp)
            os.replace(tmp, cfg.pop_save_path)
        if (t + 1) % cfg.log_every == 0 or t == cfg.outer - 1:
            losses = loss_on(pop, X, Y)
            bi = min(range(cfg.p), key=lambda i: losses[i])
            best = pop[bi]
            be = cnn_param_errors(best, teacher, hard=cfg.hard)
            quorum = max(2, int(round(cfg.cluster_quorum * cfg.p)))
            cons, _, masks = _cnn_consensus(pop, bi, cfg.cluster_eps, quorum)
            cst = _cnn_consensus_eps(cons, masks, teacher)   # per-layer, cons units only
            bmean = sum(be["mean_eps_per_matrix"]) / len(be["mean_eps_per_matrix"])
            nlay = len(best.layers)
            hid = range(nlay - 1)                             # hidden layers (exclude head)
            ncons = sum(cst[l]["n_cons"] for l in hid)
            ntot = sum(cst[l]["n_tot"] for l in hid)
            cmax = [cst[l]["max"] for l in range(nlay) if cst[l]["max"] is not None]
            cmean = [cst[l]["mean"] for l in range(nlay) if cst[l]["mean"] is not None]
            agg_max = max(cmax) if cmax else None
            agg_mean = (sum(cmean) / len(cmean)) if cmean else None
            wall = round(time.time() - t0, 1)
            fmt = lambda v: f"{v:.2e}" if v is not None else "n/a"   # noqa: E731
            log.append(dict(iter=t + 1, queries=qtot,
                            best_loss=losses[bi], best_max_eps=be["max_eps"],
                            best_mean_eps=bmean, n_consensus=ncons, n_total=ntot,
                            cons_max_eps=agg_max, cons_mean_eps=agg_mean,
                            cons_layers=[(cst[l]["n_tot"], cst[l]["n_cons"]) for l in hid],
                            wall_s=wall))
            cstr = " ".join(f"L{l + 1}:{cst[l]['n_cons']}/{cst[l]['n_tot']}" for l in hid)
            print(f"  it {t + 1:3d} | q {qtot:7d} | loss {losses[bi]:.3e} "
                  f"| best eps max {be['max_eps']:.2e} mean {bmean:.2e} "
                  f"| cons {ncons}/{ntot} [{cstr}] eps max {fmt(agg_max)} "
                  f"mean {fmt(agg_mean)} | {wall}s", flush=True)
            bM, bm = be["max_eps_per_matrix"], be["mean_eps_per_matrix"]
            print("        eps/layer (best): " + "  ".join(
                f"L{li + 1}[max {max(bM[2 * li], bM[2 * li + 1]):.2e} "
                f"mean {bm[2 * li]:.2e}]" for li in range(nlay)), flush=True)
            print("        eps/layer (cons): " + "  ".join(
                (f"L{li + 1}[max {cst[li]['max']:.2e} mean {cst[li]['mean']:.2e} "
                 f"({cst[li]['n_cons']}/{cst[li]['n_tot']})]" if cst[li]["max"] is not None
                 else f"L{li + 1}[n/a ({cst[li]['n_cons']}/{cst[li]['n_tot']})]")
                for li in range(nlay)), flush=True)

            # --- frontier SAMPLING diagnostic (attacker-side: recovered prefix +
            #     each member's own frontier weights; NO teacher). Reports per-neuron
            #     near-kink coverage of the current queries -> shows starvation. ---
            if frozen:
                fr = _cnn_frontier_layer(frozen, nlay - 1)
                if fr is not None and fr > 0:
                    with torch.no_grad():
                        # measure on the actual TRAINING buffer (not a tiny subsample):
                        # per-(member,neuron) coverage needs >> P*Cout samples to read.
                        Is = X[-min(len(X), 20000):].to(device)
                        h = _cnn_frontier_input(pop[0], Is, fr)      # shared frozen prefix
                        Wf = pop[0].layers[fr].weight
                        Cout = Wf.shape[0]
                        if Wf[0].numel() == h.shape[1]:              # dense frontier
                            cnts = []
                            for m in pop:
                                pa = F.linear(h, m.layers[fr].weight.reshape(Cout, -1),
                                              m.layers[fr].bias)
                                sd = pa.std(0).clamp_min(1e-6)
                                cnts.append(((pa.abs() / sd) < 0.1).sum(0))
                            near = torch.stack(cnts).float().mean(0)
                            print(f"        [frontier L{fr + 1} sampling] near-kink/neuron: "
                                  f"min {int(near.min())} med {int(near.median())} "
                                  f"max {int(near.max())} | starved(<10) "
                                  f"{int((near < 10).sum())}/{Cout}", flush=True)

            # --- --partial: opportunistically refine + pin the frontier's SOLVABLE
            #     channels IN PLACE every log_every iters (grad-masked hooks), no
            #     reinit / advance / restart -- stragglers keep training. EXACT MLP
            #     --partial semantics; --peel (below) does the full-layer peel. ---
            if (cfg.partial and cfg.cheat and (t + 1) % cfg.log_every == 0
                    and best is not None and cfg.act in ("relu", "leaky_relu")):
                import copy as _cp
                Lh = nlay - 1                            # hidden layers (exclude head)
                def _ncout(l): return pop[0].layers[l].weight.shape[0]
                def _psolved(l):
                    m = partial_mask.get(l, torch.zeros(_ncout(l), dtype=torch.bool,
                                                        device=device)).clone()
                    if l in frozen:
                        m = m | frozen[l][2].to(device)
                    return m
                pf = next((l for l in range(Lh)
                           if int(_psolved(l).sum()) < _ncout(l)), None)
                if pf is not None:
                    Cout = _ncout(pf)
                    if partial_exact is None:
                        partial_exact = _cp.deepcopy(best).to(device)
                    if pf not in partial_mask:
                        partial_mask[pf] = torch.zeros(Cout, dtype=torch.bool, device=device)
                    wdt = partial_exact.layers[pf].weight.dtype
                    # refresh exact prefix (< pf) + pf's frozen rows from `frozen`
                    for l in range(pf + 1):
                        if l in frozen:
                            fm = frozen[l][2].to(device)
                            partial_exact.layers[l].weight.data[fm] = frozen[l][0].to(device)[fm].to(wdt)
                            partial_exact.layers[l].bias.data[fm] = frozen[l][1].to(device)[fm].to(wdt)
                    unsolved = (~_psolved(pf)).nonzero(as_tuple=True)[0].tolist()
                    for c in unsolved:                   # seed unsolved <- current guess
                        partial_exact.layers[pf].weight.data[c] = best.layers[pf].weight.data[c].to(wdt)
                        partial_exact.layers[pf].bias.data[c] = best.layers[pf].bias.data[c].to(wdt)
                    _pt0 = time.time()
                    Wr, br, rmask = _cnn_refine_layer(
                        teacher, partial_exact, pf, input_shape, device, cfg.act,
                        only_channels=unsolved)
                    newly = [c for c in unsolved if Wr is not None and bool(rmask[c])]
                    if newly:
                        idx = torch.tensor(newly, device=device)
                        _scale_match_rows(partial_exact.layers[pf], idx, Wr, br)  # guess mag
                        partial_mask[pf][idx] = True
                        for m, opt in zip(pop, opts):    # in-place pin (grad-masked)
                            key = (id(m), pf)
                            if key not in partial_hooked:
                                live = torch.zeros(Cout, dtype=torch.bool, device=device)
                                partial_live[key] = live
                                m.layers[pf].weight.register_hook(
                                    lambda g, k=live: g * (~k).to(g.dtype).view(
                                        [-1] + [1] * (g.dim() - 1)))
                                m.layers[pf].bias.register_hook(
                                    lambda g, k=live: g * (~k).to(g.dtype))
                                partial_hooked.add(key)
                            live = partial_live[key]
                            with torch.no_grad():
                                m.layers[pf].weight[idx] = partial_exact.layers[pf].weight.data[idx].to(m.layers[pf].weight.dtype)
                                m.layers[pf].bias[idx] = partial_exact.layers[pf].bias.data[idx].to(m.layers[pf].bias.dtype)
                            live[idx] = True
                            for p in (m.layers[pf].weight, m.layers[pf].bias):
                                st = opt.state.get(p)
                                if st:
                                    if "exp_avg" in st: st["exp_avg"][idx] = 0
                                    if "exp_avg_sq" in st: st["exp_avg_sq"][idx] = 0
                    print(f"  [partial] L{pf + 1}: +{len(newly)} frozen in place "
                          f"({int(_psolved(pf).sum())}/{Cout} total)  |  "
                          f"{time.time() - _pt0:.1f}s", flush=True)

            # --- fast-peel-partial: PER-NEURON consensus peel (any mode with a
            #     committee). Every log iter: frontier = first hidden layer not fully
            #     solved; align every member to the consensus frame (channel
            #     permutation, function-preserving) so row indices agree across the
            #     committee; kink-refine the frontier's consensus channels not yet
            #     solved from the QUORUM-MEAN rows (exact prefix from partial_exact);
            #     inject every solved row into EVERY member at THAT member's own
            #     magnitude (per-member scale-match) and pin it in place (grad-masked).
            #     The frontier advances only once the whole layer is solved.
            if (cfg.fast_peel_partial and (t + 1) % cfg.log_every == 0
                    and cons is not None and cfg.p > 1
                    and cfg.act in ("relu", "leaky_relu")):
                import copy as _cp
                from align import cnn_canonicalize_, cnn_align_to_
                Lh = nlay - 1
                def _ncoutQ(l): return pop[0].layers[l].weight.shape[0]
                def _psolvedQ(l):
                    mq = partial_mask.get(l, torch.zeros(_ncoutQ(l), dtype=torch.bool,
                                                         device=device)).clone()
                    if l in frozen:
                        mq = mq | frozen[l][2].to(device)
                    return mq
                pf = next((l for l in range(Lh) if int(_psolvedQ(l).sum()) < _ncoutQ(l)), None)
                if pf is not None and pf < len(masks):
                    _pt0 = time.time()
                    # (0) align every member to the consensus frame (= best member bi's
                    #     canonical order) up to the frontier
                    ref_c = pop[bi].clone(); cnn_canonicalize_(ref_c)
                    for mi_, (m_, opt_) in enumerate(zip(pop, opts)):
                        if mi_ == bi:
                            continue
                        cc = m_.clone(); cnn_canonicalize_(cc)
                        perms = cnn_align_to_(cc, ref_c)
                        _cnn_apply_perms_(m_, perms, pf, opt=opt_, live_masks=partial_live)
                    Cout = _ncoutQ(pf)
                    if partial_exact is None:
                        partial_exact = _cp.deepcopy(cons).to(device).double()   # fp64 record of solved rows
                    elif partial_exact.layers[0].weight.dtype != torch.float64:
                        partial_exact = partial_exact.double()
                    if pf not in partial_mask:
                        partial_mask[pf] = torch.zeros(Cout, dtype=torch.bool, device=device)
                    wdt = partial_exact.layers[pf].weight.dtype
                    for l in range(pf + 1):                  # exact prefix rows from `frozen`,
                        if l in frozen:                      # but keep our own fp64 solved rows
                            fm = frozen[l][2].to(device)
                            if l in partial_mask:
                                fm = fm & ~partial_mask[l].to(device)
                            partial_exact.layers[l].weight.data[fm] = frozen[l][0].to(device)[fm].to(wdt)
                            partial_exact.layers[l].bias.data[fm] = frozen[l][1].to(device)[fm].to(wdt)
                    cand = (masks[pf].to(device) & ~_psolvedQ(pf)).nonzero(as_tuple=True)[0].tolist()
                    newly = []
                    if cand:
                        for c in cand:                       # guesses <- consensus rows
                            partial_exact.layers[pf].weight.data[c] = cons.layers[pf].weight.data[c].to(wdt)
                            partial_exact.layers[pf].bias.data[c] = cons.layers[pf].bias.data[c].to(wdt)
                        Wr, br, rmask = _cnn_refine_layer(
                            teacher, partial_exact, pf, input_shape, device, cfg.act,
                            only_channels=cand)
                        newly = [c for c in cand if Wr is not None and bool(rmask[c])]
                        if newly:
                            idx = torch.tensor(newly, device=device)
                            _scale_match_rows(partial_exact.layers[pf], idx, Wr, br)   # consensus magnitude
                            partial_mask[pf][idx] = True
                            for m_, opt_ in zip(pop, opts):
                                key = (id(m_), pf)
                                if key not in partial_hooked:
                                    live = torch.zeros(Cout, dtype=torch.bool, device=device)
                                    partial_live[key] = live
                                    m_.layers[pf].weight.register_hook(
                                        lambda g, k=live: g * (~k).to(g.dtype).view(
                                            [-1] + [1] * (g.dim() - 1)))
                                    m_.layers[pf].bias.register_hook(
                                        lambda g, k=live: g * (~k).to(g.dtype))
                                    partial_hooked.add(key)
                                live = partial_live[key]
                                with torch.no_grad():        # THIS member's magnitude
                                    _scale_match_rows(m_.layers[pf], idx, Wr, br)
                                live[idx] = True
                                for p in (m_.layers[pf].weight, m_.layers[pf].bias):
                                    st = opt_.state.get(p)
                                    if st:
                                        if "exp_avg" in st: st["exp_avg"][idx] = 0
                                        if "exp_avg_sq" in st: st["exp_avg_sq"][idx] = 0
                    ns = int(_psolvedQ(pf).sum())
                    print(f"  [fast-peel-partial] L{pf + 1}: {len(cand)} consensus candidates, "
                          f"+{len(newly)} solved & pinned in all {len(pop)} members "
                          f"({ns}/{Cout} total)  |  {time.time() - _pt0:.1f}s", flush=True)
                    if newly:                                # accuracy of the STORED fp64 rows
                        try:
                            smasks = [(_psolvedQ(l) if l < Lh else
                                       torch.zeros(partial_exact.layers[l].weight.shape[0],
                                                   dtype=torch.bool, device=device))
                                      for l in range(nlay)]
                            st_ = _cnn_consensus_eps(partial_exact, smasks, teacher.clone().double())
                            rep = "  ".join(f"L{l + 1}[max {st_[l]['max']:.2e} mean {st_[l]['mean']:.2e} "
                                            f"({st_[l]['n_cons']}/{st_[l]['n_tot']})]"
                                            for l in range(Lh) if st_[l]["max"] is not None)
                            print(f"  [stored fp64] solved rows vs teacher: {rep}", flush=True)
                        except Exception as e:
                            print(f"  [stored fp64] report skipped ({e})", flush=True)
                    if ns == Cout and pf + 1 < Lh:
                        print(f"  [fast-peel-partial] L{pf + 1} complete -> frontier advances to L{pf + 2}",
                              flush=True)
                    elif ns < Cout:
                        # how many members agree (inf-norm within cluster_eps, canonical
                        # frame of the best member) on each still-unsolved channel
                        try:
                            uns = (~_psolvedQ(pf)).nonzero(as_tuple=True)[0].tolist()
                            rows = []
                            for m_ in pop:
                                cc = m_.clone(); cnn_canonicalize_(cc); cnn_align_to_(cc, ref_c)
                                rows.append(torch.cat([cc.layers[pf].weight.reshape(Cout, -1),
                                                       cc.layers[pf].bias[:, None]], 1))
                            R_ = torch.stack(rows)                              # (P, Cout, Din+1)
                            ref_rows = R_[bi]
                            agree = ((R_ - ref_rows[None]).abs().amax(2) <= cfg.cluster_eps).sum(0)
                            print("  [agreement] L%d unsolved: " % (pf + 1)
                                  + "  ".join(f"ch{c}: {int(agree[c])}/{len(pop)} members within eps" for c in uns),
                                  flush=True)
                        except Exception as e:
                            print(f"  [agreement] report skipped ({e})", flush=True)

            # --- restart-stuck (MLP-parity port): peel-restart even when the
            #     frontier CAN'T be fully solved. Fires on (a) STAGNATION --
            #     >= restart_stuck_frac of the frontier solved AND no new
            #     channels frozen for restart_stuck_window iters -- or (b) the
            #     FINAL iter. Keeps solved channels pinned (from partial_exact,
            #     the guess-scale source of truth), cold-reinits unsolved +
            #     deeper, flushes the buffer, restarts the iter clock (fresh
            #     full --outer budget). Capped at restart_stuck_max. ---
            if (cfg.restart_stuck and cfg.partial and cfg.cheat
                    and partial_exact is not None
                    and ((t + 1) % cfg.log_every == 0 or t == cfg.outer - 1)
                    and stuck_restarts < cfg.restart_stuck_max):
                LhS = nlay - 1
                def _ncoutS(l): return pop[0].layers[l].weight.shape[0]
                def _psolvedS(l):
                    m = partial_mask.get(l, torch.zeros(_ncoutS(l),
                                                        dtype=torch.bool,
                                                        device=device)).clone()
                    if l in frozen:
                        m = m | frozen[l][2].to(device)
                    return m
                sf = next((l for l in range(LhS)
                           if int(_psolvedS(l).sum()) < _ncoutS(l)), None)
                if sf is not None:
                    cur = int(_psolvedS(sf).sum())
                    ncout = _ncoutS(sf)
                    # (re)start the stagnation clock when the frontier moves/grows
                    if stuck_pf != sf:
                        stuck_pf = sf
                        stuck_last_count = cur
                        stuck_last_growth = t
                    elif cur > stuck_last_count:
                        stuck_last_count = cur
                        stuck_last_growth = t
                    stagnant = (cur >= cfg.restart_stuck_frac * ncout
                                and t - stuck_last_growth
                                >= cfg.restart_stuck_window)
                    final_it = (t == cfg.outer - 1)
                    if stagnant or final_it:
                        why = "stagnation" if stagnant else "final iter"
                        pfz = {}
                        for l in range(sf + 1):
                            m = _psolvedS(l)
                            if bool(m.any()):
                                pfz[l] = (
                                    partial_exact.layers[l].weight.detach().clone(),
                                    partial_exact.layers[l].bias.detach().clone(),
                                    m.clone())
                        pop = _cnn_reinit_frozen_population(
                            input_shape, conv_cfgs, fc_dims, out_dim, cfg.act,
                            device, cfg.p, pfz)
                        opts = [torch.optim.Adam(
                            [p for p in n.parameters() if p.requires_grad],
                            lr=cfg.lr) for n in pop]
                        X = torch.empty(0, in_dim)
                        Y = (torch.empty(0, dtype=torch.long) if cfg.hard
                             else torch.empty(0, out_dim))
                        partial_hooked.clear()
                        partial_live.clear()
                        stuck_restarts += 1
                        stuck_last_count = cur
                        stuck_last_growth = 0
                        t = 0
                        print(f"  [restart-stuck] L{sf + 1} {cur}/{ncout} "
                              f"solved, {why}: kept solved channels pinned, "
                              f"cold-reinit unsolved+deeper, flushed buffer, "
                              f"RESTARTED iter clock (fresh {cfg.outer}-iter "
                              f"budget) [{stuck_restarts}/"
                              f"{cfg.restart_stuck_max}]", flush=True)
                        continue

            # --- freeze-reinit peel: once the frontier hidden layer reaches
            #     consensus, EXACTLY refine EVERY neuron of it with the kink refiner
            #     (verify_layer1, any layer -- see _cnn_refine_layer), then pin it
            #     (+ everything above) across a reinitialized committee so the search
            #     collapses onto the deeper layers. If the whole layer refines, we
            #     freeze the EXACT layer (all channels); else fall back to freezing
            #     the consensus average of just the quorum channels. ---
            # --- --retry: in a peel mode, when the budget ends with the frontier layer
            #     not fully peeled, reinit and go again. Partial modes: keep every solved
            #     row pinned, reinit only the unsolved rows + deeper layers. Full-layer
            #     modes: reinit the ENTIRE frontier layer (+ deeper). Buffer flushed,
            #     iteration clock restarted. Capped at cfg.retry retries.
            _peel_mode = (cfg.freeze_reinit or cfg.fast_peel or cfg.fast_peel_partial
                          or cfg.partial or bool(cfg.peel_try))
            if cfg.retry > 0 and _peel_mode and t == cfg.outer - 1 and retries_done < cfg.retry:
                _partial_mode = bool(cfg.fast_peel_partial or cfg.partial)
                LhR = nlay - 1
                def _solvedR(l):
                    n_ = pop[0].layers[l].weight.shape[0]
                    mr = torch.zeros(n_, dtype=torch.bool, device=device)
                    if l in partial_mask: mr = mr | partial_mask[l].to(device)
                    if l in frozen: mr = mr | frozen[l][2].to(device)
                    if l in exact_mask: mr = mr | exact_mask[l].to(device)
                    return mr
                rf = next((l for l in range(LhR) if int(_solvedR(l).sum()) < pop[0].layers[l].weight.shape[0]), None)
                if rf is not None:
                    cur = int(_solvedR(rf).sum()); ncout_r = pop[0].layers[rf].weight.shape[0]
                    pfz = {}
                    for l in range(rf + 1):
                        if l == rf and not _partial_mode:
                            break                          # whole frontier layer restarts
                        mr = _solvedR(l)
                        if not bool(mr.any()):
                            continue
                        Wp = pop[0].layers[l].weight.data.double().clone(); bp = pop[0].layers[l].bias.data.double().clone()
                        fz = frozen[l][2].to(device) if l in frozen else torch.zeros_like(mr)
                        if l in frozen:
                            Wp[fz] = frozen[l][0].to(device)[fz].double(); bp[fz] = frozen[l][1].to(device)[fz].double()
                        for src, msk in ((partial_exact, partial_mask.get(l)), (exact_net, exact_mask.get(l))):
                            if src is not None and msk is not None:
                                mm = msk.to(device) & ~fz
                                Wp[mm] = src.layers[l].weight.data[mm].double(); bp[mm] = src.layers[l].bias.data[mm].double()
                        pfz[l] = (Wp, bp, mr.clone())
                    if not _partial_mode:                  # frontier's partial rows are dropped
                        frozen.pop(rf, None); partial_mask.pop(rf, None); exact_mask.pop(rf, None)
                    pop = _cnn_reinit_frozen_population(input_shape, conv_cfgs, fc_dims, out_dim,
                                                        cfg.act, device, cfg.p, pfz)
                    opts = [torch.optim.Adam([p for p in n.parameters() if p.requires_grad],
                                             lr=cfg.lr) for n in pop]
                    X = torch.empty(0, in_dim)
                    Y = (torch.empty(0, dtype=torch.long) if cfg.hard else torch.empty(0, out_dim))
                    t = 0
                    partial_hooked.clear(); partial_live.clear()
                    retries_done += 1
                    print(f"  [retry] L{rf + 1} {cur}/{ncout_r} peeled at end of budget: "
                          + ("kept solved rows pinned, reinit unsolved + deeper" if _partial_mode
                             else "reinit ENTIRE frontier layer + deeper")
                          + ", flushed buffer, RESTARTED iter clock " + f"(fresh {cfg.outer}-iter budget)"
                          + f" [{retries_done}/{cfg.retry}]", flush=True)
                    continue

            if cfg.freeze_reinit and not cfg.peel_try:
                frontier = _cnn_frontier_layer(frozen, nlay - 1)
                if frontier is not None:
                    if cfg.cheat:
                        # cheat: consensus can't form at p=1 -- gate the peel on
                        # the ORACLE per-layer eps of the single member instead
                        # (same numbers as the printed "eps/layer (best)").
                        lmax = max(bM[2 * frontier], bM[2 * frontier + 1])
                        lmean = bm[2 * frontier]
                        fire = (lmax <= cfg.cheat_peel_max
                                and lmean <= cfg.cheat_peel_mean)
                        if fire:
                            print(f"  [cheat-peel] L{frontier + 1} eps max "
                                  f"{lmax:.2e} mean {lmean:.2e} within "
                                  f"({cfg.cheat_peel_max:g}, "
                                  f"{cfg.cheat_peel_mean:g}) -> peel", flush=True)
                        elif (cfg.fast_peel and cfg.p > 1 and cons is not None
                              and frontier < len(masks) and bool(masks[frontier].all())):
                            # --fast-peel (same meaning as the MLP path): the frontier
                            # layer peels as soon as the COMMITTEE fully agrees on it,
                            # even though no single member clears the eps gate; the
                            # consensus rows (quorum means) become the refiner's guesses
                            # (seed_net = cons below).
                            fire = True
                            print(f"  [fast-peel] L{frontier + 1}: full committee consensus "
                                  f"({int(masks[frontier].sum())}/{masks[frontier].numel()}) "
                                  f"-> refine + peel (guesses = quorum means)", flush=True)
                    else:
                        fx = cst[frontier]
                        ratio = fx["n_cons"] / max(fx["n_tot"], 1)
                        cm = fx["max"]
                        prec_ok = (cfg.freeze_precision <= 0 or
                                   (cm is not None and cm <= cfg.freeze_precision))
                        fire = ratio >= cfg.freeze_thresh and prec_ok
                    if fire:
                        # extend the peel through consecutive DEEPER hidden layers
                        # already in-gate (MLP --peel extend-through): refine+freeze
                        # them all in one pass instead of one per log_every.
                        last = frontier
                        while last + 1 < nlay - 1 and (
                                max(bM[2 * (last + 1)], bM[2 * (last + 1) + 1])
                                <= cfg.cheat_peel_max
                                and bm[2 * (last + 1)] <= cfg.cheat_peel_mean):
                            last += 1
                        if last > frontier:
                            print(f"  [cheat-peel] extending peel through "
                                  f"L{last + 1} (consecutive in-gate)", flush=True)
                        try:
                            import copy as _copy
                            # exact_net = persistent source of truth (exact channels where
                            # kink-refined, consensus otherwise). Seed the frontier's guess
                            # from this iter's consensus (cheat p=1: no consensus net
                            # exists -- seed from the single member).
                            seed_net = cons if cons is not None else best.clone()
                            if exact_net is None:
                                exact_net = _copy.deepcopy(seed_net).to(device).double()   # fp64 record
                            elif exact_net.layers[0].weight.dtype != torch.float64:
                                exact_net = exact_net.double()
                            # rows already solved by --fast-peel-partial / --partial are
                            # SOLVED: take them from the fp64 record, never re-refine them
                            if partial_exact is not None:
                                for l in range(last + 1):
                                    pm = partial_mask.get(l)
                                    if pm is None or not bool(pm.any()):
                                        continue
                                    Cl = exact_net.layers[l].weight.shape[0]
                                    if l not in exact_mask:
                                        exact_mask[l] = torch.zeros(Cl, dtype=torch.bool, device=device)
                                    nw = pm.to(device) & ~exact_mask[l]
                                    if bool(nw.any()):
                                        wdt_ = exact_net.layers[l].weight.dtype
                                        exact_net.layers[l].weight.data[nw] = partial_exact.layers[l].weight.data[nw].to(wdt_)
                                        exact_net.layers[l].bias.data[nw] = partial_exact.layers[l].bias.data[nw].to(wdt_)
                                        exact_mask[l] = exact_mask[l] | nw
                            # (1) KEEP every neuron we solve; (2) RETRY still-unsolved
                            # neurons in ALL layers 0..frontier (shallow->deep) with a
                            # fresh consensus guess. (3) FREEZE ONLY SOLVED neurons --
                            # never pin an unsolved neuron at its (imprecise) consensus;
                            # leave it trainable so the committee + later retries improve it.
                            fmasks, stat = {}, []
                            for l in range(last + 1):
                                Coutl = exact_net.layers[l].weight.shape[0]
                                if l not in exact_mask:
                                    exact_mask[l] = torch.zeros(Coutl, dtype=torch.bool, device=device)
                                ui = (~exact_mask[l]).nonzero(as_tuple=True)[0]
                                if len(ui):
                                    wdt = exact_net.layers[l].weight.dtype
                                    exact_net.layers[l].weight.data[ui] = seed_net.layers[l].weight.data[ui].to(wdt)
                                    exact_net.layers[l].bias.data[ui] = seed_net.layers[l].bias.data[ui].to(wdt)
                                    Wr, br, rmask = _cnn_refine_layer(
                                        teacher, exact_net, l, input_shape, device,
                                        cfg.act, only_channels=ui.tolist())
                                    if Wr is not None:
                                        newly = rmask.to(device) & (~exact_mask[l])
                                        idx = newly.nonzero(as_tuple=True)[0]
                                        if len(idx):
                                            _scale_match_rows(exact_net.layers[l],
                                                              idx, Wr, br)  # guess magnitude
                                            exact_mask[l] = exact_mask[l] | newly
                                fmasks[l] = exact_mask[l].clone()   # SOLVED ONLY -- never consensus
                                stat.append(f"L{l+1} {int(exact_mask[l].sum())}/{Coutl}")
                            print("  [kink-refine] solved " + " ".join(stat), flush=True)
                            frozen = {l: (exact_net.layers[l].weight.detach().clone(),
                                          exact_net.layers[l].bias.detach().clone(),
                                          fmasks[l].clone())
                                      for l in range(last + 1)}
                            # MLP --peel semantics: WARM reinit by default (keep the
                            # trained downstream); COLD only under --peelrestart
                            # (peel_refresh), which also restarts the clock below.
                            if cfg.peel_refresh and not cfg.fast_peel_partial:
                                pop = _cnn_reinit_frozen_population(
                                    input_shape, conv_cfgs, fc_dims, out_dim, cfg.act,
                                    device, cfg.p, frozen)
                                _wm = ""
                            else:
                                # --fast-peel-partial: the deeper layers have been training
                                # warm on the pinned prefix all along (their consensus is
                                # weight-based, e.g. L2 13/16); a cold reinit would throw
                                # that away -- keep them, pin the completed layer only
                                pop = _cnn_warm_reinit_population(
                                    pop, input_shape, conv_cfgs, fc_dims, out_dim,
                                    cfg.act, device, frozen)
                                _wm = "warm-"
                            partial_hooked.clear(); partial_live.clear()  # pop rebuilt
                            opts = [torch.optim.Adam(
                                [p for p in n.parameters() if p.requires_grad],
                                lr=cfg.lr) for n in pop]
                            print(f"  [freeze-reinit] froze L1..L{last + 1}; "
                                  f"{_wm}reinit committee onto deeper layers", flush=True)
                            refresh_now = cfg.peel_refresh
                        except Exception as e:
                            print(f"  [freeze-reinit] skipped ({e})", flush=True)
                # ALL hidden layers peeled -> the linear HEAD is a closed-form ridge
                # LSQ on the now-exact penultimate features. Solve once (exact), like
                # the MLP output-layer solve.
                if (not out_solved and len(X) > 0 and frozen
                        and all(l in frozen and bool(frozen[l][2].all())
                                for l in range(nlay - 1))):
                    try:
                        with torch.no_grad():
                            _xs, _ys = X[-30000:], Y[-30000:]
                            for _m in pop:
                                _cnn_solve_last_layer_(_m, _xs, _ys, ridge=1e-10)
                        out_solved = True
                        _oe = cnn_param_errors(pop[0], teacher, hard=cfg.hard)
                        print("  [cheat-peel] all hidden peeled -> head solved "
                              f"(closed-form LSQ) -> max_eps {_oe['max_eps']:.2e}",
                              flush=True)
                    except Exception as e:
                        print(f"  [cheat-peel] head solve skipped ({e})", flush=True)

        # --- peel-try (cfg.peel_try=k): every k iters, ASK the refiner instead
        #     of predicting with an eps gate. Solved channels accumulate in
        #     exact_net/exact_mask (frame-stable: everything below the frontier
        #     is exact+frozen); unsolved guesses are re-seeded from the current
        #     best each attempt. Early-abort after 2 fresh abstains, rotating
        #     the start channel so stubborn channels don't shadow the rest.
        #     Any attempt that solves NEW channels banks them immediately:
        #     partial freeze into the population + reinit of everything
        #     unfrozen (fresh signs for the stuck rows, fresh deeper layers).
        #     The frontier ADVANCES (peel) only when the WHOLE layer is exact
        #     and duplicate-free. ---
        if cfg.peel_try and (t + 1) % cfg.peel_try == 0:
            frontier = _cnn_frontier_layer(frozen, len(pop[0].layers) - 1,
                                           cfg.peel_advance_frac)
            if frontier is not None:
                import copy as _copy
                bi_pt = (0 if cfg.p == 1 else
                         min(range(cfg.p), key=lambda i: loss_on(pop, X, Y)[i]))
                guess = pop[bi_pt].clone()
                Coutl = guess.layers[frontier].weight.shape[0]
                if exact_net is None:
                    exact_net = _copy.deepcopy(guess).to(device)
                if frontier not in exact_mask:
                    exact_mask[frontier] = torch.zeros(Coutl, dtype=torch.bool,
                                                       device=device)
                wdt = exact_net.layers[frontier].weight.dtype
                ui = (~exact_mask[frontier]).nonzero(as_tuple=True)[0].tolist()
                for c in ui:                    # unsolved guesses <- current best
                    exact_net.layers[frontier].weight.data[c] = \
                        guess.layers[frontier].weight.data[c].to(wdt)
                    exact_net.layers[frontier].bias.data[c] = \
                        guess.layers[frontier].bias.data[c].to(wdt)
                start = peel_try_n % max(len(ui), 1)
                peel_try_n += 1
                n0 = int(exact_mask[frontier].sum())
                miss = 0
                for c in ui[start:] + ui[:start]:
                    Wr, br, rmask = _cnn_refine_layer(
                        teacher, exact_net, frontier, input_shape, device,
                        cfg.act, only_channels=[c])
                    if Wr is not None and bool(rmask[c]):
                        _scale_match_rows(exact_net.layers[frontier],       # guess magnitude
                                          torch.tensor([c], device=device), Wr, br)
                        exact_mask[frontier][c] = True
                        miss = 0                          # consecutive: reset on solve
                    else:
                        miss += 1
                        if miss >= cfg.peel_miss_abort:
                            break
                ns = int(exact_mask[frontier].sum())
                print(f"  [peel-try] L{frontier + 1}: {ns}/{Coutl} exact"
                      + (" (aborted attempt)" if miss >= cfg.peel_miss_abort
                         else ""), flush=True)
                if ns == Coutl:
                    # dedupe: two guesses may have locked the SAME teacher kink
                    # (also in flipped form) -- "all solved" would then freeze a
                    # wrong layer. Attacker-side check on the refined rows.
                    D = torch.cat(
                        [exact_net.layers[frontier].weight.view(Coutl, -1),
                         exact_net.layers[frontier].bias.view(-1, 1)], 1).float()
                    dd = torch.cdist(D, D, p=2) + 2 * torch.eye(Coutl, device=D.device)
                    df = torch.cdist(D, -D, p=2)
                    if float(torch.minimum(dd, df).min()) < 1e-3:
                        print("  [peel-try] duplicate refined channels detected; "
                              "NOT freezing", flush=True)
                    else:
                        frozen[frontier] = (
                            exact_net.layers[frontier].weight.detach().clone(),
                            exact_net.layers[frontier].bias.detach().clone(),
                            exact_mask[frontier].clone())
                        if cfg.peel_warm:            # keep deeper layers warm
                            pop = _cnn_warm_reinit_population(
                                pop, input_shape, conv_cfgs, fc_dims, out_dim,
                                cfg.act, device, frozen)
                        else:
                            pop = _cnn_reinit_frozen_population(
                                input_shape, conv_cfgs, fc_dims, out_dim, cfg.act,
                                device, cfg.p, frozen)
                        opts = [torch.optim.Adam(
                            [p for p in n.parameters() if p.requires_grad],
                            lr=cfg.lr) for n in pop]
                        print(f"  [peel-try] L{frontier + 1} fully exact -> "
                              f"froze L1..L{frontier + 1}; "
                              f"{'warm-' if cfg.peel_warm else ''}reinit committee "
                              f"onto deeper layers", flush=True)
                        refresh_now = cfg.peel_refresh
                elif ns > n0:
                    # PARTIAL freeze: bank every newly solved channel now -- pin it
                    # in the population (grad-masked). COLD (default): reinit all
                    # unfrozen params (remaining frontier rows get fresh signs;
                    # deeper layers restart). WARM (--peel-warm): keep deeper layers
                    # + unsolved frontier rows trained; only re-randomize the
                    # unsolved FRONTIER rows for flip escape. Frontier does NOT
                    # advance until the whole layer is exact (_cnn_frontier_layer).
                    frozen[frontier] = (
                        exact_net.layers[frontier].weight.detach().clone(),
                        exact_net.layers[frontier].bias.detach().clone(),
                        exact_mask[frontier].clone())
                    if cfg.peel_warm:
                        rr = ((frontier, ~exact_mask[frontier])
                              if cfg.peel_reroll else None)
                        pop = _cnn_warm_reinit_population(
                            pop, input_shape, conv_cfgs, fc_dims, out_dim, cfg.act,
                            device, frozen, reroll_frontier=rr)
                    else:
                        pop = _cnn_reinit_frozen_population(
                            input_shape, conv_cfgs, fc_dims, out_dim, cfg.act,
                            device, cfg.p, frozen)
                    opts = [torch.optim.Adam(
                        [p for p in n.parameters() if p.requires_grad],
                        lr=cfg.lr) for n in pop]
                    print(f"  [peel-try] froze {ns - n0} newly solved "
                          f"(L{frontier + 1} at {ns}/{Coutl}); "
                          f"{'warm-' if cfg.peel_warm else ''}reinit "
                          f"committee on the rest", flush=True)
                    # NOTE: a PARTIAL freeze (banking a subset) is NOT a full-layer
                    # peel, so it does NOT peel-refresh -- the outer clock keeps
                    # running. peel_refresh/--peelrestart fires only on a full-layer
                    # advance (above). Restarting on every banked channel is the
                    # "restarting too often" thrash + starves the deeper search.
                # stuck escape: warm freeze compounds while solving but has no
                # way out once the last hard channels plateau (no new solve ->
                # nothing re-rolled). After peel_stuck attempts with no progress
                # on this frontier, COLD-reinit the unfrozen params (fresh signs
                # for the stuck rows + fresh tail; the solved channels stay
                # pinned) to shake them out of their basins.
                if ns > peel_stuck_best.get(frontier, -1):
                    peel_stuck_best[frontier] = ns
                    peel_stall = 0
                elif cfg.peel_stuck > 0 and ns < Coutl:
                    peel_stall += 1
                    if peel_stall >= cfg.peel_stuck:
                        frozen[frontier] = (
                            exact_net.layers[frontier].weight.detach().clone(),
                            exact_net.layers[frontier].bias.detach().clone(),
                            exact_mask[frontier].clone())
                        pop = _cnn_reinit_frozen_population(
                            input_shape, conv_cfgs, fc_dims, out_dim, cfg.act,
                            device, cfg.p, frozen)
                        opts = [torch.optim.Adam(
                            [p for p in n.parameters() if p.requires_grad],
                            lr=cfg.lr) for n in pop]
                        peel_stall = 0
                        print(f"  [peel-try] stuck at {ns}/{Coutl} for "
                              f"{cfg.peel_stuck} attempts -> COLD-reinit unfrozen "
                              f"(fresh signs + tail) to escape", flush=True)
        if refresh_now:
            # --peelrefresh: the buffered samples were mined against the
            # pre-peel student -- stale for the reinitialized committee. Fresh
            # curriculum + full outer budget for the collapsed deeper search.
            X = torch.empty(0, in_dim)
            Y = (torch.empty(0, dtype=torch.long) if cfg.hard
                 else torch.empty(0, out_dim))
            t = 0
            refresh_now = False
            print("  [peel-refresh] emptied sample buffer; restarting outer "
                  "iterations onto the frozen prefix", flush=True)
        else:
            t += 1
    final = dict(final_max_eps=cnn_param_errors(best, teacher,
                                                hard=cfg.hard)["max_eps"],
                 wall_s=round(time.time() - t0, 1))
    return best, log, final
