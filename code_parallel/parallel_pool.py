"""Population-parallel reconstruction across GPUs.

The committee is SPLIT across GPUs: member i lives on GPU (i % ng) and is trained
there, on its home GPU, for all epochs -- with ZERO cross-GPU communication (the
members are independent, so there is nothing to synchronize). The query buffer is
REPLICATED on every GPU (each worker holds the full windowed set on-GPU), so every
member trains on the identical data the serial code would use. This is the axis
the problem is trivially parallel on: p independent trainings -> ng-way speedup,
no all-reduce, no NCCL. (The previous version sharded the *data* and all-reduced
gradients every batch; for these tiny MLPs that was pure latency and ran slower
than one GPU. See parallel_bench.py for the microbenchmark that motivates this.)

One persistent worker PROCESS per GPU (not a thread): the per-step training
kernels are tiny, so their Python/launch overhead holds the GIL and threads
serialize; separate processes each get their own interpreter and actually run
concurrently. Per-member endgame-style steps that are independent across members
(closed-form last-layer solve, staged-LBFGS polish) also run in the workers, so
they parallelize too and keep each member's Adam state intact.

Query generation, cross-member consensus/combine/maintenance, dump and the endgame
solve run in the parent, which gathers the committee (small state dicts) each iter
and keeps a full CPU copy of the queries. run.py routes multi-GPU runs here.
"""
import os
import time
import traceback
from dataclasses import asdict

import torch
import torch.multiprocessing as mp

from nets import MLP

_ERR = "__worker_error__"   # sentinel result so a dead worker raises, not hangs


class _StreamLoader:
    """Multi-threaded, order-preserving host->device prefetch for a CPU-resident
    query buffer. A single Python thread cannot gather+pin+copy fast enough to keep
    a GPU fed (it sits ~85% idle), so `nthreads` producer threads each gather a
    batch, stage it in pinned memory, and issue an async copy on their own CUDA
    stream -- the gather/copy is a C++ memcpy that releases the GIL, so the threads
    actually run in parallel across CPU cores. Batches are consumed strictly in
    `perm` order (so training is bitwise-identical to direct indexing), and a
    semaphore bounds in-flight batches so pinned/VRAM use stays fixed.

    `nthreads` is per worker; with one worker per GPU and many CPUs, e.g. 3 GPUs x
    5 threads = 15 loader threads, the feed keeps up and training goes compute-
    bound instead of transfer-starved."""

    def __init__(self, in_dim, out_dim, batch, dev, dtype=torch.float32,
                 nthreads=5, depth=None):
        import os as _os
        env = _os.environ.get("MP_LOADER_THREADS")
        self.nw = int(env) if env else nthreads
        self.batch, self.dev = batch, dev
        self.depth = depth if depth is not None else 2 * self.nw
        self.streams = [torch.cuda.Stream(device=dev) for _ in range(self.nw)]

    def epoch(self, Xs, Ys, perm):
        import threading
        n, batch, dev, nw = len(perm), self.batch, self.dev, self.nw
        nchunks = (n + batch - 1) // batch
        slots = [None] * nchunks
        ready = [threading.Event() for _ in range(nchunks)]
        sem = threading.Semaphore(self.depth)          # bound in-flight batches
        nxt = [0]; lock = threading.Lock()
        comp = torch.cuda.current_stream(dev)

        def producer(sid):
            torch.cuda.set_device(dev)                 # threads default to dev 0
            st = self.streams[sid]
            while True:
                with lock:
                    c = nxt[0]; nxt[0] += 1
                if c >= nchunks:
                    return
                sem.acquire()
                idx = perm[c * batch:(c + 1) * batch]
                px = Xs[idx].pin_memory(); py = Ys[idx].pin_memory()
                with torch.cuda.stream(st):
                    gx = px.to(dev, non_blocking=True)
                    gy = py.to(dev, non_blocking=True)
                ev = torch.cuda.Event(); ev.record(st)
                slots[c] = (gx, gy, ev, px, py)        # keep px,py alive until used
                ready[c].set()

        threads = [threading.Thread(target=producer, args=(i,), daemon=True)
                   for i in range(nw)]
        for t in threads:
            t.start()
        try:
            for c in range(nchunks):
                ready[c].wait()
                gx, gy, ev, px, py = slots[c]; slots[c] = None
                comp.wait_event(ev)                    # compute waits for this copy
                gx.record_stream(comp); gy.record_stream(comp)
                sem.release()                          # let a producer fetch one more
                yield gx, gy
        finally:
            for t in threads:
                t.join()


# ----------------------------------------------------------------- worker --
def _worker(g, dev_str, dims, cfg_d, seed, ng, cmd_q, res_q):
    """One persistent process per GPU. Owns the members with (i % ng == g),
    trains them independently on a full local copy of the query buffer. No
    cross-process communication during training."""
    from method import solve_last_layer_, solver_polish_

    dev = torch.device(dev_str)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    p = cfg_d["p"]
    epochs, batch = cfg_d["epochs"], cfg_d["batch"]
    fit_loss, lr = cfg_d["fit_loss"], cfg_d["lr"]
    window, q, outer = cfg_d["window"], cfg_d["q"], cfg_d["outer"]
    fit_delta, solverwindow = cfg_d["fit_delta"], cfg_d["solverwindow"]
    keep = window * q if window > 0 else 0        # 0 = unbounded (paper)

    owned = [i for i in range(p) if i % ng == g]
    members, opts = {}, {}
    for i in owned:
        torch.manual_seed(seed * 100003 + i)      # distinct, reproducible members
        m = MLP(dims).to(dev)
        members[i] = m
        opts[i] = torch.optim.Adam(m.parameters(), lr=lr)

    # Full query buffer, replicated on this GPU. Fall back to host RAM (streamed
    # per batch) only if the windowed set is too big for the device, mirroring the
    # serial code's behavior for very large query sets.
    cap = (window if window > 0 else outer) * q
    xbytes = cap * (dims[0] + dims[-1]) * 4
    xdev = (dev if (dev.type == "cuda" and xbytes < 8 * 1024 ** 3)
            else torch.device("cpu"))
    Xs = torch.empty(0, dims[0], device=xdev)
    Ys = torch.empty(0, dims[-1], device=xdev)
    gen = torch.Generator(device=xdev).manual_seed(seed * 7 + g)

    def _sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    # streaming = buffer lives on host RAM (too big for VRAM) and must be copied to
    # the GPU each batch; that's the case the overlapped prefetch loader accelerates.
    streaming = (xdev.type == "cpu" and dev.type == "cuda")
    loader = None

    while True:
        op, payload = cmd_q.get()
        if op == "stop":
            break
        resp = None
        try:
            if op == "append":
                I, T = payload                     # FULL new queries (CPU tensors)
                Xs = torch.cat([Xs, I.to(xdev)])
                Ys = torch.cat([Ys, T.to(xdev)])
                if keep and len(Xs) > keep:
                    Xs, Ys = Xs[-keep:], Ys[-keep:]
            elif op == "decay":
                for o in opts.values():
                    for pg in o.param_groups:
                        pg["lr"] /= 10
            elif op == "train":
                n = len(Xs)
                # per-epoch permutations shared across this worker's members (as in
                # the serial code, where members see the same batch order per epoch)
                perms = [torch.randperm(n, generator=gen, device=xdev)
                         for _ in range(epochs)]
                if streaming and loader is None:       # lazy: allocate pinned pool once
                    loader = _StreamLoader(dims[0], dims[-1], batch, dev)

                def batches(perm):
                    if streaming:                      # overlapped H2D prefetch
                        return loader.epoch(Xs, Ys, perm)
                    return ((Xs[perm[b:b + batch]], Ys[perm[b:b + batch]])
                            for b in range(0, n, batch))   # already on GPU

                for i in owned:
                    net, opt = members[i], opts[i]
                    for ep in range(epochs):
                        ep_loss, nb = 0.0, 0
                        for xb, yb in batches(perms[ep]):
                            opt.zero_grad()
                            r = net(xb) - yb
                            loss = ((r * r).mean() if fit_loss == "mse"
                                    else r.abs().mean())
                            loss.backward()
                            opt.step()
                            if fit_delta > 0:
                                ep_loss += loss.item(); nb += 1
                        if fit_delta > 0 and ep_loss / max(nb, 1) < fit_delta:
                            break
                _sync()
            elif op == "lastlayer":
                # closed-form ridge last-layer solve, per owned member, in place
                # (independent across members -> runs in parallel across GPUs).
                for i in owned:
                    solve_last_layer_(members[i], Xs, Ys)
                _sync()
            elif op == "polish":
                # staged MSE->MAE LBFGS tighten, per owned member, in place, on the
                # last `solverwindow` iters of queries (0 = all). Adam untouched.
                ksw = solverwindow * q
                Xp, Yp = ((Xs[-ksw:], Ys[-ksw:]) if ksw and len(Xs) > ksw
                          else (Xs, Ys))
                for i in owned:
                    solver_polish_(members[i], Xp, Yp)
                _sync()
            elif op == "report":
                # each worker reports its members' states + full-buffer train L1
                n = len(Xs)
                out = {}
                with torch.no_grad():
                    for i in owned:
                        net = members[i]
                        tot, cnt = 0.0, 0
                        for b in range(0, n, 8192):
                            xb = Xs[b:b + 8192].to(dev, non_blocking=True)
                            yb = Ys[b:b + 8192].to(dev, non_blocking=True)
                            tot += (net(xb) - yb).abs().sum().item()
                            cnt += yb.numel()
                        st = {k: v.detach().cpu()
                              for k, v in net.state_dict().items()}
                        out[i] = (st, tot / max(cnt, 1))
                resp = out
            elif op == "set":                      # inject a member (owner only)
                i, state = payload
                if i in members:
                    members[i].load_state_dict(state)
                    members[i].to(dev)
                    opts[i] = torch.optim.Adam(members[i].parameters(),
                                               lr=opts[i].param_groups[0]["lr"])
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception:                          # never die silently -> no hang
            resp = (_ERR, f"[gpu {g}] {op}:\n{traceback.format_exc()}")
        res_q.put((g, resp))


# ------------------------------------------------------------------- pool --
class WorkerPool:
    def __init__(self, devices, dims, cfg, seed):
        ctx = mp.get_context("spawn")
        self.ng = len(devices)
        self.p = cfg.p
        cfg_d = asdict(cfg)
        self.cmd = [ctx.Queue() for _ in range(self.ng)]
        self.res = ctx.Queue()
        self.procs = [ctx.Process(target=_worker,
                                  args=(g, str(devices[g]), dims, cfg_d, seed,
                                        self.ng, self.cmd[g], self.res),
                                  daemon=True)
                      for g in range(self.ng)]
        for pr in self.procs:
            pr.start()

    def _all(self, op, payloads=None):
        for g in range(self.ng):
            self.cmd[g].put((op, None if payloads is None else payloads[g]))
        res = {}
        for _ in range(self.ng):
            g, r = self.res.get()
            if isinstance(r, tuple) and len(r) == 2 and r[0] == _ERR:
                raise RuntimeError(f"worker {g} died during '{op}':\n{r[1]}")
            res[g] = r
        return res

    def append(self, I, T):
        # REPLICATE the full new queries onto every worker (population parallelism
        # keeps the whole buffer on each GPU, so no rows are dropped -- unlike the
        # old data-parallel shard, which dropped up to ng-1 rows per iter).
        Ic, Tc = I.detach().cpu(), T.detach().cpu()
        self._all("append", [(Ic, Tc)] * self.ng)

    def decay(self):
        self._all("decay")

    def train(self):
        self._all("train")

    def lastlayer(self):
        self._all("lastlayer")

    def polish(self):
        self._all("polish")

    def report(self):
        res = self._all("report")                  # {g: {i: (state, l1)}}
        merged = {}
        for d in res.values():
            merged.update(d)
        states = {i: merged[i][0] for i in range(self.p)}
        losses = [merged[i][1] for i in range(self.p)]
        return states, losses

    def set_member(self, i, state):
        self._all("set", [(i, state)] * self.ng)   # owner (i % ng) applies it

    def stop(self):
        for g in range(self.ng):
            try:
                self.cmd[g].put(("stop", None))
            except Exception:
                pass
        for pr in self.procs:
            pr.join(timeout=10)


# --------------------------------------------------------- reconstruct_mp --
def reconstruct_mp(teacher, dims, cfg, devices, eval_pts, seed=0, save_recon=None):
    """Population-parallel reconstruction: committee split across GPUs, query
    buffer replicated, zero cross-GPU traffic during training. Query-gen /
    consensus / combine / maintenance / endgame run in the main process; the
    parent keeps a full CPU copy of the queries for those."""
    from method import (gen_queries, gen_queries_parallel, build_consensus,
                        consensus_neuron_stats, polish_consensus, polish_lbfgs,
                        polish_f64, agreement, l1_on, aligned_pop_average,
                        align_member_to, soup_of, solve_last_layer_)
    from align import param_errors

    devs = [torch.device(d) for d in devices]
    master = devs[0]
    ng = len(devs)
    teacher = teacher.to(master)
    eval_pts = eval_pts.to(master)
    gen = torch.Generator(device=master).manual_seed(seed)
    qgens = [torch.Generator(device=d).manual_seed(seed * 131 + k)
             for k, d in enumerate(devs)]
    pool = WorkerPool(devices, dims, cfg, seed)
    # Query-gen memory scales as q*input (the learnable query tensor + its Adam
    # state, ~16 B/elem) and as p^2*q*out (the pairwise-disagreement tensor). On
    # ONE GPU a large q OOMs (e.g. q=44000 with a 12288-in net needs ~9 GB just
    # for the query optimizer); shard query-gen across the GPUs when it is heavy,
    # which bounds per-GPU memory AND parallelizes the (then-expensive) step. For
    # small q, single-GPU on master is ~2x quicker (no per-call committee clone).
    qgen_bytes = max(cfg.q * dims[0] * 16, cfg.p * cfg.p * cfg.q * dims[-1] * 4)
    qgen_shard = ng > 1 and qgen_bytes > 1_000_000_000

    # Honest heads-up for the memory-bound regime: if the windowed query set is too
    # big to sit on a GPU it must be streamed from CPU RAM every epoch. Population
    # parallelism replicates that streamed buffer on each GPU, so training becomes
    # CPU-memory-bandwidth bound and multi-GPU gives little/no training speedup
    # (it can be slower than one GPU). Query-gen and the endgame solve still
    # parallelize; training does not in this regime.
    buf_bytes = ((cfg.window if cfg.window > 0 else cfg.outer)
                 * cfg.q * (dims[0] + dims[-1]) * 4)
    if buf_bytes > 8 * 1024 ** 3:
        print(f"[parallel-mp] NOTE: query buffer ~{buf_bytes / 1e9:.0f} GB exceeds "
              f"GPU memory -> streamed from CPU RAM. This config is memory-bandwidth "
              f"bound; population parallelism won't speed up training here (may be "
              f"slower than 1 GPU). Prefer a single GPU, a smaller --window/--q so "
              f"the buffer fits on a GPU, or run independent seeds/variants one per "
              f"GPU (CUDA_VISIBLE_DEVICES=k) for real throughput.", flush=True)
    Xc = torch.empty(0, dims[0]); Yc = torch.empty(0, dims[-1])   # parent full copy
    decay_at = {int(s * cfg.outer) for s in cfg.lr_sched}
    log, t0, best, combined_done = [], time.time(), None, False
    fast_stopped, stop_hits, t = False, 0, 0

    def committee(states):
        pop = []
        for i in range(cfg.p):
            m = MLP(dims); m.load_state_dict(states[i]); pop.append(m.to(master))
        return pop

    def push(i, net):
        pool.set_member(i, {k: v.cpu() for k, v in net.state_dict().items()})

    print(f"[parallel-mp] {cfg.p} members split across {ng} GPUs "
          f"(population-parallel): {[str(d) for d in devices]}", flush=True)
    try:
        states, losses = None, None
        for t in range(cfg.outer):
            # --- query the blackbox ---
            _tq = time.time(); _t_gather = 0.0
            if t < cfg.warmstart_iters:
                I = torch.randn(cfg.q, dims[0], generator=gen,
                                device=master) * cfg.qg_init_std
            else:
                _tg = time.time()
                states, losses = pool.report()     # gather committee for query-gen
                _t_gather = time.time() - _tg
                lg = losses if (cfg.gate_kappa > 0 and t > 0) else None
                pop_m = committee(states)
                if qgen_shard:      # large q/model: split across GPUs (no OOM)
                    I = gen_queries_parallel(pop_m, cfg, dims[0], devs, qgens,
                                             losses=lg)
                else:               # small q: single-GPU master is faster
                    I = gen_queries(pop_m, cfg, dims[0], master, gen, losses=lg)
            with torch.no_grad():
                T = teacher(I)
            _t_qgen = time.time() - _tq - _t_gather
            # --- append (replicated to every worker; parent mirrors the full set) ---
            _ta = time.time()
            pool.append(I, T)
            Xc = torch.cat([Xc, I.cpu()]); Yc = torch.cat([Yc, T.cpu()])
            if cfg.window > 0:
                keep = cfg.window * cfg.q; Xc, Yc = Xc[-keep:], Yc[-keep:]
            _t_append = time.time() - _ta
            if t in decay_at:
                pool.decay()
            # --- train the population (each GPU trains its members, no comm) ---
            _tt = time.time()
            pool.train()
            _t_train = time.time() - _tt
            if os.environ.get("MP_TIMING"):
                print(f"  [timing] it {t+1}: gather {_t_gather:.1f}s | qgen "
                      f"{_t_qgen:.1f}s | append {_t_append:.1f}s | train "
                      f"{_t_train:.1f}s", flush=True)

            # --- per-iteration solver polish / closed-form last layer (in the
            #     workers, parallel across GPUs, each member independent) ---
            if cfg.solver_polish:
                pool.polish()
            if cfg.lastlayer_every and (t + 1) % cfg.lastlayer_every == 0:
                pool.lastlayer()

            # --- early stopping (App F signal) ---
            if cfg.stop_loss > 0 and (t + 1) % cfg.log_every == 0:
                states, losses = pool.report()
                bi = min(range(cfg.p), key=lambda i: losses[i])
                bl = losses[bi]
                disp = 0.0
                if cfg.stop_agree > 0:
                    pg = committee(states)
                    disp = max((pg[bi].layers[0].weight - pg[j].layers[0].weight)
                               .abs().max().item()
                               for j in range(cfg.p) if j != bi)
                if bl < cfg.stop_loss and (cfg.stop_agree <= 0 or
                                           disp < cfg.stop_agree):
                    stop_hits += 1
                    if stop_hits >= cfg.stop_patience:
                        print(f"  [early-stop] iter {t + 1}: loss {bl:.2e} "
                              f"disp {disp:.2e}", flush=True)
                        break
                else:
                    stop_hits = 0

            # --- committee maintenance (align/soup/restart), parent-side because
            #     it is cross-member; only the mutated members are pushed back ---
            souped = None
            if cfg.maint_every and (t + 1) % cfg.maint_every == 0 and t >= 5:
                states, losses = pool.report()
                pop = committee(states)
                order = sorted(range(cfg.p), key=lambda i: losses[i])
                best_now = pop[order[0]].clone()
                aligned = [best_now]
                for i in order[1:]:
                    m = pop[i].clone(); align_member_to(best_now, m); aligned.append(m)
                cand = soup_of(aligned)
                closs, wloss = l1_on([cand], Xc, Yc)[0], losses[order[-1]]
                if closs < wloss:
                    push(order[-1], cand.to(master))
                for k in range(min(cfg.restart_worst, cfg.p - 1)):
                    push(order[-1 - k], MLP(dims).to(master))
                souped = closs

            # --- logging / consensus / combine ---
            if (t + 1) % cfg.log_every == 0 or t == cfg.outer - 1:
                states, losses = pool.report()
                pop = committee(states)
                cnet = build_consensus(pop, dims, quorum_ratio=cfg.cluster_quorum)
                cc = param_errors(cnet, teacher)["max_eps"] if cnet is not None else None
                cstats = consensus_neuron_stats(pop, teacher, dims,
                                                quorum_ratio=cfg.cluster_quorum)
                combined_now = None
                if cfg.combine and not combined_done and cnet is not None:
                    polished = polish_consensus(cnet, Xc, Yc,
                                                lr=cfg.combine_polish_lr,
                                                steps=cfg.combine_polish_steps)
                    wi = max(range(cfg.p), key=lambda i: losses[i])
                    push(wi, polished.to(master))
                    combined_done, combined_now = True, t + 1
                    pe = param_errors(polished, teacher)["max_eps"]
                    print(f"  [combine] iter {t+1}: replaced worst member {wi} "
                          f"(loss {losses[wi]:.2e}) -> consensus (max_eps "
                          f"{cc:.2e}->{pe:.2e})", flush=True)
                    states, losses = pool.report(); pop = committee(states)
                bi = min(range(cfg.p), key=lambda i: losses[i])
                best = pop[bi]
                errs = param_errors(best, teacher)
                rec = {"iter": t + 1, "queries": (t + 1) * cfg.q, "best_loss": losses[bi],
                       "med_loss": sorted(losses)[cfg.p // 2], "worst_loss": max(losses),
                       "max_eps": errs["max_eps"],
                       "mean_eps": sum(errs["mean_eps_per_matrix"]) /
                       len(errs["mean_eps_per_matrix"]),
                       "agree": agreement(best, teacher, eval_pts),
                       "wall_s": round(time.time() - t0, 1), "cluster_max_eps": cc}
                if cstats is not None:
                    rec["n_consensus"] = cstats["n_consensus"]; rec["n_total"] = cstats["n_total"]
                    for k in ("l0_max", "l0_mean", "l1_max", "l1_mean",
                              "n_out_consensus", "out_total", "out_max", "out_mean"):
                        if k in cstats:
                            rec["consensus_" + k] = cstats[k]
                if souped is not None:
                    rec["soup_loss"] = souped
                if combined_now is not None:
                    rec["combined_iter"] = combined_now
                log.append(rec)
                cc_str = f"{cc:.2e}" if cc is not None else "  n/a  "
                if cstats is not None and cstats["max_eps"] is not None:
                    cons = (f"{cstats['n_consensus']}/{cstats['n_total']} "
                            f"L0[max {cstats['l0_max']:.2e} mean {cstats['l0_mean']:.2e}] "
                            f"L1[max {cstats['l1_max']:.2e} mean {cstats['l1_mean']:.2e}]")
                    om = (f"max {cstats['out_max']:.2e} mean {cstats['out_mean']:.2e}"
                          if cstats.get("out_max") is not None else "n/a")
                    cons += f" | out {cstats['n_out_consensus']}/{cstats['out_total']} [{om}]"
                else:
                    cons = f"0/{dims[1]}"
                print(f"  it {t+1:3d} | q {(t+1)*cfg.q:6d} | loss {losses[bi]:.2e} | "
                      f"max_eps {errs['max_eps']:.2e} | mean_eps {rec['mean_eps']:.2e} | "
                      f"cluster {cc_str} | cons {cons} | agree {rec['agree']:.4f} | "
                      f"{rec['wall_s']}s", flush=True)

                # --fast: dump committee + queries at the FIRST consensus and stop.
                if cfg.dump_path and cfg.stop_on_consensus and cnet is not None:
                    torch.save({
                        "dims": dims, "iter": t + 1,
                        "pop_states": [states[i] for i in range(cfg.p)],
                        "teacher_state": {k: v.cpu()
                                          for k, v in teacher.state_dict().items()},
                        "X": Xc.clone(), "Y": Yc.clone(),
                    }, cfg.dump_path)
                    print(f"  [dump] iter {t + 1} population + {len(Xc)} queries "
                          f"-> {cfg.dump_path}", flush=True)
                    fast_stopped = True
                    break

            # periodic population snapshot (inspect mid-run; no queries, overwrites)
            if (cfg.pop_save_every and cfg.pop_save_path
                    and (t + 1) % cfg.pop_save_every == 0):
                states, losses = pool.report()     # fresh post-training snapshot
                torch.save({"dims": dims, "iter": t + 1,
                            "pop_states": [states[i] for i in range(cfg.p)],
                            "teacher_state": {k: v.cpu()
                                              for k, v in teacher.state_dict().items()}},
                           cfg.pop_save_path)
                print(f"  [pop-save] iter {t + 1}: {cfg.p} members -> "
                      f"{cfg.pop_save_path}", flush=True)

        # --- final: gather, maintenance-free endgame on master ---
        states, losses = pool.report()
        pop = committee(states)
        bi = min(range(cfg.p), key=lambda i: losses[i])
        best = pop[bi]
        queries_used = (t + 1) * cfg.q
        extra = {}
        if not fast_stopped:
            if save_recon is not None:
                torch.save({"dims": dims,
                            "best_state": {k: v.cpu() for k, v in best.state_dict().items()},
                            "pop_states": [states[i] for i in range(cfg.p)],
                            "teacher_state": {k: v.cpu() for k, v in teacher.state_dict().items()},
                            "X": Xc.clone(), "Y": Yc.clone(), "cfg": asdict(cfg),
                            "seed": seed, "queries": queries_used,
                            "pre_endgame_max_eps": param_errors(best, teacher)["max_eps"]},
                           save_recon)
                print(f"  [save] reconstruction checkpoint -> {save_recon}", flush=True)
            if cfg.lastlayer_every:                 # final closed-form last-layer
                solve_last_layer_(best, Xc, Yc)
            if cfg.popavg_kappa > 0:                # aligned population averaging
                avg, k = aligned_pop_average(pop, losses, cfg.popavg_kappa)
                if avg is not None:
                    avg_loss = l1_on([avg], Xc, Yc)[0]
                    extra["popavg"] = {
                        "k": k, "avg_loss": avg_loss, "best_loss": losses[bi],
                        "avg_max_eps": param_errors(avg, teacher)["max_eps"],
                        "best_max_eps": param_errors(pop[bi], teacher)["max_eps"]}
                    best = avg if avg_loss <= losses[bi] else pop[bi]
                else:
                    extra["popavg"] = {"k": k}
            if cfg.polish_f64:
                best = polish_f64(best, Xc, Yc, cfg, gen)
            if cfg.lbfgs_polish:
                best = polish_lbfgs(best, Xc, Yc, cfg)
    finally:
        pool.stop()

    errs = param_errors(best, teacher)
    final = {"final_max_eps": errs["max_eps"],
             "final_mean_eps": sum(errs["mean_eps_per_matrix"]) /
             len(errs["mean_eps_per_matrix"]),
             "final_max_eps_per_matrix": errs["max_eps_per_matrix"],
             "final_agree": agreement(best, teacher, eval_pts),
             "queries": queries_used, "wall_s": round(time.time() - t0, 1),
             **extra}
    return best, log, final
