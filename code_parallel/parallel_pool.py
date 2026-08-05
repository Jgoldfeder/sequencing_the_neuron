"""Multiprocessing population parallelism: one persistent worker process per GPU,
each holding its committee members + optimizers + a replica of the query buffer.
Workers train in parallel (separate interpreters -> no GIL contention, real Nx).
The main process does the coupled/cheap parts (query generation, consensus,
combine, endgame) on committee weights gathered from the workers each iter.

Used by reconstruct_mp; run.py routes multi-GPU runs here.
"""
import os
import time
from dataclasses import asdict

import torch
import torch.multiprocessing as mp

from nets import MLP


# Keep the query buffer on the GPU only while it's comfortably small; above this
# many GB, stream minibatches from the shared CPU buffer instead (avoids the
# per-GPU OOM on large arch / large q -- e.g. 12288-in, q=24000, window=60 is
# ~70GB, which cannot live on a GPU). Tunable via env for testing.
_GPU_CACHE_BYTES = float(os.environ.get("MP_GPU_CACHE_GB", "6")) * (1024 ** 3)


# ----------------------------------------------------------------- worker --
def _worker(g, dev_str, dims, cfg_d, member_ids, seed, Xbuf, Ybuf, cmd_q, res_q):
    """One worker per GPU. The query buffer is a SINGLE shared CPU tensor
    (Xbuf/Ybuf) that the parent fills; workers never keep their own copy (that was
    the OOM -- N GPUs x the whole buffer). Small buffers are cached on the GPU
    once per train for speed; large buffers stream minibatches from host."""
    dev = torch.device(dev_str)
    torch.cuda.set_device(dev)
    epochs, batch = cfg_d["epochs"], cfg_d["batch"]
    fit_loss, lr = cfg_d["fit_loss"], cfg_d["lr"]
    members, opts = {}, {}
    for i in member_ids:
        torch.manual_seed(seed * 100003 + i)
        m = MLP(dims).to(dev)
        members[i] = m
        opts[i] = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator(device=dev).manual_seed(seed * 7 + g)
    cap = Xbuf.shape[0]
    cache = cap * dims[0] * 4 < _GPU_CACHE_BYTES   # keep whole buffer on GPU?

    while True:
        op, payload = cmd_q.get()
        if op == "stop":
            break
        elif op == "decay":
            for o in opts.values():
                for pg in o.param_groups:
                    pg["lr"] /= 10
            res_q.put((g, None))
        elif op == "train":
            n = payload                            # valid length of the buffer
            # same scheme as the 1-GPU path (method.py): keep the buffer on the
            # GPU when it fits, else stream minibatches from the shared CPU buffer.
            Xt = Xbuf[:n].to(dev) if cache else Xbuf
            Yt = Ybuf[:n].to(dev) if cache else Ybuf
            for i in member_ids:
                net, opt = members[i], opts[i]
                for _ in range(epochs):
                    perm = torch.randperm(n, generator=gen, device=dev)
                    if not cache:
                        perm = perm.cpu()          # index the CPU buffer
                    for b in range(0, n, batch):
                        idx = perm[b:b + batch]
                        if cache:
                            xb, yb = Xt[idx], Yt[idx]
                        else:
                            xb = Xt[idx].to(dev, non_blocking=True)
                            yb = Yt[idx].to(dev, non_blocking=True)
                        opt.zero_grad()
                        r = net(xb) - yb
                        loss = (r * r).mean() if fit_loss == "mse" else r.abs().mean()
                        loss.backward()
                        opt.step()
            if cache:
                del Xt, Yt
            torch.cuda.synchronize(dev)
            res_q.put((g, None))
        elif op == "report":                      # state_dicts (cpu) + train L1
            n = payload
            out = {}
            with torch.no_grad():
                for i in member_ids:
                    net = members[i]
                    tot, nn = 0.0, 0
                    for b in range(0, n, 4096):
                        xb = Xbuf[b:b + 4096].to(dev); yb = Ybuf[b:b + 4096].to(dev)
                        r = net(xb) - yb
                        tot += r.abs().sum().item(); nn += r.numel()
                    st = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                    out[i] = (st, tot / max(nn, 1))
            res_q.put((g, out))
        elif op == "set":                          # inject a member (combine)
            i, state = payload
            members[i].load_state_dict(state)
            members[i].to(dev)
            opts[i] = torch.optim.Adam(members[i].parameters(),
                                       lr=opts[i].param_groups[0]["lr"])
            res_q.put((g, None))


# ------------------------------------------------------------------- pool --
class WorkerPool:
    def __init__(self, devices, dims, cfg, seed):
        ctx = mp.get_context("spawn")
        self.ng = len(devices)
        self.p = cfg.p
        self.owner = {i: i % self.ng for i in range(cfg.p)}
        member_ids = [[i for i in range(cfg.p) if i % self.ng == g]
                      for g in range(self.ng)]
        cfg_d = asdict(cfg)
        # ONE shared CPU query buffer, filled by the parent, read by all workers.
        # Fixed capacity = the retained window (or the whole run if no window);
        # a ring so windowed runs overwrite the oldest queries in place. This is
        # the single copy that replaces the old N-GPUs x whole-buffer replication.
        self.cap = (cfg.window if cfg.window > 0 else cfg.outer) * cfg.q
        self.Xbuf = torch.empty(self.cap, dims[0]).share_memory_()
        self.Ybuf = torch.empty(self.cap, dims[-1]).share_memory_()
        self.count = 0
        self.cmd = [ctx.Queue() for _ in range(self.ng)]
        self.res = ctx.Queue()
        self.procs = [ctx.Process(target=_worker,
                                  args=(g, str(devices[g]), dims, cfg_d,
                                        member_ids[g], seed, self.Xbuf, self.Ybuf,
                                        self.cmd[g], self.res),
                                  daemon=True)
                      for g in range(self.ng)]
        for pr in self.procs:
            pr.start()

    def valid(self):
        return min(self.count, self.cap)

    def _all(self, op, payloads=None):
        for g in range(self.ng):
            self.cmd[g].put((op, None if payloads is None else payloads[g]))
        res = {}
        for _ in range(self.ng):
            g, r = self.res.get()
            res[g] = r
        return res

    def append(self, I, T):
        # write into the shared ring buffer (workers are idle here -> no race)
        nq = I.shape[0]
        wpos = self.count % self.cap                  # aligned: cap is a mult of q
        self.Xbuf[wpos:wpos + nq].copy_(I.detach().cpu())
        self.Ybuf[wpos:wpos + nq].copy_(T.detach().cpu())
        self.count += nq

    def decay(self):
        self._all("decay")

    def train(self):
        self._all("train", [self.valid()] * self.ng)

    def report(self):
        res = self._all("report", [self.valid()] * self.ng)
        states, losses = {}, {}
        for out in res.values():
            for i, (st, lv) in out.items():
                states[i] = st; losses[i] = lv
        return states, [losses[i] for i in range(self.p)]

    def set_member(self, i, state):
        self.cmd[self.owner[i]].put(("set", (i, state)))
        self.res.get()

    def stop(self):
        for g in range(self.ng):
            self.cmd[g].put(("stop", None))
        for pr in self.procs:
            pr.join(timeout=10)


# --------------------------------------------------------- reconstruct_mp --
def reconstruct_mp(teacher, dims, cfg, devices, eval_pts, seed=0, save_recon=None):
    """Multiprocess population-parallel reconstruction. Training runs in per-GPU
    worker processes; query-gen / consensus / combine / endgame run in the main
    process on gathered committee weights."""
    from method import (gen_queries_parallel, build_consensus,
                        consensus_neuron_stats, polish_consensus, polish_lbfgs,
                        polish_f64, agreement)
    from align import param_errors

    devs = [torch.device(d) for d in devices]
    master = devs[0]
    teacher = teacher.to(master)
    eval_pts = eval_pts.to(master)
    gen = torch.Generator(device=master).manual_seed(seed)
    qgens = [torch.Generator(device=d).manual_seed(seed * 131 + k)
             for k, d in enumerate(devs)]
    pool = WorkerPool(devices, dims, cfg, seed)
    decay_at = {int(s * cfg.outer) for s in cfg.lr_sched}
    log, t0, best, combined_done = [], time.time(), None, False
    fast_stopped = False

    def committee(states):
        pop = []
        for i in range(cfg.p):
            m = MLP(dims); m.load_state_dict(states[i]); pop.append(m.to(master))
        return pop

    # views into the single shared CPU query buffer (no separate parent copy)
    def Xc_():
        return pool.Xbuf[:pool.valid()]

    def Yc_():
        return pool.Ybuf[:pool.valid()]

    print(f"[parallel-mp] {cfg.p} members across {len(devices)} GPUs: "
          f"{[str(d) for d in devices]}", flush=True)
    for t in range(cfg.outer):
        _tg = time.time()
        states, losses = pool.report()             # current members + train L1
        _t_gather = time.time() - _tg
        _tq = time.time()
        # --- query the blackbox ---
        if t < cfg.warmstart_iters:
            I = torch.randn(cfg.q, dims[0], generator=gen,
                            device=master) * cfg.qg_init_std
        else:
            lg = losses if cfg.gate_kappa > 0 else None
            I = gen_queries_parallel(committee(states), cfg, dims[0], devs,
                                     qgens, losses=lg)
        with torch.no_grad():
            T = teacher(I)
        _t_qgen = time.time() - _tq
        pool.append(I, T)          # writes the shared buffer (windowing = ring)
        if t in decay_at:
            pool.decay()
        pool.train()

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
                polished = polish_consensus(cnet, Xc_(), Yc_(),
                                            lr=cfg.combine_polish_lr,
                                            steps=cfg.combine_polish_steps)
                wi = max(range(cfg.p), key=lambda i: losses[i])
                pool.set_member(wi, {k: v.cpu()
                                     for k, v in polished.state_dict().items()})
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

            # --fast: dump committee + queries at the FIRST consensus and stop;
            # run.py runs the staged MSE->MAE consensus solve on the dump. Same
            # trigger/payload as serial reconstruct (method.py) so --fast behaves
            # identically under --gpus. cnet uses cluster_quorum (=0.625), which
            # matches the 0.625 consensus run.py rebuilds from the dump.
            if cfg.dump_path and cfg.stop_on_consensus and cnet is not None:
                # .clone() so torch.save writes only the valid rows, not the whole
                # (possibly huge) shared ring-buffer storage the view points into.
                torch.save({
                    "dims": dims, "iter": t + 1,
                    "pop_states": [states[i] for i in range(cfg.p)],
                    "teacher_state": {k: v.cpu()
                                      for k, v in teacher.state_dict().items()},
                    "X": Xc_().clone(), "Y": Yc_().clone(),
                }, cfg.dump_path)
                print(f"  [dump] iter {t + 1} population + {pool.valid()} queries "
                      f"-> {cfg.dump_path}", flush=True)
                fast_stopped = True
                break

    # --- final: gather, endgame on master ---
    states, losses = pool.report()
    pop = committee(states)
    bi = min(range(cfg.p), key=lambda i: losses[i])
    best = pop[bi]
    queries_used = (t + 1) * cfg.q if fast_stopped else cfg.outer * cfg.q
    # --fast stopped at first consensus: skip the endgame here -- run.py loads the
    # dump, rebuilds the consensus, runs the staged MSE->MAE solve, and overwrites
    # best/final. Running polish here would just be discarded work.
    if not fast_stopped:
        if save_recon is not None:
            torch.save({"dims": dims,
                        "best_state": {k: v.cpu() for k, v in best.state_dict().items()},
                        "pop_states": [states[i] for i in range(cfg.p)],
                        "teacher_state": {k: v.cpu() for k, v in teacher.state_dict().items()},
                        "X": Xc_().clone(), "Y": Yc_().clone(), "cfg": asdict(cfg),
                        "seed": seed, "queries": queries_used,
                        "pre_endgame_max_eps": param_errors(best, teacher)["max_eps"]},
                       save_recon)
            print(f"  [save] reconstruction checkpoint -> {save_recon}", flush=True)
        if cfg.polish_f64:
            best = polish_f64(best, Xc_(), Yc_(), cfg, gen)
        if cfg.lbfgs_polish:
            best = polish_lbfgs(best, Xc_(), Yc_(), cfg)
    pool.stop()
    errs = param_errors(best, teacher)
    final = {"final_max_eps": errs["max_eps"],
             "final_mean_eps": sum(errs["mean_eps_per_matrix"]) /
             len(errs["mean_eps_per_matrix"]),
             "final_max_eps_per_matrix": errs["max_eps_per_matrix"],
             "final_agree": agreement(best, teacher, eval_pts),
             "queries": queries_used, "wall_s": round(time.time() - t0, 1)}
    return best, log, final
