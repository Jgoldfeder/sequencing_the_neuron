"""Multiprocessing population training: one student per GPU, one process per GPU.

Used by main_parallel.py. Each outer iteration spawns N worker processes (spawn-per-
iteration). Each worker loads its student's weights + optimizer state, trains for the
given number of epochs on the current data, and returns updated weights, updated
optimizer state, and its per-epoch loss trace. The PARENT owns the log file and writes
the (min/max/mean) lines from the returned traces, so logging stays a single coherent
stream even though training runs across processes.

Why processes and not threads: the students are independent, but Python's GIL serializes
kernel launches across threads, so a threaded version only reaches ~1.5x on small models.
Separate processes have no shared GIL and scale ~Nx on N GPUs.
"""
import io
import copy
import torch
import torch.nn as nn
import torch.optim as optim


def _train_worker(idx, gpu_id, model_bytes, opt_state_bytes, staged, batch_size, epochs, lr, out_q):
    """Runs in its own process. Trains one student on one GPU.

    Loss accumulation stays on-GPU and is materialized once at the end (no per-batch
    sync), so logging adds no measurable overhead.
    """
    try:
        torch.cuda.set_device(gpu_id)
        model = torch.load(io.BytesIO(model_bytes), map_location='cpu', weights_only=False).cuda(gpu_id)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if opt_state_bytes is not None:                      # resume Adam momentum
            state = torch.load(io.BytesIO(opt_state_bytes), map_location='cpu')
            optimizer.load_state_dict(state)                 # moves state to param device
            for g in optimizer.param_groups:
                g['lr'] = lr
        criterion = nn.L1Loss()

        # Stage the whole dataset on-GPU when it fits — fastest, full GPU utilization.
        # Otherwise stream batches from CPU so oversized / high-dimensional data (e.g.
        # TinyImageNet's 12288-dim inputs, tens of GB accumulated) doesn't OOM. Streaming is
        # ~2x slower (moving the dataset every epoch is memory-bandwidth-bound), but it's the
        # only option once the dataset exceeds GPU memory. Small/medium datasets pay nothing.
        dataset_bytes = sum(x.element_size() * x.nelement() + y.element_size() * y.nelement()
                            for x, y in staged)
        free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
        stage_on_gpu = dataset_bytes < 0.4 * free_bytes      # leave headroom for model/grads/acts
        if stage_on_gpu:
            staged = [(x.to(gpu_id), y.to(gpu_id)) for x, y in staged]
        num_batches = sum((x.shape[0] + batch_size - 1) // batch_size for x, _ in staged)

        trace = []
        for _ in range(epochs):
            ep = torch.zeros((), device=gpu_id)
            for X, Y in staged:
                n = X.shape[0]
                perm = torch.randperm(n, device=(gpu_id if stage_on_gpu else "cpu"))
                for b in range(0, n, batch_size):
                    sel = perm[b:b + batch_size]
                    if stage_on_gpu:
                        xb, yb = X[sel], Y[sel]              # already on GPU
                    else:
                        xb = X[sel].to(gpu_id, non_blocking=True)   # stream one batch
                        yb = Y[sel].to(gpu_id, non_blocking=True)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    (loss * 200).backward()
                    optimizer.step()
                    ep = ep + loss.detach()                  # on-GPU, no sync
            trace.append(ep)                                 # keep GPU scalar
        trace = [(t / num_batches).item() for t in trace]    # materialize once, at the end

        wbuf = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, wbuf)
        obuf = io.BytesIO()
        torch.save(optimizer.state_dict(), obuf)
        out_q.put((idx, wbuf.getvalue(), obuf.getvalue(), trace, None))
    except Exception:
        import traceback
        out_q.put((idx, None, None, None, traceback.format_exc()))


def train_population(population, gpu_ids, opt_states, batch_size=128, epochs=10, lr=1e-3, log=print):
    """Train every student in `population` for `epochs`, one student per GPU, in parallel.

    Loads returned weights back into population.subs (in place), sets each student's
    .loss, sets population.best, and writes per-epoch (min/max/mean) log lines via `log`.

    Returns the updated list of per-student optimizer-state bytes (pass back in next call
    to keep Adam momentum; pass a list of None to reset it, e.g. on an lr change).
    """
    import torch.multiprocessing as mp

    # mirror train_one_epoch's dataset check
    if population.ds is not None and len(population.datasets) == 0:
        population.datasets[0] = population.ds
    if len(population.datasets) == 0:
        raise Exception("no datasets")

    subs = population.subs
    K = len(subs)
    if K != len(gpu_ids):
        raise ValueError(f"need one student per GPU: {K} students, {len(gpu_ids)} gpus")

    # stage datasets in shared memory (parent stays alive during training, so the
    # file-descriptor sharing that torch uses for the INPUT direction is safe)
    staged = []
    for ds in population.datasets.values():
        xi = ds.inputs.detach().to('cpu').contiguous()
        yi = ds.outputs.detach().to('cpu').contiguous()
        xi.share_memory_(); yi.share_memory_()
        staged.append((xi, yi))

    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = []
    for i in range(K):
        mbuf = io.BytesIO()
        torch.save(copy.deepcopy(subs[i]).to('cpu'), mbuf)   # full model (handles any arch)
        procs.append(ctx.Process(
            target=_train_worker,
            args=(i, gpu_ids[i], mbuf.getvalue(), opt_states[i], staged, batch_size, epochs, lr, q),
        ))
    for p in procs:
        p.start()
    raw = [q.get() for _ in procs]                            # weights come back as bytes
    for p in procs:
        p.join()

    results = {}
    for idx, wbytes, obytes, trace, err in raw:
        if err is not None:
            raise RuntimeError(f"training worker {idx} (gpu {gpu_ids[idx]}) failed:\n{err}")
        results[idx] = (wbytes, obytes, trace)

    new_opt_states = list(opt_states)
    traces = []
    for i in range(K):
        wbytes, obytes, trace = results[i]
        subs[i].load_state_dict(torch.load(io.BytesIO(wbytes), map_location='cpu'))
        subs[i].loss = trace[-1]
        new_opt_states[i] = obytes
        traces.append(trace)

    # single coherent log, written by the parent (same format as train_one_epoch)
    for e in range(epochs):
        vals = [traces[i][e] for i in range(K)]
        log(f"Epoch {e+1}, Min Loss: {min(vals)}, Max Loss: {max(vals)},Mean Loss: {sum(vals)/len(vals)}")

    population.best = min(range(K), key=lambda i: traces[i][-1])
    return new_opt_states
