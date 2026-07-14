"""Multiprocessing population training: one student per GPU, one process per GPU.

Used by main_parallel.py. Each outer iteration spawns N worker processes (spawn-per-
iteration). Each worker loads its student's weights + optimizer state, trains for the
given number of epochs on the current data, and returns updated weights, updated
optimizer state, and its per-epoch loss trace. The PARENT owns the log file and writes
the (min/max/mean) lines from the returned traces, so logging stays a single coherent
stream even though training runs across processes.

Data path (chunk-wise, no cat): the accumulated dataset is a list of chunk tensors (one
per get_adv call). Workers read the chunk list directly — the chunks are the ONLY copy in
RAM (1x, not 2x). Each batch is composed of contiguous sub-slices drawn from `mix_chunks`
different chunks, so batches mix across chunks (cross-chunk shuffle) while every read stays
contiguous (fast). A background thread assembles batches on the CPU while the GPU trains.

Why processes and not threads for the students: they're independent, but Python's GIL
serializes kernel launches across threads; separate processes have no shared GIL and
scale ~Nx on N GPUs.
"""
import io
import copy
import queue
import threading
import torch
import torch.nn as nn
import torch.optim as optim


def _train_worker(idx, gpu_id, model_bytes, opt_state_bytes, staged, batch_size, epochs, lr, mix_chunks, out_q):
    """Trains one student on one GPU, chunk-wise with cross-chunk mixing.

    `staged` is a list of datasets; each dataset is a list of (X, Y) chunk tensors (CPU,
    shared memory). No dataset is ever concatenated, so memory is 1x the chunks.
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

        def epoch_batches():
            # One epoch: every sample seen once, batches mix up to `mix_chunks` chunks,
            # sub-slices are contiguous, order reshuffled each call.
            for chunks in staged:
                if not chunks:
                    continue
                nmix = min(mix_chunks, len(chunks))
                sub = max(1, batch_size // nmix)             # samples taken from each chunk per batch
                specs = [(ci, s) for ci, (x, _) in enumerate(chunks)
                         for s in range(0, x.shape[0], sub)]
                order = torch.randperm(len(specs)).tolist()
                for g in range(0, len(order), nmix):
                    grp = [specs[order[k]] for k in range(g, min(g + nmix, len(order)))]
                    xb = torch.cat([chunks[ci][0][s:s + sub] for ci, s in grp])   # contiguous sub-slices
                    yb = torch.cat([chunks[ci][1][s:s + sub] for ci, s in grp])
                    yield xb, yb

        def fill(q):
            for b in epoch_batches():
                q.put(b)
            q.put(None)

        trace = []
        for _ in range(epochs):
            q = queue.Queue(maxsize=4)                       # bounded: a few batches resident
            threading.Thread(target=fill, args=(q,), daemon=True).start()   # assemble on CPU, overlap GPU
            ep = torch.zeros((), device=gpu_id)
            nb = 0
            while True:
                item = q.get()
                if item is None:
                    break
                xb, yb = item
                xb = xb.to(gpu_id, non_blocking=True)
                yb = yb.to(gpu_id, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                (loss * 200).backward()
                optimizer.step()
                ep = ep + loss.detach()                      # on-GPU, no sync
                nb += 1
            trace.append(ep / nb)                            # keep GPU scalar
        trace = [t.item() for t in trace]                    # materialize once, at the end

        wbuf = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, wbuf)
        obuf = io.BytesIO()
        torch.save(optimizer.state_dict(), obuf)
        out_q.put((idx, wbuf.getvalue(), obuf.getvalue(), trace, None))
    except Exception:
        import traceback
        out_q.put((idx, None, None, None, traceback.format_exc()))


def train_population(population, gpu_ids, opt_states, batch_size=128, epochs=10, lr=1e-3, mix_chunks=4, log=print):
    """Train every student in `population` for `epochs`, one student per GPU, in parallel.

    Loads returned weights back into population.subs (in place), sets each student's
    .loss, sets population.best, and writes per-epoch (min/max/mean) log lines via `log`.
    Returns the updated list of per-student optimizer-state bytes (pass back next call to
    keep Adam momentum; pass a list of None to reset it, e.g. on an lr change).
    """
    import torch.multiprocessing as mp

    # Gather the chunk lists (no cat). FNN => one dataset (population.inputs); seq models
    # => one per sequence length (population.inputs_dict).
    datasets_chunks = []
    if population.inputs:
        datasets_chunks.append(list(zip(population.inputs, population.outputs)))
    for k in list(population.inputs_dict.keys()):
        datasets_chunks.append(list(zip(population.inputs_dict[k], population.outputs_dict[k])))
    if not datasets_chunks:
        raise Exception("no datasets")

    subs = population.subs
    K = len(subs)
    if K != len(gpu_ids):
        raise ValueError(f"need one student per GPU: {K} students, {len(gpu_ids)} gpus")

    # share each chunk to workers (parent stays alive during training, so fd-sharing is safe)
    staged = []
    for chunks in datasets_chunks:
        shared = []
        for x, y in chunks:
            xi = x.detach().to('cpu').contiguous()
            yi = y.detach().to('cpu').contiguous()
            xi.share_memory_(); yi.share_memory_()
            shared.append((xi, yi))
        staged.append(shared)

    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = []
    for i in range(K):
        mbuf = io.BytesIO()
        torch.save(copy.deepcopy(subs[i]).to('cpu'), mbuf)   # full model (handles any arch)
        procs.append(ctx.Process(
            target=_train_worker,
            args=(i, gpu_ids[i], mbuf.getvalue(), opt_states[i], staged,
                  batch_size, epochs, lr, mix_chunks, q),
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
