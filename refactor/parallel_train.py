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
from torch.utils.data import DataLoader, Dataset


class _BatchSlices(Dataset):
    """Dataset that returns contiguous batch-sized slices. Used with
    DataLoader(batch_size=None) so each item is a fast contiguous view of the data rather
    than 128 per-sample gathers (which thrash memory when several workers read a big shared
    tensor at once). shuffle=True then shuffles the block order."""

    def __init__(self, X, Y, batch_size):
        self.X, self.Y, self.batch_size = X, Y, batch_size

    def __len__(self):
        return (self.X.shape[0] + self.batch_size - 1) // self.batch_size

    def __getitem__(self, i):
        s = i * self.batch_size
        return self.X[s:s + self.batch_size], self.Y[s:s + self.batch_size]


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

        # Standard PyTorch input pipeline. _BatchSlices + batch_size=None makes each
        # DataLoader item a CONTIGUOUS batch-sized slice (not 128 per-sample gathers);
        # shuffle=True shuffles block order; pin_memory + a background pin thread overlap
        # the host->device transfer with GPU compute. Handles datasets of ANY size (only a
        # few batches resident, so no OOM even for tens-of-GB accumulated data).
        loaders = [DataLoader(_BatchSlices(X, Y, batch_size), batch_size=None, shuffle=True,
                              pin_memory=True, num_workers=4, persistent_workers=True,
                              prefetch_factor=4)
                   for X, Y in staged]
        num_batches = sum(len(l) for l in loaders)

        trace = []
        for _ in range(epochs):
            ep = torch.zeros((), device=gpu_id)
            for loader in loaders:
                for xb, yb in loader:
                    xb = xb.to(gpu_id, non_blocking=True)
                    yb = yb.to(gpu_id, non_blocking=True)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    (loss * 200).backward()
                    optimizer.step()
                    ep = ep + loss.detach()                  # on-GPU, no sync
            trace.append(ep / num_batches)                   # keep GPU scalar
        trace = [t.item() for t in trace]                    # materialize once, at the end

        wbuf = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, wbuf)
        obuf = io.BytesIO()
        torch.save(optimizer.state_dict(), obuf)
        out_q.put((idx, wbuf.getvalue(), obuf.getvalue(), trace, None))
    except Exception:
        import traceback
        out_q.put((idx, None, None, None, traceback.format_exc()))


def _grouped_worker(idx, gpu_id, model_bytes, opt_state_bytes, batch_size, lr, ctrl_q, out_q):
    """Persistent per-GPU worker for out-of-core training.

    Spawned ONCE and kept alive across all groups/epochs. Waits on ctrl_q for group commands
    from the parent; each command carries a shared-memory (X, Y) group the parent just loaded
    from disk. Trains one pass over the group with the fast pinned DataLoader, releases its
    references to the group, and acks so the parent can free it and load the next. On 'done'
    it sends back final weights + optimizer state + per-epoch loss trace.
    """
    try:
        torch.cuda.set_device(gpu_id)
        model = torch.load(io.BytesIO(model_bytes), map_location='cpu', weights_only=False).cuda(gpu_id)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if opt_state_bytes is not None:                      # resume Adam momentum
            state = torch.load(io.BytesIO(opt_state_bytes), map_location='cpu')
            optimizer.load_state_dict(state)
            for g in optimizer.param_groups:
                g['lr'] = lr
        criterion = nn.L1Loss()

        trace = []
        ep = torch.zeros((), device=gpu_id)
        nb = 0
        while True:
            cmd = ctrl_q.get()
            if cmd[0] == 'done':
                wbuf = io.BytesIO()
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, wbuf)
                obuf = io.BytesIO()
                torch.save(optimizer.state_dict(), obuf)
                out_q.put(('result', idx, wbuf.getvalue(), obuf.getvalue(), trace, None))
                return
            # ('group', X, Y, end_epoch): train one pass over this shared group
            _, X, Y, end_epoch = cmd
            # num_workers=4 + prefetch so batch loading/pinning overlaps GPU compute (num_workers=0
            # gives NO prefetch -> the GPU stalls on every batch; see commit cb9ea2f). Fresh loader
            # per group, so persistent_workers=False.
            loader = DataLoader(_BatchSlices(X, Y, batch_size), batch_size=None, shuffle=True,
                                pin_memory=True, num_workers=4, persistent_workers=False,
                                prefetch_factor=4)
            for xb, yb in loader:
                xb = xb.to(gpu_id, non_blocking=True)
                yb = yb.to(gpu_id, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                (loss * 200).backward()
                optimizer.step()
                ep = ep + loss.detach()
                nb += 1
            del loader, X, Y                                 # release shared-mem refs BEFORE ack
            if end_epoch:
                trace.append((ep / nb).item())
                ep = torch.zeros((), device=gpu_id)
                nb = 0
            out_q.put(('ack', idx))                          # barrier signal to parent
    except Exception:
        import traceback
        out_q.put(('result', idx, None, None, None, traceback.format_exc()))


def train_population_grouped(population, gpu_ids, opt_states, max_chunks_in_mem,
                             batch_size=128, epochs=10, lr=1e-3, log=print):
    """Out-of-core training: all chunks live on disk (population.chunk_dir); the parent streams
    them back one GROUP of `max_chunks_in_mem` chunks at a time into a single shared contiguous
    array, all GPUs train that group in parallel, then a barrier frees it and the next group
    loads. Peak RAM is ~1 group, independent of the accumulated window size. Same return
    contract as train_population.
    """
    import torch.multiprocessing as mp

    chunk_paths = list(zip(population.inputs, population.outputs))   # (px, py) file paths
    if not chunk_paths:
        raise Exception("no datasets")
    subs = population.subs
    K = len(subs)
    if K != len(gpu_ids):
        raise ValueError(f"need one student per GPU: {K} students, {len(gpu_ids)} gpus")
    M = max_chunks_in_mem if max_chunks_in_mem and max_chunks_in_mem > 0 else len(chunk_paths)

    ctx = mp.get_context('spawn')
    out_q = ctx.Queue()
    ctrl_qs = [ctx.Queue() for _ in range(K)]
    procs = []
    for i in range(K):
        mbuf = io.BytesIO()
        torch.save(copy.deepcopy(subs[i]).to('cpu'), mbuf)
        procs.append(ctx.Process(target=_grouped_worker,
                                 args=(i, gpu_ids[i], mbuf.getvalue(), opt_states[i],
                                       batch_size, lr, ctrl_qs[i], out_q)))
    for p in procs:
        p.start()

    try:
        n_groups = (len(chunk_paths) + M - 1) // M
        for e in range(epochs):
            order = torch.randperm(len(chunk_paths)).tolist()   # reshuffle chunk->group each epoch
            for gi in range(n_groups):
                grp = order[gi * M:(gi + 1) * M]
                xs = [torch.load(chunk_paths[j][0]) for j in grp]   # load M chunks from disk
                ys = [torch.load(chunk_paths[j][1]) for j in grp]
                X = torch.cat(xs).contiguous()                     # one contiguous array (1x group)
                Y = torch.cat(ys).contiguous()
                del xs, ys
                X.share_memory_(); Y.share_memory_()               # zero-copy hand-off to all workers
                end_epoch = (gi == n_groups - 1)
                for i in range(K):
                    ctrl_qs[i].put(('group', X, Y, end_epoch))
                acks = 0                                            # barrier: every GPU finishes this group
                while acks < K:
                    msg = out_q.get()
                    if msg[0] == 'ack':
                        acks += 1
                    elif msg[0] == 'result' and msg[5] is not None:
                        raise RuntimeError(f"training worker {msg[1]} failed:\n{msg[5]}")
                del X, Y                                            # free the shared group before the next
        for i in range(K):
            ctrl_qs[i].put(('done',))
        raw = [out_q.get() for _ in range(K)]
    finally:
        for p in procs:
            p.join()

    results = {}
    for msg in raw:
        _, idx, wbytes, obytes, trace, err = msg
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

    for e in range(epochs):
        vals = [traces[i][e] for i in range(K)]
        log(f"Epoch {e+1}, Min Loss: {min(vals)}, Max Loss: {max(vals)},Mean Loss: {sum(vals)/len(vals)}")
    population.best = min(range(K), key=lambda i: traces[i][-1])
    return new_opt_states


def train_population(population, gpu_ids, opt_states, batch_size=128, epochs=10, lr=1e-3,
                     max_chunks_in_mem=0, log=print):
    """Train every student in `population` for `epochs`, one student per GPU, in parallel.

    Loads returned weights back into population.subs (in place), sets each student's
    .loss, sets population.best, and writes per-epoch (min/max/mean) log lines via `log`.

    Returns the updated list of per-student optimizer-state bytes (pass back in next call
    to keep Adam momentum; pass a list of None to reset it, e.g. on an lr change).
    """
    import torch.multiprocessing as mp

    # Out-of-core: chunks are on disk (population.chunk_dir) -> stream them in groups.
    if getattr(population, "chunk_dir", None):
        return train_population_grouped(population, gpu_ids, opt_states, max_chunks_in_mem,
                                        batch_size=batch_size, epochs=epochs, lr=lr, log=log)

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
