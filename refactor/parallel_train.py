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
import os
import copy
import numpy as np
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


class _ChunkListBlocks(Dataset):
    """Contiguous batch-blocks over a LIST of in-RAM (shared-memory) chunk tensors. Exactly
    _BatchSlices, but without concatenating the chunks into one tensor -- so RAM is 1x the window
    (no cat doubling). Each item is a contiguous view chunks[ci][s:s+bs]; DataLoader(shuffle=True)
    permutes block order across all chunks (cross-chunk mixing), every read stays contiguous.
    """

    def __init__(self, chunks, batch_size):
        self.chunks = chunks                                # list of (X, Y) shared tensors
        self.bs = batch_size
        self.index = [(ci, s) for ci, (x, _) in enumerate(chunks)
                      for s in range(0, x.shape[0], batch_size)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ci, s = self.index[i]
        x, y = self.chunks[ci]
        return x[s:s + self.bs], y[s:s + self.bs]


def _disk_worker(idx, gpu_id, model_bytes, opt_state_bytes, chunks, batch_size, epochs, lr, out_q):
    """Per-GPU worker. Fed a LIST of shared-memory chunk tensors (the window, loaded from disk
    by the parent).

    Two fast paths (these inputs are ~49 KB each, so the host->device copy dominates):
      - window fits in VRAM -> STAGE it on the GPU, train from device memory. No transfer at
        all, compute-bound, scales ~Nx (measured 933k on 3x3090).
      - window too big for VRAM -> pin the ONE shared copy in place (cudaHostRegister, no
        duplication) and DMA-stream contiguous batch VIEWS, double-buffered on a side stream so
        the next batch's copy hides under the current batch's compute. ~80% of staged (738k on
        3x3090) with the data resident in RAM, not VRAM. This is the key path: a window far
        larger than VRAM trains at near-staged speed. (A plain per-batch-pinned DataLoader is
        ~1/3 this because it re-copies every batch into pinned memory on the CPU.)
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

        dbytes = sum(x.numel() * x.element_size() + y.numel() * y.element_size() for x, y in chunks)
        free, _ = torch.cuda.mem_get_info(gpu_id)
        trace = []

        if dbytes < 0.5 * free:
            # STAGE ON GPU: move the window into VRAM once, train from device memory (no per-batch
            # transfer). Contiguous batch blocks in shuffled block order.
            X = torch.cat([c[0].to(gpu_id, non_blocking=True) for c in chunks])
            Y = torch.cat([c[1].to(gpu_id, non_blocking=True) for c in chunks])
            starts = list(range(0, X.shape[0], batch_size))
            for _ in range(epochs):
                ep = torch.zeros((), device=gpu_id)
                nb = 0
                for bi in torch.randperm(len(starts)).tolist():
                    s = starts[bi]
                    xb = X[s:s + batch_size]
                    yb = Y[s:s + batch_size]
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    (loss * 200).backward()
                    optimizer.step()
                    ep = ep + loss.detach()
                    nb += 1
                trace.append((ep / nb).item())
        else:
            # PINNED-VIEW STREAM: window too big for VRAM but resident in RAM. Register the ONE
            # shared copy as pinned IN PLACE (cudaHostRegister -> no duplication, RAM stays 1x),
            # then stream contiguous batch VIEWS to the GPU as DMA transfers with NO per-batch CPU
            # copy. Each next batch is copied on a side stream while the current one computes, so
            # the transfer hides under compute. Measured ~80% of staged throughput (738k vs 933k
            # on 3x3090) with the data in RAM, not VRAM.
            cudart = torch.cuda.cudart()
            for x, y in chunks:
                cudart.cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), 0)
                cudart.cudaHostRegister(y.data_ptr(), y.numel() * y.element_size(), 0)
            index = [(ci, s) for ci, (x, _) in enumerate(chunks)
                     for s in range(0, x.shape[0], batch_size)]
            copy_stream = torch.cuda.Stream(gpu_id)

            def fetch(j):
                ci, s = index[j]
                x, y = chunks[ci]
                with torch.cuda.stream(copy_stream):
                    return (x[s:s + batch_size].to(gpu_id, non_blocking=True),
                            y[s:s + batch_size].to(gpu_id, non_blocking=True))

            try:
                for _ in range(epochs):
                    order = torch.randperm(len(index)).tolist()
                    nxt = fetch(order[0])
                    ep = torch.zeros((), device=gpu_id)
                    nb = 0
                    for k in range(len(order)):
                        torch.cuda.current_stream(gpu_id).wait_stream(copy_stream)  # await this batch's copy
                        xb, yb = nxt
                        if k + 1 < len(order):
                            nxt = fetch(order[k + 1])                                # prefetch next while we compute
                        optimizer.zero_grad()
                        loss = criterion(model(xb), yb)
                        (loss * 200).backward()
                        optimizer.step()
                        ep = ep + loss.detach()
                        nb += 1
                    trace.append((ep / nb).item())
            finally:
                for x, y in chunks:
                    cudart.cudaHostUnregister(x.data_ptr())
                    cudart.cudaHostUnregister(y.data_ptr())

        wbuf = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, wbuf)
        obuf = io.BytesIO()
        torch.save(optimizer.state_dict(), obuf)
        out_q.put((idx, wbuf.getvalue(), obuf.getvalue(), trace, None))
    except Exception:
        import traceback
        out_q.put((idx, None, None, None, traceback.format_exc()))


class _MmapChunkBlocks(Dataset):
    """Contiguous batch-blocks over memory-mapped chunk files, for windows too big for RAM.
    Only paths + a block index live in RAM; the mmaps page in on access, and the OS page cache
    holds whatever fits. Nothing is fully loaded, so it never OOMs -- speed is bounded by disk
    bandwidth (this is the only genuinely disk-bound regime)."""

    def __init__(self, chunk_paths, batch_size):
        self.paths = list(chunk_paths)
        self.bs = batch_size
        self.index = []
        for ci, (px, _py) in enumerate(self.paths):
            n = torch.load(px, mmap=True).shape[0]
            for s in range(0, n, batch_size):
                self.index.append((ci, s))
        self._mmaps = None

    def _open(self):
        if self._mmaps is None:
            self._mmaps = [(torch.load(px, mmap=True), torch.load(py, mmap=True))
                           for px, py in self.paths]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        self._open()
        ci, s = self.index[i]
        x, y = self._mmaps[ci]
        return x[s:s + self.bs], y[s:s + self.bs]


def _mmapstream_worker(idx, gpu_id, model_bytes, opt_state_bytes, chunk_paths, batch_size, epochs, lr, out_q):
    """Per-GPU worker for the window-exceeds-RAM case: stream batches from mmap'd files. Never
    loads the window into RAM (no OOM); disk-bandwidth-bound. pin_memory=False (pinning serializes
    across GPU processes); num_workers=4 prefetch overlaps disk reads with compute.
    """
    try:
        torch.cuda.set_device(gpu_id)
        model = torch.load(io.BytesIO(model_bytes), map_location='cpu', weights_only=False).cuda(gpu_id)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if opt_state_bytes is not None:
            state = torch.load(io.BytesIO(opt_state_bytes), map_location='cpu')
            optimizer.load_state_dict(state)
            for g in optimizer.param_groups:
                g['lr'] = lr
        criterion = nn.L1Loss()
        loader = DataLoader(_MmapChunkBlocks(chunk_paths, batch_size), batch_size=None, shuffle=True,
                            pin_memory=False, num_workers=4, persistent_workers=True, prefetch_factor=4)
        trace = []
        for _ in range(epochs):
            ep = torch.zeros((), device=gpu_id)
            nb = 0
            for xb, yb in loader:
                xb = xb.to(gpu_id, non_blocking=True)
                yb = yb.to(gpu_id, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                (loss * 200).backward()
                optimizer.step()
                ep = ep + loss.detach()
                nb += 1
            trace.append((ep / nb).item())
        del loader
        wbuf = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, wbuf)
        obuf = io.BytesIO()
        torch.save(optimizer.state_dict(), obuf)
        out_q.put((idx, wbuf.getvalue(), obuf.getvalue(), trace, None))
    except Exception:
        import traceback
        out_q.put((idx, None, None, None, traceback.format_exc()))


def _avail_ram_bytes():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 32 * 1024 ** 3   # conservative fallback


def train_population_grouped(population, gpu_ids, opt_states, max_chunks_in_mem,
                             batch_size=128, epochs=10, lr=1e-3, log=print):
    """Disk-backed training. Chunks live on disk (population.chunk_dir); the parent loads the
    current window into shared RAM ONCE (1x -- no cat doubling), shared across all GPU workers,
    and each worker trains with the exact working in-RAM pipeline (num_workers=4 prefetched
    loader). Same spawn-per-call structure and return contract as the in-RAM train_population.
    `max_chunks_in_mem` is accepted for config compatibility but unused.
    """
    import torch.multiprocessing as mp

    chunk_paths = list(zip(population.inputs, population.outputs))   # (px, py) file paths
    if not chunk_paths:
        raise Exception("no datasets")
    subs = population.subs
    K = len(subs)
    if K != len(gpu_ids):
        raise ValueError(f"need one student per GPU: {K} students, {len(gpu_ids)} gpus")

    # Regime by window size on disk vs available RAM:
    #  - fits in RAM  -> load ONCE into shared RAM (1x, one physical copy for all GPUs). Each
    #    worker stages it in VRAM if it fits (fastest), else pins that shared copy in place and
    #    DMA-streams batch views (near-staged speed, data stays in RAM -- can be >> VRAM).
    #  - exceeds RAM  -> don't load it (would OOM); pass paths, workers mmap-stream from disk
    #    (disk-bandwidth-bound, but it runs and never OOMs).
    window_bytes = sum(os.path.getsize(px) + os.path.getsize(py) for px, py in chunk_paths)
    if window_bytes < 0.6 * _avail_ram_bytes():
        chunks = []
        for px, py in chunk_paths:
            x = torch.load(px).contiguous()
            y = torch.load(py).contiguous()
            x.share_memory_()
            y.share_memory_()
            chunks.append((x, y))
        target, payload = _disk_worker, chunks
    else:
        log(f"[out-of-core] window ~{window_bytes/1e9:.0f}GB exceeds RAM -> mmap-streaming from "
            f"disk (disk-bandwidth-bound)")
        target, payload = _mmapstream_worker, chunk_paths

    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = []
    for i in range(K):
        mbuf = io.BytesIO()
        torch.save(copy.deepcopy(subs[i]).to('cpu'), mbuf)
        procs.append(ctx.Process(target=target,
                                 args=(i, gpu_ids[i], mbuf.getvalue(), opt_states[i],
                                       payload, batch_size, epochs, lr, q)))
    for p in procs:
        p.start()
    raw = [q.get() for _ in procs]
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
