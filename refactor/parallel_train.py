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
                              pin_memory=True, num_workers=0)
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
