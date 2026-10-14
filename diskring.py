"""diskring.py -- two-tier (host RAM + NVMe) replay ring for plain-mode training (--disk).

Disk reads are O_DIRECT (page cache bypassed) into ALIGN-aligned staging buffers, READ_THREADS parallel
readers per segment, and the segment is consumed as a zero-copy float32 view of the staging buffer.
The window is `window` blocks of `q` query rows, exactly as the in-RAM ring in method.reconstruct.
The newest `W_ram` blocks live in host RAM; the older `window - W_ram` blocks are spilled to two flat
files (X.bin / Y.bin) with one sequential block write per iteration, and streamed back SEQUENTIALLY,
one segment at a time, by a prefetch thread while the GPU trains on the previous segment. Pages are
dropped from the OS cache right after each read/write (posix_fadvise DONTNEED) so the file never
competes with the RAM tier for memory.

Training order per epoch: the buffer is cut into segments of `seg_rows` contiguous rows (RAM segments are
views, disk segments are read into one of `nbuf` staging buffers); the segment order is a fresh random
permutation, rows inside a segment are permuted, and minibatches are drawn segment by segment. Every row
is visited exactly once per epoch, as before; the only change from the all-in-RAM path is that one
minibatch comes from one ~4 GB segment instead of from the whole window (the usual shuffle-buffer pattern).
Nothing is lost on the clock as long as the disk reads faster than the GPU consumes the disk tier:
per epoch the GPU takes B/c seconds for B bytes of buffer (c = consumption rate), the disk must deliver
B - R bytes (R = RAM tier) in that time, i.e. B <= R / (1 - d/c) for disk rate d (2.5 GB/s NVMe,
c = 2.8 GB/s at batch 512 fp32 -> ~2 TB; the prefetch depth lets RAM segments absorb the slack).

Row access for the other consumers (loss ranking, last-layer solve, probes, endgame polish): `.X` / `.Y`
are tensor-like views supporting len(), slicing (sequential reads) and index-tensor gathers (row-wise
reads, slow -- callers in the hot path use `.ram_X` / `.ram_Y`, the RAM tier only).
"""
import os
import json
import concurrent.futures as cf

import torch

ALIGN = 4096            # O_DIRECT alignment (buffer address, file offset, length)
READ_THREADS = 4        # parallel O_DIRECT readers per segment (3.1 GB/s single -> 3.4 GB/s on the 1.8 TB NVMe)


class _AlignedBuf:
    """A byte buffer whose data pointer is ALIGN-aligned (for O_DIRECT reads); `.view(off, count, w)`
    exposes rows as a float32 [count, w] tensor WITHOUT copying."""

    def __init__(self, nbytes):
        self.raw = torch.empty(nbytes + 2 * ALIGN, dtype=torch.uint8)
        self.base = (-self.raw.data_ptr()) % ALIGN
        self.nbytes = nbytes

    def mv(self, a, b):
        return memoryview(self.raw[self.base + a:self.base + b].numpy()).cast("B")

    def view(self, off, count, w):
        t = self.raw[self.base + off:self.base + off + count * w * 4]
        return t.view(torch.float32).view(count, w)


class _TierView:
    """Tensor-like view of one column block (X or Y) of a DiskRing: len/shape/slicing/gather."""
    _tier = True

    def __init__(self, ring, which):
        self.ring, self.which = ring, which

    def __len__(self):
        return len(self.ring)

    @property
    def shape(self):
        return (len(self), self.ring.d if self.which == "X" else self.ring.k)

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return torch.device("cpu")

    def dim(self):
        return 2

    def __getitem__(self, key):
        return self.ring._get(self.which, key)

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, *a, **k):
        return self[0:len(self)].to(*a, **k)


class DiskRing:
    def __init__(self, path, q, window, d, k, ram_bytes, seg_bytes=4e9, nbuf=4, verbose=True):
        assert window > 0, "--disk needs a finite --window"
        self.path, self.q, self.window, self.d, self.k = path, int(q), int(window), int(d), int(k)
        self.rowb_x, self.rowb_y = self.d * 4, self.k * 4
        blk = self.q * (self.rowb_x + self.rowb_y)
        self.W_ram = max(1, min(self.window, int(ram_bytes // blk)))
        self.W_disk = self.window - self.W_ram
        self.cap_ram, self.cap_disk = self.W_ram * self.q, self.W_disk * self.q
        self.X_ram = torch.empty(self.cap_ram, self.d, dtype=torch.float32)
        self.Y_ram = torch.empty(self.cap_ram, self.k, dtype=torch.float32)
        self.n_ram = self.n_disk = 0
        self.ram_slot = self.disk_slot = 0
        self.seg_rows = max(1024, min(self.q, int(seg_bytes // self.rowb_x)))
        self.nbuf = int(nbuf)
        self._stage = None                      # staging buffers, allocated on first disk epoch
        self._tmp = _AlignedBuf(64 * self.rowb_x + 2 * ALIGN)   # small random-access reads
        self._rpool = cf.ThreadPoolExecutor(max_workers=READ_THREADS)
        self.fx = self.fy = self.fxd = self.fyd = None
        if self.W_disk > 0:
            os.makedirs(path, exist_ok=True)
            for name, rowb, attr in (("X.bin", self.rowb_x, "x"), ("Y.bin", self.rowb_y, "y")):
                fn = os.path.join(path, name)
                fd = os.open(fn, os.O_RDWR | os.O_CREAT, 0o644)          # buffered: block writes
                os.ftruncate(fd, -(-self.cap_disk * rowb // ALIGN) * ALIGN)  # size rounded up to ALIGN
                setattr(self, "f" + attr, fd)
                setattr(self, "f" + attr + "d", os.open(fn, os.O_RDONLY | os.O_DIRECT))   # direct: streaming reads
        self.X, self.Y = _TierView(self, "X"), _TierView(self, "Y")
        self.pinned_ram = pin_in_place(self.X_ram, "RAM tier X") and pin_in_place(self.Y_ram, "RAM tier Y")
        if verbose:
            print(f"  [disk] two-tier ring: window {self.window} x q {self.q} = {self.window * blk / 1e9:.0f} GB | "
                  f"RAM tier {self.W_ram} blocks ({self.W_ram * blk / 1e9:.0f} GB) | disk tier {self.W_disk} blocks "
                  f"({self.W_disk * blk / 1e9:.0f} GB -> {path}) | segment {self.seg_rows} rows "
                  f"({self.seg_rows * self.rowb_x / 1e9:.1f} GB) x {self.nbuf} staging buffers", flush=True)

    # ------------------------------------------------------------------ raw I/O
    @staticmethod
    def _pread_all(fd, mv, off):
        done, n = 0, len(mv)
        while done < n:
            r = os.preadv(fd, [mv[done:n]], off + done)
            if r <= 0:
                raise IOError(f"disk ring read failed at byte {off + done}")
            done += r

    def _read_rows(self, which, start, count, buf=None):
        """Rows [start, start+count) of the DISK tier as a float32 [count, w] VIEW of `buf` (an
        _AlignedBuf; default = the small scratch buffer). O_DIRECT, ALIGN-aligned superset range,
        READ_THREADS parallel readers; no page cache, no copy."""
        fd, rowb, w = (self.fxd, self.rowb_x, self.d) if which == "X" else (self.fyd, self.rowb_y, self.k)
        off, n = start * rowb, count * rowb
        off0 = off - off % ALIGN
        n0 = -(-(off + n - off0) // ALIGN) * ALIGN
        buf = buf or self._tmp
        assert n0 <= buf.nbytes, "read larger than the staging buffer"
        part = max(ALIGN, -(-n0 // READ_THREADS // ALIGN) * ALIGN)
        jobs = [(a, min(a + part, n0)) for a in range(0, n0, part)]
        if len(jobs) == 1:
            self._pread_all(fd, buf.mv(0, n0), off0)
        else:
            list(self._rpool.map(lambda ab: self._pread_all(fd, buf.mv(*ab), off0 + ab[0]), jobs))
        return buf.view(off - off0, count, w)

    def _write_rows(self, which, start, t):
        """Buffered block write (source = RAM-tier rows), synced and dropped from the page cache so the
        O_DIRECT reader sees it and the cache never grows."""
        fd, rowb = (self.fx, self.rowb_x) if which == "X" else (self.fy, self.rowb_y)
        mv = memoryview(t.contiguous().numpy()).cast("B")
        off, n, done = start * rowb, len(mv), 0
        while done < n:
            r = os.pwritev(fd, [mv[done:n]], off + done)
            if r <= 0:
                raise IOError(f"disk ring write failed at byte {off + done}")
            done += r
        os.fdatasync(fd)
        os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)

    # ------------------------------------------------------------------ ring protocol
    def __len__(self):
        return self.n_ram + self.n_disk

    @property
    def ram_X(self):
        return self.X_ram[:self.n_ram]

    @property
    def ram_Y(self):
        return self.Y_ram[:self.n_ram]

    def add_block(self, I, T):
        """Append one iteration's block (q rows, CPU float32). Evicts the oldest RAM block to disk
        (which in turn overwrites the oldest disk block = the one leaving the window)."""
        q = self.q
        assert len(I) == q and len(T) == q
        if self.n_ram < self.cap_ram:
            s = self.n_ram
            self.X_ram[s:s + q].copy_(I); self.Y_ram[s:s + q].copy_(T)
            self.n_ram += q
            return
        s0 = self.ram_slot * q
        if self.W_disk > 0:
            d0 = self.disk_slot * q
            self._write_rows("X", d0, self.X_ram[s0:s0 + q])
            self._write_rows("Y", d0, self.Y_ram[s0:s0 + q])
            self.disk_slot = (self.disk_slot + 1) % self.W_disk
            self.n_disk = min(self.n_disk + q, self.cap_disk)
        self.X_ram[s0:s0 + q].copy_(I); self.Y_ram[s0:s0 + q].copy_(T)
        self.ram_slot = (self.ram_slot + 1) % self.W_ram

    # ------------------------------------------------------------------ random access (views)
    def _get(self, which, key):
        n = len(self)
        ram = self.X_ram if which == "X" else self.Y_ram
        if isinstance(key, int):
            key = n + key if key < 0 else key
            return ram[key] if key < self.n_ram else self._read_rows(which, key - self.n_ram, 1)[0].clone()
        if isinstance(key, slice):
            a, b, st = key.indices(n)
            if st != 1:
                key = torch.arange(a, b, st)
            else:
                if b <= a:
                    return ram[0:0]
                parts = []
                if a < self.n_ram:
                    parts.append(ram[a:min(b, self.n_ram)])
                if b > self.n_ram:
                    da, db = max(a, self.n_ram) - self.n_ram, b - self.n_ram
                    w = self.d if which == "X" else self.k
                    rowb = self.rowb_x if which == "X" else self.rowb_y
                    out = torch.empty(db - da, w, dtype=torch.float32)
                    step = max(1, (self._tmp.nbytes - 2 * ALIGN) // rowb)       # scratch-buffer sized chunks
                    for c0 in range(0, db - da, step):
                        c = min(step, db - da - c0)
                        out[c0:c0 + c].copy_(self._read_rows(which, da + c0, c))
                    parts.append(out)
                return parts[0] if len(parts) == 1 else torch.cat(parts)
        idx = torch.as_tensor(key).long().reshape(-1)
        idx = torch.where(idx < 0, idx + n, idx)
        out = torch.empty(len(idx), self.d if which == "X" else self.k, dtype=torch.float32)
        in_ram = idx < self.n_ram
        if bool(in_ram.any()):
            out[in_ram] = ram[idx[in_ram]]
        if bool((~in_ram).any()):
            pos = torch.nonzero(~in_ram).reshape(-1)
            drow = idx[pos] - self.n_ram
            order = torch.argsort(drow)               # sorted row reads: better locality
            for p, r in zip(pos[order].tolist(), drow[order].tolist()):
                out[p].copy_(self._read_rows(which, r, 1)[0])
        return out

    # ------------------------------------------------------------------ persistence
    def meta(self):
        return {"path": self.path, "q": self.q, "window": self.window, "d": self.d, "k": self.k,
                "W_ram": self.W_ram, "W_disk": self.W_disk, "n_ram": self.n_ram, "n_disk": self.n_disk,
                "ram_slot": self.ram_slot, "disk_slot": self.disk_slot}

    def flush_ram(self):
        """Write the RAM tier next to the disk tier (X_ram.bin / Y_ram.bin + meta.json) so the whole
        window is recoverable from `path` -- replaces the X/Y entries of the end-of-run checkpoint."""
        os.makedirs(self.path, exist_ok=True)
        for name, t in (("X_ram.bin", self.ram_X), ("Y_ram.bin", self.ram_Y)):
            fd = os.open(os.path.join(self.path, name), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                mv = memoryview(t.contiguous().numpy()).cast("B"); done = 0
                while done < len(mv):
                    r = os.pwritev(fd, [mv[done:]], done)
                    if r <= 0:
                        raise IOError(name)
                    done += r
                os.fdatasync(fd)
            finally:
                os.close(fd)
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump(self.meta(), f, indent=1)
        return self.path

    def close(self):
        if getattr(self, "pinned_ram", False):
            try:
                for t in (self.X_ram, self.Y_ram):
                    torch.cuda.cudart().cudaHostUnregister(t.data_ptr())
            except Exception:
                pass
        for fd in (self.fx, self.fy, self.fxd, self.fyd):
            if fd is not None:
                os.close(fd)
        self.fx = self.fy = self.fxd = self.fyd = None
        self._rpool.shutdown(wait=True)


    # ------------------------------------------------------------------ training epochs
    def epoch_batches(self, batch, gen, device, ahead=2, depth=2, gpu=True):
        """Yield (xb, yb) device minibatches covering every row once, in segment-shuffled order.
        Disk segments are prefetched `depth` segments ahead by one reader thread into `nbuf` staging
        buffers. gpu=True (default): each segment is copied to the GPU ONCE (double-buffered, on a side
        stream, overlapping the steps of the previous segment) and minibatches are gathered on the GPU,
        so the step runs at its GPU floor (3.6 vs 5.0 ms for the 8-member 11-layer net). gpu=False:
        the CPU gather + H2D of each minibatch is done `ahead` batches early by 2 workers."""
        S = self.seg_rows
        segs = ([("ram", a, min(S, self.n_ram - a)) for a in range(0, self.n_ram, S)]
                + [("disk", a, min(S, self.n_disk - a)) for a in range(0, self.n_disk, S)])
        if not segs:
            return
        order = torch.randperm(len(segs), generator=gen, device=device).tolist()
        segs = [segs[i] for i in order]
        if self._stage is None and any(s[0] == "disk" for s in segs):
            self._stage = [(_AlignedBuf(S * self.rowb_x + 2 * ALIGN), _AlignedBuf(S * self.rowb_y + 2 * ALIGN))
                           for _ in range(self.nbuf)]
            for bx, by in self._stage:          # pinned staging: disk -> staging (O_DIRECT) -> GPU (DMA)
                pin_in_place(bx.raw); pin_in_place(by.raw)
        # disk segments get staging buffers round-robin in their consumption order
        disk_order = [i for i, s in enumerate(segs) if s[0] == "disk"]
        buf_of = {si: j % self.nbuf for j, si in enumerate(disk_order)}
        assert depth + 2 <= self.nbuf, "need nbuf >= depth + 2 staging buffers"
        reader = cf.ThreadPoolExecutor(max_workers=1)
        futs = {}

        def _load(si):
            _, a, c = segs[si]
            bx, by = self._stage[buf_of[si]]
            return self._read_rows("X", a, c, buf=bx), self._read_rows("Y", a, c, buf=by)

        def _prefetch_upto(si_cur):
            # keep every disk segment within the next `depth` segments (and the current) in flight
            for si in range(si_cur, min(len(segs), si_cur + depth + 1)):
                if segs[si][0] == "disk" and si not in futs:
                    futs[si] = reader.submit(_load, si)

        def _seg_data(si):
            kind, a, c = segs[si]
            if kind == "ram":
                return self.X_ram[a:a + c], self.Y_ram[a:a + c]
            return futs[si].result()

        try:
            yield from _segment_batches(segs, _seg_data, _prefetch_upto, batch, gen, device,
                                        S, self.d, self.k, ahead=ahead, gpu=gpu)
        finally:
            reader.shutdown(wait=True)


def ram_epoch_batches(X, Y, batch, gen, device, seg_rows, ahead=2, gpu=True):
    """The same segment-shuffled, GPU-resident epoch over plain host tensors (the in-RAM ring /
    plain buffer path when --disk is not used). Every row once per epoch."""
    n = len(X)
    S = max(1024, int(seg_rows))
    segs = [("ram", a, min(S, n - a)) for a in range(0, n, S)]
    if not segs:
        return
    order = torch.randperm(len(segs), generator=gen, device=device).tolist()
    segs = [segs[i] for i in order]

    def _seg_data(si):
        _, a, c = segs[si]
        return X[a:a + c], Y[a:a + c]

    yield from _segment_batches(segs, _seg_data, lambda si: None, batch, gen, device,
                                S, X.shape[1], Y.shape[1], ahead=ahead, gpu=gpu)


_PINNED = {}


def pin_in_place(t, label=""):
    """cudaHostRegister an existing CPU tensor's storage so H2D copies from it are true async DMA
    (no bounce memcpy: a CPU memcpy of a segment fans out over every core and starves the kernel-launch
    thread, step 3.5 -> 25 ms while it runs). Returns True on success. ~2 s per GB."""
    if not torch.cuda.is_available() or t.numel() == 0:
        return False
    try:
        import time as _t
        t0 = _t.time()
        r = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        ok = int(r) == 0
        if ok and label:
            print(f"  [disk] pinned {label} ({t.numel() * t.element_size() / 1e9:.0f} GB) in {_t.time() - t0:.0f}s", flush=True)
        return ok
    except Exception:
        return False


def _pinned_pair(S, d, k, n=3):
    """n pinned (X, Y) host buffers of S rows, cached per shape (pinning is slow and the epoch
    generator is re-entered every epoch)."""
    key = (S, d, k, n)
    if key not in _PINNED:
        _PINNED[key] = [(torch.empty(S, d, dtype=torch.float32).pin_memory(),
                         torch.empty(S, k, dtype=torch.float32).pin_memory()) for _ in range(n)]
    return _PINNED[key]


def _segment_batches(segs, seg_data, prefetch_upto, batch, gen, device, S, d, k, ahead=2, gpu=True):
    """Core of the epoch: per segment a random row order, minibatches of `batch` rows.
    gpu=True: two GPU buffers of S rows; segment j is copied into buffer j%2 on a side CUDA stream
    by a worker thread while segment j-1 trains; the copy into a buffer waits (CUDA event) for the
    main stream to finish the gathers that last read that buffer. Batches are index_selects on the GPU.
    gpu=False: 2 gather workers, `ahead` batches ahead (CPU gather + H2D per batch)."""
    # flat batch schedule: (segment index, local row indices)
    perms = [torch.randperm(c, generator=gen, device=device) for (_, _, c) in segs]
    if torch.device(device).type != "cuda":
        gpu = False                            # resident-segment path needs a CUDA device; plain gather otherwise
    if not gpu:
        tasks = []
        for si, (_, _, c) in enumerate(segs):
            p = perms[si].cpu()
            tasks += [(si, p[i:i + batch]) for i in range(0, c, batch)]

        def _fetch(si, idx):
            Xs, Ys = seg_data(si)
            return Xs[idx].to(device, non_blocking=False), Ys[idx].to(device, non_blocking=False)
        gather = cf.ThreadPoolExecutor(max_workers=ahead)
        pend = []
        try:
            prefetch_upto(0)
            pend = [gather.submit(_fetch, *tasks[j]) for j in range(min(ahead, len(tasks)))]
            for j, (si, _) in enumerate(tasks):
                prefetch_upto(si)
                xb, yb = pend.pop(0).result()
                if j + ahead < len(tasks):
                    pend.append(gather.submit(_fetch, *tasks[j + ahead]))
                yield xb, yb
        finally:
            for f in pend:
                try:
                    f.result()
                except Exception:
                    pass
            gather.shutdown(wait=True)
        return

    dev = torch.device(device)
    main = torch.cuda.current_stream(dev)
    copy_stream = torch.cuda.Stream(dev)
    NB = 3                                     # GPU segment buffers: current + 2 prefetched
    bufs = [(torch.empty(S, d, dtype=torch.float32, device=dev), torch.empty(S, k, dtype=torch.float32, device=dev))
            for _ in range(NB)]
    # Sources are pinned in place (DiskRing pins its RAM tier and staging buffers) so the segment copy is
    # a pure async DMA on the side stream. A pageable source falls back to a SINGLE-THREADED numpy memcpy
    # into a pinned bounce buffer (a torch CPU copy fans out over all cores and starves kernel launches).
    pinned = None
    free_ev = [None] * NB                      # event: main stream done reading buffer b
    copier = cf.ThreadPoolExecutor(max_workers=2)

    def _copy(si):
        nonlocal pinned
        b = si % NB
        prefetch_upto(si)                       # make sure this segment's disk read is queued
        Xs, Ys = seg_data(si)                   # waits for the disk read if needed
        c = len(Xs)
        if not (Xs.is_pinned() and Ys.is_pinned()):
            if pinned is None:
                pinned = _pinned_pair(S, d, k, NB)
            px, py = pinned[b]
            if free_ev[b] is not None:
                free_ev[b].synchronize()        # bounce buffer b reused: its last DMA must have landed
            import numpy as _np
            _np.copyto(px[:c].numpy(), Xs.numpy()); _np.copyto(py[:c].numpy(), Ys.numpy())
            Xs, Ys = px[:c], py[:c]
        with torch.cuda.stream(copy_stream):
            if free_ev[b] is not None:
                copy_stream.wait_event(free_ev[b])
            bufs[b][0][:c].copy_(Xs, non_blocking=True)
            bufs[b][1][:c].copy_(Ys, non_blocking=True)
            ev = torch.cuda.Event(); ev.record(copy_stream)
        ev.synchronize()                        # source may be reused once the DMA has landed
        return ev

    cfuts = {}
    try:
        prefetch_upto(0)
        cfuts[0] = copier.submit(_copy, 0)
        for si, (_, _, c) in enumerate(segs):
            prefetch_upto(si)
            for sj in (si + 1, si + 2):         # keep two segments of copies in flight
                if sj < len(segs) and sj not in cfuts:
                    cfuts[sj] = copier.submit(_copy, sj)
            ev = cfuts.pop(si).result()
            main.wait_event(ev)                 # segment si is on the GPU
            Xg, Yg = bufs[si % NB][0][:c], bufs[si % NB][1][:c]
            perm = perms[si]
            for i in range(0, c, batch):
                idx = perm[i:i + batch]
                yield Xg[idx], Yg[idx]          # GPU gathers; the caller's step runs on `main`
            done = torch.cuda.Event(); done.record(main)
            free_ev[si % NB] = done            # buffer may be overwritten once these gathers finished
    finally:
        for f in cfuts.values():
            try:
                f.result()
            except Exception:
                pass
        copier.shutdown(wait=True)
        main.wait_stream(copy_stream)


def load_ring_rows(path):
    """Read a flushed ring back as plain (X, Y) CPU tensors (RAM tier first, then disk tier)."""
    with open(os.path.join(path, "meta.json")) as f:
        m = json.load(f)
    d, k = m["d"], m["k"]

    def _rd(name, rows, w):
        if rows == 0:
            return torch.empty(0, w)
        t = torch.empty(rows, w, dtype=torch.float32)
        fd = os.open(os.path.join(path, name), os.O_RDONLY)
        try:
            mv = memoryview(t.numpy()).cast("B"); done = 0
            while done < len(mv):
                r = os.preadv(fd, [mv[done:]], done)
                if r <= 0:
                    raise IOError(name)
                done += r
        finally:
            os.close(fd)
        return t
    X = torch.cat([_rd("X_ram.bin", m["n_ram"], d), _rd("X.bin", m["n_disk"], d)])
    Y = torch.cat([_rd("Y_ram.bin", m["n_ram"], k), _rd("Y.bin", m["n_disk"], k)])
    return X, Y, m
