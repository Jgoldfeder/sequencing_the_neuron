"""Remove the saved query buffer (X, Y) from finished run checkpoints to free disk, keeping everything else
(member/best/teacher weights, cfg, dims, query counts, ...). Each file is rewritten to <file>.stripped.tmp, every kept
entry is checked bit-identical against the original, and only then is the original replaced. The checkpoint gets a
'buffer_stripped' entry recording what was removed. results/*.json are not touched.
usage: python strip_buffers.py [--apply] FILE [FILE ...]      (without --apply: dry run, prints what would change)"""
import datetime
import os
import subprocess
import sys

import torch


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return True


def open_by_others(f):
    out = subprocess.run(["lsof", "-t", f], capture_output=True, text=True).stdout.split()
    return [p for p in out if p != str(os.getpid())]


def strip(f, apply):
    if apply and open_by_others(f):
        print(f"  SKIP (file is open by process {open_by_others(f)})  {f}")
        return 0
    try:
        ck = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
    except Exception as e:
        print(f"  skip (not a readable torch checkpoint: {type(e).__name__})  {f}")
        return 0
    if not isinstance(ck, dict):
        print(f"  skip (not a dict checkpoint)  {f}")
        return 0
    gone = {k: list(ck[k].shape) for k in ("X", "Y") if k in ck and torch.is_tensor(ck[k])}
    if not gone:
        print(f"  skip (no X/Y)  {f}")
        return 0
    nbytes = sum(ck[k].numel() * ck[k].element_size() for k in gone)
    if not apply:
        print(f"  would free {nbytes / 1e9:6.1f} GB  {f}  (remove {gone})")
        return nbytes
    new = {k: v for k, v in ck.items() if k not in gone}
    new["buffer_stripped"] = {"date": datetime.date.today().isoformat(), "removed": gone,
                              "note": "query buffer removed to free disk; weights/config/queries kept"}
    tmp = f + ".stripped.tmp"
    torch.save(new, tmp)
    chk = torch.load(tmp, map_location="cpu", weights_only=False)
    bad = [k for k in ck if k not in gone and not same(ck[k], chk.get(k))]
    if bad or any(k in chk for k in gone):
        os.remove(tmp)
        print(f"  ABORT, mismatch in {bad}  {f}")
        return 0
    before = os.path.getsize(f)
    os.replace(tmp, f)
    print(f"  freed {(before - os.path.getsize(f)) / 1e9:6.1f} GB -> {os.path.getsize(f) / 1e6:.0f} MB  {f}", flush=True)
    return before - os.path.getsize(f)


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    files = [a for a in sys.argv[1:] if a != "--apply"]
    tot = sum(strip(f, apply) for f in files)
    print(f"{'freed' if apply else 'would free'} {tot / 1e9:.0f} GB in total")
