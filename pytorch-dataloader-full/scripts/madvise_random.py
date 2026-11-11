#!/usr/bin/env python3
"""Turn off readahead / fault-around on memory-mapped dataset files (Linux).

When samples are read in random order from mmapped files (Arrow / Hugging Face
datasets, numpy memmap, LMDB), each cold row's page fault pulls in ~100-260 KiB
around it. MADV_RANDOM makes each fault fetch one page. Worth it when the hot
data is larger than the page cache the job can hold. When it fits, readahead is
free prefetch, so leave it on.

In code, after the dataset has opened its files and before DataLoader workers
start (forked workers inherit the advice; under spawn/forkserver call it in
worker_init_fn instead):

    from madvise_random import advise_random
    nbytes = advise_random("/data/my_dataset/")   # substring of the mapped paths

Check that it reached the trainer and every worker (VmFlags 'rr' in smaps):

    python3 madvise_random.py check --tree <trainer pid> /data/my_dataset/
    python3 madvise_random.py --selftest

Stdlib only.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import mmap
import os
import sys
import tempfile


def _mappings(pid, match):
    """(start, end, path) for each mapping of `pid` whose path contains `match`."""
    out = []
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and match in parts[5]:
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
                out.append((lo, hi, parts[5].strip()))
    return out


def advise_random(match: str, advice: int | None = None) -> int:
    """madvise every mapping of this process whose path contains `match`.

    Returns the number of bytes advised (0 means nothing matched: check the path,
    and that the dataset has actually opened its files by now).
    """
    if not os.path.exists("/proc/self/maps"):
        raise OSError("advise_random needs Linux /proc/self/maps")
    if advice is None:
        advice = mmap.MADV_RANDOM
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    libc.madvise.restype = ctypes.c_int
    total = 0
    for lo, hi, path in _mappings("self", match):
        if libc.madvise(lo, hi - lo, advice) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"madvise({path}): {os.strerror(err)}")
        total += hi - lo
    return total


def advised(pid, match):
    """[(path, bytes, has_rr)] for mappings of `pid` whose path contains `match`."""
    res, cur = [], None
    with open(f"/proc/{pid}/smaps") as f:
        for line in f:
            head = line.split(maxsplit=5)
            if head and not head[0].endswith(":"):  # a mapping header, not a "Key: value" line
                cur = None
                if len(head) == 6 and match in head[5]:
                    lo, hi = (int(x, 16) for x in head[0].split("-"))
                    cur = (head[5].strip(), hi - lo)
            elif cur is not None and line.startswith("VmFlags:"):
                res.append((cur[0], cur[1], "rr" in line.split()[1:]))
                cur = None
    return res


def descendants(root):
    """root and every process below it (DataLoader workers, pin-memory helpers...)."""
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(kids.get(p, []))
    return out


def cmd_check(a):
    pids = descendants(a.pid) if a.tree else [a.pid]
    bad = 0
    for pid in pids:
        try:
            maps = advised(pid, a.match)
        except OSError as e:
            print(f"pid {pid}: unreadable ({e.strerror})")
            continue
        if not maps:
            print(f"pid {pid}: no mappings match {a.match!r}")
            continue
        ok = sum(m[2] for m in maps)
        gib = sum(m[1] for m in maps if m[2]) / 2**30
        bad += ok < len(maps)
        print(f"pid {pid}: {ok}/{len(maps)} matching mappings advised random ({gib:.2f} GiB)")
    return 1 if bad else 0


def _selftest():
    if not os.path.exists("/proc/self/maps"):
        print("selftest skipped: needs Linux /proc")
        return 0
    with tempfile.NamedTemporaryFile(prefix="madvise_selftest_") as tf:
        tf.write(b"\0" * (1 << 20))
        tf.flush()
        mm = mmap.mmap(tf.fileno(), 0, prot=mmap.PROT_READ)
        try:
            assert advised(os.getpid(), tf.name) and not any(m[2] for m in advised(os.getpid(), tf.name))
            assert advise_random(tf.name) >= 1 << 20
            assert all(m[2] for m in advised(os.getpid(), tf.name))
            assert advise_random("/no/such/path/") == 0
            assert os.getpid() in descendants(os.getpid())
        finally:
            mm.close()
    print("selftest ok")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return _selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="report which matching mappings carry MADV_RANDOM ('rr')")
    p.add_argument("--tree", action="store_true", help="also check every descendant process (workers)")
    p.add_argument("pid", type=int)
    p.add_argument("match", help="substring of the mapped file paths")
    p.set_defaults(fn=cmd_check)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
