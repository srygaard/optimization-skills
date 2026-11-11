#!/usr/bin/env python3
"""Measure a PyTorch DataLoader on its own (no model, no GPU) and classify its stalls.

    python3 loader_bench.py run     --factory pkg.mod:make_loader [--kw num_workers=8] [--step-ms 60]
    python3 loader_bench.py isolate --factory path/to/data.py:make_loader [--rows 2000] [--step-ms 60]
    python3 loader_bench.py analyze waits.json [--steps steps.json] [--num-workers 8]
    python3 loader_bench.py --selftest

--factory names a callable (or a DataLoader object) that returns the exact
DataLoader training uses: same dataset, sampler, collate_fn, sharding and worker
settings. Expose the knobs you want to sweep as its keyword arguments.

run      fetches batches and reports waits, throughput, the resolved loader knobs
         and bytes fetched from storage (Linux cgroup v2). --step-ms simulates a
         training step per batch so the prefetch queue behaves as in training;
         without it you measure peak loader throughput.
isolate  times dataset reads cold, the same reads warm, then collate_fn alone,
         in one process: I/O-bound or CPU-bound, and how many workers you need.
analyze  classifies per-step waits from a real run (JSON list, one number per
         line, or {"waits_ms": [...], "step_ms": [...]}).

In a training loop:

    from loader_bench import TimedIterator, analyze
    it = TimedIterator(train_loader)
    for batch in it: ...
    print("\\n".join(analyze(it.waits_ms, step_ms=step_times, num_workers=8)["verdicts"]))

analyze and the core of --selftest are stdlib only; run and isolate import torch.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import sys
import time
from collections import Counter

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


class TimedIterator:
    """Wraps an iterable; records the wall time each next() blocks, in ms."""

    def __init__(self, iterable):
        self._it = iter(iterable)
        self.waits_ms: list[float] = []

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.perf_counter()
        item = next(self._it)
        self.waits_ms.append((time.perf_counter() - t0) * 1e3)
        return item


# --- analysis (stdlib) ---------------------------------------------------------


def _pct(s, q):
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats(xs):
    s = sorted(xs)
    if not s:
        return {"n": 0}
    return {"n": len(s), "mean": sum(s) / len(s), "p50": _pct(s, 0.5),
            "p90": _pct(s, 0.9), "p99": _pct(s, 0.99), "max": s[-1]}


def _period(waits, stall_ms):
    """Most common gap (in steps) between stalls, if at least half the gaps match it."""
    idx = [i for i, w in enumerate(waits) if w > stall_ms]
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    if len(gaps) < 3:
        return None, 0.0
    gap = Counter(gaps).most_common(1)[0][0]
    if gap < 2:
        return None, 0.0
    tol = 1 if gap >= 4 else 0
    share = sum(abs(g - gap) <= tol for g in gaps) / len(gaps)
    return (gap, share) if share >= 0.5 else (None, share)


def analyze(waits_ms, step_ms=None, num_workers=None, stall_ms=50.0, floor_ms=2.0):
    """Classify per-step loader waits.

    step_ms, if given, is per-step wall time *including* the wait, aligned with
    waits_ms. Returns a dict with stats and a list of verdict strings.
    """
    w = [float(x) for x in waits_ms]
    st = stats(w)
    out = {"wait_ms": st, "stall_ms": stall_ms}
    if not w:
        out["verdicts"] = ["NO DATA: no waits to analyze."]
        return out
    total = sum(w)
    stalls = [x for x in w if x > stall_ms]
    k = max(1, len(w) // 100)
    top_share = sum(sorted(w)[-k:]) / total if total else 0.0
    period, pshare = _period(w, stall_ms)
    out.update(stalls=len(stalls), top1pct_share=top_share, period=period)

    verdicts = []
    material = floor_ms
    if step_ms:
        s = [float(x) for x in step_ms][: len(w)]
        ww = w[: len(s)]
        if len(s) != len(w):
            verdicts.append(f"NOTE: {len(w)} waits but {len(step_ms)} step times; used the first {len(s)}.")
        tot = sum(s)
        share = sum(ww) / tot if tot else 0.0
        ideal = min(st["p50"], 0.5)
        speedup = tot / max(tot - sum(ww) + ideal * len(ww), 1e-9)
        out.update(wait_share=share, ideal_speedup=speedup)
        material = max(floor_ms, 0.02 * stats(s)["p50"])
        if share < 0.01:
            verdicts.append(
                f"NOT THE BOTTLENECK: waits are {share:.2%} of step time; a perfect loader buys at most "
                f"{speedup:.3f}x. Loader work pays off here as less storage traffic, faster startup or "
                f"less host memory, not throughput.")
        else:
            verdicts.append(f"LOADER COST: waits are {share:.1%} of step time; a perfect loader buys at most {speedup:.2f}x.")

    if st["p50"] > material:
        verdicts.append(
            f"CHRONIC: the median step waits {st['p50']:.1f} ms. Either the prefetch queue is usually "
            f"empty (workers can't keep up: add workers or make each batch cheaper; `isolate` shows read "
            f"vs collate) or every handoff is expensive (many tensors or Python objects pickled per batch: "
            f"if more workers don't lower p50, it's this). A deeper prefetch buffer will not help.")
    elif period:
        head = f"PERIODIC: {pshare:.0%} of gaps between stalls (>{stall_ms:g} ms) are {period} steps"
        if num_workers is not None and period != num_workers:
            verdicts.append(
                f"{head}, not num_workers ({num_workers}). Something else repeats at that period: a slow "
                f"file or shard every {period} batches, a periodic eval/checkpoint/logging step, or a "
                f"background job. Line the stall steps up against those.")
        else:
            nw = " = num_workers" if num_workers == period else ""
            verdicts.append(
                f"{head}{nw}. Either the pool is too slow in aggregate (per-batch cost / num_workers > "
                f"step time: add workers or cheapen batches) or one worker is slow (tag batches with "
                f"get_worker_info().id; in_order=False lets the others pass it).")
    if stalls and (top_share > 0.5 or st["mean"] > 2 * st["p90"]):
        verdicts.append(
            f"TAIL: the slowest 1% of steps hold {top_share:.0%} of all wait (max {st['max']:.0f} ms vs "
            f"p90 {st['p90']:.1f} ms). Rare fetches outlast the prefetch buffer, and more workers won't "
            f"shorten them. Make num_workers x prefetch_factor x step time exceed the longest stall, cut "
            f"I/O per sample, or capture one (references/measurement.md section 5).")
    if not any(v.startswith(("CHRONIC", "PERIODIC", "TAIL")) for v in verdicts):
        verdicts.append(f"HEALTHY: no chronic, periodic or tail pattern (stall threshold {stall_ms:g} ms).")
    out["verdicts"] = verdicts
    return out


def _fmt(st):
    if not st.get("n"):
        return "n=0"
    return (f"n={st['n']}  mean {st['mean']:.2f}  p50 {st['p50']:.2f}  p90 {st['p90']:.2f}  "
            f"p99 {st['p99']:.2f}  max {st['max']:.1f} ms")


# --- environment helpers (Linux) ------------------------------------------------


def cgroup_memstat():
    """memory.stat of this process's cgroup v2, or None."""
    try:
        with open("/proc/self/cgroup") as f:
            rel = next((ln.strip()[3:] for ln in f if ln.startswith("0::")), None)
        if rel is None:
            return None
        with open(os.path.join("/sys/fs/cgroup" + rel, "memory.stat")) as f:
            return {p[0]: int(p[1]) for p in (ln.split() for ln in f) if len(p) == 2}
    except (OSError, ValueError):
        return None


def cgroup_delta(a, b):
    if not a or not b:
        return None
    d = lambda k: b.get(k, 0) - a.get(k, 0)  # noqa: E731
    refault = "workingset_refault_file" if "workingset_refault_file" in b else "workingset_refault"
    return {"fetched_bytes": d("file") + d("pgsteal") * PAGE,
            "refetched_bytes": d(refault) * PAGE,
            "major_faults": d("pgmajfault"),
            "anon_delta_bytes": d("anon")}


def evict(paths):
    """Drop clean, unmapped cached pages of these files/dirs on this node (no root)."""
    if not paths:
        return 0
    if not hasattr(os, "posix_fadvise"):
        print("warning: posix_fadvise unavailable here; --evict skipped", file=sys.stderr)
        return 0
    n = 0
    for p in paths:
        files = [p] if os.path.isfile(p) else [os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs]
        for f in files:
            try:
                fd = os.open(f, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    return n


# --- torch-side helpers ----------------------------------------------------------


def load_factory(spec, kw=()):
    mod_part, sep, attr = spec.rpartition(":")
    if not sep or not mod_part or not attr:
        raise SystemExit("--factory must look like pkg.module:make_loader or path/to/file.py:make_loader")
    if mod_part.endswith(".py"):
        path = os.path.abspath(mod_part)
        sys.path.insert(0, os.path.dirname(path))  # spawned workers re-import it by name
        mod = importlib.import_module(os.path.splitext(os.path.basename(path))[0])
    else:
        sys.path.insert(0, os.getcwd())
        mod = importlib.import_module(mod_part)
    obj = getattr(mod, attr)
    kwargs = {}
    for item in kw:
        k, _, v = item.partition("=")
        try:
            kwargs[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kwargs[k] = v
    return obj(**kwargs) if callable(obj) else obj


def describe(loader):
    """The knobs the loader actually resolved to: read these, not the job label."""
    import multiprocessing as mp

    d = {k: getattr(loader, k, "n/a") for k in
         ("num_workers", "prefetch_factor", "pin_memory", "persistent_workers", "in_order",
          "batch_size", "drop_last", "timeout")}
    d["dataset"] = type(loader.dataset).__name__
    d["sampler"] = type(loader.sampler).__name__
    d["batch_sampler"] = type(loader.batch_sampler).__name__ if loader.batch_sampler is not None else None
    cf = loader.collate_fn
    d["collate_fn"] = getattr(cf, "__qualname__", type(cf).__name__)
    ctx = getattr(loader, "multiprocessing_context", None)
    d["start_method"] = ctx.get_start_method() if ctx is not None else f"{mp.get_start_method()} (default)"
    return d


def _count(batch):
    """(samples, tokens, token_source), best effort."""
    try:
        if isinstance(batch, dict):
            tok = src = None
            if "attention_mask" in batch:
                tok, src = int(batch["attention_mask"].sum()), "attention_mask"
            elif "input_ids" in batch:
                tok, src = int(batch["input_ids"].numel()), "input_ids elements"
            first = next((v for v in batch.values() if hasattr(v, "shape") and len(v.shape)), None)
            return (int(first.shape[0]) if first is not None else None), tok, src
        if hasattr(batch, "shape") and len(batch.shape):
            return int(batch.shape[0]), None, None
        if isinstance(batch, (list, tuple)) and batch:
            return _count(batch[0])[0], None, None
    except Exception:
        pass
    return None, None, None


def bench(loader, batches=300, passes=1, warmup=1, step_ms=0.0, stall_ms=50.0, floor_ms=2.0):
    """Fetch `batches` per pass; with step_ms > 0, sleep that long per batch like a training step."""
    results = []
    for p in range(passes):
        it = TimedIterator(loader)
        counts = []
        for i, batch in enumerate(it):
            counts.append(_count(batch))
            if step_ms:
                time.sleep(step_ms / 1e3)
            if i + 1 >= batches:
                break
        waits = it.waits_ms
        steady, steady_counts = waits[warmup:], counts[warmup:]
        r = {"pass": p + 1, "batches": len(waits), "first_next_ms": waits[0] if waits else None}
        fetch_s = sum(steady) / 1e3
        wall_s = fetch_s + len(steady) * step_ms / 1e3
        if steady and wall_s > 0:
            r["batches_per_s"] = len(steady) / wall_s
            if all(c[0] is not None for c in steady_counts):
                r["samples_per_s"] = sum(c[0] for c in steady_counts) / wall_s
            if all(c[1] is not None for c in steady_counts):
                r["tokens_per_s"] = sum(c[1] for c in steady_counts) / wall_s
                r["token_source"] = steady_counts[0][2]
        steps = [w + step_ms for w in steady] if step_ms else None
        r["analysis"] = analyze(steady, step_ms=steps, num_workers=loader.num_workers,
                                stall_ms=stall_ms, floor_ms=floor_ms)
        r["waits_ms"] = waits
        results.append(r)
    return results


def isolate(loader, rows=2000, step_ms=None):
    """Time reads cold, the same reads warm, then collate_fn, per batch, in this process."""
    from torch.utils.data import IterableDataset

    ds = loader.dataset
    if isinstance(ds, IterableDataset):
        raise SystemExit("isolate needs a map-style dataset; time your IterableDataset's read and "
                         "transform steps separately instead.")
    src = iter(loader.batch_sampler) if loader.batch_sampler is not None else ([i] for i in loader.sampler)
    groups, n = [], 0
    for b in src:
        groups.append(list(b))
        n += len(groups[-1])
        if n >= rows:
            break
    if not groups:
        raise SystemExit("the loader's sampler yielded no indices")
    getitems = getattr(ds, "__getitems__", None)  # what the real fetcher calls, when defined

    def read(b):
        return getitems(b) if getitems else [ds[i] for i in b]

    def timed(fn, arg):
        t0 = time.perf_counter()
        res = fn(arg)
        return res, (time.perf_counter() - t0) * 1e3

    cold = [timed(read, b)[1] for b in groups]
    warm, samples = [], []
    for b in groups:
        s, t = timed(read, b)
        samples.append(s)
        warm.append(t)
    coll = [timed(loader.collate_fn, s)[1] for s in samples]

    rc, rw, cc = stats(cold), stats(warm), stats(coll)
    per_batch = rc["mean"] + cc["mean"]
    out = {"rows": n, "batches": len(groups), "rows_per_batch": len(groups[0]),
           "read_cold_ms": rc, "read_warm_ms": rw, "collate_ms": cc,
           "cold_warm_ratio": rc["mean"] / max(rw["mean"], 1e-9),
           "one_worker_batches_per_s": 1e3 / max(per_batch, 1e-9)}
    v = []
    ratio = out["cold_warm_ratio"]
    if rc["mean"] >= cc["mean"] and ratio >= 2:
        v.append(f"I/O-BOUND: cold reads are {ratio:.1f}x warm and outweigh collate "
                 f"({rc['mean']:.1f} vs {cc['mean']:.1f} ms/batch). See SKILL.md section 4 and references/storage.md.")
    elif rc["mean"] >= cc["mean"]:
        v.append(f"READ-BOUND, cold ~ warm ({ratio:.1f}x): reading costs CPU (decode/deserialize), or the "
                 f"'cold' pass hit pages an earlier run cached (use --evict or rows nobody has read).")
    else:
        v.append(f"CPU-BOUND (collate {cc['mean']:.1f} vs read {rc['mean']:.1f} ms/batch): workers scale this "
                 f"until cores run out; or move the work offline (pre-tokenize, pre-decode).")
    if rc["n"] >= 10 and rc["max"] > 10 * max(rc["p50"], 1e-3):
        v.append(f"TAIL READS: max {rc['max']:.0f} ms vs p50 {rc['p50']:.1f} ms per batch. The rare-stall "
                 f"signature lives in storage, not in your code.")
    if step_ms:
        need = math.ceil(per_batch / step_ms)
        out["workers_needed"] = need
        v.append(f"WORKERS: one worker makes a batch every ~{per_batch:.1f} ms (+ handoff), so a {step_ms:g} ms "
                 f"step needs >= {need} workers before tail headroom.")
    out["verdicts"] = v
    return out


# --- CLI -------------------------------------------------------------------------


def _read_series(path):
    with open(path) as f:
        txt = f.read().strip()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        return [float(x) for x in txt.split()], None
    if isinstance(data, dict):
        return data.get("waits_ms") or data.get("waits") or [], data.get("step_ms") or data.get("steps")
    return data, None


def cmd_run(a):
    n_ev = evict(a.evict)
    before = cgroup_memstat()
    loader = load_factory(a.factory, a.kw)
    if a.madvise:
        from madvise_random import advise_random
        print(f"madvise(MADV_RANDOM): {advise_random(a.madvise) / 2**30:.2f} GiB of mappings matching {a.madvise!r}")
    knobs = describe(loader)
    print("loader:", ", ".join(f"{k}={v}" for k, v in knobs.items()))
    if n_ev:
        print(f"evicted {n_ev} files from page cache")
    if a.passes > 1 and knobs["sampler"] == "RandomSampler":
        print("note: RandomSampler draws a new order each pass, so pass 2 is not a warm re-read; use `isolate`.")
    res = bench(loader, a.batches, a.passes, a.warmup, a.step_ms, a.stall_ms, a.floor_ms)
    delta = cgroup_delta(before, cgroup_memstat())
    for r in res:
        rate = "  ".join(f"{k} {r[k]:,.1f}" for k in ("batches_per_s", "samples_per_s", "tokens_per_s") if k in r)
        first = f"{r['first_next_ms']:.0f} ms" if r["first_next_ms"] is not None else "n/a"
        print(f"\npass {r['pass']}: {r['batches']} batches, first next() {first}  {rate}"
              + (f" ({r['token_source']})" if "token_source" in r else ""))
        print("  wait:", _fmt(r["analysis"]["wait_ms"]), f" stalls>{a.stall_ms:g}ms: {r['analysis']['stalls']}")
        if a.step_ms:
            for v in r["analysis"]["verdicts"]:
                print("  -", v)
    if not a.step_ms:
        print("\n(no --step-ms: this is peak loader throughput. Compare it with what training consumes, "
              "or pass your median step time to see starvation verdicts.)")
    if delta:
        print(f"\ncgroup: fetched ~{delta['fetched_bytes'] / 2**20:,.0f} MiB, re-fetched after eviction "
              f"{delta['refetched_bytes'] / 2**20:,.0f} MiB, major faults {delta['major_faults']:,}, "
              f"anon {delta['anon_delta_bytes'] / 2**20:+,.0f} MiB (pages other jobs cached are not counted)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"loader": {k: str(v) for k, v in knobs.items()}, "passes": res, "cgroup": delta}, f, indent=1)


def cmd_isolate(a):
    n_ev = evict(a.evict)
    loader = load_factory(a.factory, a.kw)
    if a.madvise:
        from madvise_random import advise_random
        advise_random(a.madvise)
    if n_ev:
        print(f"evicted {n_ev} files from page cache")
    r = isolate(loader, a.rows, a.step_ms)
    print(f"{r['rows']} rows in {r['batches']} batches of {r['rows_per_batch']}, one process, ms per batch:")
    print("  read cold:", _fmt(r["read_cold_ms"]))
    print("  read warm:", _fmt(r["read_warm_ms"]))
    print("  collate:  ", _fmt(r["collate_ms"]))
    for v in r["verdicts"]:
        print("  -", v)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)


def cmd_analyze(a):
    waits, steps = _read_series(a.waits)
    if a.steps:
        steps = _read_series(a.steps)[0]
    waits = waits[a.skip:]
    steps = steps[a.skip:] if steps else None
    r = analyze(waits, step_ms=steps, num_workers=a.num_workers, stall_ms=a.stall_ms, floor_ms=a.floor_ms)
    print("wait:", _fmt(r["wait_ms"]), f" stalls>{a.stall_ms:g}ms: {r.get('stalls', 0)}")
    for v in r["verdicts"]:
        print("-", v)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)


# --- selftest --------------------------------------------------------------------


class _ToyDataset:
    """Module-level so spawned workers can unpickle it."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import torch
        return {"input_ids": torch.full((16,), i), "attention_mask": torch.ones(16, dtype=torch.long)}


def _selftest():
    import random

    rng = random.Random(0)
    base = lambda: [0.2 + rng.random() * 0.2 for _ in range(400)]  # noqa: E731
    healthy = base()
    chronic = [45 + rng.random() * 20 for _ in range(400)]
    periodic = base()
    for i in range(5, 400, 8):
        periodic[i] = 80 + rng.random() * 20
    tail = base()
    tail[100], tail[300] = 3000.0, 4500.0
    cheap = [5 + rng.random() for _ in range(400)]
    cases = [
        ("healthy", healthy, None, "HEALTHY", ()),
        ("chronic", chronic, None, "CHRONIC", ()),
        ("periodic", periodic, None, "PERIODIC", ("= num_workers",)),
        ("tail", tail, None, "TAIL", ()),
        ("not-bottleneck", cheap, [1400 + x for x in cheap], "NOT THE BOTTLENECK", ()),
    ]
    for name, w, s, want, extra in cases:
        r = analyze(w, step_ms=s, num_workers=8)
        joined = " | ".join(r["verdicts"])
        assert any(v.startswith(want) for v in r["verdicts"]), (name, joined)
        assert all(e in joined for e in extra), (name, joined)
        if name == "not-bottleneck":
            assert "CHRONIC" not in joined, joined  # 5 ms of a 1.4 s step is not starvation
        if name == "periodic":
            assert r["period"] == 8, r["period"]
    assert analyze([])["verdicts"][0].startswith("NO DATA")
    print("analyze: ok")

    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch not importable: skipped run/isolate checks")
        return
    for nw in (0, 2):
        kw = {"num_workers": nw, "persistent_workers": False}
        if nw:
            kw["prefetch_factor"] = 2
        loader = DataLoader(_ToyDataset(512), batch_size=8, **kw)
        r = bench(loader, batches=20, passes=1, warmup=1, step_ms=1.0)[0]
        assert r["batches"] == 20 and r["tokens_per_s"] > 0 and r["analysis"]["verdicts"], r
        d = describe(loader)
        assert d["num_workers"] == nw, d
    iso = isolate(DataLoader(_ToyDataset(512), batch_size=8), rows=64, step_ms=5.0)
    assert iso["batches"] == 8 and iso["workers_needed"] >= 1 and iso["verdicts"], iso
    print("run/isolate: ok")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--selftest" in argv:
        _selftest()
        print("selftest ok")
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, factory=True):
        if factory:
            p.add_argument("--factory", required=True, help="pkg.mod:fn or path/file.py:fn returning the DataLoader")
            p.add_argument("--kw", action="append", default=[], metavar="K=V", help="keyword argument for the factory")
            p.add_argument("--evict", action="append", default=[], metavar="PATH",
                           help="drop these files' cached pages first (POSIX_FADV_DONTNEED)")
            p.add_argument("--madvise", metavar="SUBSTR",
                           help="apply MADV_RANDOM to mappings matching SUBSTR before iterating")
        p.add_argument("--stall-ms", type=float, default=50.0)
        p.add_argument("--floor-ms", type=float, default=2.0, help="p50 wait above this (and >2%% of a step) is chronic")
        p.add_argument("--json", help="write full results here")

    p = sub.add_parser("run", help="fetch batches; waits, throughput, knobs, bytes fetched")
    common(p)
    p.add_argument("--batches", type=int, default=300)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1, help="leading batches excluded from stats (worker start)")
    p.add_argument("--step-ms", type=float, default=0.0, help="simulated training step per batch")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("isolate", help="read cold vs warm vs collate, in one process")
    common(p)
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--step-ms", type=float, help="your step time, to estimate workers needed")
    p.set_defaults(fn=cmd_isolate)

    p = sub.add_parser("analyze", help="classify per-step waits from a real run")
    p.add_argument("waits")
    p.add_argument("--steps", help="per-step wall times incl. wait, same order")
    p.add_argument("--num-workers", type=int)
    p.add_argument("--skip", type=int, default=0, help="leading steps to drop (warmup, compile)")
    common(p, factory=False)
    p.set_defaults(fn=cmd_analyze)

    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
