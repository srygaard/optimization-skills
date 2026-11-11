#!/usr/bin/env python3
"""Summarize torch.profiler traces: where a training step's time goes.

    python3 trace_report.py run/traces/                  # every trace file in the directory, one per rank
    python3 trace_report.py rank0.pt.trace.json.gz --top 20
    python3 trace_report.py run/traces/ --rank 2         # full report for rank 2, table for all

Reads the Chrome-format JSON that torch.profiler writes (export_chrome_trace or
tensorboard_trace_handler, plain or gzipped), over the steps the schedule recorded
(ProfilerStep#N spans). Sections:

  window    steps, GPU window, warnings (warmup signs, missing collectives)
  steps     per step: host span, GPU busy, DataLoader wait and GPU idle during it
  idle      gap sizes, what the idle time overlapped, the largest gaps with host context
  syncs     blocking CUDA/HIP calls by call site: GPU busy during, GPU idle after
  regions   record_function labels: host time, GPU time of the kernels launched inside
  host ops  long ops that launched almost nothing while the GPU sat idle
  comm      collective kernels: overlapped vs exposed, collectives per step
  kernels   GPU time by kernel family, top kernels
  ranks     one row per rank, when given several files

Only kernel, memcpy and memset events count as GPU work; gpu_user_annotation spans
are labels, not work. GPU events are kept when their launch (matched by correlation
id) falls inside the profiled steps. Stdlib only, Python >= 3.10. Missing fields
print as n/a. Always exits 0.
"""

import argparse
import bisect
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
API_CATS = {"cuda_runtime", "cuda_driver"}
HOST_CATS = {"cpu_op", "user_annotation", "python_function", "cuda_runtime", "cuda_driver"}
STEP_PREFIX = "ProfilerStep#"
DATALOADER_PREFIX = "enumerate(DataLoader)#"
SYNC_RE = re.compile(
    r"^(?:cuda|hip)(?:StreamSynchronize|DeviceSynchronize|EventSynchronize|Memcpy|MemcpyWithStream)$"
    r"|^cu(?:StreamSynchronize|CtxSynchronize|EventSynchronize)$"
)
MALLOC_RE = re.compile(r"^(?:cuda|hip)Malloc(?:Async|Managed)?$|^cuMemAlloc|^cuMemCreate")
COMPILE_WORDS = ("compile", "autotune", "dynamo", "inductor")
STEADY_COMPILE_NAMES = (
    "torch-compiled region",
    "torchdynamo cache lookup",
    "compiledfunction",
    "compiledfxgraph",
)
FRAME_RE = re.compile(r"^(?P<path>[^<(][^(]*)\((?P<line>\d+)\): (?P<func>.+)$")
NON_USER_TOPS = set(sys.stdlib_module_names) | {"torch", "torchvision", "torchaudio", "triton", "numpy"}
INDUCTOR_MM_RE = re.compile(r"_(?:mm|addmm|bmm|baddbmm|mm_plus_mm)(?:_|$)")
COLLECTIVE_RE = re.compile(r"(?:nccl|rccl)(?:Dev)?Kernel_([A-Za-z]+)", re.IGNORECASE)
GAP_BUCKETS = (
    (10.0, "<10us"),
    (100.0, "10-100us"),
    (1_000.0, "100us-1ms"),
    (10_000.0, "1-10ms"),
    (float("inf"), ">10ms"),
)
SYNC_TOL_US = 10.0
MAX_CALLS_HOST_OP = 8


# --- events and intervals -------------------------------------------------------


class Ev:
    __slots__ = ("name", "cat", "ts", "dur", "end", "pid", "tid", "corr", "stream")

    def __init__(self, raw):
        args = raw.get("args") or {}
        self.name = str(raw.get("name", ""))
        self.cat = raw.get("cat")
        self.ts = float(raw.get("ts", 0.0))
        self.dur = float(raw.get("dur", 0.0) or 0.0)
        self.end = self.ts + self.dur
        self.pid = raw.get("pid")
        self.tid = raw.get("tid")
        self.corr = args.get("correlation")
        self.stream = args.get("stream", raw.get("tid"))


def merge(intervals):
    """Sorted union of (start, end) pairs."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


class Union:
    """Merged intervals with overlap and membership queries."""

    def __init__(self, intervals):
        self.spans = merge(intervals)
        self.starts = [s for s, _ in self.spans]

    def length(self):
        return sum(e - s for s, e in self.spans)

    def overlap(self, start, end):
        if end <= start:
            return 0.0
        i = max(bisect.bisect_right(self.starts, start) - 1, 0)
        total = 0.0
        while i < len(self.spans) and self.spans[i][0] < end:
            lo = max(self.spans[i][0], start)
            hi = min(self.spans[i][1], end)
            if hi > lo:
                total += hi - lo
            i += 1
        return total

    def contains(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        return i >= 0 and t <= self.spans[i][1]


def complement(union, start, end):
    """Gaps of `union` inside [start, end]."""
    gaps = []
    cur = start
    for s, e in union.spans:
        if e <= cur:
            continue
        if s >= end:
            break
        if s > cur:
            gaps.append((cur, min(s, end)))
        cur = max(cur, e)
    if cur < end:
        gaps.append((cur, end))
    return gaps


def ms(us):
    if us is None:
        return "n/a"
    return f"{us / 1e3:.2f}"


def pct(part, whole):
    if not whole:
        return "n/a"
    return f"{100.0 * part / whole:.1f}%"


# --- names ------------------------------------------------------------------------


def family(name, cat):
    if cat == "gpu_memcpy":
        return "memcpy"
    if cat == "gpu_memset":
        return "memset"
    n = name.lower()
    if "nccl" in n or "rccl" in n:
        return "comm (NCCL/RCCL)"
    # Attention before GEMM: some flash kernels are CUTLASS-built and named cutlass_*.
    if any(k in n for k in ("flash", "fmha", "sdpa", "attn")):
        return "attention"
    if n.startswith("triton_tem_") and INDUCTOR_MM_RE.search(n):
        return "GEMM (Inductor template)"
    if any(k in n for k in ("gemm", "cutlass", "cublas", "nvjet", "xmma", "cijk_", "hipblaslt", "rocblas")):
        return "GEMM (vendor library)"
    if n.startswith("triton_"):
        return "Triton (Inductor)"
    if any(k in n for k in ("multi_tensor_apply", "adam", "foreach")):
        return "optimizer"
    if "elementwise" in n or "reduce_kernel" in n:
        return "aten elementwise/reduce"
    return "other"


def user_frame(name):
    """The frame name if it is user code (not stdlib, torch or site-packages)."""
    m = FRAME_RE.match(name)
    if not m:
        return None
    path = m.group("path")
    if "site-packages/" in path or "dist-packages/" in path:
        return None
    if path.startswith("/"):
        return name
    top = path.split("/")[0]
    if top.endswith(".py"):
        top = top[:-3]
    if top in NON_USER_TOPS:
        return None
    return name


def is_compile_event(name):
    n = name.lower()
    if not any(w in n for w in COMPILE_WORDS):
        return False
    return not any(s in n for s in STEADY_COMPILE_NAMES)


# --- loading and indexing ------------------------------------------------------------


def find_traces(targets):
    paths = []
    for target in targets:
        p = Path(target)
        if p.is_file():
            paths.append(p)
            continue
        if not p.is_dir():
            print(f"[warn] no such file or directory: {target}", file=sys.stderr)
            continue
        found = sorted(q for q in p.rglob("*") if q.is_file() and q.name.endswith((".json", ".json.gz")))
        traced = [q for q in found if "trace" in q.name]
        paths.extend(traced or found)
    return paths


def load(path):
    try:
        raw = path.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw)
    except Exception as exc:
        print(f"[warn] could not read {path}: {exc}", file=sys.stderr)
        return None


class Trace:
    """One rank's trace, indexed for the report sections."""

    def __init__(self, path, data):
        self.path = path
        self.warnings = []
        dist = data.get("distributedInfo") or {}
        self.rank = dist.get("rank")
        self.world = dist.get("world_size")
        self.backend = dist.get("backend")

        host = []
        gpu_all = []
        for raw in data.get("traceEvents", []):
            if raw.get("ph") != "X":
                continue
            cat = raw.get("cat")
            if cat in GPU_CATS:
                ev = Ev(raw)
                if ev.dur > 0:
                    gpu_all.append(ev)
            elif cat in HOST_CATS:
                host.append(Ev(raw))
        self.host = host
        self.parent = build_tree(host)
        self.has_stack = any(e.cat == "python_function" for e in host)

        self.steps = find_steps(host)
        if self.steps:
            self.h0 = self.steps[0][1]
            self.h1 = self.steps[-1][2]
        else:
            self.warnings.append("no ProfilerStep# spans (no schedule / prof.step()): using the whole trace")
            spans = [(e.ts, e.end) for e in host if e.cat in ("cpu_op", "user_annotation")]
            self.h0 = min((s for s, _ in spans), default=0.0)
            self.h1 = max((e for _, e in spans), default=0.0)

        self.launch_ts = {}
        for e in host:
            if e.cat in API_CATS and e.corr is not None:
                self.launch_ts[e.corr] = e.ts
        self.gpu = []
        self.gpu_launch = []
        excluded = 0
        for g in gpu_all:
            lt = self.launch_ts.get(g.corr)
            anchor = lt if lt is not None else g.ts
            if self.h0 <= anchor <= self.h1:
                self.gpu.append(g)
                self.gpu_launch.append(lt)
            else:
                excluded += 1
        self.excluded_gpu = excluded

        self.busy = Union((g.ts, g.end) for g in self.gpu)
        if self.gpu:
            self.g0 = self.busy.spans[0][0]
            self.g1 = self.busy.spans[-1][1]
        else:
            self.g0 = None
            self.g1 = None
        # Collectives are found by kernel name, not stream: a broadcast can run on the compute stream.
        self.gpu_is_comm = [is_comm_kernel(g.name) for g in self.gpu]
        self.has_comm = any(self.gpu_is_comm)
        self.comm_streams = sorted({g.stream for g, c in zip(self.gpu, self.gpu_is_comm, strict=True) if c}, key=str)
        self.comm = Union((g.ts, g.end) for g, c in zip(self.gpu, self.gpu_is_comm, strict=True) if c)
        self.compute = Union((g.ts, g.end) for g, c in zip(self.gpu, self.gpu_is_comm, strict=True) if not c)
        if self.gpu:
            self.gaps = complement(self.busy, self.g0, self.g1)
        else:
            self.gaps = []
        self.gap_starts = [s for s, _ in self.gaps]

        in_window = [e for e in host if e.end > self.h0 and e.ts < self.h1]
        self.data_events = [
            e for e in in_window if e.cat == "user_annotation" and e.name.startswith(DATALOADER_PREFIX)
        ]
        self.data = Union((e.ts, e.end) for e in self.data_events)
        self.syncs = [i for i, e in enumerate(host) if e.cat in API_CATS and SYNC_RE.match(e.name)]
        self.syncs = [i for i in self.syncs if self.h0 <= host[i].ts <= self.h1]
        self.sync_ends = sorted((host[i].end, i) for i in self.syncs)
        self.mallocs = sum(1 for e in in_window if e.cat in API_CATS and MALLOC_RE.match(e.name))
        self.compile_events = [
            e for e in in_window if e.cat in ("cpu_op", "user_annotation") and is_compile_event(e.name)
        ]
        self.add_warnings()

    # -- helpers --

    def ancestors(self, i):
        j = self.parent[i]
        while j is not None:
            yield j
            j = self.parent[j]

    def call_site(self, i):
        """(innermost label, innermost aten op, innermost user frame) enclosing host event i."""
        label = None
        op = None
        frame = None
        for j in self.ancestors(i):
            e = self.host[j]
            if label is None and e.cat == "user_annotation" and not e.name.startswith(STEP_PREFIX):
                label = e.name
            elif op is None and e.cat == "cpu_op":
                op = e.name
            elif frame is None and e.cat == "python_function":
                frame = user_frame(e.name)
        return label, op, frame

    def idle_caused(self, start, end):
        """GPU idle after a blocking call returns, counted only if the GPU went idle during the call.

        A gap that was already open when the call began was not caused by it.
        """
        j = bisect.bisect_right(self.gap_starts, end + SYNC_TOL_US) - 1
        if j < 0:
            return 0.0
        s, e = self.gaps[j]
        if e <= end or s < start - SYNC_TOL_US:
            return 0.0
        return e - max(s, end)

    def drained_by(self, gap):
        """The sync call that returned as this gap began, if any."""
        s, e = gap
        k = bisect.bisect_left(self.sync_ends, (s - SYNC_TOL_US, -1))
        while k < len(self.sync_ends) and self.sync_ends[k][0] <= e:
            i = self.sync_ends[k][1]
            if self.host[i].ts <= s + SYNC_TOL_US:
                return i
            k += 1
        return None

    def window_idle(self, start, end):
        """GPU idle inside [start, end], clipped to the GPU window."""
        if self.g0 is None:
            return 0.0
        lo = max(start, self.g0)
        hi = min(end, self.g1)
        if hi <= lo:
            return 0.0
        return (hi - lo) - self.busy.overlap(lo, hi)

    def add_warnings(self):
        if not self.gpu:
            self.warnings.append("no GPU kernels in the profiled steps (CPU-only trace, or activities lacked CUDA)")
        spans = [e - s for _, s, e in self.steps]
        if len(spans) >= 2 and min(spans) > 0 and max(spans) / min(spans) > 1.3:
            self.warnings.append(
                f"step spans vary {max(spans) / min(spans):.2f}x ({ms(min(spans))}-{ms(max(spans))} ms): the window may"
                " still be warming up (compile, autotune, allocator) or caught a stall; compare with the"
                " unprofiled steady step time"
            )
        if self.mallocs:
            self.warnings.append(
                f"{self.mallocs} device malloc calls in the window: the caching allocator was still growing,"
                " a sign of warmup"
            )
        if self.compile_events:
            spent = Union((e.ts, e.end) for e in self.compile_events).length()
            longest = max(self.compile_events, key=lambda e: e.dur)
            self.warnings.append(
                f"torch.compile ran inside the window for {ms(spent)} ms (longest event: {longest.name},"
                f" {ms(longest.dur)} ms): a recompile or first compile, not steady state"
            )
        if (self.world or 1) > 1 and self.gpu and not self.has_comm:
            self.warnings.append(
                f"world_size {self.world} but no collective kernels in the window: if prof.step() runs per"
                " micro-batch, the window may hold only gradient-accumulation steps"
            )


def build_tree(events):
    """Parent index of each host event: the event enclosing it on the same thread."""
    by_thread = defaultdict(list)
    for i, e in enumerate(events):
        by_thread[(e.pid, e.tid)].append(i)
    parent = [None] * len(events)
    for idxs in by_thread.values():
        idxs.sort(key=lambda i: (events[i].ts, -events[i].dur))
        stack = []
        for i in idxs:
            while stack and events[stack[-1]].end <= events[i].ts:
                stack.pop()
            if stack:
                parent[i] = stack[-1]
            stack.append(i)
    return parent


def find_steps(host):
    """Host-side ProfilerStep#N spans, one per step (the longest row of each name)."""
    best = {}
    for e in host:
        if e.cat != "user_annotation" or not e.name.startswith(STEP_PREFIX):
            continue
        if e.name not in best or e.dur > best[e.name].dur:
            best[e.name] = e
    return sorted(((e.name, e.ts, e.end) for e in best.values()), key=lambda x: x[1])


# --- sections ----------------------------------------------------------------------


def section_window(t):
    print(f"== {t.path}")
    if t.rank is not None:
        print(f"   {t.backend} rank {t.rank}/{t.world}")
    print(f"   steps: {len(t.steps)}  host window {ms(t.h1 - t.h0)} ms", end="")
    if t.g0 is not None:
        print(f"  GPU window {ms(t.g1 - t.g0)} ms  ({len(t.gpu)} GPU events)", end="")
    if t.excluded_gpu:
        print(f"  ({t.excluded_gpu} launched outside the steps left out)", end="")
    print(f"  python stacks: {'yes' if t.has_stack else 'no'}")
    for w in t.warnings:
        print(f"   [warn] {w}")


def step_rows(t):
    """Per step: host span, GPU interval, busy, DataLoader wait, GPU idle during it."""
    firsts = []
    for _name, s, e in t.steps:
        starts = [g.ts for g, lt in zip(t.gpu, t.gpu_launch, strict=True) if lt is not None and s <= lt <= e]
        firsts.append(min(starts) if starts else None)
    rows = []
    known = [(i, f) for i, f in enumerate(firsts) if f is not None]
    for k, (name, s, e) in enumerate(t.steps):
        data = sum(min(d.end, e) - max(d.ts, s) for d in t.data_events if d.end > s and d.ts < e)
        data_idle = sum(t.window_idle(max(d.ts, s), min(d.end, e)) for d in t.data_events if d.end > s and d.ts < e)
        gpu_span = None
        busy = None
        compute = None
        pos = [j for j, (i, _) in enumerate(known) if i == k]
        if pos and t.g0 is not None:
            j = pos[0]
            lo = known[j][1]
            hi = known[j + 1][1] if j + 1 < len(known) else t.g1
            lo = max(lo, t.g0)
            gpu_span = hi - lo
            busy = t.busy.overlap(lo, hi)
            compute = t.compute.overlap(lo, hi)
        rows.append((name, e - s, gpu_span, busy, compute, data, data_idle))
    return rows


def section_steps(t, top):
    print("\n== Steps (GPU interval = from the step's first kernel to the next step's first kernel)")
    if not t.steps:
        print("   n/a")
        return
    print(
        f"   {'step':<18}{'host ms':>10}{'GPU ms':>10}{'busy':>8}{'compute':>9}{'data wait ms':>14}"
        f"{'GPU idle in it':>16}"
    )
    rows = step_rows(t)
    if len(rows) > top:
        spans = sorted(r[1] for r in rows)
        p90 = spans[min(len(spans) - 1, int(0.9 * len(spans)))]
        print(
            f"   {len(rows)} steps, host span median {ms(statistics.median(spans))} ms, p90 {ms(p90)} ms,"
            f" max {ms(spans[-1])} ms; the {top} slowest, in order:"
        )
        slowest = set(sorted(range(len(rows)), key=lambda i: -rows[i][1])[:top])
        rows = [r for i, r in enumerate(rows) if i in slowest]
    for name, span, gpu_span, busy, compute, data, data_idle in rows:
        busy_txt = pct(busy, gpu_span) if gpu_span else "n/a"
        compute_txt = pct(compute, gpu_span) if gpu_span else "n/a"
        print(
            f"   {name:<18}{ms(span):>10}{ms(gpu_span):>10}{busy_txt:>8}{compute_txt:>9}{ms(data):>14}"
            f"{ms(data_idle):>16}"
        )
    if t.has_comm:
        print("   busy counts any stream; compute leaves out collective kernels, which spin while waiting.")


def gap_context(t, gap):
    """Host activity covering at least half of the gap: labels, aten op, user frame, API call."""
    s, e = gap
    need = 0.5 * (e - s)
    labels = []
    op = None
    frame = None
    api = None
    for x in t.host:
        if x.dur < need or x.ts >= e or x.end <= s:
            continue
        if min(x.end, e) - max(x.ts, s) < need:
            continue
        if x.cat == "user_annotation" and not x.name.startswith(STEP_PREFIX):
            labels.append(x)
        elif x.cat == "cpu_op" and (op is None or x.dur < op.dur):
            op = x
        elif x.cat == "python_function" and user_frame(x.name) and (frame is None or x.dur < frame.dur):
            frame = x
        elif x.cat in API_CATS and (api is None or x.dur < api.dur):
            api = x
    labels.sort(key=lambda x: -x.dur)
    parts = []
    names = []
    for x in labels:
        if x.name not in names:
            names.append(x.name)
    if names:
        parts.append(" > ".join(n[:40] for n in names))
    if op is not None:
        parts.append(f"op {op.name[:40]}")
    if api is not None:
        parts.append(f"api {api.name}")
    if frame is not None:
        parts.append(f"at {frame.name[:60]}")
    if not parts:
        return "no host event covers it (GIL, GC, allocator or untraced work?)"
    return "  ".join(parts)


def section_idle(t, top):
    print("\n== Idle")
    if t.g0 is None:
        print("   n/a (no GPU events)")
        return
    window = t.g1 - t.g0
    idle = window - t.busy.length()
    print(f"   GPU busy {pct(window - idle, window)}  idle {ms(idle)} ms of {ms(window)} ms ({pct(idle, window)})")
    if t.has_comm:
        compute = t.compute.length()
        print(f"   compute streams busy {pct(compute, window)} (collective kernels excluded)")
    buckets = defaultdict(lambda: [0, 0.0])
    data_idle = 0.0
    drain_idle = 0.0
    for gap in t.gaps:
        d = gap[1] - gap[0]
        for limit, label in GAP_BUCKETS:
            if d < limit:
                buckets[label][0] += 1
                buckets[label][1] += d
                break
        if t.drained_by(gap) is not None:
            drain_idle += d
        else:
            data_idle += t.data.overlap(gap[0], gap[1])
    for _, label in GAP_BUCKETS:
        if label in buckets:
            n, total = buckets[label]
            print(f"   gaps {label:<10} n={n:<6} {ms(total):>10} ms")
    print(f"   idle that began as a sync returned (queue drained): {ms(drain_idle)} ms ({pct(drain_idle, idle)})")
    print(f"   other idle while the host waited in DataLoader __next__: {ms(data_idle)} ms ({pct(data_idle, idle)})")
    largest = sorted(t.gaps, key=lambda g: g[0] - g[1])[:top]
    if not largest:
        return
    print(f"   largest {len(largest)} gaps (offset from GPU window start):")
    for gap in largest:
        tag = ""
        i = t.drained_by(gap)
        if i is not None:
            tag = f"[after {t.host[i].name}] "
        offset = gap[0] - t.g0
        print(f"   {ms(gap[1] - gap[0]):>9} ms @{ms(offset):>9}  {tag}{gap_context(t, gap)}")


def section_syncs(t, top):
    print("\n== Syncs (host blocked until the GPU caught up)")
    if not t.syncs:
        print("   none in the window")
        return
    groups = defaultdict(lambda: {"n": 0, "dur": 0.0, "max": 0.0, "busy": 0.0, "after": 0.0, "drained": 0})
    for i in t.syncs:
        e = t.host[i]
        g = groups[(e.name,) + t.call_site(i)]
        after = t.idle_caused(e.ts, e.end)
        g["n"] += 1
        g["dur"] += e.dur
        g["max"] = max(g["max"], e.dur)
        g["busy"] += t.busy.overlap(e.ts, e.end)
        g["after"] += after
        if after > 0:
            g["drained"] += 1
    rows = sorted(groups.items(), key=lambda kv: (-kv[1]["after"], -kv[1]["dur"]))
    print(f"   {'n':>5}{'wait ms':>10}{'max ms':>9}{'GPU busy':>10}{'idle after':>12}  kind              call site")
    for (api, label, op, frame), g in rows[:top]:
        busy = g["busy"] / g["dur"] if g["dur"] else 0.0
        if g["drained"] * 2 >= g["n"]:
            kind = "drains queue"
        elif busy <= 0.1:
            kind = "GPU already idle"
        else:
            kind = "queue stays full"
        where = " | ".join(x[:50] for x in (api, label, op, frame) if x)
        print(
            f"   {g['n']:>5}{ms(g['dur']):>10}{ms(g['max']):>9}{100 * busy:>9.1f}%{ms(g['after']):>12}"
            f"  {kind:<17} {where}"
        )
    print("   The cost of a sync is the GPU idle after it returns (the queue is empty until the host launches")
    print("   again), not the wait: during the wait the GPU was running queued work.")


def section_regions(t, top):
    print("\n== Regions (record_function labels; GPU = kernels launched while the label was open, any thread)")
    by_name = defaultdict(list)
    for e in t.host:
        if e.cat != "user_annotation" or e.end <= t.h0 or e.ts >= t.h1:
            continue
        if e.name.startswith(STEP_PREFIX) or e.name.startswith("nccl:"):
            continue
        by_name[e.name].append(e)
    if not by_name:
        print("   no labels in the window (add torch.profiler.record_function around forward/backward/optimizer)")
        return
    unions = {name: Union((e.ts, e.end) for e in evs) for name, evs in by_name.items()}
    gpu_time = defaultdict(float)
    unlabeled = 0.0
    total = 0.0
    for g, lt in zip(t.gpu, t.gpu_launch, strict=True):
        total += g.dur
        hit = False
        if lt is not None:
            for name, u in unions.items():
                if u.contains(lt):
                    gpu_time[name] += g.dur
                    hit = True
        if not hit:
            unlabeled += g.dur
    rows = sorted(by_name.items(), key=lambda kv: (-gpu_time[kv[0]], -sum(e.dur for e in kv[1])))
    print(f"   {'n':>5}{'host ms':>11}{'GPU ms':>11}{'GPU share':>11}  label")
    for name, evs in rows[:top]:
        host_ms = sum(e.dur for e in evs)
        print(f"   {len(evs):>5}{ms(host_ms):>11}{ms(gpu_time[name]):>11}{pct(gpu_time[name], total):>11}  {name[:70]}")
    if t.gpu:
        print(f"   {'':>5}{'':>11}{ms(unlabeled):>11}{pct(unlabeled, total):>11}  (launched outside every label)")
        print("   Nested labels count the same kernel in each; shares can sum past 100%.")


def section_host_ops(t, slow_op_us, top):
    print(f"\n== Host ops >= {slow_op_us:.0f} us launching <= {MAX_CALLS_HOST_OP} CUDA calls (pure host work)")
    calls = [0] * len(t.host)
    for i, e in enumerate(t.host):
        if e.cat in API_CATS:
            for j in t.ancestors(i):
                calls[j] += 1
    slow = set()
    for i, e in enumerate(t.host):
        if e.cat != "cpu_op" or e.dur < slow_op_us or calls[i] > MAX_CALLS_HOST_OP:
            continue
        if e.end <= t.h0 or e.ts >= t.h1:
            continue
        slow.add(i)
    inner = set(slow)
    for i in slow:
        for j in t.ancestors(i):
            inner.discard(j)
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for i in inner:
        e = t.host[i]
        a = agg[e.name]
        a[0] += 1
        a[1] += e.dur
        a[2] = max(a[2], e.dur)
        a[3] += t.window_idle(e.ts, e.end)
    if not agg:
        print("   none")
        return
    rows = sorted(agg.items(), key=lambda kv: (-kv[1][3], -kv[1][1]))
    print(f"   {'n':>5}{'host ms':>10}{'max ms':>9}{'GPU idle meanwhile':>20}  op")
    for name, (n, dur, longest, idle) in rows[:top]:
        print(f"   {n:>5}{ms(dur):>10}{ms(longest):>9}{ms(idle):>17} ms  {name[:60]}")
    print("   Costly only when the GPU idled meanwhile; at new shapes (warmup) this is often library heuristics.")


def is_comm_kernel(name):
    n = name.lower()
    return "nccl" in n or "rccl" in n


def collective_of(name):
    m = COLLECTIVE_RE.search(name)
    if m:
        return m.group(1)
    return "other"


def section_comm(t):
    print("\n== Comm (collective kernels vs compute on other streams)")
    if not t.has_comm:
        print("   no collective kernels in the window")
        return
    comm_kernel = sum(g.dur for g, c in zip(t.gpu, t.gpu_is_comm, strict=True) if c)
    comm_span = t.comm.length()
    exposed = sum((e - s) - t.compute.overlap(s, e) for s, e in t.comm.spans)
    window = t.g1 - t.g0
    print(f"   comm streams {t.comm_streams}  kernel time {ms(comm_kernel)} ms  union {ms(comm_span)} ms")
    print(
        f"   exposed (no compute running) {ms(exposed)} ms = {pct(exposed, comm_span)} of comm,"
        f" {pct(exposed, window)} of the GPU window"
    )
    print("   Collective kernels spin while waiting for late ranks: kernel time is not cost; exposed time is.")
    if not t.steps:
        return
    counts = defaultdict(lambda: [0] * len(t.steps))
    starts = [s for _, s, _ in t.steps]
    for g, lt, c in zip(t.gpu, t.gpu_launch, t.gpu_is_comm, strict=True):
        if not c or lt is None:
            continue
        k = bisect.bisect_right(starts, lt) - 1
        if k >= 0:
            counts[collective_of(g.name)][k] += 1
    if len(t.steps) <= 12:
        header = "".join(f"{name.replace(STEP_PREFIX, 'S'):>8}" for name, _, _ in t.steps)
        print(f"   collectives per step: {'':<14}{header}")
        for coll, row in sorted(counts.items()):
            print(f"   {coll:<36}{''.join(f'{c:>8}' for c in row)}")
        return
    print(f"   collectives over {len(t.steps)} steps:")
    for coll, row in sorted(counts.items()):
        with_any = sum(1 for c in row if c)
        print(f"   {coll:<16} total {sum(row):>6}  per step min {min(row)} max {max(row)}  steps with any {with_any}")


def section_kernels(t, top):
    print("\n== Kernels")
    if not t.gpu:
        print("   n/a (no GPU events)")
        return
    fam = defaultdict(lambda: [0.0, 0])
    kern = defaultdict(lambda: [0.0, 0])
    for g in t.gpu:
        f = fam[family(g.name, g.cat)]
        f[0] += g.dur
        f[1] += 1
        k = kern[g.name]
        k[0] += g.dur
        k[1] += 1
    total = sum(v[0] for v in fam.values())
    print(f"   by family ({ms(total)} ms of kernel + copy time; streams overlap, so this can exceed busy time)")
    for name, (dur, n) in sorted(fam.items(), key=lambda kv: -kv[1][0]):
        print(f"   {ms(dur):>10} ms {pct(dur, total):>6}  n={n:<6} avg {dur / n:8.1f} us  {name}")
    print(f"   top {top} kernels")
    for name, (dur, n) in sorted(kern.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"   {ms(dur):>10} ms {pct(dur, total):>6}  n={n:<6} avg {dur / n:8.1f} us  {name[:90]}")


def summary_row(t):
    n = max(len(t.steps), 1)
    spans = [e - s for _, s, e in t.steps]
    window = (t.g1 - t.g0) if t.g0 is not None else 0.0
    busy = t.busy.length()
    data = sum(min(d.end, t.h1) - max(d.ts, t.h0) for d in t.data_events)
    data_idle = sum(t.window_idle(d.ts, d.end) for d in t.data_events)
    sync = sum(t.host[i].dur for i in t.syncs)
    comm_kernel = sum(g.dur for g, c in zip(t.gpu, t.gpu_is_comm, strict=True) if c)
    exposed = sum((e - s) - t.compute.overlap(s, e) for s, e in t.comm.spans)
    return {
        "rank": t.rank,
        "path": t.path.name,
        "steps": len(t.steps),
        "step": statistics.median(spans) if spans else None,
        "busy": pct(busy, window),
        "compute": pct(t.compute.length(), window),
        "idle": (window - busy) / n,
        "data": data / n,
        "data_idle": data_idle / n,
        "sync": sync / n,
        "comm": comm_kernel / n,
        "exposed": exposed / n,
    }


def section_ranks(rows):
    print("\n== Ranks (per-step means; ms)")
    print(
        f"   {'rank':>5}{'steps':>6}{'p50 step':>10}{'busy':>8}{'compute':>9}{'idle':>9}{'data wait':>11}"
        f"{'idle in data':>14}{'sync':>9}{'comm kern':>11}{'exposed':>9}  file"
    )
    max_data = max(rows, key=lambda r: r["data"])
    min_comm = min(rows, key=lambda r: r["comm"])
    for r in rows:
        flags = []
        if r is max_data and r["data"] > 0:
            flags.append("most data wait")
        if r is min_comm and r["comm"] > 0:
            flags.append("least comm-kernel time: others may be waiting on it")
        flag = f"  <- {'; '.join(flags)}" if flags else ""
        print(
            f"   {'-' if r['rank'] is None else r['rank']:>5}{r['steps']:>6}{ms(r['step']):>10}{r['busy']:>8}{r['compute']:>9}{ms(r['idle']):>9}"
            f"{ms(r['data']):>11}{ms(r['data_idle']):>14}{ms(r['sync']):>9}{ms(r['comm']):>11}"
            f"{ms(r['exposed']):>9}  {r['path']}{flag}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("traces", nargs="+", help="trace files (.json or .json.gz) or directories of them")
    ap.add_argument("--top", type=int, default=12, help="rows per table")
    ap.add_argument("--rank", type=int, help="print the full report for this rank (default: the first file)")
    ap.add_argument("--all", action="store_true", help="print the full report for every file")
    ap.add_argument("--slow-op-us", type=float, default=200.0, help="threshold for the host-ops table")
    args = ap.parse_args()

    paths = find_traces(args.traces)
    if not paths:
        print("No trace files found.")
        return 0
    rows = []
    reported = False
    for path in paths:
        data = load(path)
        if data is None:
            continue
        t = Trace(path, data)
        del data
        want = args.all or (args.rank is not None and t.rank == args.rank)
        if args.rank is None and not reported:
            want = True
        if want:
            section_window(t)
            section_steps(t, args.top)
            section_idle(t, args.top)
            section_syncs(t, args.top)
            section_regions(t, args.top)
            section_host_ops(t, args.slow_op_us, args.top)
            section_comm(t)
            section_kernels(t, args.top)
            print()
            reported = True
        rows.append(summary_row(t))
    if len(rows) > 1:
        section_ranks(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
