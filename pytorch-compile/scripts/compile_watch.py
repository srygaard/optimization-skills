#!/usr/bin/env python3
"""In-process per-step compile accounting, plus a self-test fixture.

Unlike the other two scripts in this directory, this one imports torch -- it runs
inside your training process.

Usage as a library:

    from compile_watch import CompileWatch
    watch = CompileWatch()
    for step, batch in enumerate(loader):
        loss = model(**batch)
        watch.step(step, shape=tuple(batch["input_ids"].shape))

It prints per-step DELTAS, not cumulative totals. By the sixth distinct shape the
running totals no longer tell you which step was expensive.

Usage as a fixture, to produce a known-good log and trace for the other scripts:

    TORCH_TRACE=./tt TORCH_LOGS=graph_breaks,recompiles \\
        python3 compile_watch.py --selftest 2> run.err
    python3 log_triage.py run.err
    python3 trace_summary.py ./tt
"""

import argparse
import os
import sys
import time


def _compile_times():
    """Cumulative per-phase compile seconds, defensively across versions."""
    try:
        import torch._dynamo.utils as u
    except Exception:
        return {}
    for kwargs in ({"repr": "csv", "aggregate": True}, {"repr": "csv"}, {}):
        try:
            raw = u.compile_times(**kwargs)
        except Exception:
            continue
        return _parse_compile_times(raw)
    return {}


def _parse_compile_times(raw):
    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            try:
                out[k] = float(v[0] if isinstance(v, (list, tuple)) else v)
            except Exception:
                continue
        return out
    if isinstance(raw, tuple) and len(raw) == 2:
        names, vals = raw
        out = {}
        for n, v in zip(names, vals):
            try:
                out[str(n)] = float(v)
            except Exception:
                continue
        return out
    if isinstance(raw, str):
        lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
        if len(lines) >= 2:
            names = lines[0].split(",")
            vals = lines[-1].split(",")
            out = {}
            for n, v in zip(names, vals):
                try:
                    out[n.strip()] = float(v)
                except Exception:
                    continue
            return out
    return {}


def _unique_graphs():
    try:
        from torch._dynamo.utils import counters
        return int(counters.get("stats", {}).get("unique_graphs", 0))
    except Exception:
        return 0


def _break_count():
    try:
        from torch._dynamo.utils import counters
        return sum(counters.get("graph_break", {}).values())
    except Exception:
        return 0


class CompileWatch:
    """Per-step compile-phase deltas and new-shape detection."""

    def __init__(self, top=6, stream=sys.stderr):
        self.top = top
        self.stream = stream
        self.prev_phases = _compile_times()
        self.prev_graphs = _unique_graphs()
        self.prev_breaks = _break_count()
        self.seen_shapes = set()
        self.t0 = time.perf_counter()

    def step(self, step=None, shape=None):
        now = time.perf_counter()
        wall = now - self.t0
        self.t0 = now

        phases = _compile_times()
        deltas = {
            k: phases[k] - self.prev_phases.get(k, 0.0)
            for k in phases
            if phases[k] - self.prev_phases.get(k, 0.0) > 0.01
        }
        self.prev_phases = phases

        graphs = _unique_graphs()
        breaks = _break_count()
        new_graphs = graphs - self.prev_graphs
        recompiled = new_graphs > 0
        new_breaks = breaks - self.prev_breaks
        self.prev_graphs, self.prev_breaks = graphs, breaks

        new_shape = shape is not None and shape not in self.seen_shapes
        if shape is not None:
            self.seen_shapes.add(shape)

        tags = []
        if new_shape:
            tags.append(f"new shape {shape}")
        if recompiled:
            tags.append(f"recompiled (+{new_graphs} graph)")
        if new_breaks:
            tags.append(f"+{new_breaks} graph breaks")
        if deltas:
            tags.append("compiling")

        label = f"step {step}" if step is not None else "step"
        suffix = ("  [" + ", ".join(tags) + "]") if tags else ""
        print(f"{label}: {wall * 1e3:8.1f} ms{suffix}", file=self.stream)

        for name, secs in sorted(deltas.items(), key=lambda kv: -kv[1])[: self.top]:
            print(f"    {name:<44}{secs:8.2f} s", file=self.stream)

        self.stream.flush()
        return {
            "wall_s": wall,
            "phase_deltas": deltas,
            "recompiled": recompiled,
            "new_shape": new_shape,
            "new_graph_breaks": new_breaks,
            "is_measurable": not (new_shape or recompiled or deltas),
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- self-test fixture -------------------------------------------------------

def selftest():
    """Produce, by construction: >=1 graph break, >=1 shape-driven recompile,
    and a non-trivial TORCH_TRACE. Runs on CPU."""
    import torch
    import torch.nn as nn

    print(f"torch {torch.__version__}", file=sys.stderr)
    print(f"TORCH_LOGS={os.environ.get('TORCH_LOGS')!r} "
          f"TORCH_TRACE={os.environ.get('TORCH_TRACE')!r}", file=sys.stderr)
    if not os.environ.get("TORCH_LOGS"):
        print("[warn] TORCH_LOGS is not set -- log_triage.py will find nothing.",
              file=sys.stderr)

    class Toy(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.a = nn.Linear(d, d)
            self.b = nn.Linear(d, d)
            # a lazily grown buffer: the RC-3 shape, in miniature
            self.register_buffer("cache", torch.zeros(8, d))

        def forward(self, x):
            h = torch.relu(self.a(x))
            # GB-1 shape: a python-scalar query inside forward
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                h = h * 1.0
            # GB-3 shape: data-dependent branch
            if (h > 0).any():
                h = h + 0.001
            n = x.shape[1]
            if self.cache.shape[0] < n:                 # regrow -> guard failure
                self.cache = torch.zeros(n, x.shape[2])
            return self.b(h) + self.cache[:n].unsqueeze(0)

    model = Toy()
    compiled = torch.compile(model)
    watch = CompileWatch()

    shapes = [(2, 8, 64), (2, 8, 64), (2, 12, 64), (2, 12, 64), (2, 20, 64), (2, 20, 64)]
    for i, shape in enumerate(shapes):
        out = compiled(torch.randn(*shape))
        out.sum().backward()
        watch.step(i, shape=shape)

    print("\nself-test done.", file=sys.stderr)
    print("  python3 log_triage.py run.err", file=sys.stderr)
    trace = os.environ.get("TORCH_TRACE")
    if trace:
        print(f"  python3 trace_summary.py {trace}", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="run a small repro that produces breaks and recompiles")
    args = ap.parse_args()
    if args.selftest:
        try:
            return selftest()
        except ImportError:
            print("torch is not installed; --selftest needs it.", file=sys.stderr)
            return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
