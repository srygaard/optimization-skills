#!/usr/bin/env python3
"""Summarize a TORCH_TRACE structured log: where compile time actually goes.

    TORCH_TRACE=/abs/dir python train.py
    python3 trace_summary.py /abs/dir
    python3 trace_summary.py /abs/dir --tlparse out/ --top-kernels 15

Prints the phase table, per-compile-id graph size and IR blowup, guard counts,
restarts, the top Triton kernels by compile time, and a verdict naming the next
reference section.

Stdlib only. Field names differ across PyTorch versions: anything missing prints
as n/a rather than raising. Always exits 0.
"""

import argparse
import gzip
import io
import json
import os
import sys
from collections import defaultdict

BOOKKEEPING = {
    "rank", "frame_id", "frame_compile_id", "attempt", "compile_id",
    "compiled_autograd_id", "has_payload", "timestamp", "pathname", "lineno",
    "thread", "training_ir", "describe_source",
}

METRIC_KEYS = [
    "graph_op_count", "graph_node_count", "graph_input_count", "ir_count",
    "guard_count", "shape_env_guard_count", "cache_size",
    "accumulated_cache_size", "entire_frame_compile_time_s",
    "backend_compile_time_s", "inductor_compile_time_s", "code_gen_time_s",
    "dynamo_time_before_restart_s", "restart_reasons", "has_guarded_code",
    "fail_type", "fail_reason", "specialize_float", "tensorify_float_attempt",
    "tensorify_float_success", "structured_logging_overhead_s",
    "triton_kernel_compile_times_us", "co_name", "frame_key", "is_forward",
]


# --- input -------------------------------------------------------------------

def find_trace_files(target, rank=None):
    if os.path.isfile(target):
        return [target]
    if not os.path.isdir(target):
        return []
    hits = []
    for name in sorted(os.listdir(target)):
        if "torch_trace" in name or name.endswith((".log", ".log.gz")):
            hits.append(os.path.join(target, name))
    if rank is not None:
        want = f"rank_{rank}"
        picked = [h for h in hits if want in os.path.basename(h)]
        if picked:
            return picked
    ranked = [h for h in hits if "rank_0" in os.path.basename(h)]
    return ranked or hits[:1]


def read_lines(path):
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
    except Exception as exc:
        print(f"[warn] could not read {path}: {exc}", file=sys.stderr)
        return []
    stream = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", errors="replace")
    return [ln.rstrip("\n") for ln in stream]


def parse_records(lines):
    """Yield (meta, payload_text). A metadata line is a line parsing as a JSON
    object; if it has has_payload, the payload is the indented run that follows."""
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        if not stripped.startswith("{"):
            i += 1
            continue
        try:
            meta = json.loads(stripped)
        except Exception:
            i += 1
            continue
        if not isinstance(meta, dict):
            i += 1
            continue
        i += 1
        payload = None
        if meta.get("has_payload"):
            buf = []
            while i < n:
                nxt = lines[i]
                if nxt.lstrip().startswith("{"):
                    try:
                        cand = json.loads(nxt.lstrip())
                        if isinstance(cand, dict):
                            break
                    except Exception:
                        pass
                if nxt and not (nxt.startswith("\t") or nxt.startswith("    ")):
                    break
                buf.append(nxt[1:] if nxt.startswith("\t") else nxt[4:])
                i += 1
            payload = "\n".join(buf)
        yield meta, payload


def kind_of(meta):
    for k in meta:
        if k not in BOOKKEEPING:
            return k
    return None


def compile_id(meta):
    f, c = meta.get("frame_id"), meta.get("frame_compile_id")
    if f is None and c is None:
        return meta.get("compile_id") or "-"
    return f"{f}/{c}"


# --- accumulation ------------------------------------------------------------

class Trace:
    def __init__(self):
        self.records = 0
        self.metrics = {}                      # compile_id -> dict
        self.phases = defaultdict(lambda: [0.0, 0])   # name -> [seconds, count]
        self.open_events = defaultdict(list)
        self.attempts = 0
        self.overhead = None
        self.kernels = {}                      # name -> seconds
        self.compile_ids = set()


def add_chromium(t, ev):
    try:
        name = ev.get("name")
        ph = ev.get("ph")
        if not name:
            return
        if ph == "X" and ev.get("dur") is not None:
            t.phases[name][0] += float(ev["dur"]) / 1e6
            t.phases[name][1] += 1
        elif ph == "B":
            t.open_events[(name, ev.get("pid"), ev.get("tid"))].append(float(ev["ts"]))
        elif ph == "E":
            stack = t.open_events.get((name, ev.get("pid"), ev.get("tid")))
            if stack:
                start = stack.pop()
                t.phases[name][0] += (float(ev["ts"]) - start) / 1e6
                t.phases[name][1] += 1
    except Exception:
        return


def ingest(t, paths):
    for path in paths:
        for meta, payload in parse_records(read_lines(path)):
            t.records += 1
            kind = kind_of(meta)
            if kind is None:
                continue
            cid = compile_id(meta)
            if meta.get("frame_compile_id") is not None:
                t.compile_ids.add(cid)
            if meta.get("attempt"):
                t.attempts = max(t.attempts, int(meta["attempt"]) + 1)

            if kind in ("compilation_metrics", "bwd_compilation_metrics"):
                body = _body(meta, kind, payload)
                if not isinstance(body, dict):
                    continue
                row = t.metrics.setdefault(cid, {})
                for k in METRIC_KEYS:
                    if k in body and body[k] is not None:
                        row[k] = body[k]
                if "structured_logging_overhead_s" in body:
                    t.overhead = body["structured_logging_overhead_s"]
                kt = body.get("triton_kernel_compile_times_us")
                if isinstance(kt, dict):
                    for name, us in kt.items():
                        try:
                            t.kernels[name] = t.kernels.get(name, 0.0) + float(us) / 1e6
                        except Exception:
                            continue

            elif kind == "chromium_event":
                body = _body(meta, kind, payload)
                if isinstance(body, dict):
                    add_chromium(t, body)
                elif isinstance(body, list):
                    for ev in body:
                        if isinstance(ev, dict):
                            add_chromium(t, ev)


def _body(meta, kind, payload):
    """Record content. The inline value is usually an empty placeholder ({}) with
    the real content in the payload, so a non-empty inline value wins and an empty
    one falls through."""
    val = meta.get(kind)
    if isinstance(val, (dict, list)) and val:
        return val
    if payload:
        try:
            return json.loads(payload)
        except Exception:
            return val if isinstance(val, (dict, list)) else None
    return val if isinstance(val, (dict, list)) else None


def load_tlparse(t, d):
    path = os.path.join(d, "chromium_events.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as fh:
            events = json.load(fh)
    except Exception:
        return False
    t.phases.clear()
    t.open_events.clear()
    for ev in events if isinstance(events, list) else []:
        if isinstance(ev, dict):
            add_chromium(t, ev)
    return True


# --- output ------------------------------------------------------------------

def fmt(v, nd=1):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, list):
        return "[" + ", ".join(str(x) for x in v) + "]" if v else "[]"
    return str(v)


def emit(t, top_kernels, only):
    print(f"== TORCH_TRACE summary")
    print(
        f"   records {t.records:,}   compile ids {len(t.compile_ids)}"
        f"   attempts {t.attempts or 1}"
        f"   logging overhead {fmt(t.overhead, 2)} s"
    )
    if t.overhead:
        print("   [note] tracing overhead is real; a traced run is not comparable")
        print("          to an untraced baseline.")
    print()

    # phases
    rows = sorted(t.phases.items(), key=lambda kv: -kv[1][0])
    print("== Phases (attribution, NOT a partition -- they overlap and do not sum)")
    if not rows:
        print("  no chromium_event records found (try --tlparse <dir>)")
    else:
        print(f"  {'phase':<46}{'total_s':>10}{'n':>7}")
        for name, (secs, n) in rows[:14]:
            print(f"  {name[:46]:<46}{secs:>10.1f}{n:>7}")
        pmap = {k: v[0] for k, v in t.phases.items()}
        cg = pmap.get("code_gen")
        aw = pmap.get("async_compile.wait")
        if cg and aw and cg > 0:
            share = 100.0 * aw / cg
            print(f"  [note] async_compile.wait / code_gen = {share:.0f}%")
            if share >= 80:
                print("         -> TRITON-KERNEL-COMPILE bound. No Dynamo or Inductor")
                print("            config flag reaches this, and more compile workers")
                print("            will not help. Find the kernel (below).")
            else:
                print("         -> CODEGEN bound (IR size). Config levers can help;")
                print("            see references/compile-time.md sections 3-4.")
    print()

    # graph size / blowup
    print("== Graph size and blowup, per compile id")
    print(f"  {'compile_id':<12}{'fn':<22}{'ops':>8}{'nodes':>8}{'ir':>10}{'blowup':>9}{'inputs':>8}")
    worst_blowup = 0.0
    for cid, m in sorted(t.metrics.items()):
        ops = m.get("graph_op_count")
        ir = m.get("ir_count")
        blow = f"{ir / ops:.1f}x" if (ops and ir) else "n/a"
        if ops and ir:
            worst_blowup = max(worst_blowup, ir / ops)
        print(
            f"  {cid:<12}{str(m.get('co_name', '-'))[:22]:<22}"
            f"{fmt(ops):>8}{fmt(m.get('graph_node_count')):>8}"
            f"{fmt(ir):>10}{blow:>9}{fmt(m.get('graph_input_count')):>8}"
        )
    if not t.metrics:
        print("  no compilation_metrics records found")
    if worst_blowup:
        print(f"  [note] blowup = ir_count / graph_op_count; worst {worst_blowup:.1f}x.")
        if worst_blowup > 20:
            print("         Above ~20x is worth attacking when codegen dominates.")
    print()

    # guards
    print("== Guards")
    print(f"  {'compile_id':<12}{'guards':>9}{'shape_env':>11}{'cache_size':>12}{'guarded':>9}")
    for cid, m in sorted(t.metrics.items()):
        print(
            f"  {cid:<12}{fmt(m.get('guard_count')):>9}"
            f"{fmt(m.get('shape_env_guard_count')):>11}"
            f"{fmt(m.get('cache_size')):>12}{fmt(m.get('has_guarded_code')):>9}"
        )
    print("  [note] cache_size is that frame's occupied slot count vs the")
    print("         recompile limit (default 8). Exceeding it falls back to eager.")
    print()

    # restarts / failures
    print("== Restarts and failures")
    any_restart = False
    for cid, m in sorted(t.metrics.items()):
        rr = m.get("restart_reasons")
        before = m.get("dynamo_time_before_restart_s")
        total = m.get("entire_frame_compile_time_s")
        if rr:
            any_restart = True
            print(f"  {cid:<12} restart_reasons = {fmt(rr)}")
            if before and total:
                print(
                    f"  {'':<12} {before:.2f} s of {total:.2f} s discarded"
                    f" ({100.0 * before / total:.0f}%)"
                )
        if m.get("fail_type"):
            print(f"  [warn] {cid} failed: {m['fail_type']}: {str(m.get('fail_reason'))[:70]}")
    if not any_restart:
        print("  none")
    else:
        print("  [note] a non-empty restart_reasons means Dynamo traced and threw")
        print("         work away. Lever: torch._dynamo.config.specialize_float=True")
        print("         (CT-3) -- but identify which float first.")
    print()

    # kernels
    if t.kernels:
        total = sum(t.kernels.values())
        print(f"== Top Triton kernels by compile time ({len(t.kernels)} kernels, {total:.1f} s)")
        print(f"  {'s':>9}{'share':>8}  kernel")
        ranked = sorted(t.kernels.items(), key=lambda kv: -kv[1])
        for name, secs in ranked[:top_kernels]:
            print(f"  {secs:>9.1f}{100.0 * secs / total:>7.1f}%  {name[:64]}")
        top_name, top_s = ranked[0]
        if total and top_s / total > 0.5:
            print(f"  [note] one kernel is {100.0 * top_s / total:.1f}% of all Triton")
            print("         compilation. The aten ops in its fused name are the")
            print("         target -- sort/argsort/topk generate code that grows with")
            print("         length. See references/compile-time.md section 5 (CT-1/CT-2).")
        print()
    else:
        print("== Top Triton kernels: no triton_kernel_compile_times_us in this trace")
        print()

    # verdict
    print("== Verdict")
    pmap = {k: v[0] for k, v in t.phases.items()}
    cg, aw = pmap.get("code_gen"), pmap.get("async_compile.wait")
    cdt = max((v for k, v in pmap.items() if "coordinate_descent" in k), default=0.0)
    if t.kernels and sum(t.kernels.values()):
        ranked = sorted(t.kernels.items(), key=lambda kv: -kv[1])
        if ranked[0][1] / sum(t.kernels.values()) > 0.5:
            print("  One generated kernel dominates compilation. Take that work out")
            print("  of the model graph and compile it separately (CT-2).")
            print("  -> references/compile-time.md section 5")
            return
    if cg and aw and aw / cg >= 0.8:
        print("  Triton-kernel-compile bound. Config flags will not help; generate")
        print("  less code (smaller IR, shorter top shape, CT-1/CT-2).")
        print("  -> references/compile-time.md sections 4-5")
    elif cdt and cg and cdt > 0.4 * cg:
        print("  Autotune search is a large share. Try coordinate_descent_tuning=False")
        print("  via list_mode_options + options= (CT-4). It is largely not cached,")
        print("  so it saves on warm runs too.")
        print("  -> references/mode-selection.md section 3")
    elif any(m.get("restart_reasons") for m in t.metrics.values()):
        print("  Dynamo is discarding and redoing work. Start with specialize_float")
        print("  (CT-3) after identifying which float triggers the restart.")
        print("  -> references/compile-time.md section 3")
    elif worst_blowup > 20:
        print(f"  IR blowup {worst_blowup:.0f}x with no restart and no dominant kernel:")
        print("  codegen bound by graph size. Reduce ops and graph inputs.")
        print("  -> references/compile-time.md section 4")
    else:
        print("  Nothing dominant in this trace. Confirm the slow step is actually")
        print("  compiling (scripts/log_triage.py on the same run's stderr) before")
        print("  attacking compile time at all.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace", nargs="+", help="TORCH_TRACE dir, log file, or .gz")
    ap.add_argument("--tlparse", metavar="DIR", help="prefer this tlparse output dir")
    ap.add_argument("--rank", type=int, help="pick this rank's trace file")
    ap.add_argument("--top-kernels", type=int, default=10)
    ap.add_argument("--compile-id", help="restrict metrics to this compile id")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    paths = []
    for target in args.trace:
        found = find_trace_files(target, args.rank)
        if not found:
            print(f"[warn] no trace files under {target}", file=sys.stderr)
        paths.extend(found)
    if not paths:
        print("No trace files found. Did you set TORCH_TRACE=/abs/dir?")
        return 0

    t = Trace()
    ingest(t, paths)
    if args.tlparse and load_tlparse(t, args.tlparse):
        pass  # tlparse phases replace in-trace ones (already merged across ranks)

    if args.compile_id:
        t.metrics = {k: v for k, v in t.metrics.items() if k == args.compile_id}

    if args.json:
        print(json.dumps({
            "records": t.records,
            "compile_ids": sorted(t.compile_ids),
            "attempts": t.attempts,
            "structured_logging_overhead_s": t.overhead,
            "phases": {k: {"seconds": v[0], "count": v[1]} for k, v in t.phases.items()},
            "metrics": t.metrics,
            "triton_kernel_compile_seconds": t.kernels,
        }, indent=2, default=str))
        return 0

    for p in paths:
        print(f"   source: {p}")
    emit(t, args.top_kernels, args.compile_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
