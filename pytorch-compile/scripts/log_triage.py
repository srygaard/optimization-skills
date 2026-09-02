#!/usr/bin/env python3
"""Triage TORCH_LOGS stderr output.

Groups graph breaks by reason, recompiles by guard text, flags recompile-limit
hits and CUDA-graph errors, and times recompile bursts.

    python3 log_triage.py run.err
    python3 log_triage.py run.err --recompiles
    cat run.err | python3 log_triage.py -

Stdlib only. Never raises on malformed input; always exits 0.
"""

import argparse
import gzip
import io
import json
import re
import sys
from collections import Counter, defaultdict

# --- anchors -----------------------------------------------------------------

RE_BREAK_SITE = re.compile(r"Graph break in user code at (.+?):(\d+)")
RE_BREAK_REASON = re.compile(r"Graph Break Reason:\s*(.+)")
RE_BREAK_NEW = re.compile(r"Graph break(?: in .*?)?:\s*(.+)")
RE_BREAK_GB_ID = re.compile(r"\b(GB\d{4})\b")
RE_SKIPPED = re.compile(r"Attempted to inline function marked as skipped")

RE_RECOMPILE = re.compile(r"Recompiling function (\S+) in (.+?):(\d+)")
RE_GUARD = re.compile(r"^\s*-\s*(?:\[?(\d+)/(\d+)\]?:?\s*)?(.+?)\s*$")
RE_TRIGGERED = re.compile(r"triggered by the following guard failure")
RE_SLOT = re.compile(r"\[(\d+)/(\d+)\]")

RE_LIMIT = re.compile(r"hit config\.(cache_size_limit|recompile_limit)\s*\((\d+)\)")
RE_LAST_REASON = re.compile(r"last reason:\s*(.+)")

RE_CUDAGRAPH_OVERWRITE = re.compile(
    r"accessing tensor output of CUDAGraphs that has been overwritten"
)
RE_CUDAGRAPH_EMPTY = re.compile(r"The CUDA Graph is empty")

# V0723 16:46:20.737000 12345 torch/_dynamo/guards.py:2 [__recompiles]
# may be prefixed by "[rank0]:" and/or other decoration, so search, don't anchor
RE_TS = re.compile(r"\b[A-Z]\d{4}\s+(\d{2}):(\d{2}):(\d{2})\.(\d{1,6})")

# noise stripped before grouping "the same reason" together
RE_ADDR = re.compile(r"0x[0-9a-fA-F]+")
RE_SHAPE = re.compile(r"\[[\d,\s]+\]")
RE_INT = re.compile(r"\b\d+\b")

FIX_HINTS = [
    (r"returned non-Tensor", "GB-1 (cache the device/capability query in __init__)"),
    (r"nonzero|Dynamic shape operator", "GB-2 (dense mask + torch.where)"),
    (r"Data-dependent branching", "GB-3 (delete the compute-skipping branch)"),
    (r"NestedUserFunctionVariable|call_id", "GB-4 (stable callable identity)"),
    (r"UserDefinedClassVariable __get__", "GB-4 step 2 (@torch.compiler.disable)"),
    (r"marked as skipped", "GB-5 (usually correct as-is)"),
    (r"assert_size_stride", "GB-7 (the op's fake/meta kernel is wrong)"),
    (r"item\(\)|\.cpu\(\)|graph_break.*sync", "GB-6 (hoist the host-device sync)"),
]

GUARD_HINTS = [
    (r"rank mismatch", "RC-2 (emit a constant rank)"),
    (r"_buffers\[", "RC-3 (pre-warm the lazily grown buffer)"),
    (r"size mismatch", "RC-1 / RC-5 (pad the width, or bucket the shape stream)"),
    (r"dtype mismatch", "make the cast unconditional"),
    (r"requires_grad", "expected at an eval/train switch"),
    (r"check_obj_id", "GB-4 (a fresh object per call)"),
    (r"==\s*-?\d+\s*$", "RC-4 (quantize the Python int)"),
]


def norm_reason(s):
    s = RE_ADDR.sub("0xADDR", s)
    s = RE_SHAPE.sub("[...]", s)
    return " ".join(s.split())[:110]


def norm_guard(s):
    s = RE_ADDR.sub("0xADDR", s)
    s = RE_INT.sub("N", s)
    return " ".join(s.split())[:110]


def hint_for(text, table):
    for pat, hint in table:
        if re.search(pat, text, re.I):
            return hint
    return None


def read_lines(path):
    """Yield lines from a file, '-' for stdin, transparently gunzipping."""
    try:
        if path == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(path, "rb") as fh:
                data = fh.read()
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
    except Exception as exc:  # unreadable file, bad gzip, permissions
        print(f"[warn] could not read {path}: {exc}", file=sys.stderr)
        return
    text = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", errors="replace")
    for line in text:
        yield line.rstrip("\n")


def ts_of(line):
    m = RE_TS.search(line)
    if not m:
        return None
    h, mi, s, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(frac) / (10 ** len(frac))


class Report:
    def __init__(self):
        self.breaks = Counter()          # normalized reason -> count
        self.break_sites = defaultdict(set)
        self.break_examples = {}
        self.gb_ids = defaultdict(set)
        self.break_events = 0

        self.guards = Counter()          # normalized guard -> count
        self.guard_examples = {}
        self.recompile_events = 0
        self.recompile_fns = Counter()
        self.recompile_times = []
        self.frame_slots = defaultdict(int)

        self.limit_hits = []
        self.cudagraph_overwrite = 0
        self.cudagraph_empty = 0
        self.saw_any_anchor = False


def parse(paths):
    r = Report()
    for path in paths:
        lines = list(read_lines(path))
        n = len(lines)
        for i, line in enumerate(lines):
            try:
                _parse_line(r, lines, i, n, line)
            except Exception:
                continue  # a single malformed line never kills the report
    return r


def _parse_line(r, lines, i, n, line):
    for m in RE_SLOT.finditer(line):
        frame, slot = int(m.group(1)), int(m.group(2))
        r.frame_slots[frame] = max(r.frame_slots[frame], slot)

    m = RE_BREAK_SITE.search(line)
    if m:
        r.saw_any_anchor = True
        r.break_events += 1
        site = f"{m.group(1).split('/')[-1]}:{m.group(2)}"
        reason = None
        for j in range(i, min(i + 6, n)):
            mr = RE_BREAK_REASON.search(lines[j]) or (
                RE_BREAK_NEW.search(lines[j]) if j > i else None
            )
            if mr:
                reason = mr.group(1).strip()
                break
        reason = reason or "(reason not captured)"
        key = norm_reason(reason)
        r.breaks[key] += 1
        r.break_sites[key].add(site)
        r.break_examples.setdefault(key, reason)
        for j in range(i, min(i + 8, n)):
            gm = RE_BREAK_GB_ID.search(lines[j])
            if gm:
                r.gb_ids[key].add(gm.group(1))
        return

    if RE_SKIPPED.search(line) and not RE_BREAK_SITE.search(line):
        r.saw_any_anchor = True
        key = norm_reason("Attempted to inline function marked as skipped")
        r.breaks[key] += 1
        r.break_examples.setdefault(key, "Attempted to inline function marked as skipped")
        r.break_events += 1
        return

    m = RE_RECOMPILE.search(line)
    if m:
        r.saw_any_anchor = True
        r.recompile_events += 1
        r.recompile_fns[m.group(1)] += 1
        t = ts_of(line)
        if t is not None:
            r.recompile_times.append(t)
        # guard lines follow "triggered by the following guard failure(s):"
        started = False
        for j in range(i + 1, min(i + 40, n)):
            nxt = lines[j]
            if RE_TRIGGERED.search(nxt):
                started = True
                continue
            if RE_RECOMPILE.search(nxt) or RE_BREAK_SITE.search(nxt):
                break
            if not started:
                continue
            gm = RE_GUARD.match(nxt)
            if gm and gm.group(3) and not nxt.strip().startswith("--"):
                g = gm.group(3).strip()
                if len(g) < 4:
                    continue
                if gm.group(1) is not None:      # "- 0/1: <guard>" carries the slot
                    frame, slot = int(gm.group(1)), int(gm.group(2))
                    r.frame_slots[frame] = max(r.frame_slots[frame], slot)
                key = norm_guard(g)
                r.guards[key] += 1
                r.guard_examples.setdefault(key, g)
            elif nxt.strip() == "":
                break
        return

    m = RE_LIMIT.search(line)
    if m:
        r.saw_any_anchor = True
        last = ""
        for j in range(i, min(i + 6, n)):
            lm = RE_LAST_REASON.search(lines[j])
            if lm:
                last = lm.group(1).strip()
                break
        r.limit_hits.append((m.group(1), m.group(2), last))
        return

    if RE_CUDAGRAPH_OVERWRITE.search(line):
        r.saw_any_anchor = True
        r.cudagraph_overwrite += 1
    if RE_CUDAGRAPH_EMPTY.search(line):
        r.cudagraph_empty += 1


def bursts(times, gap=30.0):
    """Group event timestamps into contiguous bursts separated by >gap seconds."""
    if not times:
        return []
    ts = sorted(times)
    out, start, prev, count = [], ts[0], ts[0], 1
    for t in ts[1:]:
        if t - prev > gap:
            out.append((start, prev, count))
            start, count = t, 1
        else:
            count += 1
        prev = t
    out.append((start, prev, count))
    return out


def hhmmss(t):
    return f"{int(t // 3600) % 24:02d}:{int(t // 60) % 60:02d}:{t % 60:06.3f}"


def emit(r, show_breaks, show_recompiles, by_site):
    if not r.saw_any_anchor:
        print("== No Dynamo log lines found")
        print("  [warn] no graph-break, recompile, or limit anchors of any kind.")
        print("         This proves nothing unless TORCH_LOGS was set in the")
        print("         process that wrote this file. See SKILL.md section 0.")
        print('         Try: TORCH_LOGS="graph_breaks,recompiles" ... 2> run.err')
        return

    if show_breaks:
        print(
            f"== Graph breaks: {r.break_events} events, "
            f"{len(r.breaks)} distinct reasons, "
            f"{len(set().union(*r.break_sites.values())) if r.break_sites else 0} sites"
        )
        if not r.breaks:
            print("  none")
        hints = []
        for key, count in r.breaks.most_common():
            sites = sorted(r.break_sites.get(key, []))
            shown = ",".join(sites[:3]) + ("..." if len(sites) > 3 else "")
            gb = " " + ",".join(sorted(r.gb_ids[key])) if r.gb_ids.get(key) else ""
            print(f"  {count:5d}  {key[:66]:<66}  {shown}{gb}")
            h = hint_for(r.break_examples.get(key, key), FIX_HINTS)
            if h and h not in hints:
                hints.append(h)
        for h in hints:
            print(f"  -> {h}")
        if by_site:
            print("  by site:")
            per_site = Counter()
            for key, sites in r.break_sites.items():
                for s in sites:
                    per_site[s] += r.breaks[key]
            for site, c in per_site.most_common(20):
                print(f"    {c:5d}  {site}")
        print()

    if show_recompiles:
        bl = bursts(r.recompile_times)
        print(
            f"== Recompiles: {r.recompile_events} events, "
            f"{len(r.guards)} distinct guards, {len(bl)} burst(s)"
        )
        if not r.guards and r.recompile_events:
            print("  (no guard text captured -- try TORCH_LOGS=recompiles_verbose)")
        hints = []
        for key, count in r.guards.most_common():
            print(f"  {count:5d}  {key[:70]}")
            ex = r.guard_examples.get(key, "")
            if ex and ex != key:
                print(f"         e.g. {ex[:78]}")
            h = hint_for(ex or key, GUARD_HINTS)
            if h and h not in hints:
                hints.append(h)
        for h in hints:
            print(f"  -> {h}")
        for start, end, count in bl:
            print(
                f"  burst: {hhmmss(start)} -> {hhmmss(end)}"
                f"  ({end - start:.1f} s, {count} events)"
            )
        if r.recompile_fns:
            top = ", ".join(f"{k}({v})" for k, v in r.recompile_fns.most_common(5))
            print(f"  frames: {top}")
        print()

    if r.frame_slots:
        top = max(r.frame_slots.values())
        slots = "  ".join(f"{f}->{s}" for f, s in sorted(r.frame_slots.items())[:12])
        print(f"== Cache slots (frame -> highest slot seen)\n  {slots}")
        if top >= 7:
            print(f"  [warn] max slot {top} is at or near the default limit of 8.")
            print("         Exceeding it silently falls back to EAGER for that frame.")
        else:
            print(f"  [ok] max slot {top}, under the default recompile limit of 8.")
        print()

    if r.limit_hits:
        print(f"== Recompile limit: HIT {len(r.limit_hits)} time(s)")
        print("  [warn] that frame has SILENTLY fallen back to eager. Perf cliff,")
        print("         not a crash. See references/recompiles.md section 3.")
        for name, val, last in r.limit_hits[:5]:
            print(f"    config.{name} ({val})   last reason: {last[:60]}")
        print()

    if r.cudagraph_overwrite or r.cudagraph_empty:
        print("== CUDA graph errors")
        if r.cudagraph_overwrite:
            print(f"  {r.cudagraph_overwrite}x output buffer overwritten -> RC-6 (.clone())")
        if r.cudagraph_empty:
            print(f"  {r.cudagraph_empty}x 'The CUDA Graph is empty' -> degenerate capture")
        print()

    print("== Verdict")
    bl = bursts(r.recompile_times)
    if r.limit_hits:
        print("  Recompile limit hit: some frame is running EAGER. Fix that first.")
        print("  -> references/recompiles.md section 3")
    elif len(bl) > 1:
        print(f"  {len(bl)} separate recompile bursts: shapes are still specializing.")
        print("  Read the guard text above, then the decoder table.")
        print("  -> references/recompiles.md section 2")
    elif r.recompile_events and len(bl) == 1:
        print("  One recompile burst, then quiet: automatic dynamic shapes promoted")
        print("  on the first shape transition. Expected, and a one-time cost.")
        print("  -> references/recompiles.md section 1")
    elif r.recompile_events:
        print(f"  {r.recompile_events} recompiles, no usable timestamps to group them.")
        print("  Read the guard text above against the decoder table.")
        print("  -> references/recompiles.md section 2")
    elif r.break_events:
        print("  Graph breaks only, no recompiles. Rank them by what they fragment")
        print("  before fixing -- an already-eager boundary costs ~nothing.")
        print("  -> references/graph-breaks.md section 4")
    else:
        print("  Logs present, nothing flagged. If a step is still slow, it is not")
        print("  Dynamo -- try TORCH_TRACE + scripts/trace_summary.py, or profile.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logfile", nargs="+", help="stderr log file(s), or - for stdin")
    ap.add_argument("--breaks", action="store_true", help="graph breaks only")
    ap.add_argument("--recompiles", action="store_true", help="recompiles only")
    ap.add_argument("--by-site", action="store_true", help="also group breaks by site")
    ap.add_argument("--json", action="store_true", help="emit JSON instead")
    args = ap.parse_args()

    r = parse(args.logfile)

    if args.json:
        print(json.dumps({
            "graph_break_events": r.break_events,
            "graph_breaks": dict(r.breaks),
            "graph_break_sites": {k: sorted(v) for k, v in r.break_sites.items()},
            "recompile_events": r.recompile_events,
            "guards": dict(r.guards),
            "guard_examples": r.guard_examples,
            "bursts": [
                {"start": s, "end": e, "events": c, "seconds": e - s}
                for s, e, c in bursts(r.recompile_times)
            ],
            "frame_slots": dict(r.frame_slots),
            "limit_hits": r.limit_hits,
            "cudagraph_overwrite": r.cudagraph_overwrite,
            "cudagraph_empty": r.cudagraph_empty,
            "saw_any_anchor": r.saw_any_anchor,
        }, indent=2))
        return 0

    both = not (args.breaks or args.recompiles)
    emit(r, args.breaks or both, args.recompiles or both, args.by_site)
    return 0


if __name__ == "__main__":
    sys.exit(main())
