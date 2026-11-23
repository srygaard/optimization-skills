#!/usr/bin/env python3
"""DDP communication report from torch.profiler (Kineto) chrome traces.

Stdlib only. Pass one trace per rank (all ranks of one window for the
cross-rank numbers):

    python ddp_trace.py run/profiler_trace/rank*.pt.trace.json [--json out.json]

The trace must come from a profiler stepped once per *optimizer* step, with
the training loop labelled by ``record_function("forward")`` and
``record_function("backward")`` once per micro-batch (Accelerate/HF style).

What it reports, per rank and then across ranks:

* collectives per micro-batch, split into accumulation and sync micro-batches
  (under gradient accumulation the accumulation ones must be 0, or `no_sync`
  is broken);
* gradient buckets: size, duration, NCCL algorithm/protocol;
* exposed comm: collective time not overlapped by any compute kernel;
* tail: end of the last gradient all-reduce minus end of the last backward
  compute kernel of the sync micro-batch (the formula's `t_e`);
* overlap slowdown: backward compute-kernel time of the sync micro-batch minus the
  median of the accumulation micro-batches (NCCL kernels steal SMs and HBM
  bandwidth from the backward they overlap);
* every collective (FSDP/HSDP all-gathers and reduce-scatters too) by type,
  group size and launch phase: calls, bytes of the full tensor, kernel time,
  transfer (per call the minimum across ranks) and exposed time, next to the
  forward and backward compute that can hide it;
* an OLS fit `dur = t_o + bytes / w` over the gradient buckets, using the
  per-bucket minimum across ranks when several ranks are given (a collective's
  duration includes waiting for late peers; the minimum is the transfer).

GPU work is only ``cat in {kernel, gpu_memcpy, gpu_memset}``:
``gpu_user_annotation`` events are projections of host labels and span idle
time. Kernels are attributed to a micro-batch and phase by their *launch*
time (via ``correlation``), not by when they ran.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics
import sys
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}
DTYPE_BYTES = {
    "f32": 4, "f16": 2, "bf16": 2, "f64": 8, "i32": 4, "u32": 4, "i64": 8,
    "u64": 8, "i8": 1, "u8": 1, "f8e4m3": 1, "f8e5m2": 1,
}


ARG_DTYPES = {
    "Float": "f32", "BFloat16": "bf16", "Half": "f16", "Double": "f64",
    "Int": "i32", "Long": "i64", "Byte": "u8", "Char": "i8",
}


def is_nccl(name: str) -> bool:
    return name.startswith("ncclDevKernel") or name.startswith("ncclKernel")


def nccl_meta(ev: dict) -> dict:
    """Collective, dtype, algo, proto and bytes of one NCCL kernel event."""
    name = ev["name"]
    args = ev.get("args", {})
    # torch names: allreduce, _allgather_base, _reduce_scatter_base,
    # allgather_into_tensor_coalesced, ... -> allreduce, allgather, reducescatter
    coll = (args.get("Collective name") or "").lower().lstrip("_")
    coll = re.sub(r"_base$|_into_tensor", "", coll)
    coll = coll.replace("reduce_scatter", "reducescatter").replace("all_gather", "allgather")
    parts = name.split("(")[0].split("_")[1:]  # AllReduce Sum f32 TREE LL
    dtype = next((p for p in parts if p in DTYPE_BYTES), None) or ARG_DTYPES.get(args.get("dtype") or "")
    algo = parts[-2] if len(parts) >= 3 else "?"
    proto = parts[-1] if len(parts) >= 3 else "?"
    # The full (unsharded) tensor: Out for all-gather, In for reduce-scatter.
    nelems = max(int(args.get("In msg nelems") or 0), int(args.get("Out msg nelems") or 0))
    return {
        "coll": coll or (parts[0].lower() if parts else "?"),
        "group": args.get("Group size"),
        "dtype": dtype,
        "algo": algo,
        "proto": proto,
        "bytes": nelems * DTYPE_BYTES.get(dtype or "", 0),
    }


def union_length(intervals: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def uncovered(target: list[tuple[float, float]], cover: list[tuple[float, float]]) -> float:
    """Length of the union of `target` not covered by the union of `cover`."""
    target, cover = merge(target), merge(cover)
    starts = [c[0] for c in cover]
    total = 0.0
    for s, e in target:
        length = e - s
        i = max(bisect.bisect_right(starts, s) - 1, 0)
        while i < len(cover) and cover[i][0] < e:
            cs, ce = cover[i]
            overlap = min(e, ce) - max(s, cs)
            if overlap > 0:
                length -= overlap
            i += 1
        total += max(length, 0.0)
    return total


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def analyze_rank(path: Path) -> dict:
    with open(path) as fh:
        data = json.load(fh)
    events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
    rank = (data.get("distributedInfo") or {}).get("rank")
    if rank is None:
        m = re.search(r"rank(\d+)", path.name)
        rank = int(m.group(1)) if m else -1

    launch_ts: dict[int, float] = {}
    ann: dict[str, list[tuple[float, float]]] = {}
    gpu: list[dict] = []
    for e in events:
        cat = e.get("cat")
        if cat in LAUNCH_CATS:
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                launch_ts[corr] = e["ts"]
        elif cat == "user_annotation":
            ann.setdefault(e["name"], []).append((e["ts"], e["ts"] + e.get("dur", 0)))
        elif cat in GPU_CATS:
            gpu.append(e)

    forwards = sorted(ann.get("forward", []))
    if not forwards:
        raise SystemExit(f"{path}: no record_function('forward') labels; cannot split micro-batches")
    fwd_starts = [s for s, _ in forwards]
    backwards = sorted(ann.get("backward", []))
    opt_steps = [s for s, _ in ann.get("Optimizer.step#AdamW.step", [])] or [
        s for name, spans in ann.items() if name.startswith("Optimizer.step#") for s, _ in spans
    ]

    n_mb = len(forwards)
    mbs = [
        {
            "index": i,
            "sync": False,
            "compute": [],
            "compute_bwd_us": 0.0,
            "bwd_compute": [],
            "fwd_compute_us": 0.0,
            "nccl": [],
            "gpu_first": None,
            "gpu_last": None,
        }
        for i in range(n_mb)
    ]
    for ts in opt_steps:
        i = bisect.bisect_right(fwd_starts, ts) - 1
        if 0 <= i < n_mb:
            mbs[i]["sync"] = True

    def phase_of(ts: float) -> str:
        for name in ("backward", "forward", "optimizer_step"):
            spans = ann.get(name, [])
            j = bisect.bisect_right([s for s, _ in spans], ts) - 1
            if j >= 0 and spans[j][0] <= ts <= spans[j][1]:
                return name
        return "other"

    unattributed = 0
    for e in gpu:
        corr = (e.get("args") or {}).get("correlation")
        lts = launch_ts.get(corr)
        if lts is None:
            unattributed += 1
            continue
        i = bisect.bisect_right(fwd_starts, lts) - 1
        if i < 0:
            continue
        mb = mbs[i]
        s, d = e["ts"], e.get("dur", 0.0)
        mb["gpu_first"] = s if mb["gpu_first"] is None else min(mb["gpu_first"], s)
        mb["gpu_last"] = s + d if mb["gpu_last"] is None else max(mb["gpu_last"], s + d)
        phase = phase_of(lts)
        if is_nccl(e["name"]):
            meta = nccl_meta(e)
            meta.update(ts=s, dur=d, phase=phase)
            mb["nccl"].append(meta)
        else:
            mb["compute"].append((s, s + d))
            if phase == "backward":
                mb["compute_bwd_us"] += d
                mb["bwd_compute"].append((s, s + d))
            elif phase == "forward":
                mb["fwd_compute_us"] += d

    # Drop a trailing micro-batch cut by the window end (no GPU work, or no backward).
    mbs = [m for m in mbs if m["gpu_first"] is not None and m["compute_bwd_us"] > 0]

    out_mbs = []
    for m in mbs:
        grads = [c for c in m["nccl"] if c["coll"] == "allreduce" and c["phase"] == "backward"]
        comm_iv = [(c["ts"], c["ts"] + c["dur"]) for c in m["nccl"]]
        exposed = uncovered(comm_iv, m["compute"])
        # Every collective type (FSDP all-gathers, reduce-scatters, ...) by launch phase.
        by_coll: dict[str, dict] = {}
        for c in m["nccl"]:
            g = f"/{c['group']}" if c.get("group") else ""
            k = by_coll.setdefault(f"{c['coll']}{g}@{c['phase']}", {"n": 0, "bytes": 0, "kernel_us": 0.0, "durs_us": [], "iv": []})
            k["n"] += 1
            k["bytes"] += c["bytes"]
            k["kernel_us"] += c["dur"]
            k["durs_us"].append(c["dur"])
            k["iv"].append((c["ts"], c["ts"] + c["dur"]))
        for k in by_coll.values():
            k["exposed_ms"] = uncovered(k.pop("iv"), m["compute"]) / 1e3
        tail = post = None
        if grads and m["bwd_compute"]:
            last = max(grads, key=lambda c: c["ts"] + c["dur"])
            last_end = last["ts"] + last["dur"]
            # Gradient compute the last bucket waited on: backward kernels that
            # started before it. Kernels after it are the reducer's finalize
            # (bucket->grad copies, division), reported as post-comm work.
            before = [e for s_, e in m["bwd_compute"] if s_ <= last["ts"]]
            after = [e for s_, e in m["bwd_compute"] if s_ >= last_end]
            if before:
                tail = last_end - max(before)
            post = (max(after) - last_end) if after else 0.0
        out_mbs.append(
            {
                "index": m["index"],
                "sync": m["sync"],
                "span_ms": (m["gpu_last"] - m["gpu_first"]) / 1e3,
                "gpu_busy_ms": union_length(m["compute"] + comm_iv) / 1e3,
                "compute_bwd_ms": m["compute_bwd_us"] / 1e3,
                "compute_fwd_ms": m["fwd_compute_us"] / 1e3,
                "collectives": len(m["nccl"]),
                "by_coll": by_coll,
                "grad_allreduces": len(grads),
                "grad_bytes": sum(c["bytes"] for c in grads),
                "comm_ms": sum(c["dur"] for c in m["nccl"]) / 1e3,
                "exposed_comm_ms": exposed / 1e3,
                "tail_ms": None if tail is None else tail / 1e3,
                "post_comm_ms": None if post is None else post / 1e3,
                "buckets": [
                    {"bytes": c["bytes"], "dur_us": c["dur"], "algo": c["algo"], "proto": c["proto"], "dtype": c["dtype"]}
                    for c in grads
                ],
                "other_collectives": sorted({f"{c['coll']}@{c['phase']}" for c in m["nccl"] if c not in grads}),
            }
        )

    accum = [m for m in out_mbs if not m["sync"]]
    sync = [m for m in out_mbs if m["sync"]]
    accum_bwd = [m["compute_bwd_ms"] for m in accum]
    acc_med = median(accum_bwd)
    for m in sync:
        m["overlap_slowdown_ms"] = None if acc_med is None else m["compute_bwd_ms"] - acc_med
    return {
        "rank": rank,
        "path": str(path),
        "micro_batches": len(out_mbs),
        "sync_micro_batches": len(sync),
        "unattributed_gpu_events": unattributed,
        "accum": {
            "collectives_per_mb": [m["collectives"] for m in accum],
            "grad_allreduces_per_mb": [m["grad_allreduces"] for m in accum],
            "compute_bwd_ms_median": acc_med,
            "compute_bwd_ms_p10_p90": _p10_p90(accum_bwd),
            "span_ms_median": median([m["span_ms"] for m in accum]),
            "compute_fwd_ms_median": median([m["compute_fwd_ms"] for m in accum]),
            "exposed_comm_ms_median": median([m["exposed_comm_ms"] for m in accum]),
            "grad_reductions_per_mb": [
                sum(v["n"] for k, v in m["by_coll"].items()
                    if k.endswith("@backward") and k.split("/")[0].split("@")[0] in ("reducescatter", "allreduce"))
                for m in accum
            ],
        },
        "accum_mbs": accum,
        "sync": sync,
    }


def _p10_p90(xs: list[float]) -> tuple[float, float] | None:
    if len(xs) < 2:
        return None
    s = sorted(xs)
    return (s[int(0.1 * (len(s) - 1))], s[int(0.9 * (len(s) - 1))])


def ols(points: list[tuple[float, float]]) -> dict | None:
    """Fit y = a + b x; returns t_o (us), w (GB/s), r2."""
    if len(points) < 3:
        return None
    xs, ys = zip(*points)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in points)
    return {
        "t_o_us": a,
        "w_GBps": (1.0 / b) / 1e3 if b > 0 else None,  # bytes/us -> GB/s
        "r2": 1 - ss_res / ss_tot if ss_tot else None,
        "n": len(points),
    }


def cross_rank(ranks: list[dict]) -> dict:
    """Per-bucket min/max across ranks, matched by order within each sync micro-batch."""
    n_sync = min(r["sync_micro_batches"] for r in ranks)
    per_step = []
    fit_points = []
    for k in range(n_sync):
        bucket_lists = [r["sync"][k]["buckets"] for r in ranks]
        n_b = min(len(b) for b in bucket_lists)
        mins, maxs = [], []
        for j in range(n_b):
            durs = [b[j]["dur_us"] for b in bucket_lists]
            mins.append(min(durs))
            maxs.append(max(durs))
            fit_points.append((bucket_lists[0][j]["bytes"], min(durs)))
        exposed = [r["sync"][k]["exposed_comm_ms"] for r in ranks]
        tails = [r["sync"][k]["tail_ms"] for r in ranks if r["sync"][k]["tail_ms"] is not None]
        posts = [r["sync"][k]["post_comm_ms"] for r in ranks if r["sync"][k]["post_comm_ms"] is not None]
        slowdowns = [r["sync"][k]["overlap_slowdown_ms"] for r in ranks if r["sync"][k].get("overlap_slowdown_ms") is not None]
        per_step.append(
            {
                "sync_mb": k,
                "buckets": n_b,
                "transfer_ms_sum_min": sum(mins) / 1e3,
                "wait_ms_sum_max_minus_min": (sum(maxs) - sum(mins)) / 1e3,
                "exposed_ms_min": min(exposed),
                "exposed_ms_median": median(exposed),
                "exposed_ms_max": max(exposed),
                "tail_ms_min": min(tails) if tails else None,
                "tail_ms_median": median(tails),
                "post_comm_ms_median": median(posts),
                "overlap_slowdown_ms_median": median(slowdowns),
            }
        )
    return {
        "ranks": len(ranks),
        "per_sync_step": per_step,
        "fit_min_across_ranks": ols(fit_points),
        "by_coll": {kind: by_coll_cross(ranks, kind) for kind in ("accum_mbs", "sync")},
    }


def by_coll_cross(ranks: list[dict], kind: str) -> dict:
    """Per collective@phase, medians over micro-batches of one kind.

    transfer: per call the minimum duration across ranks (matched by order),
    summed over the micro-batch; exposed: the median rank's uncovered time.
    """
    n = min(len(r[kind]) for r in ranks)
    keys = sorted({k for r in ranks for m in r[kind][:n] for k in m["by_coll"]})
    out = {}
    for key in keys:
        transfer, exposed, kernel, calls, mib = [], [], [], [], []
        for i in range(n):
            per_rank = [r[kind][i]["by_coll"].get(key) for r in ranks]
            if any(x is None for x in per_rank):
                continue
            n_calls = min(len(x["durs_us"]) for x in per_rank)
            transfer.append(sum(min(x["durs_us"][j] for x in per_rank) for j in range(n_calls)) / 1e3)
            exposed.append(median([x["exposed_ms"] for x in per_rank]))
            kernel.append(median([x["kernel_us"] / 1e3 for x in per_rank]))
            calls.append(per_rank[0]["n"])
            mib.append(per_rank[0]["bytes"] / 2**20)
        if transfer:
            out[key] = {
                "calls": median(calls),
                "mib": median(mib),
                "kernel_ms": median(kernel),
                "transfer_ms": median(transfer),
                "exposed_ms": median(exposed),
            }
    windows = {
        "forward": median([median([m["compute_fwd_ms"] for m in r[kind][:n]]) for r in ranks if r[kind][:n]]),
        "backward": median([median([m["compute_bwd_ms"] for m in r[kind][:n]]) for r in ranks if r[kind][:n]]),
    }
    return {"micro_batches": n, "collectives": out, "compute_window_ms": windows}


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def report(ranks: list[dict], xr: dict | None) -> None:
    for r in sorted(ranks, key=lambda r: r["rank"]):
        acc = r["accum"]
        bad = [c for c in acc["collectives_per_mb"] if c]
        print(f"== rank {r['rank']}: {r['micro_batches']} micro-batches, {r['sync_micro_batches']} sync")
        print(
            f"   accumulation micro-batches: collectives {acc['collectives_per_mb']}"
            + ("  <-- NONZERO: no_sync is not suppressing the all-reduce" if any(acc["grad_allreduces_per_mb"]) else "")
            + (f"  (non-gradient collectives on {len(bad)} accumulation micro-batches)" if bad and not any(acc["grad_allreduces_per_mb"]) else "")
        )
        if any(acc["grad_reductions_per_mb"]):
            print(
                f"   gradient reductions in accumulation backward: {acc['grad_reductions_per_mb']}"
                "  (expected 0 when gradient sync is skipped; FSDP without sync skipping, or HSDP's in-node reduce-scatter, is the exception)"
            )
        print(
            f"   accum forward compute: median {fmt(acc['compute_fwd_ms_median'])} ms, exposed comm median {fmt(acc['exposed_comm_ms_median'])} ms"
        )
        print(
            f"   accum backward compute: median {fmt(acc['compute_bwd_ms_median'])} ms"
            f" (p10-p90 {', '.join(fmt(x) for x in acc['compute_bwd_ms_p10_p90'] or ())}), span median {fmt(acc['span_ms_median'])} ms"
        )
        for m in r["sync"]:
            sizes = [b["bytes"] / 2**20 for b in m["buckets"]]
            # Kernel-name algo/proto is the default specialization, not
            # necessarily what ran (NCCL 2.30); the dtype is reliable.
            algos = sorted({f"{b['dtype']}/kernel:{b['algo']}_{b['proto']}" for b in m["buckets"]})
            print(
                f"   sync mb {m['index']:3d}: {m['grad_allreduces']} buckets {m['grad_bytes'] / 2**20:8.1f} MiB {algos} | "
                f"comm {fmt(m['comm_ms'])} exposed {fmt(m['exposed_comm_ms'])} tail {fmt(m['tail_ms'])} post-comm {fmt(m['post_comm_ms'])} "
                f"overlap-slowdown {fmt(m.get('overlap_slowdown_ms'))} ms | span {fmt(m['span_ms'])} ms | other {m['other_collectives']}"
            )
            print("      bucket MiB: " + " ".join(f"{s:.1f}" for s in sizes))
            print("      bucket us : " + " ".join(f"{b['dur_us']:.0f}" for b in m["buckets"]))
    if xr:
        print(f"== across {xr['ranks']} ranks (per sync micro-batch)")
        for s in xr["per_sync_step"]:
            print(
                f"   sync {s['sync_mb']}: {s['buckets']} buckets, transfer(min) {fmt(s['transfer_ms_sum_min'])} ms, "
                f"peer-wait(max-min) {fmt(s['wait_ms_sum_max_minus_min'])} ms, exposed min/med/max "
                f"{fmt(s['exposed_ms_min'])}/{fmt(s['exposed_ms_median'])}/{fmt(s['exposed_ms_max'])} ms, "
                f"tail min/med {fmt(s['tail_ms_min'])}/{fmt(s['tail_ms_median'])} ms, post-comm med {fmt(s['post_comm_ms_median'])} ms, "
                f"overlap-slowdown med {fmt(s['overlap_slowdown_ms_median'])} ms"
            )
        f = xr["fit_min_across_ranks"]
        if f:
            print(f"   fit dur = t_o + bytes/w (min across ranks): t_o {fmt(f['t_o_us'], 1)} us, w {fmt(f['w_GBps'], 1)} GB/s, r2 {fmt(f['r2'], 3)}, n {f['n']}")
        print("== every collective, by type and launch phase (medians over micro-batches; transfer = per-call min across ranks, summed)")
        for kind, label in (("accum_mbs", "accum"), ("sync", "sync")):
            bc = xr["by_coll"][kind]
            w = bc["compute_window_ms"]
            print(f"   {label}: {bc['micro_batches']} micro-batches, compute forward {fmt(w['forward'], 1)} ms, backward {fmt(w['backward'], 1)} ms")
            for key, v in bc["collectives"].items():
                print(
                    f"      {key:28s} calls {v['calls']:5.0f}  {v['mib']:9.1f} MiB  kernel {fmt(v['kernel_ms'], 1):>7s}  "
                    f"transfer {fmt(v['transfer_ms'], 1):>7s}  exposed {fmt(v['exposed_ms'], 1):>6s} ms"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, help="also write the full result as JSON")
    args = ap.parse_args()
    ranks = []
    for p in args.traces:
        print(f"loading {p} ...", file=sys.stderr)
        ranks.append(analyze_rank(p))
    xr = cross_rank(ranks) if ranks else None
    report(ranks, xr)
    if args.json:
        args.json.write_text(json.dumps({"ranks": ranks, "cross_rank": xr}, indent=1))


if __name__ == "__main__":
    main()
