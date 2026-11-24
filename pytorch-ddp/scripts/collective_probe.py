#!/usr/bin/env python3
"""Size sweep of NCCL collectives, with the fit `ms = t_o + bytes / w`.

The Job B probe for projects that have no probe of their own. It runs
standalone (no model, no trainer), so launch it in Job B's allocation, with
the same NCCL environment and the same visible GPUs as the training arms:

    torchrun --nnodes N --nproc-per-node G ... collective_probe.py \
        [--ops all_reduce,all_gather,reduce_scatter] [--groups world,node] \
        [--sizes-mb 1,4,16,64,256,1024] [--dtype fp32] [--json out.json]

Groups: `world` (all ranks), `node` (the ranks of one node: HSDP's shard
group), `replica` (one rank per node with the same local rank: HSDP's
replica group). Every group is created, and used once, before any timing.
All instances of a group run at once (every node's `node` group, every
local rank's `replica` group), as they do under HSDP, so they share the NICs.

Sizes are the full tensor: the all-reduced buffer, an all-gather's output, a
reduce-scatter's input (each rank holds 1/group_size of the last two). Each
point is the median over trials of (N back-to-back calls / N), the slowest
rank per trial. Per collective and group, rank 0 prints the rows, the fit
`t_o` (us) and `w` (GB/s, algorithm bandwidth), and `w * c_k` (bus bandwidth,
the number to compare with the link speed; `c_k` as in F9).

It measures the fabric, not the trainer's communicators: a communicator the
trainer creates later can still see a degraded topology (gate 2,
`nccl_topo.py` on the trainer's NCCL log).
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def bus_factor(op: str, n: int) -> float:
    return 2 * (n - 1) / n if op == "all_reduce" else (n - 1) / n


def make_groups(names, world, local):
    """Returns {name: (group, size)} for this rank. All ranks create all groups."""
    nodes = world // local
    out = {}
    for name in names:
        if name == "world":
            out[name] = (None, world)
        elif name == "node":
            for node in range(nodes):
                g = dist.new_group(list(range(node * local, (node + 1) * local)))
                if dist.get_rank() // local == node:
                    out[name] = (g, local)
        elif name == "replica":
            for lr in range(local):
                g = dist.new_group([node * local + lr for node in range(nodes)])
                if dist.get_rank() % local == lr:
                    out[name] = (g, nodes)
        else:
            raise SystemExit(f"unknown group {name}")
    return out


def timer(device):
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def run(fn, n):
            start.record()
            for _ in range(n):
                fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / n

    else:

        def run(fn, n):
            t = time.perf_counter()
            for _ in range(n):
                fn()
            return (time.perf_counter() - t) * 1e3 / n

    return run


def fit(rows):
    pts = [(r["bytes"], r["ms"]) for r in rows]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if not sxx:
        return None
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return {"t_o_us": (my - slope * mx) * 1e3, "w_GBps": 1e-6 / slope if slope > 0 else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops", default="all_reduce")
    ap.add_argument("--groups", default="world")
    ap.add_argument("--sizes-mb", default="1,4,16,64,256,1024")
    ap.add_argument("--dtype", default="fp32", choices=DTYPES)
    ap.add_argument("--calls", type=int, default=10, help="back-to-back calls per trial")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ranks-per-node", type=int, help="default: LOCAL_WORLD_SIZE")
    ap.add_argument("--json")
    a = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl" if use_cuda else "gloo")
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = a.ranks_per_node or int(os.environ.get("LOCAL_WORLD_SIZE", world))
    if world % local:
        raise SystemExit(f"{world} ranks do not split into nodes of {local}")
    dtype = DTYPES[a.dtype]
    esize = torch.tensor([], dtype=dtype).element_size()
    run = timer(device)

    groups = make_groups(a.groups.split(","), world, local)
    for g, _ in groups.values():  # create every communicator now
        dist.all_reduce(torch.ones(1, device=device), group=g)

    results = []
    for gname, (g, n) in groups.items():
        if n < 2:
            if rank == 0:
                print(f"== group {gname} has 1 rank: nothing to time")
            continue
        for op in a.ops.split(","):
            rows = []
            for mb in (float(s) for s in a.sizes_mb.split(",")):
                numel = int(mb * 2**20) // esize // n * n
                full = torch.ones(numel, dtype=dtype, device=device)
                shard = torch.ones(numel // n, dtype=dtype, device=device)
                fn = {
                    "all_reduce": lambda: dist.all_reduce(full, group=g),
                    "all_gather": lambda: dist.all_gather_into_tensor(full, shard, group=g),
                    "reduce_scatter": lambda: dist.reduce_scatter_tensor(shard, full, group=g),
                }[op]
                run(fn, a.warmup)
                trials = []
                for _ in range(a.trials):
                    dist.barrier()
                    t = torch.tensor([run(fn, a.calls)], device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)  # the slowest rank sets the pace
                    trials.append(t.item())
                ms = statistics.median(trials)
                rows.append({"bytes": numel * esize, "ms": ms})
                del full, shard
            f = fit(rows)
            results.append({"group": gname, "size": n, "op": op, "dtype": a.dtype, "rows": rows, "fit": f})
            if rank == 0:
                c = bus_factor(op, n)
                print(f"== {op} {a.dtype}, group {gname} ({n} ranks), c_k = {c:.3f}")
                for r in rows:
                    algbw = r["bytes"] / r["ms"] / 1e6
                    print(f"  {r['bytes'] / 2**20:8.1f} MiB  {r['ms']:9.3f} ms  alg {algbw:7.1f} GB/s  bus {algbw * c:7.1f} GB/s")
                if f:
                    print(f"  fit: t_o {f['t_o_us']:.1f} us, w {f['w_GBps']:.1f} GB/s (bus {f['w_GBps'] * c:.1f} GB/s)")
    if rank == 0 and a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
