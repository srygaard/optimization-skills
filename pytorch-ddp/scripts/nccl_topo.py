#!/usr/bin/env python3
"""Check the NVLink topology NCCL detected for every communicator.

NCCL detects the node topology when a communicator is created and keeps the
paths it chose for that communicator's whole life. If a GPU's NVLinks look
inactive at that moment (a failed link, or links that read as down while they
sleep), NCCL routes that communicator through PCIe or the CPU, often several
times slower, and nothing errors. Sub-communicators (HSDP shard and replica
groups, tensor-parallel groups) are created later than the world communicator,
so they can see a worse topology than it did.

Input: NCCL logs written with NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS covering
INIT and GRAPH, ideally one file per process (NCCL_DEBUG_FILE=<dir>/%h.%p.log).

    nccl_topo.py <log dir or files ...>

Prints, per host, each distinct communicator view (ranks, channels, NVLink
bandwidth per local GPU, GPU pairs without an NVLink path) with how many
communicators had it, then flags:
  * GPUs whose NVLink bandwidth is below the host's other GPUs in every
    communicator (a persistent fault: check `nvidia-smi nvlink -s`);
  * communicators that saw less NVLink than the best view of the same GPUs
    (a transient view: that communicator runs on the degraded paths).
Stdlib only.
"""

import collections
import glob
import os
import re
import sys

PREFIX = re.compile(r"^(?P<host>[^:\s]+):(?P<pid>\d+):(?P<tid>\d+) \[\d+\] NCCL INFO (?P<msg>.*)$")
INIT = re.compile(
    r"ncclCommInitRank\w* comm (?P<addr>0x[0-9a-f]+) rank \d+ nranks (?P<nranks>\d+) .*?commId (?P<cid>0x[0-9a-f]+) - Init (?P<what>START|COMPLETE)"
)
PATHS = re.compile(r"^(?P<src>GPU/[\w-]+) :(?P<rest>.*)$")
HOP = re.compile(r"(?P<dst>(?:GPU|NVS)/[\w-]+) \((?P<hops>\d+)/(?P<bw>[\d.]+)/(?P<type>\w+)\)")
CHANNELS = re.compile(r"(?P<coll>\d+) coll channels, .*?(?P<nvls>\d+) nvls channels")


def parse(path):
    """Yield one dict per communicator init found in one log file."""
    open_comms = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = PREFIX.match(line.rstrip("\n"))
            if not m:
                continue
            key = (m["host"], m["pid"], m["tid"])
            msg = m["msg"].strip()
            mi = INIT.search(msg)
            if mi and mi["what"] == "START":
                open_comms[key] = dict(
                    host=m["host"], nranks=int(mi["nranks"]), cid=mi["cid"], file=os.path.basename(path),
                    nvs={}, pairs={}, coll=None, nvls=None,
                )
                continue
            comm = open_comms.get(key)
            if comm is None:
                continue
            if mi and mi["what"] == "COMPLETE":
                yield comm
                del open_comms[key]
                continue
            mp = PATHS.match(msg)
            if mp:
                src = mp["src"]
                comm["nvs"].setdefault(src, 0.0)
                for h in HOP.finditer(mp["rest"]):
                    bw, typ, dst = float(h["bw"]), h["type"], h["dst"]
                    if dst.startswith("NVS/") and typ == "NVL":
                        comm["nvs"][src] = max(comm["nvs"][src], bw)
                    elif dst.startswith("GPU/") and dst != src:
                        comm["pairs"][tuple(sorted((src, dst)))] = (typ, bw)
                continue
            mc = CHANNELS.search(msg)
            if mc:
                comm["coll"], comm["nvls"] = int(mc["coll"]), int(mc["nvls"])


def main(args):
    files = []
    for a in args or ["."]:
        files += sorted(glob.glob(os.path.join(a, "*.log"))) if os.path.isdir(a) else [a]
    comms = {}
    for p in files:
        for c in parse(p):
            # Every rank prints the same fused topology: keep one view per (host, communicator).
            comms.setdefault((c["host"], c["cid"]), c)
    if not comms:
        sys.exit("no communicator inits found (need NCCL_DEBUG=INFO with INIT and GRAPH)")

    best = collections.defaultdict(float)  # (host, gpu) -> best NVLink GB/s seen in any communicator
    for c in comms.values():
        for g, bw in c["nvs"].items():
            best[(c["host"], g)] = max(best[(c["host"], g)], bw)

    by_host = collections.defaultdict(list)
    for c in comms.values():
        by_host[c["host"]].append(c)

    flagged = []
    for host in sorted(by_host):
        print(f"== {host}")
        views = collections.Counter()
        for c in by_host[host]:
            gpus = sorted(c["nvs"])
            nvl = tuple(round(c["nvs"][g]) for g in gpus)
            bad_pairs = sum(1 for typ, _ in c["pairs"].values() if typ != "NVL")
            views[(c["nranks"], c["coll"], c["nvls"], tuple(gpus), nvl, bad_pairs)] += 1
            low = [g for g in gpus if c["nvs"][g] < 0.9 * best[(host, g)]]
            if low or bad_pairs:
                flagged.append((c, low, bad_pairs))
        print(f"   {'comms':>5} {'ranks':>5} {'coll':>4} {'nvls':>4}  {'non-NVL GPU pairs':>17}  NVLink GB/s per local GPU")
        for (nranks, coll, nvls, gpus, nvl, bad), n in sorted(views.items(), key=lambda kv: (-kv[0][0], -kv[1])):
            per = " ".join(f"{g.split('-')[-1]}:{v}" for g, v in zip(gpus, nvl))
            print(f"   {n:5d} {nranks:5d} {str(coll):>4} {str(nvls):>4}  {bad:17d}  {per}")
        host_best = [best[(host, g)] for (h, g) in best if h == host]
        top = max(host_best) if host_best else 0
        for (h, g), bw in sorted(best.items()):
            if h == host and bw < 0.9 * top:
                print(f"   !! GPU {g}: at most {bw:.0f} GB/s of NVLink in every communicator (others {top:.0f}):"
                      " persistent fault, check `nvidia-smi nvlink -s`")

    if flagged:
        print("\n== communicators that saw less NVLink than the same GPUs elsewhere, or GPU pairs off NVLink")
        for c, low, bad in sorted(flagged, key=lambda t: (t[0]["host"], t[0]["file"])):
            lows = " ".join(f"{g.split('-')[-1]}:{c['nvs'][g]:.0f}/{best[(c['host'], g)]:.0f}" for g in low)
            print(f"   {c['host']} {c['file']} nranks {c['nranks']} commId {c['cid']} coll {c['coll']} nvls {c['nvls']}"
                  f"  non-NVL pairs {bad}  low GPUs (seen/best) {lows or '-'}")
    else:
        print("\nevery communicator saw the same NVLink topology")


if __name__ == "__main__":
    main(sys.argv[1:])
