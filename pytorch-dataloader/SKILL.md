---
name: pytorch-dataloader
description: >-
  Diagnose and fix PyTorch input pipelines that starve the GPU: DataLoader
  wait, low or bursty GPU utilization, periodic or rare multi-second stalls,
  eval slower than train, slow time to first step, worker host-memory blowups.
  Use when tuning num_workers, prefetch_factor, pin_memory, non_blocking,
  persistent_workers or in_order; when next() wait may be a CUDA sync; when
  samples come from mmap, Arrow / HF datasets, LMDB or shards on network
  storage (Lustre, BeeGFS, VAST/NFS; Mila and Alliance/DRAC clusters) or are
  staged to $SLURM_TMPDIR; when data outgrows page cache or local disk; for
  chunk shuffling or shuffle buffers; when scaling to more nodes or faster
  GPUs; or when one DDP rank waits on another's late batch.
---

# PyTorch DataLoader: stop starving the GPU

Prove the loader is the bottleneck, name the shape of the starvation, apply the
lever for that shape. **Measure before and after on the same node, back to
back:** loader throughput varies more between nodes and runs than most fixes
change it.

## 1. Is the loader the bottleneck?

Time how long the loop blocks in `next()`, every step, on every rank:

```python
import time


class TimedIterator:
    """Records how long each next() blocks, in ms."""

    def __init__(self, iterable):
        self._it = iter(iterable)
        self.waits_ms = []

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.perf_counter()
        item = next(self._it)
        self.waits_ms.append((time.perf_counter() - t0) * 1e3)
        return item
```

Wrap what the loop actually consumes (after `accelerate.prepare` or any
device-placement wrapper). Record step wall time beside it; drop each iterator's
first `next()`. On long runs, log each interval's mean, p90 and max and then clear
`waits_ms`; a capped deque would silently drop old stalls and the run's max.

- **Wait share = Σwait / Σstep.** Under ~1%, a faster loader won't raise
  throughput. A perfect loader buys at most Σstep / (Σstep − Σwait).
- **Read mean, p90 and max, not the median.** Prefetch keeps the median near
  zero whenever the queue is ahead, even when the mean is large.
- **Under DDP, a late batch on one rank shows up on the others as all-reduce
  time.** Check every rank; take the per-step max.
- **Measure train and eval separately.** Eval has no backward, so its shorter
  steps drain the queue faster: an eval loader can starve on a config that keeps
  up with train.
- **Rule out a CUDA sync.** A blocking copy *inside* `next()` reads as loader
  wait: a near-constant cost every step that is really queued GPU work draining.
  A sync *before* it (`loss.item()`) hides a slow loader. Iterate the same loader
  with `time.sleep(step_s)` in place of the step: near-zero wait there but steady
  wait in training means a sync (§6). Take compute time from CUDA events, not
  step − wait.

## 2. Triage

| you see | it is | go to |
|---|---|---|
| high p50 wait; almost every step waits | **chronic**: workers can't keep up | read or CPU? below; §3 |
| stalls every N steps, N = `num_workers` | **periodic**: pool too slow, or one slow worker | §3 |
| rare multi-second stalls; mean ≫ p90; often all ranks at once | **tail**: storage hiccup outlasts the prefetch buffer | §3 prefetch, §4 |
| a near-constant wait every step with `pin_memory=True` | **blocking H2D copy** | §6 |
| minutes before step 1; huge host memory at startup | **setup** | §7 |
| epoch 2 no faster than epoch 1; wait rises as data outgrows cache or disk | **data bigger than the node** | §4, §5 |
| the loader kept up until a faster GPU or more nodes | **demand outgrew supply** | §8 |
| little wait, but GPU busy and MFU low | **not the loader**: batch shape or compute | out of scope |

**Read or CPU?** In one process, time `dataset[i]` over a few thousand indices
with the files evicted first (cold, §9), the same indices again (warm), and
`collate_fn` alone. Warm much faster than cold means I/O-bound: go to §4. Collate dominating
means CPU-bound: add workers, or move the work offline (pre-tokenize,
pre-decode). Workers needed ≥ (read + collate per batch) ÷ step time, plus
headroom.

More workers fix chronic and periodic starvation; a deeper buffer fixes tail
stalls. Neither fixes the other.

## 3. Workers and prefetch

- **`num_workers`:** raise it until cores, host memory or storage run out. If
  per-batch cost ÷ `num_workers` > step time the pool is too slow; otherwise one
  worker is (tag batches with `get_worker_info().id`).
- **Queue depth = `num_workers × prefetch_factor`; size it in seconds.** Depth ×
  step time should exceed the longest stall to hide. The default of 2 is often
  too shallow for network storage. Batch bytes × depth must fit in host memory.
- **`persistent_workers=True`** when iterating more than once (epochs, periodic
  eval); otherwise every new iterator pays worker startup and first file opens.
- **Threads:** workers + main-process threads ≤ cores per rank; set
  `OMP_NUM_THREADS = cores_per_rank − num_workers`. Don't pin CPUs or NUMA nodes
  unless each rank owns whole NUMA nodes and an A/B shows a win.
- **`in_order=False` (torch ≥ 2.6)** lets ready batches pass one late worker. It
  doesn't help a slow pool, a sync, or a stall hitting every worker, and it
  breaks resume-by-skipping and position-matched eval outputs.

## 4. Storage

**Where will each read be served from?** First row that fits the bytes training
actually reads (the columns it uses, not the size on disk):

| fits in | do |
|---|---|
| the job's page cache (memory limit − anon, per node, shared by its ranks) | read in place; later epochs are warm |
| node-local disk | stage once per node (§5) |
| the sum of local disks across nodes | partition across nodes (§5) |
| none | chunk-shuffled reads every epoch, plus a fixed cached subset (§5) |

Page cache is per node: another node having read the same rows doesn't make them
warm for you.

**Find the filesystem:** `findmnt -T /path/to/data -o TARGET,FSTYPE,OPTIONS`.
Free local disk: `df -h $SLURM_TMPDIR`, at job start (DRAC shares it between
jobs on a node).

| cluster (as of 2026-11) | network storage | `$SLURM_TMPDIR` |
|---|---|---|
| Mila | BeeGFS (`/network/scratch`) | local disk; request with `--tmp` |
| Narval, Rorqual, Fir, TamIA | Lustre | local SSD / NVMe |
| Nibi | VAST over NFS | local NVMe |
| Trillium | VAST | **RAM disk**, counted against job memory |
| Killarney | VAST over NFSv3 | ~350 GB (L40S), ~800 GB (H100) |

- **Every network filesystem:** each cold read is a round trip, and
  `open`/`stat`/`listdir` hit a shared metadata server. Scratch file quotas (1M
  DRAC, 10M Mila) rule out one file per sample.
- **BeeGFS:** in buffered mode `read()` bypasses the page cache, so repeated
  reads aren't cached; mmap uses it.
- **Lustre:** striping is fixed when a file is created (`lfs setstripe` the
  directory, then copy in). Avoid huge directories and `ls -l`/`du`/`find` over
  big trees.
- **VAST / NFS:** every `open()` revalidates with the server, and an `fsync` per
  commit serializes writers, possibly to a standstill. One stream doesn't fill a
  mount: use several sequential streams.

**Random reads from mmapped files** (Arrow / HF `datasets`, memmap, LMDB) are
the usual I/O-bound loader: each cold row is a page fault, and readahead pulls a
whole window of neighbouring pages to serve one small row. Prefer a
chunk-shuffled order or staging (§5). If the order must stay random, whether
those extra pages are waste depends on whether they survive until a later batch
reads them:
- **The data fits in the job's page cache: keep readahead.** A page holds many
  rows and a random epoch touches most pages, so the neighbours are read later
  in the same epoch. Readahead fetched them in a few large requests instead of
  one round trip each, and later epochs come from cache anyway.
- **The data exceeds the page cache: turn it off.** The neighbours are evicted
  before any batch reads them, so every fault wastes bandwidth and pushes out
  pages still needed. `mm.madvise(mmap.MADV_RANDOM)` on the mapping, before
  workers fork, makes each fault fetch only its own page.

| mistake on network storage | fix |
|---|---|
| one file per sample | pack into shards (tar, LMDB, Arrow, HDF5), or stage and extract locally |
| `open()` in every `__getitem__` | open once per worker, lazily, never before fork |
| `glob` / `listdir` / `getsize` on every rank | a manifest built once, offline |
| every rank scanning or copying the same files | once per node, then broadcast |
| whole-file loads on every rank | `torch.load(..., mmap=True)`, `np.load(..., mmap_mode="r")` |
| writing with a sync per commit, or many writers | LMDB `sync=False, metasync=False` + one `env.sync()`; a few writers |
| venv, temp or cache directories on network storage | a container image; `TMPDIR` and caches on `$SLURM_TMPDIR` |

## 5. Staging and read order

**Stage once per node.**
- **Price it:** bytes ÷ the node's sequential rate, paid by every job while the
  GPUs idle. At modest read rates a deeper prefetch queue may hide the network
  reads without staging: try it first.
- `srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES <copy>` before launch, or
  local rank 0 copies behind a barrier. Never once per rank.
- Copy a few large files with a few streams; extract locally. Stage only the
  columns and splits you read, val included.
- Copy to a temporary name, rename, write a marker; fall back to the network
  path when the marker is missing. Apptainer `--cleanenv` drops `SLURM_TMPDIR`.

**Chunk-shuffled read order** (assumes the data isn't sorted on disk):
1. Cut the on-disk order into contiguous chunks of MBs, aligned to shards.
2. Permute chunks with an RNG seeded by (seed, epoch); deal them to ranks
   disjointly, with equal step counts.
3. Walk each rank's chunks in windows of k; permute rows within a window.
4. Precompute it as an index array behind a batch sampler; resume is a skip.

Reads stay sequential, so leave readahead on. To check "unsorted enough",
compare the spread of per-chunk means (length, label, source) with that of
same-size chunks from a random permutation: ≫ 1 means raise k or rewrite the
data once shuffled. Bucket lengths inside the window. Libraries do shard
shuffle plus a buffer: HF `ds.shuffle(seed=42, buffer_size=10_000)` with
`ds.set_epoch(epoch)` every epoch; with any of them, split by node *and* worker
or rows repeat.

**Many epochs, data bigger than local disk.** If it fits the sum of the local
disks, node n stages part n once and reshuffles within it (re-stage when the
node count changes). If it fits nothing, stream chunk order every epoch,
staging window w+1 while training on w. Cache a fixed subset, not LRU: with a
new permutation every epoch, LRU holding a fraction f hits ~f²/2 of reads, a
fixed subset f.

## 6. Host-to-device copy

- **`pin_memory=True` needs `non_blocking=True`.** A blocking copy syncs the
  stream, so the host waits for all queued GPU work every step. An
  Accelerate-prepared loader copies each batch inside `next()`, and
  `DataLoaderConfiguration.non_blocking` defaults to `False`. Pass
  `Accelerator(dataloader_config=DataLoaderConfiguration(non_blocking=True))`,
  and keep `pin_memory=True` on the DataLoader, as Accelerate's docs recommend.
- **Accelerate ≤ 1.15.0: `skip_first_batches` drops `non_blocking`.** The loader
  it returns for a resumed epoch copies blocking again. Check `_non_blocking` on
  it; fixed on main (PR #4296), unreleased as of 2026-11.
- **Pageable copies sync too:** `torch.tensor(0.0, device="cuda")` copies from
  pageable memory and waits for the queued step. Use
  `torch.zeros((), device=...)`.
- **Hidden syncs:** `.item()`, `.tolist()`, `.cpu()`, `print(t)`, `if t:`,
  `nonzero`, boolean masks. `torch.cuda.set_sync_debug_mode("warn")` flags most.
- **Custom batch types need a `pin_memory()` method,** or stay unpinned silently;
  check `is_pinned()`.
- **Pinned memory ≈ batch bytes × queue depth,** locked, for both loaders.
- **Side-stream copy overlap rarely pays:** compare batch bytes ÷ H2D bandwidth
  with the step time before building one.

## 7. Setup and host memory

- **Build preprocessing caches in a CPU job,** never inside the GPU job.
- **Prepare once per node and broadcast a manifest,** so the other ranks only
  mmap the result.
- **Size host memory from anon, not MaxRSS** (which includes page cache). Keep
  per-sample metadata in numpy or Arrow arrays: refcount writes to Python objects
  grow forked workers' memory.

## 8. Faster GPUs and more nodes

A loader that keeps up today can starve after a GPU upgrade or a scale-out,
with no code change.

- **A faster GPU needs a faster loader.** Each node must deliver bytes per step
  ÷ step time, so a shorter step means proportionally more bytes and samples per
  second. Re-derive both rules from the new step time:
  - workers needed ≥ (read + collate per batch) ÷ step time;
  - depth × step time ≥ the longest stall. The same `prefetch_factor` now covers
    fewer seconds.
- **More nodes multiply the load on shared storage.** The aggregate rate is the
  per-node rate × nodes. Every rank and worker adds `open`/`stat` traffic to the
  same metadata server. Shared storage doesn't grow with your job, but local disk
  and page cache do, so staging and per-node partitions (§5) scale and network
  reads don't.
- **A step waits for the slowest rank.** If each rank independently stalls on a
  fraction p of steps, about 1 − (1 − p)^ranks of steps stall. A stall that is
  rare on one node becomes routine at scale, so buffer depth matters more.
- **A reshuffling `DistributedSampler` sends each node rows from the whole
  dataset over the epochs.** A node's page cache then only helps if the whole
  dataset fits in it. A fixed partition per node shrinks each node's working set
  to 1/nodes, so adding nodes makes the data more likely to fit.
- **Check before scaling.** Run the §1 GPU-free check with the target step time,
  and compare the aggregate rate with what the shared storage delivers.

## 9. Measurement rules

**The page cache outlives the job.** Pages read on this node by an earlier job,
an earlier arm or another rank stay cached. A "cold" rerun then reads partly from
memory, and an A/B can flip: the arm whose node held more of the data wins.
- **Evict the files before every cold run** (no root needed; this node only):

  ```python
  import os
  from pathlib import Path


  def evict(path):
      """Drop a file's pages from this node's page cache."""
      with Path(path).open("rb") as f:
          os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
  ```

  Evict before the dataset opens: pages a live process still maps stay cached.
  Check with `fincore <file>`.
- **Or give each cold run files or rows this node has never read.** Arms that
  share rows warm them for each other.
- **Run the arms on one node in ABBA order, with equal memory per rank.** The
  memory limit caps the page cache, so different requests mean different cache
  budgets.

Also: a barrier before multi-rank timed loops · don't share the node with your
own jobs · exclude the first `next()` and compile steps · repeat short runs (one
stall in a short eval swings the result) · read knobs from the resolved
DataLoader, not the config.
