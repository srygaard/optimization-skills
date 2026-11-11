---
name: pytorch-dataloader-full
description: >-
  Diagnose and fix PyTorch input pipelines that starve the GPU: DataLoader wait,
  low or bursty GPU utilization, periodic or rare multi-second stalls, eval
  slower than train, slow time to first step, worker host-memory blowups. Use
  when tuning num_workers, prefetch_factor, pin_memory, non_blocking,
  persistent_workers, in_order, worker affinity or OMP_NUM_THREADS; when next()
  wait may be a CUDA sync; when samples come from mmap, Arrow / HF datasets,
  LMDB or shards on network storage (Lustre, BeeGFS, GPFS, VAST/NFS; Mila and
  Alliance/DRAC clusters) or staged to $SLURM_TMPDIR; when data outgrows page
  cache or local disk over epochs; when weighing madvise or readahead,
  chunk/block shuffling or shuffle buffers, length bucketing, token-budget
  batches or packing; for pinned memory and side-stream copies; when one DDP
  rank waits on another's late batch; or with Mosaic Streaming, WebDataset,
  LitData or DALI. Also for GPU-free loader benchmarks and A/B hygiene.
---

# DataLoader diagnostics: GPU starvation

A router. Prove the loader is the bottleneck, name the shape of the starvation,
apply the lever for that shape. Depth lives in `references/`.

**Hard rule:** measure before and after on the same node, back to back. Loader
numbers move more between nodes and runs than most fixes move them: the same
code measured 1.7-3.2× apart across nodes, while two runs on one node agreed
within 0.04%. See [measurement.md](references/measurement.md).

---

## 0. Is the loader the bottleneck at all?

Time how long the training loop blocks in `next()`, every step, on every rank:

```python
from loader_bench import TimedIterator     # scripts/loader_bench.py (or copy the class)
it = TimedIterator(train_loader)           # wrap the loader the loop really consumes,
for batch in it: ...                       # i.e. after accelerate.prepare / device placement
```
```bash
python3 scripts/loader_bench.py analyze waits.json --steps steps.json --num-workers 8
```

- **Wait share = Σwait / Σstep.** Under ~1%, faster batches will not raise
  throughput. Measured: p90 wait 5-7 ms of a 1.4 s step, and a change that cut
  storage traffic 36× moved throughput by zero. Loader work there pays off as
  less storage load, faster startup and less host memory, not tokens/s.
- **A perfect loader buys at most Σstep / (Σstep − Σwait).** One workload: wait
  was 54.6% of wall time, so ≤2.18×.
- **Read mean, p90 and max, never the median alone.** Prefetch keeps p50 at
  ~1 ms whenever the queue is ahead, including in a run whose mean was 63 ms.
- **Diagnose train and eval separately; budget them together.** Short eval
  steps drain the queue faster: eval sat at ~42% GPU idle while train, on the
  same loader, was fine. The two loaders still share the node's memory, page
  cache and storage: [measurement.md](references/measurement.md) §1, "Train and
  eval".
- **Under DDP, a late batch on one rank shows up on the others** as all-reduce /
  backward-sync time, not as loader wait. Check every rank before blaming comms.
- **Rule out the non-loader causes first:** tokens per step (a packing-fill fix,
  82% → 99.5%, was worth 1.15× alone) and eval cadence.
- `next()` wait is an upper bound on data-caused GPU idle. It is near exact if
  the loop syncs every step (`loss.item()`); if the CPU runs ahead, queued GPU
  work hides part of it. Confirm with a trace when the number decides something.
- **Measure the wait; don't infer it from step time.** Compute jitters too: the
  run with the best loader stats in one sweep had the worst MFU.

---

## 1. Triage: what shape is the starvation?

A CUDA sync inside `next()` reads as loader wait, and a sync just before it hides
wait. Rule that out before picking a row: [measurement.md](references/measurement.md)
§1, "Is the wait really the loader?".

| you observe | it is | first check | go to |
|---|---|---|---|
| p50 wait is tens of ms; almost every step waits | **chronic**: workers can't keep up | `loader_bench.py isolate`: read or collate? | §2, §3 |
| stalls recur every N steps, N = `num_workers` | **periodic**: the pool is too slow in aggregate, or one worker is slow | per-batch cost ÷ `num_workers` vs step time | §2, §3 |
| rare 0.5-10 s stalls; mean ≫ p90; often all ranks at the same step | **tail**: a storage hiccup outlasts the prefetch buffer | stall capture: workers in `D` state on page faults | §3, §4 |
| a steady few-to-hundreds of ms every step with `pin_memory=True` | **blocking H2D copy** | `cudaStreamSynchronize` inside `next()` in a trace | §5 |
| minutes before step 1; huge host anon at startup | **setup** | `py-spy dump`, cgroup `memory.stat` | §6 |
| GPU busy but MFU low; step time swings with batch shape | **batch shape**, not I/O | per-shape step time, fill fraction | §7 |
| epoch 2 no faster than epoch 1; wait rises once the data outgrows page cache or local disk | **data bigger than the node** | bytes read vs memory limit − anon; `df -h $SLURM_TMPDIR` | [staging.md](references/staging.md) |

**Chronic vs tail decides the lever.** More workers fix chronic and periodic
starvation but not the tail: max fetch went 434 → 758 ms as workers went 1 → 8.
A deeper buffer absorbs tail stalls but cannot fix chronic starvation, because a
buffer that drains faster than it fills is empty in steady state.

---

## 2. I/O-bound or CPU-bound?

```bash
python3 scripts/loader_bench.py isolate --factory mypkg.data:make_train_loader \
    --rows 2000 --step-ms 60 [--evict /data/cache]
```

Times dataset reads cold, the same rows again warm, then `collate_fn` alone, in
one process, and prints how many workers the step time needs.

- **Warm ≫ cold → I/O-bound.** Measured 3.3× and 5.3× on two datasets read in
  random order from a network filesystem. Go to §4.
- **Collate dominates → CPU-bound.** Workers scale this near-linearly until
  cores run out; otherwise move the work offline (pre-tokenize, pre-decode).
- **Reads ~1-2 ms with a rare ~1 s outlier, collate tight** → the tail lives in
  storage, not in your code.
- **Workers needed ≥ (read + collate per batch) / step time**, plus headroom for
  the tail.

"Warm" on a network filesystem means *this node's* page cache: rows another
node had just read were as slow as fresh ones. A "cold" pass is warm if an
earlier job left the pages cached; `--evict` drops them (no root needed).

---

## 3. Workers, prefetch, threads, affinity

- **`num_workers`.** One rank, 1 → 2 → 4 → 8 workers: 0.47 → 0.86 → 1.49 →
  2.06 M tok/s, p50 wait 53 → 0.23 ms. Going 4 → 8 took an eval phase from 42% to
  0.5% GPU idle (1.7× eval throughput).
- **Queue depth is `num_workers × prefetch_factor` batches.** Size it in
  seconds: depth × step time should exceed the longest stall you want hidden
  (one config: 8 × 16 = 128 batches ≈ 20 s). With reads over a network
  filesystem the default of 2 had 1-9 stalls >50 ms per 222 steps; 4 and 8 had
  none in six runs. Check that batch bytes × depth fits in host memory.
- **Sweeping workers also changes depth.** To separate more fetchers from a
  deeper buffer, hold the product constant.
- **Periodic, one slow worker or a slow pool?** If per-batch cost ÷
  `num_workers` > step time, the pool is too slow. If not, one worker is: tag
  batches with `get_worker_info().id`.
- **`in_order=False` (torch ≥ 2.6) is for one late batch, not a default.**
  - **It fixes** head-of-line blocking: ready batches from other workers pass a
    late one, and the epoch's set of batches is unchanged.
  - **It doesn't fix** a slow pool, a sync, or a storage stall that hits every
    worker at once, which is what the captured rare stalls were.
  - **Costs besides determinism:**
    - Resuming by skipping N batches repeats or misses up to
      `num_workers × prefetch_factor` of them.
    - Anything matched by position breaks. Keep eval in order unless its
      outputs carry ids.
    - The order of shapes under bucketing and compile is no longer
      reproducible.
  - Not measured in isolation here.
- **`persistent_workers=True`** when the loader is iterated more than once
  (epochs, periodic eval). The first `next()` of a fresh iterator cost 14-18 s
  (worker start plus first file opens).
- **Threads.** Workers plus main-process intra-op threads should be at most the
  cores per rank. PyTorch gives each worker one torch thread, but not the main
  process or non-torch libraries. One launcher set `OMP_NUM_THREADS = cores_per_rank − num_workers`.
- **Don't pin CPUs or NUMA nodes unless each rank owns whole NUMA nodes.** It
  never helped: eval 0.90× at 4 workers; `--cpu-bind=cores` made the worst
  fetch 6× longer (8.5 vs 1.4 s); GPU-local binding on a scattered shared
  allocation squeezed rank + workers + 16 OMP threads onto 3-7 cores, ~33%
  slower; NVML masks reaching outside the job's cgroup crashed
  `sched_setaffinity` (EINVAL). Forked workers inherit the parent's mask.

---

## 4. Random reads from memory-mapped files

This is the usual cause of an I/O-bound loader: samples read in random order
from mmapped files (Arrow / HF `datasets`, numpy memmap, LMDB) on network
storage. Each cold row is a page fault, and kernel fault-around pulled
~100-260 KiB to serve a ~140 B row.

```python
from madvise_random import advise_random   # scripts/madvise_random.py
advise_random("/path/to/dataset/files")    # after the dataset opens, before workers start
```

- **Measured:** per-row latency 2.5× lower, 27× fewer bytes; real loader 2.69×
  cold throughput (1.7-3.2× across node pairs). When the data was ~2× the
  page cache: 4.5× rows/s, because without it 64% of fetched bytes were pages
  re-fetched after eviction.
- **Decide per data source, and per loader.** Use it when the hot data exceeds
  what the job can cache (memory limit − anon, shared by every rank on the
  node). When the data fits, readahead is free prefetch; leave it alone. An eval
  loader that reads in on-disk order wants readahead even when train doesn't.
- **It cuts bytes, not requests:** ~0.7 reads per row either way, 4 KiB instead
  of ~100 KiB. The storage must sustain the IOPS.
- **`posix_fadvise(POSIX_FADV_RANDOM)` does nothing for mmap page faults.**
- Apply after opening the files: opening scans metadata sequentially, which
  readahead helps.
- Forked workers inherit the advice. Under spawn/forkserver (the Linux default
  from Python 3.14), call it in `worker_init_fn`. Verify the workers got it:
  `python3 scripts/madvise_random.py check --tree <pid> <path>`.
- It can be a storage-only win: on a compute-bound 4×H100 job, zero throughput
  change but 36× less traffic to shared storage.

Striping, dense per-epoch files, measuring bytes actually fetched:
[storage.md](references/storage.md).

- **Or change the read order.** Chunk-shuffled reads stay sequential, keep
  readahead useful, and are random enough if the data isn't sorted on disk:
  [staging.md](references/staging.md) §3.
- **Or move the data.** Stage once per node; when it doesn't fit, partition it
  across nodes or cache a fixed subset: [staging.md](references/staging.md) §2, §4.
- **Filesystems differ.** BeeGFS `read()` skips the page cache, NFS revalidates
  on every `open()`, and Lustre stripes per file. Mila/DRAC table and common
  mistakes: [network-fs.md](references/network-fs.md).

---

## 5. Host-to-device copy

- **`pin_memory=True` is not enough; the copy must be `non_blocking=True`.**
  An Accelerate-prepared loader copies each batch inside `next()`, and
  `DataLoaderConfiguration.non_blocking` defaults to `False`. The blocking copy
  showed as ~72 ms/step of `cudaStreamSynchronize` inside `next()`. That time
  went to draining queued GPU work, not to moving bytes. Fixing it took mean
  wait from 160.9 → 1.5 ms (1.01× train, 1.05× eval). Pass
  `Accelerator(dataloader_config=DataLoaderConfiguration(non_blocking=True))`
  and keep `pin_memory=True`, as Accelerate's docs recommend.
- **Accelerate ≤ 1.15.0: `skip_first_batches` drops `non_blocking`.** The
  resumed epoch's loader copies blocking again. Check `_non_blocking` on it;
  fixed on main (PR #4296), unreleased as of 2026-11.
- **A pageable copy syncs too.** `torch.tensor(0.0, device="cuda")` copied 4
  bytes and then waited 63-73 ms for the queued forward and backward. Allocate on
  the device instead (`torch.zeros((), device=...)`).
- **Custom batch types stay unpinned unless they define `pin_memory()`.** No
  error is raised; check `is_pinned()` on what reaches the copy.
- **Pinned memory ≈ batch bytes × queue depth,** and it is locked. A deep
  `prefetch_factor` multiplies it.
- **Check batch bytes before building copy/compute overlap.** A side-stream
  prefetcher hid 81% of the copy and gained 1.00×: the batch was 800 KB, ~65 µs.

Hidden syncs, non-blocking correctness, side streams and reading a trace:
[host-to-device.md](references/host-to-device.md).

---

## 6. Setup and host memory

- **Build preprocessing caches in a separate CPU job**, never inside the GPU
  job. An edit to data-prep code silently changed cache keys and the GPU job
  rebuilt for ~40 min before step 1.
- **Prepare once per node and broadcast a manifest**, so the other ranks just
  mmap the same files. Time to first step went from 26-40 min to ~2 min, and
  host anon peak from ~195 to 57-66 GiB (4 ranks).
- **Size host memory from anon, not MaxRSS.** MaxRSS includes page cache, which
  fills any limit. Summing per-worker RSS double-counts copy-on-write pages.
- **Anon that creeps up with forked workers** usually comes from refcount writes
  to inherited Python objects. Keep per-sample metadata in numpy/Arrow arrays.
- **Breaking out of an Accelerate-prepared loader loop early** (e.g. a max eval
  steps) leaves it registered on the shared `GradientState`. This can silently
  disable gradient accumulation. Release it explicitly.
- HF `datasets` and pyarrow traps (`filter` → each `map` worker re-opens the
  whole table, ~13 GB per worker, OOM): [storage.md](references/storage.md) §5.

---

## 7. Batch shape

The loader also decides whether batches keep an overhead-bound GPU busy.

- **Token-budget batching:** rows per bucket = T // bucket length. Fixed
  72-row length buckets swung per-step compute 5.6× and pinned memory to the
  largest bucket. Token budgets kept one shape per bucket (no extra compiles),
  cut the swing to 1.08×, and gave +22.5% throughput on a small model. The
  effective batch changes, so retune LR (√ scaling for Adam-family) and express
  warmup in tokens.
- **Pick bucket boundaries by DP over the real length histogram.** Padding fell
  from 12.6% to 9.05%. With token-budget batch sizes, the padding-optimal grid
  was also the throughput-optimal one. Alignment finer than the attention
  kernel's block size bought nothing (152 vs 160: within 1%).
- **Packing** gives one shape and ~zero padding but needs segment-aware
  attention. Check fill fraction first.
- **Log the shapes that reach `forward`**, not the ones the sampler was
  configured for.

Details and the LR rule: [batching.md](references/batching.md).

---

## 8. Levers, ranked

| lever | measured effect | cost | reach for it when |
|---|---|---|---|
| `non_blocking=True` with pinned memory | wait 161 → 1.5 ms/step | none | wait ≈ constant every step |
| more `num_workers` | 4.3× at 8 vs 1; eval 1.7× | cores, host memory | chronic, periodic |
| `prefetch_factor` 2 → 4-16 | stalls >50 ms: up to 9 → 0 per 222 steps | batch bytes × depth | tail stalls |
| `madvise(MADV_RANDOM)` | 2.7× cold; 4.5× when data > cache; 27-36× fewer bytes | none; per source | random reads, mmap, network storage |
| stage to node-local disk, once per node | mean wait 12-76 → 1-1.5 ms (short runs) | copy time every job; local disk ≥ bytes read | data fits local disk |
| chunk-shuffled read order | sequential vs random LMDB reads: 319 vs 13.7 MB/s, 8 readers (a build job, not a loader) | sampler plan; mixing bounded by the window | data exceeds cache or disk; on-disk order unsorted |
| caches built offline, prep once per node | first step 26-40 → ~2 min; anon ~3× lower | refactor | slow startup, OOM at startup |
| token-budget batch sizes | +22.5% throughput | LR retune | variable lengths, overhead-bound GPU |
| stripe files across storage targets | 1.9× plain mmap; +13% on top of madvise | re-copy data | access advice impossible |
| dense per-epoch file / read-ordered store | ~2× less traffic than madvise | offline build, disk, versioning | storage IOPS are the limit |
| CPU / NUMA pinning | 0.67-1.00× | crashes on some allocations | rank owns whole NUMA nodes and an A/B shows a win |

Every row was measured on one workload. Re-measure on yours.

---

## 9. Scripts

`analyze` is stdlib-only; `run` and `isolate` import torch.

| command | gives you |
|---|---|
| `loader_bench.py run --factory pkg.mod:make_loader --step-ms 60` | GPU-free repro: waits, verdict, resolved knobs, bytes fetched (`--madvise`, `--evict` for A/Bs) |
| `loader_bench.py isolate --factory pkg.mod:make_loader --step-ms 60` | read vs collate, cold vs warm, workers needed |
| `loader_bench.py analyze waits.json --steps steps.json` | verdict for a real run's per-step waits |
| `madvise_random.py check --tree <pid> /data/path` | whether the advice reached every worker |

`--factory` names a callable that returns the exact training DataLoader. Expose
the knobs you want to sweep as its keyword arguments (`--kw num_workers=8`).

---

## 10. Measurement rules (short form)

Same node, back to back, ABBA when node page cache matters · a fresh seed or
evicted files for every cold run · a barrier before multi-rank timed loops, or
nothing contends · never share the node with your own other jobs · exclude the
first `next()` and compile steps · repeat (one stall in 48 eval steps read as
0.63×; the rerun was 1.00×) · read knobs from the resolved DataLoader, not the
job label · bytes fetched from cgroup `memory.stat`, not `/proc/<pid>/io`.
Full versions, with the incident behind each: [measurement.md](references/measurement.md).

---

## 11. Reference map

| open | when |
|---|---|
| [measurement.md](references/measurement.md) | instrumenting, standalone harnesses, A/B traps, bytes fetched, capturing a rare stall |
| [storage.md](references/storage.md) | I/O-bound: mmap and page cache, madvise, layouts, staging, startup, host memory, HF `datasets` traps |
| [batching.md](references/batching.md) | token-budget batching, bucket boundaries, packing, LR scaling |
| [network-fs.md](references/network-fs.md) | which filesystem you're on, Mila/DRAC clusters, BeeGFS / Lustre / VAST-NFS behaviour, mistakes on network storage |
| [staging.md](references/staging.md) | staging to local disk, chunk-shuffled read order, many epochs over data bigger than the node |
| [host-to-device.md](references/host-to-device.md) | pinned memory, `non_blocking`, hidden syncs, side-stream copies, trace signatures |
| [frameworks.md](references/frameworks.md) | using or choosing a loader other than the torch DataLoader (Mosaic Streaming, WebDataset, LitData, HF streaming, DALI, …) |

---

*Numbers come from two transformer pretraining workloads, one reading Arrow /
Hugging Face datasets and one reading LMDB shards, both from network
filesystems (BeeGFS, and VAST over NFS), on NVIDIA A100/H100 and AMD MI300A. Treat them as orders of
magnitude and re-measure on yours. The mechanisms are general; the numbers are
not.*
