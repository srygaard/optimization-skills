# Measuring a DataLoader

Loader effects are often smaller than run-to-run noise, and several of the most
convincing early results in the source investigations were artifacts. Read this
before trusting any number.

## 1. Instrument the training loop

- **Wrap the iterator the loop actually consumes**, after `accelerate.prepare`
  or any device-placing wrapper. That is where a blocking host-to-device copy
  hides (SKILL.md §5). `TimedIterator` in `scripts/loader_bench.py` is a drop-in.
- **Is the wait really the loader?** `next()` time is how long the host blocked,
  not why. A sync distorts it in both directions:
  - **Inflated:** a sync *inside* `next()` waits on the previous step's queued GPU
    work. The usual source is a blocking or pageable H2D copy in a
    device-placing wrapper. Measured: ~72 ms/step. It is steady every step, even
    with a full prefetch queue, and `analyze` reports it as chronic starvation.
    More workers won't touch it.
  - **Deflated:** a sync *before* `next()` (`loss.item()`, logging) gives the
    queue time to refill, so a slow loader can show ~0 wait.
  - **Overlapping:** with no syncs the host runs ahead, and its wait overlaps
    GPU work still queued. Wait rose 3.9 → 160.9 ms/step after one sync was
    removed, with device time unchanged (455 vs 450 ms). Removing a different
    sync exposed the loader's own 72 ms one: fixing one sync can move the wait
    elsewhere.

  Checks, cheapest first:
  1. `loader_bench.py run --step-ms <real step>` uses no CUDA. Wait ≈ 0 there
     but steady in training points to a sync, not the loader.
  2. Put a diagnostic `torch.cuda.synchronize()` right before `next()` and time
     both: the sync holds the GPU backlog, `next()` holds only loader time. This
     serializes the loop; diagnose with it, don't ship it.
  3. Wrap the raw DataLoader (before device placement) and time the
     `.to(device)` separately.
  4. `torch.cuda.set_sync_debug_mode("warn")` flags `.item()`, `.cpu()` and
     blocking H2D copies inside `next()`. It misses `torch.cuda.synchronize()`
     and non-blocking copies from pageable memory
     ([host-to-device.md](host-to-device.md) §2).
  5. Ground truth: GPU idle between steps from CUDA events, or a trace with
     `next()` annotated (`torch.profiler.record_function`). Look for
     `cudaStreamSynchronize` or `Memcpy HtoD (Pageable -> Device)` inside it.

  Fixes: [host-to-device.md](host-to-device.md).
- **Record per step, per rank, per phase:** wait ms, step wall ms (including the
  wait), real tokens or samples, batch shape. Keep per-step values, not interval
  averages: the stall signature is in the distribution.
- **Exclude the first `next()` of every iterator** (worker start and first file
  opens: 14-18 s measured). Exclude compile steps by the *first appearance of
  each shape*, not by a fixed warmup count.
- **Aggregate DDP ranks with the per-step max**: the slowest rank sets the step.
- **Report compute-side throughput separately, from device time**: CUDA events
  around the compute region, read after one synchronize. Without it, loader
  stalls leak into compute A/Bs: in one run, dataloader stalls added 86 ms/step
  to the mean, while median step times and kernel-level numbers were
  unaffected.
- **Don't derive compute time as (step − wait).** Once the host runs ahead, the
  wait overlaps queued GPU work instead of adding to it. One A/B subtracted the
  wait and reported 1.56×; device time showed 1.01×.
- **Check that the instrumentation is live.** A per-batch line logged at
  `debug` under an `INFO` logger vanished silently while its counters kept
  updating. Totals that should match a known count (steps × batch size) catch this.

### Train and eval: diagnose apart, budget together

Most jobs run two loaders. Each one gets its own diagnosis, but they share one node.

**Separately: each loader gets its own numbers and verdict.**
- **One `TimedIterator` per loader, and every step tagged with its phase.** Compute wait share, p90, stall counts and the `analyze` verdict per phase, against that phase's step times. A pooled number is dominated by whichever phase has more steps.
- **The same wait means more in eval.** Eval steps have no backward, so they drain the queue faster and hide less behind queued GPU work.
  - With 4 workers, eval sat at 40-48% GPU idle (wait ~99-125 ms) while train on the same loader was fine. 8 workers gave eval 1.7×.
  - A 72 ms blocking copy cost train 1.01× (~450 ms steps hid it) but eval 1.05× (~73 ms steps).
- **Size each loader's knobs from its own step time.** Workers needed and prefetch depth in seconds (SKILL.md §3) both use it, so a depth that covers seconds of train covers far less of eval.
- **Eval samples are small.** One stall in a 48-step eval read as 0.63×; the rerun was 1.00×. Use more eval steps, or repeat.
- **Exclude per pass, count per run.**
  - Each eval pass makes a new iterator, so its first `next()` pays the startup again (unless `persistent_workers`).
  - The first eval step recompiles the forward (grad mode changes): 23-75 s, once per run.
  - Keep both out of steady-state statistics, but add them to eval's wall time: a startup cost × the number of eval passes can be minutes.

**Together: they share the node's memory, cache and storage.**
- **Weight by wall time.** A loader's starvation costs the run its wait share × its phase's share of wall time. Eval at 40% idle in 5% of the run costs less than train at 5% idle.
- **Host memory adds up.** Persistent workers of both loaders stay alive through the other phase, and the paused train iterator keeps its full prefetch queue, pinned, through every eval pass. Budget workers and batch bytes × depth for both.
- **One page-cache budget.** Reading the val split can evict train pages when train barely fits. Look for wait right after each eval pass that the rest of the epoch doesn't show (reasoning, not measured). A small val split that fits stays warm across passes on that node.
- **Storage advice and staging go per loader.** Choose advice by each loader's read order: `MADV_RANDOM` cut a randomly read val split's traffic 23×, but an eval read in on-disk order wants readahead. Stage the val split too; every eval pass rereads it.
- **DDP:** ranks that finish eval unevenly wait in the metric all-reduce or gather, not in `next()`. Breaking out of a prepared eval loader early can disable gradient accumulation (SKILL.md §6).

## 2. Run the loader without the GPU

- **Build the exact DataLoader training builds** (same dataset config, sampler,
  collate, sharding) and only fetch: `loader_bench.py run`. CPU nodes are
  cheaper and remove GPU-side confounders. `--step-ms <median step>` adds a
  simulated consumer so the prefetch queue behaves as in training. Without it
  you measure peak loader throughput.
- **Multi-rank harnesses need a barrier before the timed loop.** Preparation was
  serialized by a file lock, so each rank started timing as soon as its own
  setup finished. The ranks never overlapped, and "linear scaling" was measured
  with no contention at all.
- **Cold vs warm:** `loader_bench.py isolate`, or two passes over a sampler
  whose order is deterministic across epochs (`RandomSampler` draws a new order
  each epoch, so its second pass is not warm).
- **The ceiling from the other side:** compute-only throughput (device time per
  step), or a run that cycles a few real batches of every shape already on the
  GPU. The gap to the real run is the most a loader fix can recover.

## 3. A/B traps (each one bit a real investigation)

| trap | symptom | fix |
|---|---|---|
| Arms on different nodes | Same code, 1.7-3.2× apart in cold throughput | One node, back to back. Two runs on one node agreed within 0.04% |
| Your own jobs share a node | A sweep landed on one node, several jobs at once; the worst results were exactly the overlapping pairs | Submit sequentially, or exclusive / singleton scheduling |
| Page cache outlives the job | A same-seed rerun read partly warm; one dataset showed "super-linear" 1.84× scaling at 2 ranks | A fresh seed per cold run, or evict files (`--evict`, `POSIX_FADV_DONTNEED`) |
| Node cache history | Pages earlier jobs left in cache are free hits. An A/B and its reversed-order rerun gave opposite answers; both times, the job on the warmer node read less | ABBA order on one node, or answer storage questions in a CPU harness on evicted files |
| Memory limit caps page cache | Arms with different memory requests get different cache budgets | Equal memory per rank in every arm |
| Mixed hardware pools | The first comparison of two datasets ran on different node types | Constrain to one node type |
| Shared-storage load drifts | Cold throughput 194k vs 372k tok/s, same node, same config | Repeat; compare within a run (cold/warm ratio) |
| Small samples | One stall in a 48-step eval → 0.63×; rerun 1.00× | Hundreds of steps; repeat; report stall counts beside the mean |
| Mislabeled arms | Notes said 8 workers, the log said 10; `prefetch_factor` was hardcoded and hand-edited per job; an Accelerate YAML `store_true` flag was silently ignored | Print the resolved DataLoader parameters at startup and read them back (`loader_bench.py` prints them) |
| Arms re-read each other's rows | Arms sharing one file ran 2-6× too fast, superlinear with readers | Each arm its own never-read file; disjoint rows within a file |
| Step time as a loader proxy | Best loader stats in a sweep, worst MFU | Measure wait directly |
| Setup dwarfs the timed loop | A large source needed 15-20 min per rank before its first fetch; 1 h limits failed | Set time limits from measured setup × ranks |
| Harness rot | After a rebase, every queued job died within 40 s | Smoke-test locally with a few batches before queuing |

## 4. Bytes fetched from storage

Linux cgroup v2, `memory.stat` of the job's cgroup (`/sys/fs/cgroup` + the
`0::` path in `/proc/self/cgroup`), sampled before and after:

| quantity | from |
|---|---|
| bytes brought into page cache (incl. later evicted) | Δ`file` + Δ`pgsteal` × page size |
| bytes re-fetched after eviction (thrashing) | Δ`workingset_refault_file` × page size |
| read requests (≈ IOPS) | Δ`pgmajfault` |

`loader_bench.py run` prints these. Slurm nests cgroups; the binding limit is
the ancestor with the smallest finite `memory.max`. For one file only this
process reads, `fincore` growth is exact until the cache fills.

Don't use: `/proc/<pid>/io` `read_bytes` (counts readahead I/O only: +39 MB
with readahead, **+0** under `MADV_RANDOM` for the same 300 page touches),
`rchar` (only `read()` syscalls), cgroup `io.stat` (block devices only, blind to
network filesystems), or NIC counters (whole node, every job).

## 5. Capturing a rare stall

Per-step numbers cannot explain a 5-second stall. Capture one:

- **A watchdog thread** that, when a step passes a threshold, dumps every
  thread's stack (`faulthandler`, or `py-spy dump --pid`), each worker's state
  and `/proc/<pid>/wchan`, pressure-stall deltas (`/proc/pressure/io` or the
  cgroup's `io.pressure`), GC pause times, and its own wake-up lag.
- **Profile rolling windows and export only those containing a slow step.**
- **Reading it:** workers in `D` state with `wchan` = `wait_on_page_bit_common`
  (`folio_wait_bit_common` on newer kernels) are blocked on page faults into
  the mmapped files, so the stall is storage, not code. `io.pressure` `full` ≈
  the stall length confirms it. No GC pause plus a small watchdog lag means the
  process was not frozen.
- **Traps:** don't judge steps inside the profiler's warmup or the compile
  window (step 1 got flagged); disarm the watchdog at loop exit or teardown
  reads as a stall.
- Measured: ~1 stall >5 s per 1,000 steps (~0.35% of throughput), all ranks at
  once, from page faults on network scratch outlasting an 8 × 4 batch buffer.

## 6. Tools

`py-spy dump --pid` · `/proc/<pid>/wchan` · `/proc/<pid>/status` (`RssAnon`,
`RssFile`) · `/proc/<pid>/smaps_rollup` (`Private_Dirty`) · `/proc/<pid>/smaps`
(`VmFlags`, `rr` = random-read advice) · `fincore` · filesystem layout tools
(`lfs getstripe`, `beegfs-ctl --getentryinfo`, `mmlsattr`) · `srun --jobid=<id> --overlap`
to inspect a running Slurm job.
