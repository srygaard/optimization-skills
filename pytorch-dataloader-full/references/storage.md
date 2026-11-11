# Storage, setup and host memory

For loaders that `isolate` shows are I/O-bound, or that take minutes to start
or blow up host memory.

## 1. Cost model for random reads

- **A cold row costs a storage round trip plus the fault-around transfer.**
  Measured ~0.8 ms/row serially on a network filesystem, with ~256 KiB moved to
  serve a ~140 B row.
- **Cold reads per token scale with 1 / bytes per row.** A dataset with 2.3×
  shorter rows ran 0.52× as fast cold but 0.83× warm. Each read cost the same;
  it just needed more of them.
- **Page cache is per node and charged to the job's cgroup.** A job can cache at
  most its memory limit minus anon, shared by every rank on the node, however
  much RAM the node has.
- **On a network filesystem, "warm" means this node only.** Re-reading rows
  another node had just read was no faster than a fresh read (no
  storage-server cache benefit).
- **Random sampling still touches most pages.** An epoch that reads ~14% of
  rows touched an estimated 82-98% of 4 KiB pages (11-28 rows per page). So
  fault-around pages are wasted only if evicted before the epoch reaches them,
  i.e. when the data exceeds the cache.

## 2. `madvise(MADV_RANDOM)`

`scripts/madvise_random.py` advises every mapping whose path matches a
substring (it walks `/proc/self/maps`, which covers mappings created by pyarrow,
LMDB and similar libraries). For a mapping you own: `mm.madvise(mmap.MADV_RANDOM)`;
for `numpy.memmap`: `arr._mmap.madvise(mmap.MADV_RANDOM)` (private attribute).

Eight readers, one never-read 30 GiB shard, evicted before every run:

| access | default layout | 4-way striped | KiB fetched / row |
|---|---|---|---|
| plain mmap | 11.2k rows/s | 21.6k | 175 |
| mmap + `posix_fadvise(RANDOM)` | 11.4k | 24.3k | 181 |
| **mmap + `madvise(MADV_RANDOM)`** | **39.3k** | **44.6k** | **7.8** |
| `os.pread` of each row's exact bytes | 14.3k | 14.6k | 0 (no reuse) |

When the data didn't fit (cache:data ≈ 0.47, steady state): readahead fetched
78 KiB per row, and 64% of fetched bytes were re-fetches after eviction.
`MADV_RANDOM` fetched 3.4 KiB per row with 8% re-fetched, and read 4.5× the
rows/s. Readahead got worse as the cache shrank (37 → 70 KiB per row going from
cache:data 0.68 to 0.47), while `MADV_RANDOM` barely moved.

- **Choose per data source by fit.** If the data doesn't fit in cache, use
  `MADV_RANDOM`. If it fits, readahead is free prefetch. Whether readahead wins
  later in an epoch on cacheable data is reasoning, not a measurement.
- **Bytes, not requests:** both modes do ~0.7 reads per row at steady state.
  One workload projected to 40 GPUs needed ~65k random 4 KiB reads/s from
  storage with it. Ask whether your storage can serve that.
- **Verify** in each worker: `VmFlags ... rr` in `/proc/<pid>/smaps`
  (`madvise_random.py check --tree <pid> <path>`).

## 3. Things that don't work, or bite

- `posix_fadvise(POSIX_FADV_RANDOM / SEQUENTIAL)`: no effect on mmap faults
  (kernel 5.15, measured). It applies to `read()` I/O only.
- `POSIX_FADV_NOREUSE`: a no-op before Linux 6.3.
- `MADV_HUGEPAGE` on file-backed pages needs `CONFIG_READ_ONLY_THP_FOR_FS` and
  filesystem support. It didn't apply on a network filesystem.
- `pread` of exact bytes: 2.7× slower than madvised mmap on network storage. It
  needs two round trips per row (offset, then data) and gets no cache reuse.
  Holding offsets in RAM makes it one round trip.
- Mixing `read()` and mmap on one hot file: on BeeGFS, a `pread` run dropped the
  file's mmapped pages from cache (mechanism unverified).
- Absolute latency differed 2× between nodes. Compare access modes within one
  job only.

## 4. Layout and placement

Ranked roughly by effort.

- **Stripe across storage targets** (`lfs setstripe`, `beegfs-ctl --setpattern`)
  and *copy* the files in; moving them keeps the old layout. 1.9× for plain mmap,
  +13% on top of `MADV_RANDOM`: striping adds bandwidth, and madvised reads are
  latency-bound.
- **Node-local staging:** when it pays, how to copy, and why a background copy
  can't outrun a random sampler: [staging.md](staging.md) §2.
- **Arrow string columns cost ~2 pages per cold row** (offset entry plus data).
  A flat file with offsets held in RAM costs ~1. Not tested end to end.
- **Column projection is already free.** Arrow column buffers are separate, so
  reading one column never faults the others. Rewriting a one-column copy
  doesn't help random reads.
- **Reordering once** (`flatten_indices()`, a shuffled rewrite) matches at most
  one epoch when every epoch draws a new order. Read in a new order, it is the
  baseline again.
- **Dense per-epoch file.** If an epoch's selection is a function of the seed,
  gather the selected rows (needed columns only) into a compact file that fits
  page cache. In the source investigation it was ~7× smaller than the split and
  needed ~2× less storage traffic per epoch than `MADV_RANDOM`. The cost: a
  prepare stage, disk per epoch, and cache keys over seed, sources and sampling
  config.
- **Precomputed read-ordered batch store** (batches planned offline, records
  written in read order, so reads are sequential). One project built it and then
  shelved it: ~1.3× the storage, an offline repack, and artifact versioning
  against tokenizer and bucket config. A deeper prefetch queue plus dynamic
  batching handled the stalls instead. It is the ceiling, at the highest
  operational cost.
- **Shards plus a global random sampler have no locality.** A chunk-shuffled
  read order keeps reads sequential while staying random enough for unsorted
  data, and keeps length grouping: [staging.md](staging.md) §3.

## 5. Startup and one-time setup

- **Build caches in a CPU job chained before the GPU job,** and prove the cache
  is warm before launching (same code, config and lockfile fingerprints). A data
  code edit silently changed cache keys, and the GPU job rebuilt in-job; a cold
  build also left ~31 GiB of freed-but-retained memory per preparing rank.
- **Prepare once per node.** Local rank 0 prepares and broadcasts a picklable
  manifest (files, fingerprint, row count). Other ranks mmap the same files
  (~5 s) and fall back to preparing only if the manifest fails validation.
  Running every rank's preparation concurrently instead would multiply the
  setup memory peak by the rank count.
- **Hugging Face `datasets`** (seen on 4.8 and 5.0; re-check after upgrades):
  - `filter` leaves an indices mapping. Each worker of a later
    `map(num_proc=N)` re-opens every Arrow file of the table when it unpickles
    its shard: ~13 GB private per worker, ~15 min of re-reads, OOM at 22
    workers. One batched `map` that drops rows as it writes gives row-identical
    output at ~0.8 GB per worker.
  - A *cached* `filter` still rebuilds its indices in memory on each call
    (~8 B per kept row: 15 GiB and 459 s for 1.9 B rows). If a `map` follows,
    open the map's cache file directly.
  - `flatten_indices()` rewrites every row even when there is no indices
    mapping. Check `ds._indices is None` first.
  - `map` on an empty dataset writes no file and returns a slice of its input's
    files. Check row counts after opening cache files directly.
  - `Dataset.from_file` plus `concatenate_datasets` gives a new fingerprint.
    Restore it, or every later cached step misses.
- **pyarrow `ChunkedArray.take()`** with indices spanning every chunk
  concatenates the whole column first: gathering 14% of a 30 GiB shard needed
  ~14 GiB. Take per chunk with sorted local indices.

## 6. Host memory

- **MaxRSS includes page cache.** Read `anon` and `file` from the cgroup's
  `memory.stat`; `file` is reclaimable.
- **Forked workers share pages copy-on-write.** 4 ranks × 8 workers each showed
  ~41 GiB anon, which sums to ~1.4 TB, but the cgroup held ~4 × 41 GiB. A
  worker's private cost is `Private_Dirty` in `smaps_rollup`, or worker
  `RssAnon` − parent `RssAnon`.
- **Anon creep** (+8 GiB over 1,150 steps, undiagnosed). The usual cause is
  refcount updates dirtying inherited Python objects, which gives each worker
  private copies. Keep per-sample state in numpy/Arrow arrays. To test, sample
  each worker's `Private_Dirty` over time.
- **Release as a diagnostic.** `gc.collect()` +
  `pa.default_memory_pool().release_unused()` + glibc `malloc_trim(0)` freed
  31 GiB per rank before the setup fixes and ~0.1 GiB after. A large release
  means some phase allocated and freed a lot; fix that phase.
- **Sizing:** anon peak (setup included) + headroom for creep + the page cache
  you intend. Too little cache shows first as rising refaults, then as bytes
  fetched and loader wait, well before any OOM.
