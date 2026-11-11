# Other loader frameworks

The mechanisms in this skill (page cache per node, read order against readahead,
staging, prefetch depth in seconds, syncs in the copy) don't change with the
framework. Only the knob names change. This file maps each mechanism to each
framework's knob.

Names and defaults were checked against each project's docs or source on
2026-11-09. Re-check them after upgrading. Nothing here is measured in this
skill's workloads.

## 1. Pick one

| framework | data on disk | adds over the torch DataLoader | measure with |
|---|---|---|---|
| torch DataLoader + torchdata `StatefulDataLoader` | anything | `StatefulDataLoader`: mid-epoch `state_dict()` | `loader_bench.py run` / `isolate` |
| `torchdata.nodes` | anything | composable stages on threads or processes | `TimedIterator` |
| SPDL | anything | thread-based async pipeline; fastest with free-threaded Python | `TimedIterator` |
| Mosaic Streaming | MDS shards (convert) | shuffle algorithms, local shard cache with eviction, deterministic resume across node counts | `run` (`StreamingDataLoader`) |
| LitData | chunks written by `optimize()` (convert) | chunk + item shuffling, cache limit, resume | `run` (`StreamingDataLoader`) |
| WebDataset | tar shards (convert) | shard shuffle + sample buffer, optional download cache | `TimedIterator` (`WebLoader` wraps a DataLoader, isn't one) |
| Megatron-Energon | WebDataset shards via `energon prepare` | multimodal samples, dataset blending, savable loader state | `run` (`SavableDataLoader`) |
| HF `datasets` | Arrow / Parquet / files | exact or buffered shuffle, sharded streaming, `state_dict()` | `run` (pass to a DataLoader) |
| FFCV | `.beton` (convert) | quasi-random page order; inactive since 2023 | `TimedIterator` |
| NVIDIA DALI | files, TFRecord, … | decode and augment on the GPU, so the bottleneck has a different shape | `TimedIterator` |
| Ray Data | Parquet, files, … | streaming execution across CPU nodes | `TimedIterator` |
| Lance | Lance columnar (convert) | fast random access, fragment-aware samplers | `run` (pass to a DataLoader) |

**About the "measure with" column:**
- `loader_bench.py run` reads `dataset`, `sampler`, `collate_fn` and `num_workers` off the loader, so it needs a torch DataLoader or a subclass.
- `isolate` refuses an `IterableDataset`. Every streaming framework above is one, so time their read and decode stages separately.
- `TimedIterator` wraps any iterable.

tf.data and Grain belong to other ecosystems, and Petastorm is in maintenance mode. They are out of scope here.

## 2. Knob map

Every block lists the same fields, in this order:
- **order:** the read order ([staging.md](staging.md) §3);
- **cache:** local cache and eviction ([staging.md](staging.md) §4);
- **split:** across nodes, ranks and workers, and whether ranks get equal steps;
- **resume:** mid-epoch;
- **workers:** the worker, prefetch and pin equivalents (SKILL.md §3, §5);
- **traps**, and the documented **example**.

**torch DataLoader / torchdata**
- **order:** the sampler. `RandomSampler`, `DistributedSampler`, or a precomputed chunk plan ([staging.md](staging.md) §3). `in_order` (SKILL.md §3).
- **cache:** none beyond the OS page cache.
- **split:** `DistributedSampler`. With `drop_last=False` it pads so ranks are equal. Call `sampler.set_epoch(epoch)` every epoch, or the order repeats.
- **resume:** `StatefulDataLoader` (a DataLoader subclass) `state_dict()` / `load_state_dict()`, plus `snapshot_every_n_steps`.
- **`torchdata.nodes` stages:** `ParallelMapper(source, map_fn, num_workers, method="thread"|"process", in_order=True)`, `Prefetcher(source, prefetch_factor)`, `PinMemory`, `Loader` (has `state_dict`). DataPipes and DataLoader2 were removed in 0.10.
- **example:** `tn.ParallelMapper(node, map_fn=map_and_collate, num_workers=num_workers, method="process", in_order=True)`, then `tn.Prefetcher(node, prefetch_factor=num_workers * 2)`. The docs warn that threads can contend on the GIL without free-threaded Python.

**SPDL**
- **order:** `DistributedRandomSampler(n, rank=, world_size=)`, then `.shuffle(seed=epoch)`.
- **cache:** none.
- **split:** equal counts per rank by default (`ddp_drop_last_distributed_round=True`).
- **resume:** none found.
- **workers:** `pipe(op, concurrency=, output_order="completion")`, `add_sink(buffer_size)`, `build(num_threads=)`.
- **traps:** the default output order is completion order, i.e. what `in_order=False` gives you. Set `output_order` if order matters.
- **example:** `PipelineBuilder().add_source(src).pipe(download, concurrency=4).pipe(preprocess, concurrency=2).add_sink(3).build(num_threads=4)`.

**Mosaic Streaming**
- **order:** `shuffle=True` with `shuffle_algo` (`py1e` default; `py1br`, `py1s`, `py2s`, `naive`; `py1b` is deprecated), `shuffle_block_size` and `shuffle_seed`. The docs call `naive` "the most random shuffle possible, but … the least download-efficient".
- **cache:**
  - `local` is the shard cache directory: put it on `$SLURM_TMPDIR`.
  - `cache_limit=None` (the default) never evicts. A limit below 4× the largest shard raises an error, and too low a limit means shards "continually evicted and redownloaded".
  - `predownload` defaults to 8 × `batch_size`.
- **split:** ranks come from the environment, and partitions pad by repeating samples, so counts are equal.
- **resume:**
  - `StreamingDataLoader` (a DataLoader subclass) `state_dict()`.
  - Keep `num_canonical_nodes` the same across runs. On the first run, canonical and physical node counts must divide one another. On resume, the global batch must divide by the device count.
- **traps:**
  - A local or shared-filesystem `remote` is *copied* into `local`, which is staging by another name: size `$SLURM_TMPDIR` for it, or leave `remote` unset to read `local` in place.
  - With `cache_limit` below the data and a new order each epoch, expect most shards to be fetched again every epoch ([staging.md](staging.md) §4).
- **example:** `StreamingDataset(remote=..., local=..., cache_limit='10mb', predownload=8, ...)`; for resume, `num_canonical_nodes=4  # explicitly set to the same as Run 1`.

**LitData**
- **order:** `StreamingDataset(input_dir, shuffle=True, seed=42, item_shuffle_window=...)`.
  - Chunk order is shuffled.
  - Items inside a chunk are shuffled in windows (`"full"` gives a full permutation).
  - Per the source (0.2.76), an `input_dir` on NFS, Lustre or GPFS switches to a window shuffle: each worker reads whole chunks in sequence, and chunk order is shuffled within a small window.
  - The docs warn the chunked shuffle "is **not** a substitute for a fully shuffled file-level DataLoader when the source has strong structure" (the check is in [staging.md](staging.md) §3).
- **cache:** `cache_dir`, `max_cache_size`, `max_pre_download`. By default the cache may take 75% of free disk, leaving at least 50 GB.
- **split:** automatic. `drop_last` (True under DDP) gives every process the same amount.
- **resume:** `StreamingDataLoader` (a DataLoader subclass) `state_dict()`.
- **traps:**
  - On a DRAC node whose local disk is shared with other jobs, set `max_cache_size` explicitly.
  - Local paths are mmapped in place. A `local:` prefix copies chunks into the cache instead.
- **example:** `StreamingDataset("s3://my-bucket/train", shuffle=True, drop_last=True, seed=42, item_shuffle_window=256)`; `StreamingDataset(..., max_cache_size="10GB", max_pre_download=4)`.

**WebDataset**
- **order:** `WebDataset(urls, shardshuffle=100)`, plus `.shuffle(bufsize, initial)` (1000 and 100 by default). An unset `shardshuffle` warns and does *no* shard shuffle. The docs: "A larger buffer size … provides better randomness but requires more memory".
- **cache:** `cache_dir` / `cache_size` (or `WDS_CACHE`, `WDS_CACHE_SIZE`). The oldest-accessed file is evicted first. Local files are never cached.
- **split:**
  - The default `nodesplitter` refuses multi-node runs. Pass `nodesplitter=wds.split_by_node`, or `resampled=True`, which the FAQ recommends for DDP.
  - For equal steps, `.with_epoch(n)` repeats, then truncates. `.with_length()` changes no sample counts.
- **resume:** none.
- **example:** `wds.WebDataset(shard_urls, resampled=True, cache_dir=..., nodesplitter=wds.split_by_node)`; `loader.unbatched().shuffle(2000).batched(20).with_epoch(200)`.

**Megatron-Energon**
- **order:** `get_train_dataset(path, batch_size=, shuffle_buffer_size=, max_samples_per_sequence=, worker_config=)`. `max_samples_per_sequence` is the chunk length: how many samples are read in sequence from one shard. The docs say "a high value reduces the shuffling randomness".
- **cache:** `cache_pool=FileStoreCachePool(parent_cache_dir=..., max_cache_size_gbytes=...)`. Point it at local disk.
- **split:** `WorkerConfig(rank, world_size, num_workers, data_parallel_group)`. Train datasets repeat forever by default, so the training loop sets the step count.
- **resume:** `get_savable_loader(...)` returns a `SavableDataLoader` (a DataLoader subclass) with `save_state_rank()` / `restore_state_rank()` and `save_state_global()` / `restore_state_global()`.
- **example:** `get_train_dataset('/my/dataset/path', batch_size=1, shuffle_buffer_size=100, max_samples_per_sequence=100, worker_config=WorkerConfig.default_worker_config())`.

**HF `datasets`**
- **order:**
  - Map-style `Dataset.shuffle(seed)` is exact, through an indices mapping. Reads stop being contiguous. `flatten_indices()` rewrites the data contiguously, which matches one epoch only ([storage.md](storage.md) §4).
  - `IterableDataset.shuffle(seed, buffer_size=1000)` shuffles shard order and adds a buffer. In 5.x the buffer is fed from up to `max_buffer_input_shards` (10) interleaved shards.
  - `to_iterable_dataset(num_shards=N)` turns an Arrow dataset into N contiguous shards, which with `.shuffle(...)` is the chunk shuffle of [staging.md](staging.md) §3.
  - `set_epoch(epoch)` makes the seed `seed + epoch`.
- **cache:** streaming has no shard cache.
- **split:**
  - Each DataLoader worker gets a subset of the shards, so give each rank at least `num_workers` shards.
  - `split_dataset_by_node(ds, rank=, world_size=)`. Map-style: a contiguous chunk per rank. Iterable: whole shards if `num_shards % world_size == 0`; otherwise every rank reads everything and keeps every world_size-th example, which has no locality.
  - 5.x adds `strategy="auto"|"shards"|"examples"`.
  - Use a fixed `seed` when shuffling in a distributed run.
  - Equal step counts are not documented: enforce them yourself.
- **resume:** `IterableDataset.state_dict()` with `StatefulDataLoader`. It restarts at the start of the current shard and skips forward. Shuffle-buffer contents are lost and refilled.
- **example:** `ds.shuffle(seed=42, buffer_size=10_000)`; `ds.to_iterable_dataset(num_shards=1024)`; `split_dataset_by_node(ds, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))`.

**FFCV**
- **order:** `Loader(fname, batch_size, order=OrderOption.SEQUENTIAL|RANDOM|QUASI_RANDOM, os_cache=True, distributed=False, batches_ahead=3)`. For data bigger than RAM, the docs recommend `QUASI_RANDOM` with `os_cache=False`, "memory-efficient but not truly random".
- **split:** `distributed=True`, but not with `QUASI_RANDOM` (raises `NotImplementedError`).
- **resume:** none.
- **traps:** last release in 2023; treat it as unmaintained.

**NVIDIA DALI**
- **order:** `fn.readers.file(..., random_shuffle=True, initial_fill=1024)` shuffles "only within the prefetch buffer": "samples that are far apart in the dataset rarely end up in the same batch". `shuffle_after_epoch=True` shuffles globally and can't be combined with `random_shuffle` or `stick_to_shard`.
- **cache:** decoder cache (`fn.decoders.image(cache_size=...)`, mixed backend). `stick_to_shard=True` makes it much smaller.
- **split:** `shard_id`, `num_shards`, `pad_last_batch`; on the iterator, `reader_name` and `last_batch_policy`.
- **resume:** `Pipeline(enable_checkpointing=True)`, `pipe.checkpoint()`, `iterator.checkpoints()`.
- **workers:** `num_threads`, `prefetch_queue_depth`, `py_num_workers`.
- **traps:** decoding runs on the GPU, so the CPU-worker triage of SKILL.md §1-3 doesn't carry over directly. Profile the pipeline's GPU time.

**Ray Data**
- **order:** `read_*(shuffle="files")`, `ds.randomize_block_order(seed=)`, or a global `ds.random_shuffle(seed=)`. The docs: "the more global the shuffle is, the more expensive the shuffling operation".
- **split:** `ds.streaming_split(n, equal=True)`: exactly equal rows per iterator, dropping data if needed.
- **workers:** `iter_torch_batches(batch_size=, prefetch_batches=1, local_shuffle_buffer_size=, pin_memory=False)`. `pin_memory` is alpha.
- **resume:** `DataIterator.state_dict()` is alpha and has no public load. Plan for no resume.

**Lance**
- **order:** `LanceDataset(ds, batch_size, columns=[...], batch_readahead=16, sampler=...)`, an `IterableDataset`.
  - `ShardedFragmentSampler(rank, world_size, randomize=False, seed=0)` gives each rank whole fragments: chunk-local reads.
  - `ShardedBatchSampler` interleaves batches instead.
- **cache:** `cache=True` caches on disk with no size limit, which is risky on a small local disk.
- **traps:** Lance is not fork-safe. Use `SafeLanceDataset` with `get_safe_loader`.
- **example:** `LanceDataset("train.lance", columns=["image", "prompt"], batch_size=128, batch_readahead=8, sampler=ShardedFragmentSampler(rank=1, world_size=8))`.

## 3. On Mila and DRAC

- **DRAC compute nodes have no internet.** Point `remote` / `input_dir` / shard URLs at a shared-filesystem path, and check how the framework handles one:
  - Mosaic copies it into `local`;
  - LitData mmaps it in place;
  - WebDataset doesn't cache local files.
- **Put every framework cache on `$SLURM_TMPDIR`,** not on the `$HOME` or scratch defaults, and set a size limit. Local disk is shared on DRAC. On Trillium it is a RAM disk counted against the job's memory ([network-fs.md](network-fs.md) §2).
- **Convert formats in a CPU job** with a few writers and local temporary files ([network-fs.md](network-fs.md) §4).

## 4. When a framework needs its own skill

Split one out only when its bottleneck has a different shape from the DataLoader's
(DALI is the likely case) *and* there are measurements for it. A description that
is too full to trigger this skill on framework questions is also a reason.
