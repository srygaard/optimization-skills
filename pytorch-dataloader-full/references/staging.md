# Read order, staging and data bigger than the node

Where each read is served from (page cache, local disk, or the network) decides
most of a loader's I/O cost. This file covers moving the data closer (staging),
changing the read order so network reads are sequential (chunk shuffling), and
what to do across epochs when the data fits neither.

## 1. Where will each read be served from?

Take the first row that fits:

| the bytes training reads fit in | do |
|---|---|
| the job's page cache: memory limit − anon, per node, shared by its ranks | read in place; later epochs are warm on this node |
| node-local disk | stage once per node at job start (§2) |
| the sum of the local disks across the job's nodes | partition the data across nodes (§4) |
| none of these | chunked read order streamed every epoch (§3), plus a static partial cache (§4) |

Measure the inputs, don't assume them:
- **Bytes actually read:** the columns training uses, not the dataset's size on disk.
- **The memory limit and anon peak:** from the cgroup ([storage.md](storage.md) §6).
- **Free local disk:** `df -h $SLURM_TMPDIR` at job start.
- **One node's sequential read rate from the network:** time a copy of a shard this node has never read.

Which filesystem you're on and how it behaves: [network-fs.md](network-fs.md).

## 2. Pre-staging to node-local disk

- **Price it first.** Stage time ≈ bytes ÷ the node's sequential rate, paid at the start of every job while the GPUs idle. Weigh that against what the network reads cost the run.
  - Measured, one workload, short benchmark runs: skipping staging gave mean wait 12-76 ms against 1-1.5 ms staged. A small dataset's copy probably finished early in the run.
  - The same workload with no staging but `prefetch_factor` 4-8 got ~1-2 ms. A deeper queue is the cheaper first try when the read rate is modest (SKILL.md §3).
- **Copy once per node, never once per rank.** Use `srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES <copy>` before the training launch, or local rank 0 copies while the other ranks wait at a barrier.
- **Copy a few large files.**
  - Archives or shards (tar, LMDB, Arrow, HDF5) stream sequentially. Extract them on local disk, where metadata is cheap.
  - Copying many small files from network storage pays a metadata round trip per file.
  - **For a read-only set of many small files,** a filesystem image (squashfs) avoids extraction. Mount it on each node, e.g. Apptainer `--bind data.sqsh:/data:image-src=/`, or `squashfuse`. Per-file metadata then lives inside one file. Not measured here.
- **Use a few parallel streams, one file each, not dozens.** One node's mount caps a single stream, and past a knee more streams contend ([network-fs.md](network-fs.md) §3). Find the knee with 1, 2, 4 and 8 streams on one node.
- **Stage only what you read:** the projected columns, the tokenized form, the splits training reads. Stage the val split too: it is usually small and every eval pass rereads it. Every byte staged is copy time and local disk.
- **Eval needs no shuffle.** Read it in on-disk order: sequential I/O, with readahead working for you.
- **Know where it lands.**
  - Mila: request local disk with `--tmp`.
  - DRAC: local disk is shared with other jobs on the node unless you hold the whole node.
  - tmpfs and RAM disks (DRAC `/tmp`, Trillium `$SLURM_TMPDIR`) count against the job's memory.
  - Check free space before copying and fail fast, rather than running out halfway through.
- **Containers:** Apptainer `--cleanenv` drops `SLURM_TMPDIR`. Pass the path in another variable (e.g. `TMPDIR`) and bind-mount it.
- **Make staging atomic and checkable.**
  - Copy to a temporary name, rename it into place, then write a marker with the source fingerprint (sizes, row counts). The loader checks the marker, and falls back to the network path if it is missing.
  - To run without staging, keep the expected local layout and fill it with symlinks to the network copies. Pointing the root at the network tree instead broke every path when its layout differed.
- **Overlapping the copy with training helps only if copy order = read order.**
  - A background copy in a fixed order can't stay ahead of a uniformly random sampler.
  - It also re-transfers whole shards that random reads had already pulled in part.
  - Pair a background copy with a read order that reads the staged shards first (§3), and throttle it: it shares the node's network path with the loader.
- **Multi-node:** N nodes staging the same data put N× the load on shared storage at once. Alternatives: partition the data across nodes (§4), stagger the starts, or use Slurm `sbcast` for one archive (not measured here). Local disk doesn't survive the job, so chained jobs pay again.
- **Judge staging by the fraction of reads served locally** over the run length you care about, not by whether the copy eventually finishes.

## 3. Chunk-shuffled read order, one epoch

A global random sampler over shards has no locality: per-batch shard histograms were
uniform from step 1. On network storage that means one round trip per row, and
read amplification under mmap (SKILL.md §4).

Chunk shuffling keeps reads sequential and puts the randomness in two places: the
order of chunks, and the mixing of rows inside a window of chunks. This assumes the
on-disk order is not sorted (by length, label, source or time); the check is below.

**The plan.** Build it once per epoch; it is a pure function of (seed, epoch, rank):
1. Cut the on-disk row order into contiguous chunks. Align them to files or shards, and to the format's natural block inside a file (an Arrow record batch, an LMDB key range).
2. Permute the chunks with an RNG seeded by (seed, epoch).
3. Deal chunks to ranks so each rank's set is disjoint. Give every rank the same number of steps (drop or pad the remainder).
4. Walk each rank's chunks in windows of k. Permute the rows inside a window and emit its batches, then move to the next window.
5. Store the result as an index array behind a map-style `Sampler` or batch sampler. Resuming is then a skip to step s, and the order doesn't depend on worker timing.

**I/O**
- **Each chunk is read once, contiguously, within one window.** Page cache is per node, so the locality has to hold across the node's window, not per worker. Workers can split a chunk's rows between them and still hit pages the first reader pulled in.
- **Leave readahead on for these files.** `MADV_RANDOM` (SKILL.md §4) is for random order; under chunk order readahead is the prefetch. For LMDB, open with `readahead=True`.
- **On NFS, check the mount's read-ahead window first** ([network-fs.md](network-fs.md) §1). If it is small (128 KB is a common default), read each chunk with large reads, or hint its range with `WILLNEED`, rather than trusting the kernel's prefetch.
- **Optionally prefetch the next window** with `madvise(MADV_WILLNEED)` or `posix_fadvise(POSIX_FADV_WILLNEED)` on its byte range, or with a background read. Not measured here.
- **Measured in build jobs, not in a training loader** (cold LMDB over NFS):
  - one sequential reader per shard: 319 MB/s with 8 readers, against 13.7 MB/s for 8 random readers;
  - a cursor scan in key order: 232 MB/s from one reader, against 1.5 MB/s random.

  A cached dense file read with 8 readers gave 3.7M rows/s in read order, 2.1M in random order.

**Randomness**
- **A batch draws from at most k chunks,** and neighbouring batches come from the same window. Each epoch has a new chunk order and new windows.
- **This approaches a full shuffle only if each chunk looks like the whole dataset.** If chunks are homogeneous (sorted or clustered data), batches are too. That is the block-shuffle condition analysed by CorgiPile (block shuffle plus a buffer shuffle; convergence degrades as blocks get more homogeneous).

**Is the on-disk order unsorted enough?** Measure, don't assume:
- For each covariate that matters (length, label, source, time, cluster), compute per-chunk means, or per-chunk class shares.
- Compute the same for chunks of the same size cut from a randomly permuted index.
- Compare the spread across chunks (std of the means):
  - **ratio ≈ 1:** chunk order is fine.
  - **ratio ≫ 1:** the data is clustered. Raise k, or rewrite the data once in shuffled order. The rewrite is one sequential pass, and chunk shuffling on the copy then mixes well.
- When it has to be certain, compare loss curves of short runs: chunk shuffle against a global shuffle, same seeds.

**Sizing**
- **Chunk:** large enough that reading it is bandwidth-bound, i.e. at least the stripe or chunk size and the readahead window (MBs, not KBs). Measure throughput against chunk size on a file this node has never read.
- **Window:** k × chunk bytes must fit in RAM if rows are decoded, or in the job's page-cache share if they're mmapped. It should hold many batches.
- **Trade-off:**
  - bigger chunks give better I/O and less mixing at the same k;
  - bigger k gives more mixing, more memory, and a longer fill at every epoch start.

**Pitfalls**
- **Fill delay and narrow mixing at every epoch start and every resume.** The first batches come from the first k chunks only. Prefetch the first window before step 1.
- **Length bucketing goes inside the window.** Bucket the window's rows, emit full batches, and carry leftovers into the next window (or drop them at epoch end). One project's length-grouping sampler was incompatible with its shard-sequential staging and fell back to uniform random reads; bucketing per window keeps both.
- **DDP:** disjoint chunks per rank, and equal step counts per rank, especially with token-budget batches, whose count varies ([batching.md](batching.md) §2).
- **The `IterableDataset` variant** must split chunks by rank *and* by `get_worker_info()`, or rows repeat. Its batch order interleaves workers, which makes resuming harder. Prefer the precomputed map-style plan.

Libraries that implement this, with their knobs: [frameworks.md](frameworks.md) §2.

## 4. Many epochs, data bigger than local storage

- **If each epoch samples a subset,** a dense per-epoch file of that epoch's rows may fit where the full set doesn't ([storage.md](storage.md) §4).
- **Partition across nodes** when the data fits the sum of the local disks. Node n stages part Pₙ once, and its ranks reshuffle within Pₙ every epoch.
  - Under DDP the global batch still draws from every partition at every step. What stays fixed is which rows share a rank, which matters for per-rank state (BatchNorm statistics, per-rank bucketing).
  - Cut partitions at random, not as sorted ranges. Balance them in bytes and in steps.
  - Changing the node count on resume, or replacing a node, means re-partitioning and re-staging. Key the partition by node count.
- **Stream every epoch in chunk order** when nothing fits (§3).
  - Required rate per node = bytes per epoch ÷ epoch time ÷ nodes. Compare it with the measured sequential rate.
  - Chunk order makes the reads bandwidth-bound instead of request-bound.
  - Stage window w+1 to local disk or RAM while training on window w, then delete w.
- **Cache a fixed subset, not the most recent shards.** With a new permutation each epoch, an LRU cache holding a fraction f of the data only hits rows read again within f·N reads of their last use: roughly f²/2 of all reads. A fixed subset staged at job start and kept for the run serves a fraction f. (Reasoning, not measured. It fits the measured result that 64% of fetched bytes were re-fetches once the data was ~2× the page cache.) Implement it as "read the shard from local disk if staged, else from the network", in chunk order.
- **Shrink the bytes:** projected columns, pre-tokenized data, smaller dtypes. Compress on local disk only if workers have spare CPU to decompress; check with `loader_bench.py isolate`.
- **Resume:** keep the plan a pure function of (seed, epoch, rank) and checkpoint the step. Don't checkpoint buffer contents.
