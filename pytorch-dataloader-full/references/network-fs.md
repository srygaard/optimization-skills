# Network filesystems

Which filesystem a path lives on, how each kind behaves under a data loader, and
the mistakes that make network storage slow. For mmap read amplification and
`madvise`, see [storage.md](storage.md). For read order and staging, see
[staging.md](staging.md).

## 1. Find out what you're on

```bash
findmnt -T /path/to/data -o TARGET,FSTYPE,OPTIONS
```

- **`FSTYPE`:**
  - network: `nfs`/`nfs4` (VAST, PowerScale, other NAS), `lustre`, `beegfs`, `gpfs`;
  - local: `xfs`/`ext4`;
  - RAM: `tmpfs`.
- **NFS `OPTIONS` worth reading:**
  - `vers=` (3 or 4.x);
  - `nconnect=` (TCP connections per mount; absent means one);
  - `rsize=`/`wsize=`;
  - `actimeo=`/`ac*` (attribute cache);
  - `local_lock=`.
- **Layout:** `lfs getstripe <file>` (Lustre); `beegfs-ctl --getentryinfo <file>` (BeeGFS; some sites need `--cfgFile=<client conf>`).
- **Client read-ahead for an NFS mount** (readable without root): `cat /sys/class/bdi/$(mountpoint -d /mnt/point)/read_ahead_kb`.
- **Local space:** `df -h $SLURM_TMPDIR`. Check it at job start, not from the docs: on DRAC the disk is shared with other jobs on the node.

## 2. Mila and DRAC clusters

As of 2026-11. Check the current pages before relying on a row: [Mila storage](https://docs.mila.quebec/technical_reference/clusters/mila/storage/), Alliance [storage](https://docs.alliancecan.ca/wiki/Storage_and_file_management) and [node-local storage](https://docs.alliancecan.ca/wiki/Using_node-local_storage). Confirm with `findmnt`.

| cluster | network storage | node-local `$SLURM_TMPDIR` | notes |
|---|---|---|---|
| Mila | `/network/scratch`: BeeGFS, `buffered` client mode | local disk, TB-scale; request it with `--tmp` | the docs don't name the scratch filesystem; BeeGFS was seen with `beegfs-ctl` on compute nodes (2026) |
| Narval, Rorqual, Fir | Lustre | local SSD / NVMe | |
| Nibi | VAST (all-flash) over NFS | local NVMe | NFS version not documented |
| Trillium | VAST | **RAM disk**, counted against the job's memory | `home` and `project` are read-only from jobs |
| Killarney | VAST over **NFSv3** | small: ~350 GB (L40S nodes), ~800 GB (H100 nodes) | |
| TamIA | Lustre | local SSD | |
| Vulcan | Dell PowerScale | not documented | treat as NFS until `findmnt` says otherwise |

Béluga, Cedar, Graham and Niagara are retired (2025-26). Scratch has per-user file-count
quotas: 1M on most DRAC clusters, 10M on Mila. A dataset stored as one file per sample
runs into them, and into the metadata costs below.

## 3. How each kind behaves

**All network filesystems**
- **Page cache is per client node.** Rows another node had just read were as slow as fresh reads (BeeGFS, measured).
- **Every cold read is a network round trip,** and `open`, `stat` and `readdir` go to a metadata server shared by every user. With small files, per-file costs dominate.
- **Bandwidth is shared with every other job.** Cold throughput varied ~2× between runs on one node with the same config. Compare arms within one job or back to back ([measurement.md](measurement.md)).

**BeeGFS (Mila scratch, `buffered` mode)**
- **`read()` doesn't use the Linux page cache.** It is served from BeeGFS's own client buffers: a 64 MiB `pread` left 0 bytes resident. Repeated `read()`s of the same data get no reuse.
- **mmap does use the page cache.** Mixing `read()` and mmap on one hot file dropped cached pages ([storage.md](storage.md) §3).
- **The default scratch layout is 1 storage target with 64 KiB chunks,** so one server serves each file. Striping is in [storage.md](storage.md) §4.
- **Opening a large Arrow file reads its metadata:** 174 MiB for a 30 GiB shard, before any row.

**Lustre (Narval, Rorqual, Fir, TamIA)**

Mechanisms below follow the Alliance [Tuning Lustre](https://docs.alliancecan.ca/wiki/Tuning_Lustre) page and INCD's [Lustre best practices](https://wiki.incd.pt/books/filesystem-user-guide/page/lustre-best-practices).
- **Metadata lives on the MDS, data on the OSTs, and every user shares the MDS.**
  - Many small files cost an MDS round trip each, and so do `stat` loops, `ls -l`, `ls --color`, `du`, `find` and shell globs over a big tree.
  - Use plain `ls`, `lfs find` and `lfs df`, or better, a manifest built once.
- **One huge directory is a lock hotspot.** Opening files in it contends on the parent directory's lock. Spread files over subdirectories (a common rule: about √N directories for N files) or pack them into shards.
- **Many processes opening one file at the same moment contend too.** All ranks × workers opening every shard at startup is exactly this.
  - Open read-only (for LMDB, `readonly=True, lock=False`).
  - Let one process `stat` or scan the files and broadcast the result (SKILL.md §6).
- **Many writers appending to one file serialize on its lock.** Give each file one writer: rank 0, or one file per rank.
- **Striping is set per file when it is created.**
  - `lfs setstripe` on a directory affects new files only: copy the data in, or `lfs migrate`.
  - Match the stripe count to how many processes read the file at once. A large file read at random by every worker benefits from wide striping. A shard read by one worker at a time (a chunked read order, [staging.md](staging.md) §3) needs stripe count 1: parallelism then comes from different shards on different OSTs.
  - `lfs df` shows how many OSTs there are; striping can't help with only one.

**VAST and other NFS (Nibi, Trillium, Killarney, Vulcan)**
- **Close-to-open consistency:** every `open()` revalidates attributes with the server, so opening a file per sample costs a round trip each time. NFSv3 is stateless and has no delegations to skip it.
- **`fsync` / commit is a synchronous round trip.** Many writers that each sync per commit serialize on the server's commit path. Measured, LMDB with default sync and 16 writers: zero bytes written over 90 s, every writer in `D` state.
- **Read-ahead is set per mount by the admins, and it may be small.** The window lives in `/sys/class/bdi/<dev>/read_ahead_kb` (§1).
  - It applies to `read()` and to mmap faults on that mount, unless `MADV_RANDOM` is set.
  - Newer kernels default it to 128 KB. VAST's [read-ahead tuning guide](https://docs.vastdata.com/docs/vast-quick-nfs-read-ahead-tuning) calls that too small for sequential reads and suggests starting at 4 MB.
  - Changing it needs root. If it is small, don't rely on the kernel for sequential prefetch:
    - read each chunk with large reads of several MB (`readinto` into a reused buffer);
    - or hint the chunk's range with `posix_fadvise(POSIX_FADV_WILLNEED)` / `madvise(MADV_WILLNEED)`;
    - and run several streams.
  - A large window hurts random small reads. That is the over-fetch [storage.md](storage.md) measures as bytes fetched ≫ bytes used.
- **One mount's TCP connections cap a node's throughput.** Use several sequential streams, not one. Measured, cold LMDB reads over NFS, 1 → 8 readers: sequential, one reader per shard, 92 → 319 MB/s; random 1.5 → 13.7 MB/s.
- **Past the knee, more streams cost more than they add.** A store build with 16 writers took 2.0× as long as with 4.
- **All-flash backends make random reads cheaper, not free.** Each read is still a network round trip, so fewer, larger reads still win.

**Node-local disk (`$SLURM_TMPDIR`)**
- Local latency and a local page cache.
- Shared with other jobs on the node unless you have the whole node.
- Wiped at job end, so each job stages again.

**tmpfs / RAM disk** (DRAC `/tmp`, Trillium `$SLURM_TMPDIR`)
- Files are memory: they count against the job's limit together with anon memory and the page cache.
- Staging 100 GiB there leaves 100 GiB less for everything else.

## 4. Mistakes on network storage

| mistake | what you see | fix |
|---|---|---|
| one file per sample | slow `open`/`stat` per sample, file-quota errors, slow `ls` | pack into shards (tar, LMDB, Arrow, HDF5), or stage and extract locally |
| `open()` in every `__getitem__` | a metadata round trip per sample; worse on NFS (revalidation on open) | open once per worker, lazily on first use; never open LMDB/HDF5 handles before workers fork |
| `glob` / `listdir` / `exists` / `getsize` over the dataset, on every rank | minutes before the first step; metadata server load | list once offline, ship a manifest (paths, sizes, row counts); on Lustre, `lfs find` instead of `find` |
| millions of files in one directory | slow opens and listings; directory lock contention on Lustre | subdirectories (about √N), or pack into shards |
| uniform random reads of mmapped files | I/O-bound loader; ~100-260 KiB fetched to serve a ~140 B row | `MADV_RANDOM` (SKILL.md §4) or a chunked read order ([staging.md](staging.md) §3) |
| many small `read()`s on BeeGFS `buffered` | every read goes to storage, nothing cached (64 MiB `pread` → 0 resident) | mmap, or fewer, larger reads |
| every rank opening and scanning the same files at startup | 386-665 s per rank opening raw files to reach a cache | prepare once per node, broadcast a manifest (SKILL.md §6) |
| every rank copying the dataset to local disk | N× the network traffic; ranks contend | one copy per node ([staging.md](staging.md) §2) |
| loading a whole file on every rank (`torch.load`, `np.load`) | host memory × ranks; a full read per rank | `torch.load(..., mmap=True)`, `np.load(..., mmap_mode="r")` |
| writing a store on NFS with per-commit sync, `writemap=True` under a memory limit, or many writers | full stall with writers in `D` state; ~0.4 of 16 cores busy in reclaim; 16 writers 2.0× slower than 4 | LMDB `sync=False, metasync=False` with one `env.sync(True)` before close (plain `env.sync()` is skipped under `sync=False`); `writemap=False`; ~4 writers; temporary fragments on local disk |
| the Python environment (venv, conda, site-packages) on network storage | slow imports in every rank, and again in every spawned worker; metadata server load | a container image (one file), or build or copy the environment into `$SLURM_TMPDIR` at job start |
| every rank appending to one log or metrics file | logging steps stall on the file's lock | one writer (rank 0) or one file per rank; open once, write buffered |
| temp or cache directories left on network storage | `APPTAINER_TMPDIR` on scratch made an image conversion I/O-bound until it hit the time limit | point `TMPDIR`, `APPTAINER_TMPDIR` and framework caches at `$SLURM_TMPDIR` |
| reading progress from a log on NFS | log lags; a line written 4 min earlier wasn't there yet | check on the node: `du` on the output, `ps` for worker states |
| assuming another node's reads warmed your cache, or co-locating your own jobs | "warm" runs no faster; spikes when two of your jobs share a node | [measurement.md](measurement.md) §3 |
