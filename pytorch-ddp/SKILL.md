---
name: pytorch-ddp
description: >-
  Tune PyTorch data-parallel training (DistributedDataParallel, and FSDP /
  HSDP), single- or multi-node, with two calibration jobs and a set of
  formulas instead of a knob sweep: decide whether communication is the
  bottleneck at all, then pick the strategy, visible GPUs/NICs, gradient
  accumulation and micro-batch size, bucket_cap_mb or FSDP unit size and
  resharding, comm hooks or reduce dtype (bf16/fp16/noop/PowerSGD),
  gradient_as_bucket_view, batched_grad_copy, static_graph and NCCL
  settings from measurements. Use when multi-GPU or multi-node scaling
  efficiency is below what you expect, when choosing DDP or FSDP knobs on
  a new project or cluster, when gradient accumulation looks like it
  all-reduces every micro-batch (no_sync not working), when
  torch.compile + DDP/FSDP misbehaves, when NCCL bandwidth, NVLink or
  GPUDirect RDMA is suspect, when one slow rank stalls the job, or when
  deciding if a model has outgrown DDP (FSDP/HSDP/ZeRO).
metadata:
  version: 3.3.0
---

# Tuning data-parallel training (DDP, FSDP, HSDP)

**Core principle: predict first, then measure only what the prediction says
can matter.** Two calibration jobs supply every input the formulas need.
The formulas say which change can buy more than ~1% of the step, and one
validation job tests only those changes.

The order of payoff is the same on every cluster and for every strategy:

1. Gradient sync actually skipped on accumulation micro-batches (DDP
   `no_sync`, FSDP `set_requires_gradient_sync(False)`). A silent break
   multiplies the per-sync communication by `ga`.
2. The fabric doing what it should: NVLink up on every visible GPU *in
   every communicator*, RDMA / GPUDirect on, every fast NIC carrying
   traffic.
3. The ranks: one slow GPU or one stalled host is paid by *every* rank at
   the sync.
4. The compute: a sharded strategy can change the kernels themselves
   (F11's `c_mb`), and that can cost more than all of its communication.
5. Only then the knobs. On a healthy fabric, where transfer hides behind
   compute, all of them together are usually worth 0-2%. On a slow
   fabric, where comm is exposed, the knobs that cut bytes or syncs per
   token (`ga` or `b`, lower precision on the wire) recover a large share
   of it. Granularity (buckets, units) and NCCL env stay near ties in both
   cases.

Most tuning that does nothing was spent on item 5 while items 1-4 were
broken, or while communication was already under 1% of the step.

Files:

* `formulas.md`: the formulas for any data-parallel strategy, with
  inputs, the decision each drives, and caveats. A strategy enters
  through its collective inventory.
* `ddp.md`: DDP's collectives, its correctness gate (`no_sync` and the
  compile traps), F3 bucket size, F14 tail, F7 comm hook, its knobs, and
  when to leave it.
* `fsdp.md`: FSDP2 and HSDP: choosing the strategy, the collectives, the
  correctness gate, what sharding does to the compute, unit size,
  resharding, mixed precision, the working set, the knobs.
* `scripts/collective_probe.py` (under `torchrun`): Job B's probe. A size
  sweep of all-reduce, all-gather and reduce-scatter on the world, node
  or replica group, with the `t_o`/`w` fit. Standalone, so a project
  needs no probe code: run it as one more step of Job B's allocation, with
  the training arms' NCCL environment and visible GPUs.
* `scripts/nccl_topo.py <nccl log dir>`: the NVLink topology NCCL detected
  for every communicator; flags GPUs with NVLink down and communicators
  that saw less NVLink than the same GPUs elsewhere.
* `scripts/ddp_trace.py rank*.json`: any `torch.profiler` trace, one per
  rank, from a loop that labels `forward` and `backward` per micro-batch
  with `record_function`. Per micro-batch: collectives by type, group and
  phase, transfer (min across ranks), exposed comm, forward and backward
  compute; for DDP buckets, tail and overlap slowdown.

`sync_cost`, `c_mb`, imbalance and stalls come from per-rank step timings
("Reading the numbers"), in whatever form the project's trainer logs them.

## The process

| step | GPU time | produces |
|---|---|---|
| 0. Preflight | none | strategy, correctness guard, visible GPUs, node placement, memory fit, first guess |
| 1. Job A: 1-GPU ceiling | 1-GPU arms inside Job B's allocation | throughput ceiling, compute and memory constants |
| 2. Job B: target shape | >= 2 nodes | every comm, rank and stall input |
| 3. Predict | none | the step ledger (F11) and the decision sheet |
| 4. Validate | one job, only if step 3 finds a change worth >= 1% | the changes that ship |
| 5. Production node count | only if needed | the scale-out check |

### Step 0. Preflight (no GPU time)

1. **Strategy** by memory (F12; `fsdp.md`, "Choosing the strategy"): DDP
   whenever it fits, since it communicates least and leaves the compute
   alone; else HSDP, else FSDP.
2. **Correctness guard in code**, from the strategy file: for DDP,
   compile in place on the DDP object (`ddp_model.compile()`, not
   `torch.compile(ddp_model)`), so no helper can strip DDP off and break
   `no_sync` (`ddp.md`). If the code keeps `torch.compile(DDP(model))`,
   assert at train start and after every checkpoint save that `no_sync`
   still resolves to `DistributedDataParallel.no_sync`. For FSDP, build the
   optimizer after sharding and clip with the sharded API (`fsdp.md`).
   With process groups beyond the world one (HSDP, tensor parallel),
   create their communicators at startup, before any in-process NVML
   sampler starts (gate 2).
3. **Visible GPUs.** Map GPU <-> NIC <-> CPU socket (`nvidia-smi topo -m`,
   `ibstat`, `lspci -tv`, or NCCL's topology dump). If a job uses fewer
   GPUs than a node has, choose them so every fast NIC has a visible GPU
   on its PCIe switch, and balance sockets where possible (F9). Note any
   NICs that must not carry NCCL traffic (management, storage, slow
   Ethernet).
4. **Node placement.** Find each node's leaf switch (e.g. `scontrol show
   topology`). If the fabric may be oversubscribed, keep Job B on one
   leaf (Slurm: `--switches=1@<max wait>`), and record the leaves of every
   run. Keep a list of nodes with
   known-bad GPUs and exclude them.
5. **First guess.** F9 estimates each collective's transfer from the
   topology. Compared with the compute window it must hide behind (F15;
   from an earlier 1-GPU run, or `6 P` FLOPs per token at the expected
   MFU, forward ~1/3 of it), it gives the regime:
   * **Hidden** (transfer shorter than its window): expect a comm cost of
     a few ms plus the overlap slowdown. If F2 with `a = 1` is already under
     ~1%, Job B is a check, not a search.
   * **Exposed:** the comm share is roughly `(transfer - window) / step`.
     Plan `B_sync` and `b` (F2) and precision on the wire before anything
     else.

### Step 1. Job A: single-GPU ceiling

One GPU, no wrapper, production config, at two micro-batch sizes (`b` and
`b/2`, or `2b` if it fits); read only steady-state steps (after compile
and autotune warmup). Run it as 1-GPU arms of Job B's allocation: no
extra queue wait, and the ceiling comes from the same node as the
multi-GPU arms. Run it apart only if the scheduler cannot split an
allocation, and skip it if a 1-GPU benchmark of the same model on the
same hardware exists. If the model does not fit on one GPU, see `fsdp.md`
("Process changes").

| read | feeds |
|---|---|
| tok/s per GPU `R`, step time `T_1` | F2, F11 |
| accumulation micro-batch time, forward and backward, at both `b` | F11 (`d_load`), F13, F15 windows |
| peak allocated memory, at both `b` | F12 |

### Step 2. Job B: the target shape

Setup:

* >= 2 nodes, because the inter-node fabric is what the formulas
  describe.
* The visible GPUs from step 0.
* Production `ga`, which must be >= 2 to measure `sync_cost`.
* NCCL logging: `NCCL_DEBUG=INFO`,
  `NCCL_DEBUG_SUBSYS=INIT,ENV,NET,GRAPH,TUNING`, one file per process
  (`NCCL_DEBUG_FILE=<dir>/%h.%p.log`), and a topology dump
  (`NCCL_TOPO_DUMP_FILE`).
* A size sweep (1 MiB to 1 GiB) of each collective type the strategy
  uses, in the dtype it uses and on each communicator it uses: all-reduce
  of fp32 for DDP; all-gather and reduce-scatter for FSDP (HSDP: in the
  shard group too). `scripts/collective_probe.py` does this if the
  trainer has no probe of its own.
* A profiler window of a few optimizer steps on all ranks after warmup,
  stepped once per optimizer step.

Arms, run back to back in one allocation: the strategy file's arms
(`ddp.md`: `base`, optional `noop`; `fsdp.md`: `base`, a resharding
control). For a sharded strategy, add a DDP arm when DDP fits: it is the
reference that measures `c_mb`.

**Gates, in order.** If one fails, fix it and rerun Job B before reading
anything else:

1. The strategy file's **correctness gate** passes in the trace: the
   collective counts per micro-batch match (no gradient reduction on
   accumulation micro-batches when sync is skipped).
2. The NCCL log shows:
   * **NVLink between every pair of local GPUs, in every communicator**
     (`scripts/nccl_topo.py`). NCCL detects the topology when a
     communicator is created and keeps the paths for its whole life. A GPU
     whose NVLinks are down, or read as down at that moment, sends the
     communicator over PCIe (`PHB`/`SYS` paths, fewer channels, no NVLS):
     several times slower, no error. A GPU low in every communicator is a
     hardware fault (`nvidia-smi nvlink -s`): exclude its node. A
     communicator low where others on the same GPUs are fine was created
     while something in the same process polled NVML from a thread (a
     telemetry or GPU-stats sampler): NCCL's link-state reads then fail
     and count as links down. Sub-communicators (HSDP, tensor parallel)
     are created lazily, at their first collective, so they are the usual
     victims. Create every communicator at startup, before such threads
     start (one small collective on each process group);
   * the transport you expect: IB/RoCE with GPUDirect RDMA where the
     fabric has it. `NET/Socket` means TCP through the host, which is a
     fallback unless the cluster has no RDMA;
   * every intended NIC in the channel lines;
   * the collectives completing. A NIC that initialises can still have
     no path to the peer (look for retry or completion errors).
3. The probe's bandwidth matches earlier healthy runs of the same shape,
   or, without any, its large-message `w_k` is at least ~0.7x
   `B_node / c_k` (F9).
4. No rank or node stalls on most steps. If one does, its hosts are
   unhealthy; fix that first.

Then read the inputs:

| read | from | feeds |
|---|---|---|
| `t_o,k`, `w_k` per collective type (fit of each size sweep) | probe | F9, F0, DDP F3 |
| `t_k`, `w_eff,k`, call sizes, per collective and phase | trace (min across ranks) | F2, F9 check, F15 |
| exposed comm per collective, overlap slowdown, DDP tail | trace | F15, F6, DDP F7/F14 |
| forward and backward compute per micro-batch | trace | F15 windows, `c_mb` |
| `sync_cost` (gives `a`), loaded accumulation micro-batch (gives `d_load`, `c_mb`) | per-rank timings | F11, F2 |
| per-rank compute speed, slowest rank, imbalance wait | per-rank timings | F10 |
| stalled steps and their extra time | per-rank timings | F5 |
| algorithm, protocol and channels per message size | NCCL `TUNING` lines | fit range, F6 |

### Step 3. Predict (no GPU time)

First fill in the **step ledger (F11)**: the 1-GPU step, node load, the
strategy's per-micro-batch cost `c_mb`, `sync_cost`, imbalance and stalls.
Only `sync_cost` and the comm part of `c_mb` respond to comm knobs. If
they are under ~1% of the step, skip every comm-knob row below and work on
the largest other term. If `c_mb` is large but its comm is hidden, diff
the compute by kernel against the DDP or 1-GPU trace (F11).

Then find the **regime (F15)** of each collective: is its transfer shorter
than the compute window it must hide behind (hidden), or longer
(exposed)? The regime decides which rows below can pay.

Then go through the decision sheet: the rows below, plus the strategy
file's rows (`ddp.md`, `fsdp.md`). A default costs no arm.

| decision | formula | change when | default |
|---|---|---|---|
| strategy | F12, F11 | DDP does not fit, or a sharded strategy's `c_mb` exceeds its memory benefit | DDP if it fits |
| visible GPUs per node | F9 | a fast NIC has no visible GPU | the allocation's GPUs |
| drain a GPU or node | F10, gate 2 | one rank is > ~1% slower than the median on every step, or a GPU has no NVLink | keep |
| fix hosts before tuning | F5 | expected stall loss at the target `W` exceeds the comm share | - |
| tokens per sync, `B_sync = ga * b` | F2, F4 | per-sync comm share > target at the target `W` | production `ga` |
| micro-batch `b` | F13, F12, F2 (`C_mb`) | the fixed cost share is far from ~3%, per-micro-batch comm is exposed, or `b` exceeds `b_max` | production `b` |
| NCCL env (CTAs, algorithm, protocol) | F6, F0 | hidden regime, the overlap slowdown is large next to the comm cost, **and** the change does not stretch transfer | NCCL's choice |

In the exposed regime the order is: tokens per collective first (no
numerics change), then lower precision on the wire (check the loss curve).
Changing `B_sync` changes the global batch, and so the optimization: keep
the global batch fixed unless that trade-off is intended, and compare
throughput only at equal global batch.

Step 4 gets only the changes predicted to buy >= ~1% of the step, plus
free changes with a clear mechanism (visible GPUs, draining a slow GPU).
Smaller effects sit inside run-to-run noise and cannot be confirmed
anyway.

### Step 4. Validate (one job)

Run in one allocation: `base`, one arm per change from step 3, and one arm
with all of them. Judge each arm by its trace (transfer, exposed comm and
overlap slowdown; F0) and by `sync_cost` and `c_mb`. Throughput from a short arm is
dominated by whichever steps happened to stall. Accept a change when its
direction matches the prediction and its size is within ~25%. Otherwise
record the miss against the formula before shipping anything.

### Step 5. At the production node count

If production runs on more nodes than Job B, predict before running:

* the comm share: F4;
* stalls: F5;
* imbalance: F10, which grows with the expected maximum over `W` ranks;
* uplink limits: F9's blocking term, if the job spans leaves.

Run a job at the target size only if a prediction lands near a threshold,
or the fabric may be blocking.

## Reference

### Fabric checks

* **NVLink in every communicator** (gate 2, `scripts/nccl_topo.py`):
  GPU-to-GPU paths of type `NVL`, NVLS channels where the platform has
  NVLink SHARP, and the same channel count on every node.
* The transport line should show IB/RDMA, not sockets. A wrong
  `NCCL_IB_HCA` or `NCCL_SOCKET_IFNAME` falls back silently. `=` in
  `NCCL_IB_HCA` makes the match exact.
* GPUDirect RDMA should be enabled for every GPU/NIC pair in use. If it is
  disabled by distance, `NCCL_NET_GDR_LEVEL` can force it. A/B that
  change: across a CPU socket link it can be slower than staging.
* `ENV` lines confirm which `NCCL_*` overrides took effect.
* The `TUNING` and per-size algorithm lines show the algorithm, protocol
  and channel count NCCL actually chose. **Do not read them off the
  profiler's kernel name**, which reflects a default specialization (the
  dtype in the name is reliable).
* With a subset of a node's GPUs, check in the topology dump that every
  fast NIC still has a visible GPU on its PCIe switch.
* On an oversubscribed fat tree, traffic that crosses leaves is capped by
  the uplinks, not the NICs. If transfer could be uplink-bound, compare
  the same shape on one leaf and across leaves.

### Reading the numbers

* **The world_size=1 run is the ceiling.** Compare multi-GPU runs against
  it, never 1x2 against 2x1 (both carry a collective).
* **`sync_cost` is the cost of the sync:** `t_sync_mb - t_accum_mb` per
  rank (with backward timings: `backward_sync - backward_nosync/(ga-1)`),
  then the min across ranks per step, then the median over steps. One
  rank's reading includes waiting for its peers, and one stall wrecks a
  mean.
* **`c_mb` is the strategy's cost on every micro-batch:** the
  accumulation micro-batch time minus a same-allocation DDP arm's (or
  minus the 1-GPU one times `1 + d_load`).
* **NCCL kernel time is not cost.** A collective kernel spins while
  waiting for late peers and runs concurrently with compute.
* **Transfer**: per call, the minimum duration across ranks, summed. The
  spread across ranks (peer wait) is skew, not bandwidth.
* **Exposed comm**: collective time not covered by any compute kernel.
  In the hidden regime it overstates the cost (F15); `sync_cost` and
  `c_mb` are the cost. **Tail** (DDP): the end of the last bucket minus the end of the backward
  compute it waited on.
* **Overlap slowdown**: compute time added to the compute a collective
  overlaps; for DDP, the sync micro-batch's backward compute minus an
  accumulation micro-batch's. It is summed kernel time and mostly does not
  lengthen the step. It tells you which knobs *could* matter, not what
  they would save.
* **Compute diff**: kernel time per micro-batch by kernel name, against
  the DDP or 1-GPU trace. It separates what a strategy changed in the
  compute from what its collectives cost; for a compiled kernel that
  regressed, compare the generated code (`fsdp.md`).
* **Transfer is a goal of its own** (F0). Hidden transfer stays hidden
  only at the current `B_sync` and `b`.
* **Stragglers**: a step where one rank's host time jumps while every
  other rank's sync jumps by the same amount is one late rank paid by all.
  Count them, and exclude them from knob comparisons.
* Compare arms run back to back in one allocation, since node placement
  alone moves throughput by several %. Profile steady state only, and step
  the profiler once per optimizer step: per micro-batch, the window misses
  the sync or holds a partial accumulation cycle.
* **Step-timing hygiene.**
  * Drop steps whose host time is well below their device time: the host
    clock stopped before the GPU work. This typically hits the profiler's
    warm-up step and the last step, which read 5-25% short.
  * Drop the first steps after warmup on transports that ramp up (e.g.
    TCP sockets).
  * Read medians of the remaining steady steps. A single short artifact
    otherwise becomes the "fast" reference, and every normal step looks
    like a stall.

### NCCL environment

Only after the ledger says comm matters. A/B one setting at a time on the
same nodes, confirm it in the `ENV` and `TUNING` lines, and judge it with
F0. The usual pattern:

* Fewer CTAs or channels (`NCCL_MAX_CTAS`, CTA policies) lower the
  overlap slowdown but stretch transfer.
* NVLink-SHARP / NVLS algorithms can raise channel count, and with it the
  slowdown.
* Protocol and algorithm overrides rarely beat NCCL's own choice for
  large messages.

A setting that lowers `sync_cost` by stretching transfer only moves the
cost into the margin you will need when scaling out.

### Traps that cost a wave each

* **Name the bandwidth.** Algorithm bandwidth is bytes / time; bus
  bandwidth is that times `c_k` (F9), and it is what nccl-tests and many
  probes print. Mixing them is a ~2x error on all-reduce.
* **Costs from untraced steps, causes from the trace.** The profiler slows
  many small kernels more than a few large ones (FSDP's copies, foreach
  ops), so take `sync_cost` and `c_mb` from the wall clock of clean steps.
* **Attribute a kernel by the CPU op that launched it**, not by its name:
  `multi_tensor_apply` serves the optimizer, gradient clipping and FSDP's
  copies alike, and NCCL kernel names show a default specialization.
* **Dynamic shapes can silently end in eager mode.** When every new shape
  recompiles, the recompile limit is reached and the rest of the run is
  eager: grep the log for `recompile_limit` (or `TORCH_LOGS=recompiles`).
* **NVML numbers physical GPUs, CUDA the visible ones.** With a subset of
  a node's GPUs, per-rank telemetry must open NVML by PCI bus id
  (`torch.cuda.get_device_properties(i).pci_bus_id`), or it reads another
  GPU.
* **A full node can be power-limited.** With every GPU busy, CPU clocks
  can collapse and host-bound steps straggle (F5): if stalls dominate,
  compare `G` with fewer GPUs per node in one allocation.
* **`#SBATCH` lines after the first command are ignored** without a
  warning. Confirm `--exclude` and `--dependency` in `scontrol show job`.
