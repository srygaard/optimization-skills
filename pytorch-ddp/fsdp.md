# FSDP2 and HSDP: strategy specifics

The process in `SKILL.md` and the formulas in `formulas.md` apply
unchanged. This file covers what is specific to sharding with FSDP2
(`torch.distributed.fsdp.fully_shard`); HSDP is `fully_shard` over a 2-D
device mesh (replicate x shard):

* choosing the strategy;
* the collectives, as inputs to the common formulas;
* the correctness gate and the traps;
* what sharding does to the compute (`c_mb`), which can cost more than
  its communication;
* unit size, resharding, mixed precision, the working set, the tail;
* HSDP, the knobs, and the process changes.

## Choosing the strategy (step 0)

Pick by memory, before any speed question. Symbols are as in
`formulas.md`; `m_static` is ~16-20 B/param with fp32 AdamW.

| fits in 0.9 x HBM with a useful micro-batch | use |
|---|---|
| `m_static * P` | **DDP**: the least communication, and the compute you benchmarked |
| `m_static * P / G` | **HSDP**, shard within a node and replicate across nodes |
| `m_static * P / W` | **FSDP** across all ranks |
| none of these | add activation checkpointing, then tensor/pipeline parallelism; CPU offload is a last resort |

Add the working set (F12-FSDP below) to each row.

**Do not shard a model that fits in DDP to gain speed.** Sharding adds
work on every micro-batch and saves only at the sync. In one calibration,
configured right, FSDP2 ran within 0.5% of DDP and HSDP 1-3% behind
FSDP; with a compile setting that was fine for DDP, FSDP2 was 20%
slower (`c_mb` below). Shard for memory: a model that does not fit, or a
larger micro-batch, and measure `c_mb` before trusting the result.

## What FSDP2 and HSDP communicate

`b_p` and `b_r` are the bytes per element of `param_dtype` and
`reduce_dtype`; `n_u` is the number of units (modules given to
`fully_shard`, plus the root).

| strategy | collective | `X_k` | calls | cadence | window |
|---|---|---|---|---|---|
| FSDP | all-gather of parameters | `P * b_p` | `n_u` | every micro-batch | forward |
| FSDP | the same in backward (`reshard_after_forward=True`) | `P * b_p` | `n_u` | every micro-batch | backward |
| FSDP | reduce-scatter of gradients | `P * b_r` | `n_u` | sync, if sync is skipped on accumulation; else every micro-batch | backward |
| HSDP | all-gather / reduce-scatter in the shard group | as FSDP | `n_u` | as FSDP | as FSDP |
| HSDP | all-reduce of each gradient shard across replicas | `P * b_r / G` | `n_u` | sync, with `set_requires_all_reduce(False)` on accumulation | backward |

Inputs to the common formulas:

* `C_mb` = the all-gathers (F2): `ga` does not amortise them, the
  micro-batch `b` does;
* `C_sync` = the reduce-scatter (and HSDP's replica all-reduce);
* the forward all-gathers hide only behind forward, ~1/3 of a
  micro-batch, so they enter the exposed regime first (F15);
* F9 with `c_k = (W-1)/W` (ring) and `n_k = n_u`: one call per unit, so
  the per-call overhead counts.

With bf16 parameters and fp32 reduction, a sync micro-batch moves the same
bytes as DDP's fp32 all-reduce. FSDP's extra traffic is the all-gathers on
every *accumulation* micro-batch.

## Correctness gate (replaces the `no_sync` check)

Count collectives per micro-batch in the trace (`ddp_trace.py` lists them
by type, group size and phase). Expected, with `n_u` units:

* **forward:** `n_u` all-gathers, every micro-batch;
* **backward:**
  * `n_u` all-gathers if `reshard_after_forward=True`, 0 if `False`;
  * `n_u` reduce-scatters on the sync micro-batch;
  * 0 reduce-scatters on accumulation micro-batches when
    `set_requires_gradient_sync(False)` is set there (wrappers'
    accumulation contexts usually do it);
* **HSDP:** replica all-reduces on the sync micro-batch only, when
  `set_requires_all_reduce(False)` is set for accumulation.

Also read the dtype in the collective kernel names. A wrapper's "bf16
mixed precision" can set `reduce_dtype` to bf16 too, which is a numerics
decision (F7-FSDP), not only a speed one.

Silent-failure traps:

* **Optimizer built before sharding.** `fully_shard` replaces parameters
  with DTensors, so an optimizer created earlier holds stale tensors.
  Build the optimizer after sharding.
* **Hand-written gradient norms.** A norm over `.to_local()` gradients
  clips each shard by its own norm, with no error. Use
  `torch.nn.utils.clip_grad_norm_` on the DTensor gradients.
* **Checkpointing is collective.** Every rank must take part: a
  rank-0-only `state_dict()` hangs, and a full state dict on every rank
  can run out of memory. Use `torch.distributed.checkpoint` (sharded), or
  `get_model_state_dict(..., StateDictOptions(full_state_dict=True,
  cpu_offload=True))`.
* **Tied weights** (embedding and output head) must live in the same unit,
  usually the root.
* **`reduce_dtype=bf16`** sums gradients in bf16, and the error grows with
  `W`. Check the loss curve, as with DDP's bf16 hook.

## `c_mb`: what sharding does to the compute

F11's `c_mb` is the strategy's cost on every micro-batch. For FSDP it has
three parts, and only the first two are comm:

1. exposed all-gathers (F15);
2. the overlap slowdown from the all-gathers, paid on every micro-batch (F6);
3. **changes to the compute itself**, which the comm formulas cannot
   see:
   * `MixedPrecisionPolicy` changes what enters every unit: weights in
     `param_dtype` (no autocast casts; GEMMs can get faster), and with
     `cast_forward_inputs=True` (the default) every floating-point input
     of every unit, such as fp32 position tables or masks, is cast to
     `param_dtype`;
   * **under `torch.compile(dynamic=True)`, sizes that are fixed in the
     model (head count, head dim) can become runtime symbols** in FSDP2's
     per-unit graphs, while the same model under DDP keeps them as
     constants. The generated pointwise kernels then compute every index
     with runtime division and modulo and cannot vectorise;
   * FSDP2's all-gather copy-in and copy-out (`_foreach_copy_`, two or
     more `multi_tensor_apply` launches per unit per pass). Do not mistake
     them for the optimizer: fused and foreach AdamW work on DTensors, and
     each rank updates only its shard, so the optimizer step gets cheaper
     by about the shard factor.

In one calibration (1B transformer, all comm hidden), (1) and (2) were
~3% of the micro-batch and (3) was +27%: GEMMs and attention got ~5%
faster, but the compiled rotary-embedding kernels became 3-8x slower per
call because `num_heads` and `head_dim` were symbols. Compiling with
automatic dynamic shapes (`dynamic=None`: only dimensions that actually
change become symbolic) brought compute to +3% over DDP, and FSDP2 to
within 0.5% of DDP end to end. Measure (3) before tuning any comm knob:

* run a same-allocation DDP arm when DDP fits, and diff kernel time per
  micro-batch by kernel name against it (or against the 1-GPU trace);
* if compiled kernels regress, compare their generated code between the
  two arms (`TORCH_TRACE` + tlparse, or `TORCH_LOGS=output_code`): size
  symbols (`ks0`, `ks1`, ...) in the index math where the model's sizes
  are fixed are the sign above;
* compile FSDP2 models with `dynamic=None` rather than `True`, and check
  that `None` reaches `torch.compile`: some launchers replace it with
  `False` (`accelerate launch` does, through
  `ACCELERATE_DYNAMO_USE_DYNAMIC`), which recompiles on every new shape
  until the recompile limit and then runs eager;
* an eager arm shows how much compile still gains under sharding;
* attribute `multi_tensor_apply` kernels by their enclosing CPU op
  (`_fused_adamw_`, `_foreach_norm` vs `_foreach_copy_`), not by kernel
  name.

## Formulas specific to sharding

`X_p = P * b_p` is the parameter bytes on the wire, and `X_r = P * b_r`
the gradient bytes.

### F3-FSDP. Unit size replaces bucket size

```text
s_u* ~= sqrt(X * t_o * w)          per collective type, as F3 (ddp.md)
blocks per unit = max(1, round(s_u* / block_bytes))
```

Each unit is one collective of each type: per-call overhead against the
exposed first all-gather and last reduce-scatter. One transformer block
per unit is the usual default. Merge blocks only if `s_u*` is several
blocks; split only if a block is far larger than `s_u*`. Larger units
also raise the working set (F12-FSDP). Fit `t_o` and `w` to all-gather
and reduce-scatter sweeps **on the communicator that runs them** (HSDP:
the shard group), not to all-reduce.

### Resharding

```text
reshard_after_forward=True:   backward all-gather: + its transfer and overlap slowdown, every micro-batch
reshard_after_forward=False:  + P * b_p memory (unsharded params from forward to backward)
reshard_after_forward=G:      reshard to the node; the backward all-gather stays on NVLink
```

The backward all-gather usually hides, so its cost is its overlap
slowdown. Per
micro-batch that is small (~3% of backward in one calibration), but it is
paid on every micro-batch, so `False` gained ~2% there for `P * b_p` of
memory.

### F7-FSDP. Mixed precision, split in two

* `param_dtype=bf16` halves every all-gather and is almost always right,
  but it changes the compute too (`c_mb` above).
* `reduce_dtype` is F7's question: bf16 halves the reduce-scatter, at a
  numerics cost that grows with `W`. Keep fp32 unless the reduce-scatter
  is exposed, and check what the wrapper actually set.

### F12-FSDP. Working set

```text
M_work ~= k_u * unit_bytes_unsharded                    (current + prefetched units, k_u ~ 2-3)
        + P * b_grad   if gradient sync is skipped during accumulation (unsharded grads)
        + P * b_p      if params stay unsharded across micro-batches or until backward
```

Once activations dominate, sharding saves only the static term. Predict
the saving rather than the FSDP peak alone, since a DDP peak taken from
another run carries its own error into the prediction:

```text
DDP peak - FSDP peak ~= m_static,DDP * P  -  m_static * P / W  -  M_work
```

### F14-FSDP. Tail

```text
tail ~= t_o + last_unit_grad_bytes / w_RS
```

The last reduce-scatter (usually the root unit: embeddings and head) is
exposed. Give a large embedding its own unit.

## HSDP

With shard group = node (`G` GPUs) and replicate group = nodes:

* all-gathers and reduce-scatters stay inside the node, on NVLink;
* only the replica all-reduce of each GPU's gradient shard crosses the
  network: `2(n-1)/n * P * b_r / G` per GPU, so per node the same as DDP;
* it runs on sync micro-batches only, if `set_requires_all_reduce(False)`
  is used for accumulation.

HSDP removes the cross-node all-gathers, not the compute change: expect
FSDP's `c_mb`, not DDP's speed. In one calibration, with healthy
communicators, it ran 1-3% behind FSDP: its in-node all-gathers were
faster, but FSDP's hid too, while the in-node communicator's extra
channels (32 + NVLS against the world's 16) slowed the backward more
(F6) and the replica all-reduce added a few ms per sync. It adds two
failure modes:

* **Its communicators are created later than the world one**, lazily, at
  the first FSDP collective. NCCL reads each GPU's NVLink state through
  NVML when it creates a communicator; if a thread in the same process is
  polling NVML at that moment (a telemetry or GPU-stats sampler started
  with training), some reads fail and NCCL counts those links as down. The
  shard group then runs on PCIe paths with a few channels and no NVLS for
  its whole life, with no error: in one calibration HSDP ran 11% slower
  than FSDP for this alone (its in-node all-gather 5x slower), and a
  probe's in-node all-gather fell from 333 to 13 GB/s. Create every mesh communicator at startup, before such
  threads (one small collective per mesh group), check every communicator
  (`scripts/nccl_topo.py`), and probe the shard group, not only the
  world.
* **The shard groups only meet at the replica all-reduce.** Within a node
  the in-node collectives keep the GPUs in step, but nothing synchronises
  the nodes until the sync micro-batch. One slower shard group (a slower
  GPU, or a degraded communicator) accumulates its lag over the whole
  accumulation cycle and the other nodes wait for it there: the replica
  all-reduce shows as mostly peer wait (F10), not transfer.

Compare HSDP with plain FSDP in Job B before assuming it is faster.

## Knobs

| knob | trade | decide with |
|---|---|---|
| strategy (DDP, HSDP, FSDP) | memory vs. comm and compute changes | strategy table, F11 `c_mb` |
| units (which modules get `fully_shard`) | fewer collectives vs. exposed first/last unit and working-set memory | F3-FSDP, F12-FSDP |
| `reshard_after_forward` | `True`: least memory, all-gather again in backward. `False`: one all-gather per micro-batch, params unsharded until backward. Int (e.g. `G`): backward all-gather inside the node | resharding, F12-FSDP |
| `set_requires_gradient_sync(False)` on accumulation | skip the reduce-scatter vs. full unsharded gradients | F12-FSDP |
| `set_requires_all_reduce(False)` (HSDP) | reduce-scatter in-node every micro-batch, replica all-reduce only on sync | HSDP |
| `set_reshard_after_backward(False)` + skipped sync | no all-gathers after the first micro-batch: comm like DDP, memory like DDP for params and grads | F12-FSDP |
| explicit prefetch (`set_modules_to_forward_prefetch`, `set_modules_to_backward_prefetch`) | more overlap vs. one more unsharded unit in memory | the trace: exposed all-gathers |
| `MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype, cast_forward_inputs)` | bytes vs. numerics, and what dtype the compute sees | F7-FSDP, `c_mb` |
| HSDP shard group size | more sharding (memory) vs. more traffic off NVLink | strategy table, F12-FSDP |
| `torch.compile` dynamic shapes | `True`: no recompiles, but fixed model sizes can become symbols under FSDP2. `None`: one recompile per graph when a shape first changes, fixed sizes stay constants | `c_mb`, generated code |
| activation checkpointing | ~1/3 more compute vs. a larger `b` | F2 (a larger `b` amortises all-gathers) |
| CPU offload (`OffloadPolicy`) | fits a larger model; host-link bound | only when nothing else fits |

DDP knobs that disappear: `bucket_cap_mb`, `gradient_as_bucket_view`,
`static_graph` and DDP comm hooks.

## Process changes

* **Job A (ceiling).** A model that needs FSDP usually does not fit on one
  GPU. Use one node with the strategy sharded over NVLink as the
  reference, and measure the multi-node gap against it. Single-node FSDP
  still has comm and the compute changes, so this bounds the *inter-node*
  cost, not all of it.
* **Job B (target shape).**
  * Probe all-gather and reduce-scatter size sweeps on every communicator
    the strategy uses.
  * Arms: `base` and, as the comm control (there is no `noop` hook for
    FSDP), `reshard_after_forward=False`: the difference isolates the
    backward all-gathers.
  * A DDP arm whenever DDP fits: it measures `c_mb` and gives the
    kernel-diff reference.
  * An eager arm if compile is suspect.
  * For HSDP, a plain FSDP arm.
* **Reading the trace.** `ddp_trace.py` reports every collective by type,
  group size and phase, with transfer, exposure and the forward and
  backward compute that hide them. Its bucket, tail and `t_o`/`w` fit
  cover DDP's gradient all-reduces only.
