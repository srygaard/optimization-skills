# DDP: strategy specifics

What is specific to `DistributedDataParallel`:

* its collectives, as inputs to the common formulas in `formulas.md`;
* its correctness gate;
* the formulas only DDP has: bucket size (F3), tail (F14), comm hook
  (F7);
* its knobs and bucketing behaviour.

The process is in `SKILL.md`.

## What DDP communicates

| collective | bytes | when | hides behind |
|---|---|---|---|
| gradient all-reduce, in buckets | `S` = gradient bytes on the wire: `4P` in fp32, `2P` with a bf16/fp16 comm hook (F7) or bf16 parameters | the sync micro-batch only, with `no_sync` on the others | that micro-batch's backward, from the first full bucket on |
| buffer broadcast | buffer bytes (usually small) | forward, if buffer sync is on | nothing (it is before compute) |

Inputs to the common formulas:

* per-micro-batch comm `C_mb = 0`;
* per-sync comm `C_sync = transfer` (the all-reduce);
* window: the sync micro-batch's backward after the first bucket
  (F15).

## Correctness gate: is the all-reduce firing only where it should?

Under gradient accumulation the all-reduce must fire on the micro-batch
that steps the optimizer, and **zero** times on the others. Check it in the
trace (`ddp_trace.py` prints the count per micro-batch), not in the code,
which looks right when it is broken.

The common trap is with `torch.compile(DDP(model))`. Wrappers such as
Accelerate resolve `getattr(model, "no_sync", nullcontext)`. Anything that
writes the bare model into the compiled wrapper's `_orig_mod` makes
`no_sync` resolve to `nullcontext`. Numerics stay correct, nothing warns,
and every micro-batch all-reduces. Writers seen in practice:

* `accelerator.unwrap_model(model)`, whose default is
  `keep_torch_compile=True`;
* `accelerator.get_state_dict(model)`, whose default is `unwrap=True`. It
  fires at the first checkpoint save, so short benchmarks never see it.
  Pass `unwrap=False` and strip the `_orig_mod.` / `module.` prefixes.

Guard it in code: at train start and after every checkpoint save, assert
that `no_sync` still resolves to `DistributedDataParallel.no_sync`.

The trap exists only in this wrap order. Reversing it
(`DDP(torch.compile(model))`) avoids the trap but loses all overlap,
because the compiled backward emits every gradient at the end of one
graph. Keep `torch.compile(DDP(model))` and fix the writers.

Other rules:

* `torch._dynamo.config.optimize_ddp = "python_reducer"` silently no-ops
  `no_sync()`. Keep the default `ddp_optimizer`.
* A raw `torch.autograd.Function` in the model becomes a higher-order op
  that DDPOptimizer cannot compile. Wrap kernels as
  `torch.library.custom_op`.
* With variable-length inputs, make sure compile runs with dynamic
  shapes: `True`, or `None` (automatic) if it reaches `torch.compile`.
  Some launchers turn `None` into `False` (`accelerate launch` does), so
  every new shape recompiles until the recompile limit, then runs eager.
  Under DDP `True` is fine; under FSDP2 prefer `None` (`fsdp.md`).

## Job B arms for DDP

* `base`, long enough to count stalls (~100+ steps).
* `noop` (`torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook`,
  a short run), optional. It is the floor: DDP machinery without
  communication. Use only its `sync_cost`: with no sync to realign the
  ranks, its step time is not a throughput reference at many ranks.
* With `ga = 1`, `sync_cost` cannot be computed from timings: the `noop`
  arm is then required.

## Formulas specific to DDP

Symbols as in `formulas.md`. In addition: `n_b` buckets, `s_last` the last
bucket reduced, `H_eff` achievable HBM bandwidth (~0.75 x spec).

### F3. Bucket size (pytorch#118421)

```text
s_b* = sqrt(S * t_o * w)
k    = bucket_cap / s_b*;   overhead(k) / overhead(1) = (k + 1/k) / 2
```

* **Decides:** `bucket_cap_mb`. Change it only when `k` is outside
  `[0.5, 2]`. Inside that range the penalty is at most +25% of a term
  worth a few ms.
* `s_b*` scales as `sqrt(P)` and `sqrt(w)`, so only very large models or
  very slow fabrics move it far.
* Fit `t_o` and `w` over the sizes the buckets actually use. NCCL switches
  algorithm with message size, so a 1 MiB-1 GiB fit can mix algorithms.
  Same-size buckets in one trace cannot separate `t_o` from `w`.
* It overestimates the penalty of large buckets when `t_o` is small
  relative to the backward time left to hide the tail.
* **Hidden regime only (F15).** Once transfer exceeds the backward window,
  the last bucket waits behind the queue, so its tail is not an extra
  cost. F3 then gets both sign and size wrong; use F15 instead. In the
  exposed regime, bucket size acts only through the call count
  (`n_b * t_o`) and through how late the first bucket starts: a few % of
  the step at most, in either direction.

### F14. Tail: a remainder effect

```text
tail_min ~= t_o + s_last / w_eff       s_last = the last bucket reduced (the remainder)
```

* **Decides:** nothing by itself. It explains tail changes: the tail
  follows the remainder left after filling buckets, not the cap.
* The median tail is larger because it includes imbalance (F10).

### F7. Gradients in bf16 on the wire

```text
bf16 hook pays  <=>  dE + dTau > C_cast
  dE ~= e * S / (2 w_eff),  C_cast ~= 3S / H_eff + launches      (S: the fp32 size, 4P)
  =>  w_eff < e * H_eff / 6           (S cancels: a fabric question, not a model-size one)
```

* **Decides:** `comm_hook=bf16` (or fp16). It pays only when comm is
  exposed (`e` large) on a slow fabric. In the exposed regime (F15) it
  saves about half the transfer, all of it exposed time. If `e` is large
  because `B_sync` is small, raise ga first.
* Numerics: NCCL sums in the low precision, and the error grows with `W`.
  Check the loss curve.

## Bucketing facts (recent torch)

* Iteration 0 reduces one all-parameter bucket. Buckets are rebuilt in
  gradient-ready order from iteration 1.
* `bucket_cap_mb=None` gives a small (1 MiB) first bucket. Any explicit
  value makes the first bucket that size too, and some wrappers always
  pass an explicit value.
* `bucket_cap_mb_list` (torch >= 2.14) sets per-bucket caps and is never
  rebuilt.
* Under `torch.compile`, DDPOptimizer splits the graph at `bucket_cap_mb`
  and ignores `bucket_cap_mb_list`. Bucket size is then also a graph-split
  count, so check the accumulation micro-batch's backward time when
  comparing sizes.
* `_get_ddp_logging_data()["bucket_sizes"]` reports the initial
  assignment. Read the real sizes from the trace, or check bucketing
  without a GPU: a comm hook that records bucket sizes, on CPU with gloo.
  That is how the facts above were established.

## Decision rows for DDP

Add these to the decision sheet in `SKILL.md` step 3.

| decision | formula | change when | default |
|---|---|---|---|
| `bucket_cap_mb` | F3 (hidden), F15 (exposed) | hidden regime and `k = cap / s_b*` outside `[0.5, 2]`; in the exposed regime it is a near tie | current value |
| bf16 / fp16 comm hook | F7 | exposed regime, or `w_eff < e * H_eff / 6` | off |
| copy and graph knobs | the table below | the stated mechanism applies | defaults |

## Other knobs

| knob | what it does | when it pays |
|---|---|---|
| `gradient_as_bucket_view` | `.grad` aliases the bucket | Rarely for speed, and peak memory usually does not drop. Breaks `grad.detach_()` and anything holding `.grad` across steps. |
| `batched_grad_copy` (torch >= 2.14) | one `_foreach_copy_` per bucket instead of per parameter | Very many small parameters. |
| `forward_sync_buffers` / `broadcast_buffers` | broadcasts buffers on every syncing forward | Only with large or rank-dependent buffers. Keep it on for safety. |
| `static_graph` | caches the used-parameter set | Avoid it with compiled DDP (DDPOptimizer). |
| `find_unused_parameters` | walks the autograd graph each backward | Only if the model truly skips parameters. |
| bf16 / fp16 comm hook | halves the bytes on the wire | F7: slow fabric with exposed comm. Check the loss curve: the sum is done in low precision. |
| PowerSGD | low-rank compression | Slow fabrics only. Extra memory, and it changes optimization. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | allocator | Not DDP, but it recovers fragmentation under dynamic shapes. |

## When to leave DDP

Every rank holds the full weights, gradients and optimizer state: ~20+
bytes/parameter with fp32 AdamW and DDP buckets (F12), before
activations. When that plus a minimal micro-batch no longer fits, or
memory rather than speed is what stops you, move on:

* ZeRO-1 shards the optimizer state;
* FSDP or HSDP shard everything (HSDP shards within a node and replicates
  across nodes): `fsdp.md`;
* tensor or pipeline parallelism.

Sharding is not free even when its communication hides: it can change
the compute itself (`fsdp.md`). Do not move a model that fits in DDP to
FSDP for speed.
