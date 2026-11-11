# Batch shape

The sampler and collator decide how much useful work each step gives the GPU.
On an overhead-bound model (small model, ~8% MFU), batch shape moved throughput
more than any I/O fix did.

## 1. Diagnose

- **Real tokens vs padded shape.** FLOPs, memory and step time follow
  `rows × padded_len`, not the non-pad token count. Log both per step.
- **Fill fraction** for packed batches: real tokens ÷ token budget.
- **Per-shape step time and throughput, with loader wait excluded.**
- What fixed-row length buckets looked like in one run (6 buckets, 72 rows):
  real tokens per step varied 5.1×, step time 7.9×. The 11% of steps from
  large buckets took 28% of wall time. Peak memory was set by the largest shape,
  although 84% of steps used the smallest.

## 2. Token-budget (dynamic) batch size

- **rows(bucket) = max(1, T // bucket_len).** Still one shape per bucket, so the
  compile and recompile budget is unchanged. Padded compute per step swung 1.08×
  instead of 5.56×.
- Measured (60M model, 1 GPU): fixed 72 rows 184k tok/s → T=43,848 205k →
  T=88,000 226k (**+22.5%**); MFU 7.06 → 8.30%. Peak memory ≈ a fixed floor
  plus a per-token slope, so memory caps T.
- **Take the knee, not the max.** In a later per-bucket sweep, T≈70k matched
  T≈131k within 1-5% per bucket at about half the memory. Two larger configs
  crashed in a compiler autotune on fresh shapes.
- Long buckets are less efficient per token (attention ∝ rows·L², so at fixed
  rows·L the cost grows with L). A fitted `step ≈ c0 + c1·T + c2·T·L` predicted
  an 8% spread across buckets; the measured spread was 29%. Sweep; don't
  extrapolate the fit.
- **DDP:** with variable rows per batch, give every rank the same number of
  steps (drop or pad the trailing `num_batches % world_size`), or one rank hangs
  at the end of the epoch.

### LR when the effective batch changes

Effective batch N = real tokens per step × grad accumulation × world size. With
a mean-reduced loss, a bigger N lowers gradient variance without changing its
magnitude, so LR goes **up**:

- Adam-family: LR ∝ √N (SDE scaling rules for adaptive optimizers). Linear
  scaling is the SGD rule and runs hot for Adam at large ratios.
- Recompute warmup and total steps in tokens: larger N means fewer optimizer
  steps per epoch.
- Near the critical batch size, √ over-predicts. Sweep {1, 1.5, 2}× around the
  prediction and keep the hottest stable one.

This is a rule, not a measurement from the source workloads.

## 3. Bucket boundaries

- **Padding = Σ(bucket ceiling − length)**, a 1-D quantization problem. Solve it
  exactly by DP over a boundary grid, on the real length histogram, with each
  item weighted by how often an epoch draws it.
- Measured: near-uniform 6 buckets 12.6% padding → DP grid 9.05%. The optimal
  shape had two tight buckets bracketing the dominant length spike and wide
  buckets for the sparse tail. Even with ideal per-bucket batch sizes, the old
  grid cost +4.1% epoch time.
- **With token-budget batch sizes, padding-optimal = throughput-optimal.** A DP
  minimizing predicted epoch time returned the same boundaries.
- **Alignment:** attention kernels quantize sequence length at their block size
  (64 there), so boundaries inside one block band cost the same. 152 vs 160 was
  within 1%. Multiples of 8 were enough; finer grids gave ~0.1 points.
- **Bucket by the length you pad to.** Items assigned by a worst-case bound, but
  padded to their realized length, made true waste 13.1% vs 9.05% analytic.
  Simulate with real batches.
- **Enforce `final length ≤ top bucket` in code.** One over-length item creates
  an extra, unbounded shape (and a recompile). A hand-maintained margin that
  happens to match today is not a guarantee.
- A boundary that rarely fills a batch hardly matters: one 128-length batch in
  25 steps.
- Re-derive per tokenizer and data mix; the length spike moves.

## 4. Packing

One constant shape and ~zero padding. It needs attention that respects segment
boundaries (varlen kernels or a block-diagonal mask), plus per-segment
position and loss bookkeeping, which is a model change, not a loader change.
Check fill first: a packing-budget fix (81.8 → 99.5% fill) was worth 1.15×; at
99.5% there was nothing left.

## 5. Verify what reaches the model

Log the shapes `forward` sees and the rows the sampler yields per bucket. One
run reported a `(72, 736)` batch where the sampler intended `(59, 736)`. The
logs showed `accelerate.prepare` had left the custom `batch_sampler` alone
(no re-sharding or re-batching), so the cause was a stale build. Without those
logs it would have been guesswork. After any wrapper (`prepare`,
`DistributedSampler`), confirm it did not re-batch.
