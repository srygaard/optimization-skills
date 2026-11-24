# Tuning formulas (any data-parallel strategy)

Working models for the process in `SKILL.md`. For each one:

* the formula;
* the decision it drives;
* caveats that apply on any cluster.

They hold for any data-parallel strategy: DDP, FSDP, HSDP. A strategy
enters only through its **collective inventory** (below), which its own
file gives: `ddp.md` or `fsdp.md`. Formulas that exist for one strategy
only live in its file: bucket size (F3), tail (F14) and comm hook (F7) in
`ddp.md`, and unit size, resharding and the sharded working set in
`fsdp.md`.

Inputs come from Job A (the 1-GPU ceiling), Job B (the target multi-node
shape) or the node and fabric topology. The constants are cluster-specific,
so measure them; ranges quoted below are what one calibration found, not
defaults. F1 and F8 were folded into F11 and F12.

## The collective inventory

A strategy is a list of collectives `k`, each with:

| field | meaning |
|---|---|
| `X_k` | bytes of the full (unsharded) tensor per call set |
| `n_k` | calls per set (DDP buckets, FSDP units) |
| cadence | every micro-batch, or the sync micro-batch only |
| window | the compute it can hide behind: forward or backward |

Examples:

| strategy | collective | `X_k` | cadence | window |
|---|---|---|---|---|
| DDP | gradient all-reduce | `S` (`4P` fp32, `2P` bf16) | sync | backward |
| FSDP | parameter all-gather | `P * b_p` | every micro-batch | forward |
| FSDP | the same, again (if resharded after forward) | `P * b_p` | every micro-batch | backward |
| FSDP | gradient reduce-scatter | `P * b_r` | sync (if sync is skipped on accumulation) | backward |
| HSDP | all-gather / reduce-scatter in the shard group, replica all-reduce | see `fsdp.md` | | |

## Symbols

| symbol | meaning | from |
|---|---|---|
| `P` | parameters | model |
| `S` | gradient bytes on the wire: `4P` in fp32, `2P` in bf16/fp16 (a comm hook, or bf16 parameters) | model, Job B |
| `W`, `n`, `G` | ranks, nodes, GPUs per node | shape |
| `b` | tokens (or samples) per micro-batch | config |
| `ga` | gradient accumulation steps | config |
| `B_sync = ga*b` | tokens per GPU between gradient syncs | config |
| `R` | throughput per GPU on 1 GPU | Job A |
| `T_1` | 1-GPU optimizer step time | Job A |
| `t_mb` | 1-GPU accumulation micro-batch (forward + backward) | Job A |
| `d_load` | loaded / unloaded `t_mb` - 1 (node power, clocks, host contention with every GPU busy) | Job B vs A |
| `c_mb` | what the strategy adds to every micro-batch (F11) | Job B |
| `t_o,k`, `w_k` | per-call overhead and bandwidth of collective `k`, fit to its size sweep | Job B probe |
| `t_k` | in-step transfer of collective `k`: per call the min across ranks, summed | Job B trace |
| `w_eff,k` | `X_k / t_k` | Job B |
| `exposed`, `e` | comm not covered by compute; `e = exposed / transfer` | Job B trace |
| `slowdown` | overlap slowdown: compute time the collectives add to the compute they overlap | Job B trace |
| `sync_cost` | wall-clock cost of the sync micro-batch, min across ranks, median over steps | Job B |
| `a` | `sync_cost / t_sync`, the unhidden share, in `(0, 1]` | Job B |
| `N_nic`, `B_nic` | fast NICs sharing a PCIe switch with a *visible* GPU; per-NIC bandwidth (e.g. 400 Gb/s = 50 GB/s) | topology |
| `c_k` | bytes leaving a node per byte of `X_k` (F9) | NCCL algorithm |
| `p`, `D_stall` | chance a rank stalls in a step; extra time of a stalled step | Job B, ~100+ steps |
| `s_r` | rank r's accumulation micro-batch / the median rank's | Job B |
| `L`, `h` | layers, hidden size | model |

## 1. Where the step goes (evaluate first)

### F11. Clean-step ledger (with F1's stall term)

```text
T_clean(W) = T_1 + ga * t_mb * d_load + ga * c_mb + sync_cost + I
T_step(W)  = T_clean(W) + P_stall * D_stall
```

* **Decides:** whether communication tuning can matter at all.
  Communication knobs act only on `sync_cost` and the comm part of
  `c_mb`. If those are under ~1% of `T_clean`, no comm knob is worth an
  arm; work on the largest other term: `c_mb`'s compute part, `d_load`
  (node health), `I` (F10) or stalls (F5).
* `c_mb` is the strategy's cost on *every* micro-batch: exposed
  per-micro-batch comm, the overlap slowdown from collectives on every
  micro-batch, and any change to the compute itself (dtypes, how the
  compiler specialises shapes, extra copies). Measure it as the
  accumulation micro-batch time minus a same-allocation DDP arm's, or
  minus `t_mb * (1 + d_load)` when DDP does not fit. It is ~0 for DDP with
  a working `no_sync`.
* **Diff compute before blaming comm.** When `c_mb` is large, diff kernel
  time per micro-batch by kernel name against the DDP or 1-GPU trace
  (`fsdp.md`, "`c_mb`"). Hidden comm costs only its overlap slowdown; the rest is
  compute the strategy changed, which no comm knob fixes.
* Do not add `slowdown + exposed` into this sum. They are summed kernel time,
  usually several times `sync_cost`, and they explain *why* `sync_cost`
  moved, not how much it costs.

## 2. Communication

### F9. Transfer per collective

```text
from the NICs:   t_k ~= f_c * (n_k * t_o,k + c_k * X_k / B_node)
                 B_node = min(N_nic * B_nic,  B_uplink_leaf / n_nodes_leaf_active)
from the probe:  t_k ~= f_c * (n_k * t_o,k + X_k / w_k)
f_c  = in-step / probe transfer, ~1-1.5;   t_o ~0.1-0.2 ms on InfiniBand until a probe measures it
c_k  = bytes leaving a node per byte of X_k, set by the algorithm NCCL picks (TUNING lines):
       all-reduce   ring 2(W-1)/W;  tree / NVLS-tree ~1 at 2 nodes (reduce in-node first), ~2 at many
       all-gather, reduce-scatter   ring (W-1)/W
```

* **Decides:**
  * which GPUs to make visible when a job uses fewer than the node has:
    cover every fast NIC;
  * with F15, whether each collective runs in the hidden or the exposed
    regime;
  * whether to keep nodes on one leaf switch.
* Use the NIC form before any multi-node run, and the probe form once a
  size sweep of each collective type exists: `w_k` already holds `c_k`
  and the protocol efficiency (typically 0.85-0.95 of the NIC rate).
  Before the algorithm is known, assume ring.
* **Keep `n_k * t_o`.** A collective split into many calls (DDP buckets,
  one FSDP unit per layer) pays the per-call overhead on every call. With
  ~30 calls it can be a third of the transfer, and leaving it out makes
  the NIC form 1.3-1.75x short.
* `f_c` is contention with the compute the collective overlaps (SMs,
  HBM): ~1 on slow host-driven transports, up to ~1.5 on fast GPU-driven
  ones, larger for big fp32 all-reduces than for bf16 all-gathers.
* Every term assumes the topology NCCL detected is the real one. A GPU
  whose NVLink is down, or reads as down when the communicator is created,
  sends that communicator over PCIe: several times slower, no error
  (`SKILL.md`, gate 2).
* The uplink term matters only on oversubscribed fabrics, with many nodes
  per leaf sending across at once.

### F15. Exposure: the hidden and the exposed regime

```text
t_window,k ~= compute of k's window that runs after k's first call can start
              DDP: t_bwd_sync * (1 - s_first / S);  FSDP forward all-gather: t_fwd
exposed_k  ~= max( tail_k,  n_k * t_o,k + X_k / w_eff,k - t_window,k )
hidden regime:   t_k < t_window,k    only the tail shows (DDP F14); granularity formulas (DDP F3, FSDP unit size) apply
exposed regime:  t_k > t_window,k    exposed grows 1:1 with transfer
```

* **Decides:** which knobs can pay at all.
  * **Hidden regime:** the comm cost is a few ms plus the overlap slowdown.
    Granularity (DDP buckets, FSDP units) and NCCL env act only at the
    margin, and transfer (F0) is the margin kept for scaling out.
  * **Exposed regime:** only what cuts bytes or calls per token pays: more
    tokens per collective (F2), lower precision on the wire (DDP F7, FSDP
    `reduce_dtype`), more NIC bandwidth (F9).
* Collectives that share a window share the network: sum their `t_k`
  against that window (FSDP: backward all-gathers and reduce-scatters).
* Forward is ~1/3 of a micro-batch, so collectives that must hide behind
  forward (FSDP all-gathers) enter the exposed regime first.
* The threshold is soft: from ~0.8x of the window, exposure grows before
  transfer exceeds it (calls queue behind each other).
* In the hidden regime, *measured* exposed comm overstates the cost: it
  also counts NCCL running while compute idles for other reasons (host
  gaps, late peers). Use `sync_cost` and `c_mb` for cost.
* `a = sync_cost / t_sync` (F2) is not a constant across regimes. It is
  small while hidden, and ~`1 - t_window / transfer` once exposed.

### F2. Critical batch for comm

```text
C_mb   = sum of t_k over per-micro-batch collectives      (0 for DDP)
C_sync = sum of t_k over sync-only collectives            (DDP: the all-reduce)
comm share ~= a * R * (C_mb / b + C_sync / B_sync)
rules:  b >= a * R * C_mb / f_target          B_sync >= a * R * C_sync / f_target
B_crit = C_sync * R                           tokens of compute one sync's comm costs
```

* **Decides:** the micro-batch `b`, which amortises per-micro-batch comm,
  and `B_sync = ga * b`, which amortises per-sync comm, to keep comm under
  a target fraction `f_target`.
* DDP has only the `B_sync` rule. FSDP's all-gathers run every
  micro-batch, so raising `ga` does not amortise them: `b` is the lever.
* `S` grows with `P` and `R` falls with it, so `B_crit` barely moves with
  model size on a given cluster.
* Before Job B, take `t_k` from F9 and use `a = 1`. That is conservative:
  it assumes nothing hides.
* `a` depends on the regime (F15). In the exposed regime, compute the
  share directly: `exposed (F15) / T_step`.

### F4. Scaling out at a fixed global batch

```text
B_sync = B_global / W       per-sync share grows ~ linearly in W
                            per-micro-batch share is flat in W at a fixed b
t_k ~ flat in W             (ring: (W-1)/W or 2(W-1)/W of X_k per node)
```

* **Decides:** the node count at which comm crosses `f_target` for a given
  global batch, and so whether to raise the global batch or accept the
  cost.
* NCCL may switch algorithm as `W` grows (e.g. tree at 2 nodes, ring
  beyond), which moves `c_k` and `f_c` together; read the TUNING lines at
  the target size. One calibration: transfer +8% from 2 to 4 nodes.
* Shorter steps also raise the stall loss (F5), which often bites before
  comm does.

### F0. Transfer efficiency: the second objective

```text
eta_t = (X_k / w_probe,k) / t_k          1 = wire speed of the full-fabric probe
judge a comm knob on  (exposed + slowdown)  AND  transfer
```

* **Decides:** whether to reject a knob that lowers `sync_cost` by
  stretching transfer. Hidden transfer becomes exposed as soon as `B_sync`
  or `b` shrinks (F4).
* Score environment-changing arms against the default fabric's
  `w_probe`, because they change the probe too.
* In-step transfer runs below the probe on fast GPU-driven transports
  (NCCL contends with compute for SMs and HBM), and close to the probe on
  slow host-driven ones such as sockets.

### F6. Overlap slowdown (qualitative)

```text
slowdown ~ kappa * (NCCL CTAs / SMs) * t_overlap
```

* **Decides:** whether CTA or channel knobs (`NCCL_MAX_CTAS`, algorithm)
  are worth an arm: only when the slowdown is large next to the comm cost and
  transfer is well hidden. Always check transfer with F0.
* A collective on every micro-batch (FSDP) pays its slowdown `ga` times per
  step, so it can matter even when every collective is hidden.
* NCCL picks the channel count itself, and algorithm or NIC changes can
  change it. Read it from the TUNING lines (`channel{Lo..Hi}`) before
  attributing a slowdown change. Do not use the form quantitatively.

## 3. Ranks

### F10. Rank imbalance: the slowest GPU sets the pace

```text
I ~= (s_max - s_median) * ga * t_mb  +  I_rand
grows with W like the expected max of W draws
gain from removing the slowest rank ~= (s_max - s_second) * ga * t_mb
```

* **Decides:** whether to drain or replace a GPU or node: one rank more
  than ~1% slower than the median, the same rank every step.
* Imbalance is a fixed fraction of compute, so ga does not amortise it.
* `I_rand` (random per-step jitter) stays after the slowest rank is
  removed. Predict the gain from the second-slowest rank, not to zero.
* Strategies with collectives inside forward or backward (FSDP) also
  synchronise the ranks at every unit; a slow rank then shows as peer
  wait on every collective rather than only at the sync. With HSDP that
  holds within a shard group only: the groups meet at the replica
  all-reduce, so a slower group's lag piles up over the accumulation
  cycle, as with DDP.

### F5. Stalls (one late host, everyone pays)

```text
P_stall(W) = 1 - (1 - p)^W ~= W * p
loss       = P_stall * D_stall / T_step
```

* **Decides:** whether to fix the hosts (power headroom, fewer GPUs per
  node, CPU pinning, loader jitter) before anything else at the target
  `W`. Also: do not shorten steps (lower ga) to scale out, since a
  fixed-length hiccup is a larger share of a short step.
* `p` is higher early in a run and for short, host-bound steps. Measure
  it from steady-state steps of the production step length.

## 4. Memory and micro-batch

### F12. Peak memory per GPU

```text
M_peak ~= m_static * P / W_shard + M_work + m_act * L * h * b
W_shard = 1 (DDP),  G or the shard group (HSDP),  W (FSDP)
b_max   = (0.9 * HBM - m_static * P / W_shard - M_work) / (m_act * L * h)
change strategy when  m_static * P / W_shard + M_work + m_act * L * h * b_min > 0.9 * HBM
```

* **Decides:** the micro-batch ceiling, and the strategy (DDP, ZeRO-1,
  HSDP, FSDP).
* `m_static` covers fp32 weights, gradients and AdamW state, ~16-20
  bytes/param (DDP adds its buckets). ZeRO-1 shards only the optimizer
  state.
* `M_work` is the strategy's working set: ~0 for DDP; unsharded units,
  and unsharded gradients when sync is skipped, for FSDP (`fsdp.md`).
* `m_act` depends on the architecture, precision and kernels. Fit
  `m_static` and `m_act` from Job A's two micro-batch sizes. Once
  activations dominate, sharding saves only the static term.
* `gradient_as_bucket_view` does not reliably lower DDP's peak.

### F13. Fixed cost per micro-batch

```text
t_mb(b) ~= t_fix + beta * b
pick b so t_fix / t_mb(b) ~ 3%, then ga = B_sync / b  (B_sync from F2)
```

* **Decides:** how to split `B_sync` into `b x ga`. For DDP, comm (F2) and
  imbalance (F10) do not depend on the split; for FSDP, a larger `b` also
  amortises the per-micro-batch all-gathers (F2), up to F12's `b_max`.
* Going past the ~3% point rarely pays: larger GEMMs gain little, and
  activation memory grows.
