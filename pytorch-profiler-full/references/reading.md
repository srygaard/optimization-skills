# Reading a trace

Three ways in. Use the text report first, because an agent can read it and it applies the traps below for you.
Use `key_averages()` for per-op tables, and Perfetto to look at one step by eye.

## 1. `scripts/trace_report.py`

```bash
python3 scripts/trace_report.py traces/                 # every trace file: full report for the first, a row per rank
python3 scripts/trace_report.py traces/ --rank 1        # full report for rank 1
python3 scripts/trace_report.py rank0.pt.trace.json.gz --top 20 --slow-op-us 500
```

Stdlib only. It reads `.json` and `.json.gz`, and loads each file whole: a 140 MB trace took ~28 s and ~0.8 GB of
memory. Run it in a CPU job rather than on a shared login node.

| section | what it tells you | read it as |
|---|---|---|
| header and `[warn]` | rank, steps, GPU window; warmup signs (step spans vary, `cudaMalloc` or compile events in the window), no collectives under DDP, events left out | fix the window before reading anything else |
| Steps | per step: host span, GPU interval, busy on any stream, busy on compute only, DataLoader wait, GPU idle during it | compute busy is the honest number under DDP; idle during the DataLoader wait means input-bound |
| Idle | gap-size histogram; idle that began as a sync returned; idle while the host waited for data; the largest gaps with what the host was doing | few big gaps mean discrete stalls; thousands of tiny ones add up to launch overhead |
| Syncs | blocking calls grouped by call site: wait, GPU busy during it, GPU idle after it, kind | the cost is the idle after, not the wait |
| Regions | per `record_function` label: host time, GPU time of the kernels launched inside it | where GPU time goes, by your own labels |
| Host ops | long ops that launched ≤ 8 CUDA calls, and how long the GPU idled meanwhile | host work that leaves the GPU idle; ignore rows with no idle |
| Comm | collective kernel time, exposed time, collectives per step | exposed is the cost; per-step counts check gradient accumulation |
| Kernels | GPU time by family, top kernels | the hotspot list |
| Ranks | one row per file | the straggler: most data wait, least collective-kernel time |

**How it counts:**
- **GPU work is kernels, memcpy and memset only.** `gpu_user_annotation` spans are host labels projected onto the
  GPU timeline, and they cover idle time. Counting them once turned a 75.5% busy / 24.5% idle window into "96.9%
  busy", which hid all three gaps over 10 ms (4×H100).
- **GPU work is selected by when it was launched.** Kernels run behind the host, so the report keeps the GPU events
  whose launch (matched by correlation id) falls inside the `ProfilerStep` spans. Work launched just outside the
  window is left out, and the header says how many events.
- **Launches come from both APIs.** Triton and Inductor kernels launch through the driver API (`cuLaunchKernel`,
  category `cuda_driver`); cuBLAS and aten kernels launch through the runtime API (`cudaLaunchKernel`, category
  `cuda_runtime`). Scanning only the runtime API makes real launches look like host stalls.
- **Each `ProfilerStep#N` appears twice:** once as a host `user_annotation`, once as a short `gpu_user_annotation`.
  The report keeps the host span.
- **Per-step GPU intervals run from a step's first kernel to the next step's first kernel.** They add up to the GPU
  window.
- **Busy vs compute:** busy counts any stream. Compute leaves out collective kernels, which spin while they wait
  for other ranks. On a 2×2 A100 run, busy read 100% while 80.6% of the window was exposed communication.

## 2. Event categories in a Kineto trace

| `cat` | what | notes |
|---|---|---|
| `cpu_op` | aten ops, autograd nodes, `Torch-Compiled Region: i/j`, `CompiledFunction` | host time; nest per thread |
| `user_annotation` | `record_function` labels, `ProfilerStep#N`, `enumerate(DataLoader)#...`, `Optimizer.step#...`, `nccl:*` | host spans |
| `gpu_user_annotation` | the same labels, projected onto the GPU timeline | **not work** |
| `cuda_runtime`, `cuda_driver` | API calls: launches, `cudaMemcpyAsync`, `cudaStreamSynchronize`, `cudaMalloc` | `args.correlation` links a call to its kernel |
| `kernel`, `gpu_memcpy`, `gpu_memset` | GPU work, one lane (`tid`) per stream | `args`: `correlation`, `grid`, `block`, `stream`; memcpy has `bytes` and the copy kind, e.g. `Memcpy HtoD (Pageable -> Device)` |
| `python_function` | Python frames, only with `with_stack=True` | see the caveat below |
| `ac2g` flows | launch → kernel arrows | what Perfetto draws |
| `overhead` | the profiler's own work (`Activity Buffer Request`) | ignore |

The top level of the file has `distributedInfo` (backend, rank, world size, NCCL version) and `deviceProperties`.

**`python_function` names identify a function, not a line.**
- A function entered while tracing is named by its `def` line.
- A frame already running when tracing began, such as the training loop's function, is named by the line where it
  stood at that moment, which is the `prof.step()` line.
- So a sync in the loop body shows up as `train.py(NNN): main`. To find the line, use the enclosing
  `record_function` label and aten op (the report prints both), or `torch.cuda.set_sync_debug_mode("warn")`, which
  warns with a traceback.

## 3. Attributing GPU time

- **By label, through the correlation id.** Map each kernel's `args.correlation` to its launch call, then check
  which labels were open at the launch time on any thread. Backward kernels are launched from the autograd thread,
  which holds none of your labels. A label's host span is not its GPU time.
- **Under `torch.compile`, module scopes don't work.** `nn.Module` rows covered ~11% of kernel time on a compiled
  model (MI300A), because compiled regions don't nest under module scopes. Use loop-level labels plus kernel names.
- **To identify one kernel:**
  - **input shapes** (`record_shapes=True`): `[B, H, S, D]` is recognizable;
  - **grid size:** a grid dimension containing B×H;
  - **launch count:** a kernel launched layers × steps times. "No other kernel launched exactly 14 times" identified
    the two flex-attention kernels on a 7-layer model over 2 steps.
- **Inductor kernel names:**
  - `triton_poi_*` is pointwise;
  - `triton_per_*` is a persistent reduction;
  - `triton_red_*` is a reduction;
  - `triton_tem_*` is a template (GEMM or attention).
  - The suffix lists the fused aten ops, which is the way back to source. See the `pytorch-compile` skill,
    `references/tooling.md` §4.
- **Vendor names:**
  - `nvjet_*` and `sm90_xmma_*` are cuBLAS GEMMs;
  - `ncclDevKernel_<Collective>_*` are NCCL collectives;
  - on ROCm, `Cijk_*` are hipBLASLt GEMMs.
- **Kernel families are a name match, and they drift:**
  - **FA4 counted as GEMM:** its attention kernels are named `kernel_cutlass_kernel_flash_attn...`. A rule that
    tested `cutlass` before `flash` counted all of FA4 attention as GEMM, about 85 ms too much GEMM in a 3-step
    window (4×H100).
  - **GEMMs change family under `max-autotune`:** Inductor moves many GEMMs from the vendor library into Triton
    templates, so a mode change moves time between families, not just its amount. The report splits "GEMM (vendor
    library)" from "GEMM (Inductor template)".

## 4. `key_averages()` tables

Per-op aggregates, printed from the profiler object. These are the documented calls:

```python
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
```

- `self_*` excludes time in child ops; `*_total` includes it.
- `group_by_input_shape` needs `record_shapes=True`, and `group_by_stack_n` needs `with_stack=True`.
- Call it in the `on_trace_ready` handler, or after the `with` block, since only the last cycle is kept.
- **Host and GPU columns don't reconcile.**
  - Host time on an op is launch cost plus any blocking. On one MI300A run, `aten::mm` took ~54 s of host time
    against 4.4 s of kernel time. Not explained.
  - A table has no timeline, so it can't show idle gaps, overlap or syncs. Use the report for those.

## 5. Perfetto

`chrome://tracing` may be blocked on managed machines. `ui.perfetto.dev` opens `.json` and `.json.gz` traces.

- **Your trace stays local.** Perfetto's docs describe the UI as client-only: the trace is parsed in the browser
  (WebAssembly) and kept in browser memory, not uploaded. Check your site's policy anyway before opening traces from
  private code.
- **Size:** about 2 GB is the typical in-browser limit. Keep `active` short rather than reaching for the native
  `trace_processor` server.
- **Use Perfetto v57 or later.** Earlier builds silently dropped PyTorch events that overlapped on one track. The
  hosted UI is current.
- **What to look at:**
  - one process per device, with one row per CUDA stream;
  - the CPU process, with one row per thread (the main thread and `pt_autograd_*`);
  - the arrows from launch to kernel (`ac2g` flows);
  - select a range to get a summary of the events inside it.
- **Keys:** W/S zoom, A/D pan.
- **Find the obvious first:** the longest empty stretch of the compute stream, and what the main thread was doing
  above it.
