
<!-- revisions suggested by: https://claude.ai/chat/611e46f9-a718-40c5-b836-b0bddd5577cc -->

# 1-Week Preflight (30 min/day) — to guarantee readiness

### Day 1 — Shapes/Strides

* Write 3 tensor transforms (`reshape`, `permute`, `view`) and **predict** the resulting stride layout by hand; verify in NumPy/PyTorch (`.strides`).

### Day 2 — BLAS/GEMM Intuition

* Code a naive `C = A @ B` in Python/C; estimate FLOPs (`2MNK`) and memory bytes touched; compare to `numpy.dot` time.

### Day 3 — Memory Hierarchy

* Sketch your CPU cache sizes; pick a tile size for blocked matmul that fits L2; run once and note cache-miss drop (cachegrind).

### Day 4 — Numerics

* Implement softmax naive vs log-sum-exp; create a case that overflows without stabilization.

### Day 5 — GPU Model Quickstart

* Explain (in your notes) threads → warps (32) → blocks → grid; list **2** ways to improve occupancy and **2** ways to improve memory coalescing.

### Day 6 — Autodiff at a Glance

* Hand-derive $d/dx$ of $y = \mathrm{LayerNorm}(x)$, at least symbolically; check against PyTorch gradients on random tensors.

### Day 7 — Tool Sanity

* Install/verify: `perf`, `valgrind`, PyTorch, Triton; if you have NVIDIA, run:

```bash
nsys profile python -c "import torch; x=torch.randn(1).cuda(); print(x)"
```

---

## Quick Self-Checks (pass if “yes”)

* [ ] Can you tell if a kernel is memory-bound or compute-bound with just time, tensor size, and device peak BW/GFLOPs (using arithmetic intensity)?
* [ ] Can you predict when a `reshape`/`view` is free vs copy from strides?
* [ ] Do you know 2 fixes when FP16 blows up (e.g., BF16, loss scaling, log-space)?
* [ ] Can you read an Nsight Compute summary and state occupancy and warp execution efficiency in one sentence?

---

# 12-Week Micro-Syllabus (30 mins/day, Mon–Fri; weekend optional)

Each week has a theme. Repeat the five daily patterns below; weekend = tiny project tying it together.

---

## Week 1 — Profiling Fundamentals (Python/CPU)

* **Mon:** Profile a slow Python loop vs NumPy (`timeit`, `cProfile`, `line_profiler`).
* **Tue:** Flamegraph or `scalene` hotspot hunt; remove one hotspot.
* **Wed:** Measure allocation overheads; preallocate vs append.
* **Thu:** Numba `@njit` vs pure Python; when does it help/hurt?
* **Fri:** `perf stat -d` on a CPU-bound script; note IPC, cycles, cache misses.
  **Weekend:** turn a 500 ms function into <200 ms and write a 5-bullet memo (before/after, why).

## Week 2 — CPU Locality & Vectorization

* **Mon:** Compare row-major vs column-major access; cachegrind miss counts.
* **Tue:** Stride experiments: contiguous vs strided NumPy; measure.
* **Wed:** SIMD: compile simple C loop with `-O3 -march=native`, check `objdump` for vectors.
* **Thu:** Numba `prange` (multithread) vs OpenMP in C; speedup vs cores.
* **Fri:** BLAS > homemade: naive GEMM vs `numpy.dot`; reason about when to call libraries.
  **Weekend:** micro-project: pairwise L2 distances for 10k vectors — go from naive → blocked (tiling) → BLAS.

## Week 3 — PyTorch Performance 101

* **Mon:** `torch.profiler` on a tiny model; find top ops.
* **Tue:** Dataloader perf: `num_workers`, `pin_memory`, batch size; measure step time.
* **Wed:** `torch.compile` (if available) on an op-heavy model; compare.
* **Thu:** Mixed precision (`autocast`/`GradScaler`); track speed & accuracy drift.
* **Fri:** CPU vs GPU break-even batch size; chart step time vs batch.
  **Weekend:** write a 1-pager: “3 changes that gave the most speed”.

## Week 4 — GPU Basics: Measurement & Memory

* **Mon:** Host↔Device transfers: `torch.cuda.Event` timings; cut copies.
* **Tue:** Kernel launch overhead; fuse a couple elementwise ops in PyTorch and measure.
* **Wed:** Memory throughput ceiling: bandwidth test (copy/axpy); compare to theoretical peak.
* **Thu:** Nsight Systems pass (`nsys profile python …`); identify bottleneck regions.
* **Fri:** Nsight Compute (`ncu`) on a single kernel; note occupancy, throughput, warp efficiency.
  **Weekend:** reduce runtime of a simple GPU pipeline by ≥20% by batching/fusing/reusing buffers.

<!-- Week 4-5 transition: Add one session on CUDA memory model (global/shared/registers) before jumping into kernels -->

## Week 5 — CUDA from Scratch: Elementwise & Reductions

* **Mon:** Write a vector add kernel; validate + measure.
* **Tue:** SAXPY (`y = a*x + y`); ensure coalesced loads/stores.
* **Wed:** Reduction (sum) with shared memory; avoid bank conflicts.
* **Thu:** Warp-shuffle reduction; compare to shared-mem version.
* **Fri:** Use `compute-sanitizer` for race detection; fix an injected bug.
  **Weekend:** aim for ≥70% of device memory bandwidth on SAXPY.

## Week 6 — CUDA Tiling & Memory Hierarchy

* **Mon:** Matrix transpose naive vs shared-mem tiled; measure coalescing impact.
* **Tue:** Tiled GEMM (16×16); compare to cuBLAS for sanity.
* **Wed:** Tune block size / tile shape; track occupancy vs perf tradeoff.
* **Thu:** Introduce double buffering or register tiling (small).
* **Fri:** Document cache/bank conflict learnings in your diary.
  **Weekend:** hit a concrete target (e.g., >20% of cuBLAS for small M,N,K).

## Week 7 — Triton Basics (Fused Elementwise)

* **Mon:** Install & write a Triton elementwise add kernel.
* **Tue:** Triton bias+activation fusion; benchmark vs PyTorch eager.
* **Wed:** Triton softmax across last dim (classic kata); verify numerics.
* **Thu:** Vary `BLOCK_SIZE`/`num_warps`; measure occupancy/perf.
* **Fri:** Add masking for ragged tails; unit tests on random sizes.
  **Weekend:** Triton kernel that beats PyTorch by ≥1.5× for a specific tensor shape.

## Week 8 — Triton Reductions & LayerNorm

* **Mon:** Triton row-wise mean/var reduction.
* **Tue:** Triton layernorm (fused affine).
* **Wed:** Profile with Triton’s perf counters; track memory BW.
* **Thu:** Try autotuning decorators; log best configs.
* **Fri:** Stress different shapes (small vs large rows).
  **Weekend:** write a benchmark script that runs CPU/Torch/Triton variants and spits a neat table.

## Week 9 — Attention Building Blocks

* **Mon:** CUDA or Triton matmul wrapper for fixed tile sizes.
* **Tue:** QKᵀ contraction micro-kernel; verify vs Torch.
* **Wed:** Softmax + dropout fusion (skip dropout if short on time).
* **Thu:** KV product; ensure good memory access.
* **Fri:** Stitch a tiny attention forward; compare to Torch scaled-dot.
  **Weekend:** hit a target speedup on one head (e.g., 512×64 dims).

## Week 10 — End-to-End Model Perf Hygiene

* **Mon:** Mixed precision + grad checkpointing tradeoffs.
* **Tue:** Gradient accumulation vs bigger batch; measure throughput.
* **Wed:** DDP or single-GPU micro-batch — find your sweet spot.
* **Thu:** Kernel launch aggregation (CUDA graphs) if available.
* **Fri:** Create a perf checklist you can reuse on any model.
<!-- Week 10: Include inference optimization techniques (quantization, TensorRT) -->


## Week 11 — Debugging & Correctness Under Speed Pressure

* **Mon:** Add property-based tests (random tensors, tolerances).
* **Tue:** Numerical stability (log-sum-exp, eps handling); compare FP16 vs FP32.
* **Wed:** Memory leaks: steady-state VRAM check across 1000 steps.
* **Thu:** Determinism runs; seed + compare outputs.
* **Fri:** Build a minimal CI script that fails on regressions.

## Week 12 — Capstone & Portfolio

**Pick one:**

* Triton layernorm/softmax beating PyTorch on a chosen shape, **or**
* CUDA tiled GEMM hitting a % of cuBLAS, **or**
* End-to-end inference pipeline (tokenizer → model → post-proc) with profiled bottlenecks removed.

**Deliverables:** repo with README, reproducible benchmark script, and a short “Perf Story” (numbers, screenshots, what you learned).

---

# Daily Kata Bank (rotate forever)

## CPU

* Replace Python loops with vectorized NumPy; measure 1M-element ops.
* Reorder loops to improve cache locality; show cachegrind miss deltas.
* Parallelize with Numba `prange`; find diminishing returns point.

## Profiling

* Draw a quick flamegraph; eliminate one frame from the hot path.
* Compare `perf stat` for two codepaths; explain the IPC difference in one paragraph.

## CUDA (progressive)

* Vector add → SAXPY → Reduction → Scan → Transpose → Tiled GEMM.
* For each: coalescing, shared-mem tiling, occupancy tuning, measure GB/s or GFLOP/s.

## Triton

* Fused elementwise chain (bias+gelu+dropout).
* Softmax / LayerNorm / small matmul; autotune `num_warps` and `BLOCK_SIZE`.

## PyTorch

* Dataloader tuning; aim for GPU utilization >90% in profiler.
* Try AMP + `torch.compile`; verify speed/accuracy.

<!-- Daily kata bank: Add "implement paper technique X in Triton" challenges -->


