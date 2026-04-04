# Tensor Core GEMM Experiment Plan

## Summary

`tensor_core_gemm` is a standalone batched 2D GEMM experiment target.

It measures many independent `C = A * B` matrix multiplications in three modes:

- `fp64_ref_no_tc`
- `fp32_no_tc`
- `tc32_tf32`

The experiment keeps `raw` and `padded` layout variants side by side:

- `raw`
  - physical sizes equal logical sizes
- `padded`
  - `m`, `n`, and `k` are independently rounded up to multiples of `16`
  - `A` and `B` tail regions are zero-filled
  - accuracy is evaluated only on the logical `m x n` region of `C`

The goal is to isolate Tensor Core behavior on dense batched GEMM without mixing in SPD solvers, fused contractions, or IPC scene logic.

## Core Contract

Each case is defined as:

- `A(m x k) * B(k x n) -> C(m x n)`
- column-major storage
- batched execution through strided-batched GEMM
- no epilogue, activation, bias, or GEMV side path

The experiment is implemented as a separate subproject under:

- `experiments/tensor_core_gemm/`

and is enabled from the root build through:

- `UIPC_BUILD_TENSOR_CORE_GEMM=ON`

with fixed targets:

- `tensor_core_gemm_core`
- `tensor_core_gemm_test`
- `tensor_core_gemm_bench`

and fixed output root:

- `output/tensor_core_gemm/<run_id>/`

Recommended standalone configure command:

```powershell
cmake -S . -B build_tcg `
  -DUIPC_BUILD_TENSOR_CORE_GEMM=ON `
  -DUIPC_BUILD_TENSOR_CORE_LAB=OFF `
  -DUIPC_BUILD_TESTS=OFF `
  -DUIPC_BUILD_BENCHMARKS=OFF `
  -DUIPC_BUILD_EXAMPLES=OFF `
  -DUIPC_WITH_CUDA_BACKEND=OFF `
  -DUIPC_WITH_CUDA_MIXED_BACKEND=OFF `
  -DUIPC_CUDA_ARCHITECTURES=native
```

## Data Model

The GEMM experiment uses:

- `GemmShape { m, n, k }`
- `GemmLayoutVariant { Raw, Padded }`
- `GemmCaseSpec`
  - `shape_tag`
  - `shape_group`
  - `layout_variant`
  - `batch_count`
  - `seed`
  - `m`, `n`, `k`
  - `physical_m`, `physical_n`, `physical_k`
- `GemmCaseData`
  - `a_fp64`
  - `b_fp64`
  - `reference_fp64`

`reference_fp64` is populated from the `fp64_ref_no_tc` execution path before cross-mode comparison.

## Shape Inventory

### `uipc` Common Square

- `3x3 * 3x3`
- `6x6 * 6x6`
- `9x9 * 9x9`
- `12x12 * 12x12`
- `24x24 * 24x24`
- `48x48 * 48x48`

### `uipc` Common Rect

- `3x12 * 12x3`
- `12x3 * 3x12`
- `9x9 * 9x12`
- `12x9 * 9x12`
- `3x3 * 3x24`
- `24x3 * 3x24`
- `9x9 * 9x24`
- `24x9 * 9x24`

### Friendly Square

- `16x16 * 16x16`
- `32x32 * 32x32`
- `64x64 * 64x64`
- `96x96 * 96x96`
- `128x128 * 128x128`

### Awkward Square

- `5x5 * 5x5`
- `7x7 * 7x7`
- `15x15 * 15x15`
- `20x20 * 20x20`
- `40x40 * 40x40`

### Friendly Rect

- `16x16 * 16x32`
- `32x16 * 16x32`
- `32x32 * 32x64`
- `64x64 * 64x128`

### Awkward Rect

- `5x7 * 7x3`
- `7x15 * 15x5`
- `15x20 * 20x7`
- `20x40 * 40x15`

## Backend Semantics

- `fp64_ref_no_tc`
  - double-precision `cublasDgemmStridedBatched`
- `fp32_no_tc`
  - `CUBLAS_COMPUTE_32F_PEDANTIC`
- `tc32_tf32`
  - prefer `cuBLASLt`
  - otherwise fall back to `FAST_TF32 + TENSOR_OP`

Trace fields are always emitted:

- `impl_path`
- `tensor_core_requested`
- `tensor_core_verified`

## Metrics And Reporting

Each aggregated row must include:

- `shape_tag`
- `shape_group`
- `layout_variant`
- `m`
- `n`
- `k`
- `physical_m`
- `physical_n`
- `physical_k`
- `batch`
- `mode`
- `impl_path`
- `tensor_core_requested`
- `tensor_core_verified`
- `time_us`
- `speedup_vs_fp32_no_tc`
- `speedup_vs_fp64_ref_no_tc`
- `rel_error`
- `abs_linf`
- `nan_inf_count`
- `logical_gflops`
- `physical_gflops`
- `padding_overhead_ratio`

Accuracy comparison always reads only the logical `m x n` region.

## Test And Batch Policy

`tensor_core_gemm_test` must:

- cover all `uipc_common_*` shapes
- cover at least one representative case per extra shape group
- test both `raw` and `padded`
- validate finite output and trace behavior
- validate `fp32_no_tc` and `tc32_tf32` against the same `fp64_ref_no_tc` reference
- report `unsupported` for `tc32_tf32` on `SM < 80`

`tensor_core_gemm_bench` must:

- default to a `smoke` profile with one batch size per shape
- expose a `full` profile with three batch sizes per shape
- classify batch presets from logical GEMM FLOP count

`profile_gemm_tensor_core.py` must profile representative cases:

- `12x12 * 12x12`
- `16x16 * 16x16`
  - `raw`
  - `padded`
- `24x24 * 24x24`
- `48x48 * 48x48`
- `16x16 * 16x32`
- `64x64 * 64x128`
- `15x15 * 15x15`

and record `blocked_by_permissions` when counters cannot be collected.
