# Tensor Core Lab: Tensor Core Coverage Upgrade Plan

## Scope

This document is the source of truth for the Tensor Core coverage upgrade inside `experiments/tensor_core_lab/`.

Scope limits:

- only `experiments/tensor_core_lab/`
- no changes to `src/backends/cuda_mixed/`
- no public API changes to the three execution modes:
  - `fp64_ref_no_tc`
  - `fp32_no_tc`
  - `tc32_tf32`

The key semantic change is internal and operator-specific:

- contraction operators may use a Tensor-Core-optimized implementation family
- the SPD family currently stays on the baseline algorithm and only changes the GEMM request mode to `FAST_TF32 + TENSOR_OP`

## Implementation Families

`tc32_tf32` may resolve to one of two internal implementation families:

- `tc_blas`
  - Tensor-Core-first cuBLASLt or cuBLAS matmul path
  - used for matrix-friendly operators and GEMM-dominated SPD updates
- `tc_wmma`
  - custom WMMA microkernel
  - used where fixed-size operators are too launch-bound for library calls

`fp64_ref_no_tc` and `fp32_no_tc` remain baseline implementations.

## Execution Tracing Contract

Every benchmarked operator run must expose:

- `impl_path`
  - `baseline`
  - `tc_blas`
  - `tc_wmma`
- `tensor_core_requested`
  - `yes`
  - `no`
- `tensor_core_verified`
  - `yes`
  - `no`
  - `blocked_by_permissions`

Interpretation:

- `tensor_core_requested=yes`
  - the implementation intentionally selected a Tensor-Core-oriented code path
- `tensor_core_verified=yes`
  - hardware-level profiling confirmed tensor-pipe activity
- `tensor_core_verified=blocked_by_permissions`
  - the run selected a Tensor-Core-oriented path, but Nsight Compute counters were not available on the machine

## Default Selection Rules

- `fp64_ref_no_tc`
  - always `baseline`
- `fp32_no_tc`
  - always `baseline`
- `tc32_tf32`
  - prefer `tc_blas`
  - use `tc_wmma` for fixed-size fused kernels where it is implemented
  - fall back to `baseline` only when an operator has not yet been upgraded
  - the selected path must be visible in output through `impl_path`

## Operator Coverage

### `fem12_local`

Target formula:

- `H = D^T (K D)`

Upgrade direction:

- keep padded `16x16` storage
- replace the baseline two-launch GEMM sequence with a fused WMMA kernel
- keep the baseline cuBLAS path as the `fp32_no_tc` reference

Current implementation state:

- `tc32_tf32` uses `tc_wmma`
- `fp32_no_tc` and `fp64_ref_no_tc` use the baseline cuBLAS path

### `joint24_local`

Target formula:

- `H = Jr^T Hr Jr + Jt^T Ht Jt`

Upgrade direction:

- keep padded `32x32` storage
- move the `tc32_tf32` path to Tensor-Core-first BLAS execution
- collapse the old "four GEMMs plus explicit add kernel" shape into a BLAS-dominated path

Current implementation state:

- `tc32_tf32` uses `tc_blas`
- `fp32_no_tc` and `fp64_ref_no_tc` use the baseline path
- the current implementation still uses multiple Tensor-Core-oriented matmuls rather than a fully grouped cuBLASLt submission

### SPD Family

Covered operators:

- `abd12_factorize`
- `abd12_inverse`
- `abd12_solve`
- `mas48_factorize`
- `mas48_inverse`
- `mas48_solve`

Current rules:

- `fp32_no_tc` and `fp64_ref_no_tc` stay on the baseline blocked Cholesky family
- `tc32_tf32` currently stays on the same baseline blocked Cholesky family
- the only SPD difference in `tc32_tf32` is that GEMM calls go through the `FAST_TF32 + TENSOR_OP` request path

#### `abd12_factorize`

Current implementation state:

- `tc32_tf32` uses the baseline blocked factorization path
- trace reports:
  - `impl_path=baseline`
  - `tensor_core_requested=yes`

#### `mas48_factorize`

Current implementation state:

- `tc32_tf32` uses the baseline blocked factorization path
- trace reports:
  - `impl_path=baseline`
  - `tensor_core_requested=yes`

#### `abd12_inverse` and `mas48_inverse`

Current implementation state:

- `tc32_tf32` reuses the baseline factorization and inverse formula
- the dense product still goes through the `FAST_TF32 + TENSOR_OP` request path via standard GEMM dispatch

#### `abd12_solve` and `mas48_solve`

Current implementation state:

- `tc32_tf32` uses the same forward/backward substitution path as the baseline modes
- the SPD family no longer uses a separate inverse-based solve algorithm

## Profiling Contract

Profiling is handled by `experiments/tensor_core_lab/tools/profile_tensor_core.py`.

The helper profiles representative pairs:

- `fem12_local`
  - `fp32_no_tc`
  - `tc32_tf32`
- `joint24_local`
  - `fp32_no_tc`
  - `tc32_tf32`
- `abd12_factorize`
  - `fp32_no_tc`
  - `tc32_tf32`
- `mas48_factorize`
  - `fp32_no_tc`
  - `tc32_tf32`

Default Nsight Compute metrics:

- `smsp__inst_executed_pipe_tensor.sum`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`

If GPU performance counters are blocked:

- the profiling helper must not fail the workflow
- the result must record `tensor_core_verified=blocked_by_permissions`

## Reporting Contract

Aggregated benchmark output must include:

- `impl_path`
- `tensor_core_requested`
- `tensor_core_verified`
- `time_us`
- `speedup_vs_fp32_no_tc`
- `speedup_vs_fp64_ref_no_tc`
- `rel_error`
- `abs_linf`
- `nan_inf_count`
- `symmetry_error`

Markdown summaries must make it obvious whether a speedup came from:

- a baseline path
- a Tensor-Core-oriented BLAS path
- a WMMA microkernel path

## Validation Rules

Correctness:

- keep all existing Catch2 operator tests
- for `tc32_tf32`, assert the selected implementation path where the upgrade is implemented
- require finite output and `nan_inf_count == 0`
- keep symmetry and error thresholds per operator family

Benchmark smoke:

- every operator family must emit valid rows for all supported modes
- `joint24_local` `tc32_tf32` must remain faster than `fp32_no_tc`
- contraction `tc32_tf32` paths must report a non-baseline `impl_path`
- SPD `tc32_tf32` paths must report:
  - `impl_path=baseline`
  - `tensor_core_requested=yes`
- `smoke` is the default regression profile; higher-condition `full` coverage remains an explicit stress run

Profiling:

- when counters are available, representative `tc32_tf32` cases should show nonzero tensor-pipe activity
- the matching `fp32_no_tc` cases should not show the same tensor-pipe signature

## Non-Goals

- no attempt to retrofit this work into IPC runtime integration in this pass
- no global sparse `SpMV`
- no fused vector PCG operations
- no `half` or `bf16` path in this pass
