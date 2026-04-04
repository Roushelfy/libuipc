# Tensor Core Lab: Directory Layout And Build Plan

## Status

This document is the source of truth for the independent `tensor_core_lab` experiment.

Current repo state is **mostly aligned** with this plan.

Tensor Core implementation strategy is defined separately in
`experiments/doc/tensor_core_lab_tensor_core_upgrade_plan.md`.

Pure batched GEMM experiments are defined separately in
`experiments/doc/tensor_core_gemm_experiment_plan.md`.

Known gaps in the current scaffold:

- `tests/` and `bench/` use a few small host wrapper `.cpp` files on Windows so Catch2 and Google Benchmark registration remain reliable with CUDA translation units
- the backend API is still exposed through one shared header even though the implementation is now split into `fp64_ref.cu`, `fp32_simt.cu`, and `tc32_tf32.cu`
- benchmark aggregation is implemented, but the report format should still be treated as experimental and may be refined after the first real measurement campaign
- not every Tensor-Core-oriented operator is fully on its final optimized kernel strategy yet; use the upgrade plan as the operator-level source of truth

This file defines the **decision-complete target layout and build contract** that the implementation must converge to.

`tensor_core_lab` is the structured-operator experiment target.

`tensor_core_gemm` is a separate batched 2D GEMM experiment target.

## Target Directory Layout

```text
experiments/
  doc/
    tensor_core_lab_layout_and_build.md
    tensor_core_lab_tensor_core_upgrade_plan.md
    tensor_core_gemm_experiment_plan.md
  tensor_core_lab/
    CMakeLists.txt
    README.md
    include/tcl/
      mode.h
      op_kind.h
      case_spec.h
      tensor_shape.h
      result_record.h
      device_info.h
      metrics.h
      runner.h
      backend_api.h
    src/
      common/
        device_info.cu
        cuda_check.h
        event_timer.cu
        json_writer.cpp
        rng.cpp
        padding.cpp
        metrics.cpp
      generators/
        mas48_cases.cpp
        abd12_cases.cpp
        fem12_local_cases.cpp
        joint24_local_cases.cpp
      backends/
        fp64_ref.cu
        fp32_simt.cu
        tc32_tf32.cu
      ops/
        chol_factor.cu
        inverse_from_factor.cu
        solve_from_factor.cu
        fem12_assemble.cu
        joint24_assemble.cu
      registry/
        op_registry.cpp
        case_registry.cpp
      cli/
        bench_main.cpp
        test_main.cpp
    tests/
      smoke_device.cu
      mas48_factorize.cu
      mas48_inverse.cu
      mas48_solve.cu
      abd12_factorize.cu
      abd12_inverse.cu
      abd12_solve.cu
      fem12_assemble.cu
      joint24_assemble.cu
      unsupported_arch.cu
    bench/
      mas48_bench.cu
      abd12_bench.cu
      fem12_local_bench.cu
      joint24_local_bench.cu
    tools/
      run_matrix.py
      aggregate.py
      emit_markdown.py
      profile_tensor_core.py
  tensor_core_gemm/
    CMakeLists.txt
    README.md
    include/tcg/
      gemm_case.h
      registry.h
      runner.h
    src/
      cli/
        bench_main.cpp
        test_main.cpp
      generators/
        gemm_cases.cpp
      registry/
        shape_registry.cpp
      runner.cu
    tests/
      gemm_cases.cu
    bench/
      gemm_bench.cu
    tools/
      run_gemm_matrix.py
      aggregate_gemm.py
      emit_gemm_markdown.py
      profile_gemm_tensor_core.py
```

## Directory Responsibilities

### `experiments/tensor_core_lab/include/tcl/`

- Internal public headers for this subproject only.
- Nothing here is installed to the repo root `include/`.
- Namespace is fixed to `uipc::tensor_core_lab`.
- No IPC scene, engine, backend, or geometry-layer types may leak into this boundary.

### `experiments/tensor_core_lab/src/common/`

- Shared infrastructure only.
- Allowed responsibilities:
  - CUDA error handling
  - device capability query
  - deterministic RNG
  - padding helpers
  - benchmark/test result writing
  - metrics computation
  - event timing
- Not allowed:
  - case generation logic
  - operator-specific math

### `experiments/tensor_core_lab/src/generators/`

- Generates realistic but IPC-independent input batches.
- Every generator must first produce an `fp64 master` dataset.
- `fp32_no_tc` and `tc32_tf32` input are derived by casting from the same `fp64 master`.
- No generator may depend on `Scene`, `Engine`, `SimSystem`, or asset loading.

### `experiments/tensor_core_lab/src/backends/`

- Exactly three backend implementations:
  - `fp64_ref.cu`
  - `fp32_simt.cu`
  - `tc32_tf32.cu`
- Responsibilities:
  - create and own compute handles
  - select math mode / compute type
  - expose a uniform matmul and support-capability API
  - report the selected implementation path and Tensor Core tracing metadata
- `fp32_simt` must explicitly disable Tensor Core use.
- `tc32_tf32` must not silently fall back to non-Tensor-Core execution when the GPU is unsupported; it must report `unsupported`.
- `tc32_tf32` may internally select:
  - `tc_blas`
  - `tc_wmma`

### `experiments/tensor_core_lab/src/ops/`

- Contains only tested operator implementations.
- No case generation, file writing, CLI parsing, or report formatting.
- `48x48` and `12x12` factorization/inverse/solve share one blocked Cholesky family.
- `fem12_local` and `joint24_local` share a batched contraction convention.

### `experiments/tensor_core_lab/src/registry/`

- `op_registry.cpp`
  - canonical list of supported operators
  - maps operator ids to implementation entrypoints
- `case_registry.cpp`
  - canonical list of smoke / full / stress case presets
  - no ad hoc benchmark-local case definitions outside the registry

### `experiments/tensor_core_lab/tests/`

- Catch2 correctness / stability / unsupported checks only.
- No performance assertions here.
- One test translation unit per operator family.

### `experiments/tensor_core_lab/bench/`

- Google Benchmark performance entrypoints only.
- Measures GPU operator execution time, not data generation or H2D/D2H setup.
- One benchmark translation unit per operator family.

### `experiments/tensor_core_lab/tools/`

- Python batch-run tooling only.
- `run_matrix.py`
  - runs the benchmark executable over the approved preset matrix
  - defaults to the stable `smoke` profile
  - `full` is explicit
- `aggregate.py`
  - parses raw benchmark output and emits normalized CSV/JSON summary
- `emit_markdown.py`
  - converts aggregated results into a stable markdown report
- `profile_tensor_core.py`
  - runs Nsight Compute on a small approved set of representative cases
  - records whether tensor-pipe activity was hardware-verified or blocked by permissions

## Public Build Contract

## Top-Level CMake Option

Root `CMakeLists.txt` must define:

```cmake
option(UIPC_BUILD_TENSOR_CORE_LAB "Build independent tensor core lab" OFF)
```

The pure GEMM experiment is gated separately by:

```cmake
option(UIPC_BUILD_TENSOR_CORE_GEMM "Build independent batched GEMM experiment targets" OFF)
```

After `add_subdirectory(apps)` it must add:

```cmake
if(UIPC_BUILD_TENSOR_CORE_LAB)
    add_subdirectory(experiments/tensor_core_lab)
endif()
```

## Subproject CMake Rules

`experiments/tensor_core_lab/CMakeLists.txt` must:

1. call `enable_language(CUDA)`
2. `find_package(Eigen3 REQUIRED)`
3. `find_package(CUDAToolkit REQUIRED COMPONENTS cublas cublasLt)`
4. `find_package(Catch2 CONFIG REQUIRED)`
5. `find_package(benchmark CONFIG REQUIRED)`
6. define one static library:
   - `tensor_core_lab_core`
7. define two executables:
   - `tensor_core_lab_test`
   - `tensor_core_lab_bench`

`tensor_core_lab_core` must link:

- `CUDA::cudart`
- `CUDA::cublas`
- `CUDA::cublasLt`
- `Eigen3::Eigen`

`tensor_core_lab_test` must additionally link:

- `Catch2::Catch2`

`tensor_core_lab_bench` must additionally link:

- `benchmark::benchmark`

All three targets must set:

- `CXX_STANDARD 20`
- `CUDA_STANDARD 20`
- `CUDA_STANDARD_REQUIRED ON`
- `CUDA_SEPARABLE_COMPILATION ON`
- `CUDA_RESOLVE_DEVICE_SYMBOLS ON`
- `CUDA_ARCHITECTURES ${UIPC_CUDA_ARCHITECTURES}`

All three targets must call:

- `uipc_target_set_output_directory(...)`

They must **not** call:

- `uipc_add_test`
- `uipc_add_benchmark`
- `uipc_target_add_backend_dependency`

## Package Resolution Constraint

Even when `UIPC_BUILD_BENCHMARKS=OFF`, enabling `UIPC_BUILD_TENSOR_CORE_LAB=ON` must still pull `benchmark` into the generated vcpkg manifest.

That means the manifest-generation logic must treat:

- `UIPC_BUILD_BENCHMARKS`
- `UIPC_BUILD_TENSOR_CORE_LAB`
- `UIPC_BUILD_TENSOR_CORE_GEMM`

as separate frontends that both require the `benchmark` dependency.

## Target Names And Output Locations

Fixed target names:

- `tensor_core_lab_core`
- `tensor_core_lab_test`
- `tensor_core_lab_bench`

Binary output location follows existing repo conventions:

- Windows: `build_xxx/Release/bin/`
- Linux: `build_xxx/Release/bin/`

Experiment runtime output is fixed to:

```text
output/tensor_core_lab/<run_id>/
  meta.json
  raw/
    bench.json
    test.log
  tables/
    summary.csv
    summary.md
  per_op/
    mas48_factorize.csv
    mas48_inverse.csv
    mas48_solve.csv
    abd12_factorize.csv
    fem12_local.csv
    joint24_local.csv
```

## Execution Modes And Data Layout

Fixed execution modes:

- `fp64_ref_no_tc`
- `fp32_no_tc`
- `tc32_tf32`

### Mode Semantics

- `fp64_ref_no_tc`
  - GPU double baseline
  - no Tensor Core use
- `fp32_no_tc`
  - GPU float baseline
  - explicit non-Tensor-Core math path
- `tc32_tf32`
  - float storage
  - TF32 Tensor Core compute path
  - may internally resolve to `tc_blas` or `tc_wmma`
  - unsupported on `SM < 80`

### Fixed Physical Shapes

- `48x48`
  - no padding
- `12x12`
  - pad to `16x16`
  - metrics only read logical `12x12`
- `fem12_local`
  - logical forms: `9x12`, `9x9`, `12x12`
  - all internal contraction storage padded to `16x16`
- `joint24_local`
  - logical result: `24x24`
  - internal storage padded to `32x32`
  - metrics only read logical `24x24`

This padding policy is fixed to keep `fp32_no_tc` and `tc32_tf32` physically identical, so padding does not pollute the Tensor Core comparison.

## Operator Inventory

## `mas48`

- `mas48_factorize`
- `mas48_inverse`
- `mas48_solve`

Input characteristics:

- `16` logical nodes
- `3 DOF` per node
- total `48x48` SPD matrix
- structure must resemble cluster Hessian assembly, not a generic random dense matrix

Implementation contract:

- all three ops share a blocked Cholesky implementation family
- `inverse` is derived from factorization plus solve against identity
- `solve` is derived from factorization plus RHS solve

## `abd12`

- `abd12_factorize`
- `abd12_inverse`
- `abd12_solve`

Input characteristics:

- `12x12` SPD block
- must resemble `J^T H J + M` style ABD local blocks
- logical shape is `12x12`, physical storage is `16x16`

Implementation contract:

- same blocked Cholesky family as `mas48`, specialized for the `12x12` logical case

## `fem12_local`

- computes `H = D^T K D`
- logical shapes:
  - `D`: `9x12`
  - `K`: `9x9`
  - `H`: `12x12`
- internal storage is padded to `16x16`

Input characteristics:

- generated from realistic tetrahedron-derived `dFdx`
- `K` must be SPD and condition-controlled

## `joint24_local`

- computes `H = Jr^T Hr Jr + Jt^T Ht Jt`
- logical shapes:
  - `Jr`: `9x24`
  - `Hr`: `9x9`
  - `Jt`: `3x24`
  - `Ht`: `3x3`
  - `H`: `24x24`
- internal storage is padded to `32x32`

Input characteristics:

- generated from ABD-style Jacobian structure
- not a generic random dense `24x24`

## Test Matrix

## Smoke Profile

- one representative case per operator family
- small batch sizes
- condition scales:
  - `1e2`
  - `1e4`
- purpose:
  - correctness regression
  - support gating
  - CI-friendly runtime

## Full Profile

Fixed benchmark batch presets:

- `mas48`: `256`, `2048`, `8192`
- `abd12`: `4096`, `32768`, `131072`
- `fem12_local`: `8192`, `65536`, `262144`
- `joint24_local`: `4096`, `32768`, `131072`

Fixed condition-scale presets:

- `1e2`
- `1e4`
- `1e6`

Fixed seed groups:

- `stable`
- `typical`
- `stress`

## Correctness Metrics

Required metrics:

- `rel_error`
- `abs_linf`
- `nan_inf_count`
- `symmetry_error`

Operator-specific metrics:

- factorization:
  - `rel_recon = ||A - L L^T||_F / ||A||_F`
- inverse:
  - `rel_identity = ||I - A A^{-1}||_F / ||I||_F`
- solve:
  - `rel_residual = ||Ax-b||_2 / ||b||_2`
- assembly:
  - `rel_fro_to_fp64 = ||H_mode - H_fp64||_F / ||H_fp64||_F`

## Performance Metrics

Required output fields:

- `impl_path`
- `tensor_core_requested`
- `tensor_core_verified`
- `time_us`
- `speedup_vs_fp32_no_tc`
- `speedup_vs_fp64_ref_no_tc`
- `rel_error`
- `abs_linf`
- `nan_inf_count`

Additional recommended fields:

- `effective_gflops`
- `iterations`
- `batch_count`
- `condition_scale`
- `gpu_name`
- `sm`
- `cuda_runtime`

## Pass / Fail Policy

### `tensor_core_lab_test`

- all tests must pass
- `SM < 80`:
  - `tc32_tf32` must return `unsupported`
  - it must not silently fall back

### `tensor_core_lab_bench`

- benchmark does not hard-fail the build
- benchmark only emits raw and aggregated data

### Default Numeric Gates

- `fp64_ref_no_tc`
  - reference path
  - no cross-mode gate beyond finite output
- `fp32_no_tc`
  - `nan_inf_count = 0`
- `tc32_tf32`
  - `nan_inf_count = 0`
  - for `cond <= 1e4`, default target is `rel_error <= 5e-4`

## Canonical Build Commands

Use a dedicated build directory:

```powershell
cmake -S . -B build_tcl `
  -DUIPC_BUILD_TENSOR_CORE_LAB=ON `
  -DUIPC_BUILD_TESTS=OFF `
  -DUIPC_BUILD_BENCHMARKS=OFF `
  -DUIPC_BUILD_EXAMPLES=OFF `
  -DUIPC_WITH_CUDA_BACKEND=OFF `
  -DUIPC_WITH_CUDA_MIXED_BACKEND=OFF `
  -DUIPC_CUDA_ARCHITECTURES=native
```

Build targets:

```powershell
cmake --build build_tcl --config Release --target tensor_core_lab_test --parallel 8
cmake --build build_tcl --config Release --target tensor_core_lab_bench --parallel 8
```

Run targets:

```powershell
.\build_tcl\Release\bin\tensor_core_lab_test.exe
.\build_tcl\Release\bin\tensor_core_lab_bench.exe --benchmark_out=output\tensor_core_lab\manual\raw\bench.json --benchmark_out_format=json
python experiments/tensor_core_lab/tools/run_matrix.py --build_dir build_tcl --config Release --run_root output/tensor_core_lab/nightly
```

## Explicit Non-Goals

- no IPC `Scene`, `Engine`, `World`, or `SimSystem` integration
- no global sparse `SpMV`
- no `dot`, `axpy`, or fused PCG vector updates
- no installation of this subproject's headers to the public repo include tree
- no reuse of `apps/tests` or `apps/benchmarks` helper macros

## Implementation Order

Fixed implementation sequence:

1. make the directory and build contract exact
2. make `tests/` and `bench/` independently buildable
3. implement `fem12_local`
4. implement `joint24_local`
5. implement the shared blocked Cholesky family
6. wire `abd12`
7. wire `mas48`
8. finish aggregation and markdown emission

This order is mandatory because it gives a working contraction baseline before the more error-prone factorization family lands.
