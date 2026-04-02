# Mixed Precision Validation

This page documents the validation workflow used by `cuda_mixed`. It focuses on the supported scripts, build matrix, and result interpretation, not on freezing current benchmark numbers into the docs.

## Validation Goals

The mixed-precision workflow checks three things:

1. The backend still initializes and advances representative scenes.
2. Runtime changes are measurable and attributable.
3. Solution quality remains understandable when compared against an `fp64` reference build.

## Build Matrix

The repository convention is one build directory per level:

- `build_impl_fp64`
- `build_impl_path1`
- `build_impl_path2`
- `build_impl_path3`
- `build_impl_path4`
- `build_impl_path5`
- `build_impl_path6`
- `build_impl_path7`

Example baseline build:

```shell
cmake -S . -B build_impl_fp64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DUIPC_BUILD_BENCHMARKS=ON \
  -DUIPC_BUILD_TESTS=OFF \
  -DUIPC_BUILD_EXAMPLES=OFF \
  -DUIPC_BUILD_GUI=OFF \
  -DUIPC_WITH_CUDA_BACKEND=OFF \
  -DUIPC_WITH_CUDA_MIXED_BACKEND=ON \
  -DUIPC_CUDA_MIXED_PRECISION_LEVEL=fp64

cmake --build build_impl_fp64 --config Release --target mixed_stage2 --parallel 8
```

Example compare-path builds in PowerShell:

```powershell
$levels = "path1","path2","path3","path4","path5","path6","path7"
foreach ($level in $levels) {
    cmake -S . -B "build_impl_$level" `
      -DCMAKE_BUILD_TYPE=Release `
      -DUIPC_BUILD_BENCHMARKS=ON `
      -DUIPC_BUILD_TESTS=OFF `
      -DUIPC_BUILD_EXAMPLES=OFF `
      -DUIPC_BUILD_GUI=OFF `
      -DUIPC_WITH_CUDA_BACKEND=OFF `
      -DUIPC_WITH_CUDA_MIXED_BACKEND=ON `
      "-DUIPC_CUDA_MIXED_PRECISION_LEVEL=$level"

    cmake --build "build_impl_$level" --config Release --target mixed_stage2 --parallel 8
}
```

Turn `-DUIPC_WITH_CUDA_BACKEND=ON` on only when you also want optional `cuda` baseline entries in the same benchmark flow.

## Stage1 Smoke Benchmark

Primary entrypoints:

- `apps/benchmarks/mixed/mixed_stage1_benchmark.cpp`
- `apps/benchmarks/mixed/tools/run_stage1_matrix.py`

Stage1 is a lightweight smoke matrix:

- `abd_gravity`
- `fem_gravity`
- `fem_ground_contact`

It registers init-only and 20-frame advance cases, with telemetry both on and off. The benchmark can optionally include `cuda` baseline entries through `UIPC_BENCH_ENABLE_CUDA_BASELINE=1`.

Example:

```shell
python apps/benchmarks/mixed/tools/run_stage1_matrix.py \
  --build_fp64 build_impl_fp64 \
  --build_path1 build_impl_path1 \
  --config Release \
  --with_cuda_backend ON \
  --enable_cuda_baseline
```

## Stage2 Perf and Quality Benchmark

Primary entrypoints:

- `apps/benchmarks/mixed/mixed_stage2_benchmark.cpp`
- `apps/benchmarks/mixed/tools/run_stage2_matrix.py`
- `apps/benchmarks/mixed/tools/run_stage2_all_paths.py`

Stage2 expands the workload:

- performance runs on heavier scenarios
- `fp64` reference solution dumps
- compare-path solution dumps
- telemetry on/off overhead checks

`run_stage2_all_paths.py` is the most complete orchestration script for the current path set.

Example:

```shell
python apps/benchmarks/mixed/tools/run_stage2_all_paths.py \
  --build_fp64 build_impl_fp64 \
  --build_path1 build_impl_path1 \
  --build_path2 build_impl_path2 \
  --build_path3 build_impl_path3 \
  --build_path4 build_impl_path4 \
  --build_path5 build_impl_path5 \
  --build_path6 build_impl_path6 \
  --build_path7 build_impl_path7 \
  --config Release \
  --with_cuda_backend ON \
  --run_root output/benchmarks/all_paths_audit
```

Useful switches:

| Option | Meaning |
|---|---|
| `--compile` | Configure and build each path before running |
| `--with_cuda_backend ON` | Compile `cuda` alongside `cuda_mixed` so optional baseline entries can be registered |
| `--dump_surface ON` | Also dump debug surface meshes |

## Asset-Level Regression Benchmark

Primary entrypoint:

- `apps/benchmarks/uipc_assets/run_uipc_assets_benchmark.py`

This flow runs curated dataset scenes under `cuda_mixed`, records timing output, and compares compare-path solution dumps against an `fp64` reference dump.

Example:

```shell
python apps/benchmarks/uipc_assets/run_uipc_assets_benchmark.py run \
  --build_fp64 build_impl_fp64 \
  --build_path1 build_impl_path1 \
  --build_root . \
  --compare_levels path1 path2 path3 path4 path5 path6 path7 \
  --config RelWithDebInfo \
  --run_root output/benchmarks/uipc_assets/mixed_audit
```

This workflow enables `extras/debug/dump_solution_x` during quality reference / compare runs and then compares the resulting `x.*.mtx` files offline.

## Metrics and Outputs

The mixed benchmark pipeline relies on a small set of recurring outputs.

| Metric / artifact | What it is used for |
|---|---|
| `rel_l2_x` | Relative solution-difference metric against the `fp64` reference |
| `abs_linf_x` | Absolute max-norm solution difference |
| `nan_inf_count` | Hard failure signal for invalid numeric output |
| `timer_frames.json` | Frame-level timing tree produced when telemetry is enabled |
| timer hotspots | Aggregated timing bottlenecks derived from the timer tree |
| telemetry on/off overhead | Measures instrumentation cost separately from solver-path changes |
| `x.*.mtx` | Offline solution-dump comparison inputs |

Interpret `rel_l2_x` carefully when the reference solution norm is very small. In those cases `abs_linf_x` and `nan_inf_count` are usually more informative.

## Debug and Profiling Switches

Relevant scene config keys:

| Key | Meaning |
|---|---|
| `extras/debug/dump_solution_x` | Dump the current solve vector `x` as MatrixMarket files |
| `extras/debug/dump_linear_system` | Dump global matrix and RHS |
| `extras/debug/dump_surface` | Dump debug surfaces during solve |
| `extras/debug/dump_linear_pcg` | Dump per-iteration PCG vectors for `linear_pcg` |

Relevant compile-time option:

| Option | Meaning |
|---|---|
| `UIPC_WITH_NVTX` | Enables optional NVTX markers in `cuda_mixed` for external profiling tools |

NVTX is a profiling aid only. It is not part of the mixed-precision correctness contract.

## Recommended Validation Sequence

1. Build `fp64` and the target compare paths with `UIPC_WITH_CUDA_MIXED_BACKEND=ON`.
2. Run Stage1 smoke first to catch obvious initialization or convergence regressions.
3. Run Stage2 perf / quality for the affected paths.
4. Run `uipc_assets` comparisons before claiming path-level quality is acceptable.
5. Only use `cuda` baseline entries when you specifically need side-by-side backend comparison.
