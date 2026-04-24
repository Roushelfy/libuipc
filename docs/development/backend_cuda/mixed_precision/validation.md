# Mixed Precision Validation

This page documents the validation workflow used by `cuda_mixed`. It focuses on the supported scripts, build matrix, and result interpretation, not on freezing current benchmark numbers into the docs.

For the exact runtime environment needed by manual Python scripts, viewers, C++ tests, and each `fp64/path1..path6` build, see [Mixed Backend Path 运行手册](running_paths.md). That page is the first place to check when a manual run behaves differently from `uipc_assets`.

## Validation Goals

The mixed-precision workflow checks three things:

1. The backend still initializes and advances representative scenes.
2. Runtime changes are measurable and attributable.
3. Solution quality remains understandable when compared against an `fp64` reference build.

## Build Matrix

The repository convention is one build directory per level:

- `build/build_impl_fp64`
- `build/build_impl_path1`
- `build/build_impl_path2`
- `build/build_impl_path3`
- `build/build_impl_path4`
- `build/build_impl_path5`
- `build/build_impl_path6`

On Windows, the recommended entrypoint is the provisioning script:

```powershell
powershell -File scripts/setup_mixed_uipc_assets_builds.ps1
```

It provisions `build/build_impl_fp64` through `build/build_impl_path6`, configures them with `RelWithDebInfo`, enables `UIPC_BUILD_PYBIND=ON`, and schedules the levels serially while each `cmake --build` uses `--parallel`.

The script keeps the default worker count conservative: it uses a default cap of 8 workers and can be overridden with `-Parallel` when the machine has enough memory headroom.

Manual single-level equivalent:

```shell
cmake -S . -B build/build_impl_fp64 \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DUIPC_BUILD_BENCHMARKS=ON \
  -DUIPC_BUILD_TESTS=OFF \
  -DUIPC_BUILD_EXAMPLES=OFF \
  -DUIPC_BUILD_GUI=OFF \
  -DUIPC_BUILD_PYBIND=ON \
  -DUIPC_WITH_CUDA_BACKEND=OFF \
  -DUIPC_WITH_CUDA_MIXED_BACKEND=ON \
  -DUIPC_CUDA_MIXED_PRECISION_LEVEL=fp64

cmake --build build/build_impl_fp64 --config RelWithDebInfo --parallel 8
```

Turn `-DUIPC_WITH_CUDA_BACKEND=ON` on only when you also want optional `cuda` baseline entries in the same benchmark flow.

## Asset-Level Regression Benchmark

Primary entrypoint:

- `apps/benchmarks/mixed/uipc_assets/cli.py`

This flow is now the primary mixed-precision validation entrypoint. It runs selected `uipc-assets` scenes under `cuda_mixed`, records pipeline timer trees, compares each path against an `fp64` reference, and can optionally export OBJ sequences for visual inspection.

`uipc_assets` expects each build root to contain `python/src`, so the per-level builds used here must be configured with `UIPC_BUILD_PYBIND=ON`.

Example:

```shell
python apps/benchmarks/mixed/uipc_assets/cli.py run \
  --manifest apps/benchmarks/mixed/uipc_assets/manifests/full.json \
  --levels fp64 path1 path2 path3 path4 path5 path6 \
  --build fp64=build/build_impl_fp64 \
  --build path1=build/build_impl_path1 \
  --build path2=build/build_impl_path2 \
  --build path3=build/build_impl_path3 \
  --build path4=build/build_impl_path4 \
  --build path5=build/build_impl_path5 \
  --build path6=build/build_impl_path6 \
  --run_root output/benchmarks/mixed/uipc_assets/mixed_audit
```

Useful subcommands:

| Command | Meaning |
|---|---|
| `list` | Show available assets, tags, perf frames, quality frames, notes |
| `resolve` | Print the final selection after `scene/tag/manifest/all` filters |
| `run` | Execute perf + quality collection, then write reports |
| `compare` | Recompute quality metrics from an existing run |
| `report` | Re-render Markdown / JSON / CSV summary outputs |
| `export` | Export per-frame OBJ sequences for selected assets / paths |

## Metrics and Outputs

The mixed benchmark pipeline relies on a small set of recurring outputs.

| Metric / artifact | What it is used for |
|---|---|
| `rel_l2_x` | Relative solution-difference metric against the `fp64` reference |
| `abs_linf_x` | Absolute max-norm solution difference |
| `nan_inf_count` | Hard failure signal for invalid numeric output |
| `timer_frames.json` | Frame-level timing tree produced when telemetry is enabled |
| `stage_summary.json` | Canonical per-stage timer aggregation for one asset + one path |
| pipeline report tables | `perf_by_stage.csv` and `perf_by_asset.csv` for cross-path comparison |
| `x.*.mtx` | Offline solution-dump comparison inputs |
| OBJ sequences | Optional per-frame visual dumps for regression inspection |

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

1. Provision `fp64` and the target compare paths with `scripts/setup_mixed_uipc_assets_builds.ps1`, or manually build them with `UIPC_WITH_CUDA_MIXED_BACKEND=ON` and `UIPC_BUILD_PYBIND=ON`.
2. Use `resolve` or `run --dry-run` to verify the final asset selection and build mapping.
3. Run `uipc_assets` perf / quality comparisons for the affected paths.
4. Inspect `summary.md`, `perf_by_stage.csv`, and `quality.csv`.
5. If quality is suspicious, re-run with `export` to inspect per-frame OBJ output.
