# Mixed Precision

The `cuda_mixed` backend is a separate backend module for compile-time mixed-precision experiments. It is loaded with `Engine("cuda_mixed")` and keeps the production `cuda` backend untouched.

This doc set treats the current implementation under `src/backends/cuda_mixed/` as the source of truth. `claude_plan.md` is useful as design history, but if a plan note disagrees with code, the code wins.

## Backend Role

`cuda_mixed` is not a runtime mode inside `cuda`. It has its own backend target, entry point, build flags, and benchmark flow.

| Backend | Role |
|---|---|
| `cuda` | Main CUDA backend |
| `cuda_mixed` | Experimental backend for precision-path builds and validation |

Minimal runtime usage is the same as any other backend:

```cpp
Engine engine{"cuda_mixed"};
World  world{engine};
```

## Build-Time Contract

Mixed precision is selected at configure time.

| CMake option | Meaning |
|---|---|
| `UIPC_WITH_CUDA_MIXED_BACKEND` | Builds the `cuda_mixed` backend |
| `UIPC_CUDA_MIXED_PRECISION_LEVEL` | Selects `fp64`, `path1`, `path2`, `path3`, `path4`, `path5`, or `path6` |
| `UIPC_WITH_NVTX` | Enables optional NVTX markers inside `cuda_mixed` |

Invalid values for `UIPC_CUDA_MIXED_PRECISION_LEVEL` fail at CMake configure time.

Example:

```shell
cmake -S . -B build/build_impl_path2 \
  -DUIPC_WITH_CUDA_MIXED_BACKEND=ON \
  -DUIPC_CUDA_MIXED_PRECISION_LEVEL=path2 \
  -DUIPC_WITH_NVTX=OFF

cmake --build build/build_impl_path2 --config Release
```

Important constraints:

- Precision selection is compile-time only.
- The backend does not switch precision at runtime.
- The backend does not auto-fallback to another path after validation or convergence issues.
- This documentation only covers the CMake workflow for mixed precision in this pass.

## Precision Policy

`mixed_precision/policy.h` exposes the active build contract through `ActivePolicy = PrecisionPolicy<kBuildLevel>`.

| Alias | Meaning |
|---|---|
| `AluScalar` | Kernel-side compute precision for gradients, Hessians, and local algebra |
| `StoreScalar` | Global matrix / vector storage precision |
| `EnergyScalar` | Shared energy / line-search / reporter precision |
| `PcgAuxScalar` | PCG auxiliary vectors such as `r`, `z`, `p`, and `Ap` |
| `SolveScalar` | Solve vector `x` precision |
| `PcgIterScalar` | PCG iteration scalars such as `rz`, `alpha`, and `beta` |

Current path matrix:

| Path | `AluScalar` | `StoreScalar` | `PcgAuxScalar` | `SolveScalar` | `PcgIterScalar` |
|---|---|---|---|---|---|
| `fp64` | `double` | `double` | `double` | `double` | `double` |
| `path1` | `float` | `double` | `double` | `double` | `double` |
| `path2` | `float` | `float` | `double` | `double` | `double` |
| `path3` | `float` | `float` | `float` | `double` | `double` |
| `path4` | `float` | `float` | `float` | `double` | `double` |
| `path5` | `float` | `float` | `float` | `float` | `float` |
| `path6` | `float` | `float` | `float` | `float` | `double` |

Additional compile-time policy flags:

| Flag | Meaning |
|---|---|
| `preconditioner_no_double_intermediate` | Enabled in `path4`, `path5`, and `path6` |
| `full_pcg_fp32` | Enabled only in `path5` |

In practice:

- `path4` keeps the same headline type matrix as `path3`, but removes double intermediates inside supported preconditioners.
- `path5` extends fp32 all the way into the solve vector and PCG iteration scalars.
- `path6` is a diagnostic split path: it keeps `SolveScalar=float` like `path5`, but restores `PcgIterScalar=double` to isolate iteration-scalar sensitivity.

## Core Files

Three files define the mixed-precision contract:

| File | Role |
|---|---|
| `mixed_precision/build_level.h` | Maps CMake compile definitions to `MixedPrecisionLevel` and `kBuildLevel` |
| `mixed_precision/policy.h` | Defines `PrecisionPolicy<L>` and the active aliases used across the backend |
| `mixed_precision/cast.h` | Provides `safe_cast`, `downcast_gradient`, and `downcast_hessian` with debug-time safety checks |

`cast.h` is the main boundary between compute and storage domains. In debug builds it checks for NaN/Inf input and narrowing overflow before allowing a downcast.

## Next Pages

- [Mixed Precision Scope](precision_scope.md) maps the current implementation by precision domain.
- [Coverage Fill 2026-04-15](coverage_fill_20260415.md) records the mixed-coverage patch that enabled FEM MAS on `cuda_mixed`, unified shared energy precision, and removed key ABD / friction double bridges.
- [Mixed Precision Validation](validation.md) describes the benchmark and comparison workflow used to audit the paths.
