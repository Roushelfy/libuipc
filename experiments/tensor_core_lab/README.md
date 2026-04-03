# Tensor Core Lab

`tensor_core_lab` is a repo-local experiment target for small, structured CUDA linear-algebra kernels.

Current coverage:

- independent CUDA/Catch2/Google Benchmark targets
- three execution modes:
  - `fp64_ref_no_tc`
  - `fp32_no_tc`
  - `tc32_tf32`
- four operator families:
  - `fem12_local`
  - `joint24_local`
  - `abd12_factorize` / `abd12_inverse` / `abd12_solve`
  - `mas48_factorize` / `mas48_inverse` / `mas48_solve`
- two internal `tc32_tf32` implementation families:
  - `tc_blas`
  - `tc_wmma`
- `tools/run_matrix.py` writes:
  - `meta.json`
  - `raw/bench.json`
  - `raw/test.log`
  - `tables/summary.csv`
  - `tables/summary.md`
  - `per_op/*.csv`

Known implementation notes:

- `tc32_tf32` requires `SM80+`; unsupported devices report `unsupported` instead of silently falling back
- `fem12_local` and `joint24_local` use Tensor-Core-oriented internal implementations under `tc32_tf32`
- the SPD family currently uses the baseline blocked Cholesky algorithm with `FAST_TF32 + TENSOR_OP` request semantics rather than a separate Tensor-Core-specific algorithm
- benchmark rows report:
  - `impl_path`
  - `tensor_core_requested`
  - `tensor_core_verified`
- `12x12` logical blocks use `16x16` physical storage
- `24x24` logical blocks use `32x32` physical storage
- a few host wrapper translation units exist only to make test / benchmark registration stable on Windows
- `tools/profile_tensor_core.py` can run representative Nsight Compute checks and records `blocked_by_permissions` when counters are not available

Recommended configure command:

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

Recommended batch run:

```powershell
python experiments/tensor_core_lab/tools/run_matrix.py --build_dir build_tcl --config Release
```

The default profile is `smoke`.

Run the full matrix explicitly:

```powershell
python experiments/tensor_core_lab/tools/run_matrix.py --build_dir build_tcl --config Release --profile full
```

`full` currently includes higher-condition exploratory cases. Use it when you want stress behavior, not as the default regression path.

Representative tensor-pipe profiling:

```powershell
python experiments/tensor_core_lab/tools/profile_tensor_core.py --build_dir build_tcl --config Release
```
