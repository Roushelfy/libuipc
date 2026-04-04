# Tensor Core GEMM

`tensor_core_gemm` is a standalone batched 2D GEMM experiment target.

It complements `tensor_core_lab` by measuring only dense `C = A * B` workloads in:

- `fp64_ref_no_tc`
- `fp32_no_tc`
- `tc32_tf32`

Each registered shape is benchmarked in two layout variants:

- `raw`
- `padded`

Output is written under:

- `output/tensor_core_gemm/<run_id>/`

Recommended configure command:

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

Recommended smoke run:

```powershell
python experiments/tensor_core_gemm/tools/run_gemm_matrix.py --build_dir build_tcg --config Release
```
