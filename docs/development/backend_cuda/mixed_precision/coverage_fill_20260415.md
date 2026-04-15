# `cuda_mixed` Coverage Fill 2026-04-15

This note records the code-side coverage fill that followed the benchmark audit for `cuda_mixed`.

It does not change the path table, add a new algorithm, or change `cuda` backend behavior. The patch only closes precision-coverage gaps inside `src/backends/cuda_mixed/`.

## What Changed

### 1. Shared energy precision is now explicit

`src/backends/cuda_mixed/mixed_precision/policy.h` now defines:

```cpp
using EnergyScalar = ActivePolicy::AluScalar;
```

This alias is now used across the shared mixed energy surfaces:

- `line_search/line_searcher.h/.cu`
- `affine_body/abd_line_search_reporter.h/.cu`
- `finite_element/fem_line_search_reporter.h/.cu`
- `dytopo_effect_system/*line_search_reporter*`
- `contact_system/contact_reporter.h`
- contact energy buffers and managers
- ABD / FEM / inter-primitive `ComputeEnergyInfo` and `EnergyInfo` wrappers

The important boundary is:

- scene config, `dt`, material coefficients, and similar host-facing inputs remain `Float`
- local kernel evaluation stays in `AluScalar`
- shared energy buffers, reporter buffers, and line-search totals now stay in `EnergyScalar`

This removes the previous mixed friction / energy situation where a kernel could compute in fp32 locally and then be forced back through `Float` just because the shared interface was typed on `double`.

### 2. FEM MAS is available on mixed paths

The mixed backend no longer compile-time disables MAS on `path2` through `path8`.

The implementation changes are:

- `finite_element/fem_mas_preconditioner.cu`
  - removed the path2~8 compile-time disable
  - synced `mesh_part` validation with `cuda`
  - added Empty FEM vertex `mesh_part` validation
  - kept the existing `contact_aware` extension path
- `finite_element/fem_diag_preconditioner.cu`
  - now defers to MAS when any FEM geometry carries `mesh_part`
- `finite_element/mas_preconditioner_engine.h/.cu`
  - matrix-input blocks now come from `StoreScalar`
  - inverse / residual / solution buffers use `PcgAuxScalar`
  - hard-coded `Matrix3d` / `float3` / `double3` / `DenseVectorView<Float>` bindings were removed from the preconditioner core

Behavioral result:

- partitioned FEM vertices use MAS
- unpartitioned FEM vertices use diagonal fallback inside the same mixed preconditioner
- hybrid scenes that previously failed mixed initialization now follow the same routing intent as `cuda`

### 3. ABD store / ALU coverage was tightened

The major ABD storage boundary changes are:

- `affine_body/abd_linear_subsystem.h`
  - local gradient / Hessian buffers are now store-domain fixed-size types
  - `diag_hessian()` returns store-domain matrices
- `affine_body/abd_linear_subsystem.cu`
  - targeted writeback paths now downcast directly to `StoreScalar`
  - the dytopo path no longer bounces through `Float` before returning to ALU/store domains
- `affine_body/abd_jacobi_matrix.h`
- `affine_body/details/abd_jacobi_matrix.inl`
  - `ABDJacobi` and `ABDJacobiStack` helpers are templated on scalar type, so joint / inter-ABD math can stay in `AluScalar` until the final store-domain write

The targeted constitution / constraint writeback fixes include:

- `affine_body/constitutions/arap.cu`
- `affine_body/constitutions/ortho_potential.cu`
- `affine_body/constitutions/affine_body_fixed_joint.cu`
- `affine_body/constitutions/affine_body_spherical_joint.cu`
- ABD BDF energy / gradient / Hessian paths
- `affine_body/constraints/soft_transform_constraint.cu`

The preserved `Float` areas are still intentional:

- body `q` / `dq`
- mass / inertia
- host-side articulation / external-force bridge state

### 4. Friction uses the active mixed call chain

The active mixed friction path is:

- `contact_system/contact_models/ipc_simplex_frictional_contact.cu`
- `contact_system/al_simplex_frictional_contact.cu`
- `contact_system/al_vertex_half_plane_frictional_contact.cu`

This patch does not route work through `ipc_simplex_frictional_contact_function.h`.

What changed on the active path:

- local friction energy / gradient / Hessian algebra stays in `AluScalar`
- `edge_edge_mollifier_threshold` and `need_mollify` use ALU-typed inputs on the active chain
- final friction energy writes go to `EnergyScalar`

That removes the specific double round-trip that used to make the benchmark conclusion for friction and line search more pessimistic than the path table suggested.

## What Did Not Change

The patch intentionally does not touch:

- the path table itself
- `cuda` backend logic
- `path8` SpMV / PCG accumulation semantics
- a stronger FEM preconditioner beyond the existing MAS
- benchmark manifests or external asset datasets

So the remaining high-value precision questions are now narrower:

1. how much line-search gain is still limited by control flow and repeated evaluation rather than precision alone
2. how much `path8` is still bounded by float-accumulation SpMV
3. whether additional mixed coverage is needed in host bridges or diagnostic / export code

## Static Audit Checklist

After this patch, these repo-local checks should be clean:

```powershell
rg -n "UIPC_MAS_ENGINE_DISABLED|defined\\(UIPC_MIXED_LEVEL_PATH[2-8]\\)" src/backends/cuda_mixed/finite_element

rg -n "BufferView<Float>.*ener|CBufferView<Float>.*ener|DeviceBuffer<Float>.*ener|DeviceVar<Float>.*ener|std::optional<Float> m_energy|vector<Float>\\s+m_energy_values|Float compute_energy\\(|void\\s+energy\\(Float" src/backends/cuda_mixed

rg -n "safe_cast<Float>\\(E_|safe_cast<Float>\\(shape_E|safe_cast<Float>\\(K" src/backends/cuda_mixed/affine_body src/backends/cuda_mixed/contact_system
```

Expected remaining `Float` hits are limited to:

- scene / config / state bridges
- normal-contact helper inputs that still intentionally read baseline `Float` thresholds
- solve / state export paths that are not part of the new mixed-coverage patch

## Verification Status

Repo-local verification added with this patch:

- `apps/tests/sim_case/84_cuda_mixed_fem_mas_hybrid_smoke.cpp`
- `apps/tests/sim_case/85_cuda_mixed_abd_ramp_friction_smoke.cpp`

Build verification in this workspace was partially completed:

- `cuda_mixed` for `build_impl_path6` was rebuilt far enough to catch and fix a new `ABDJacobi` template-definition error introduced during this patch
- after the fix, the target progressed substantially further, but a full no-timeout clean build of every mixed path was not completed inside this turn

So the current state is:

- the precision-coverage patch is implemented in source
- targeted static audits are aligned with the intended design
- mixed smoke tests are added to the repo
- full benchmark re-run and full multi-path build confirmation should be done as the next validation pass
