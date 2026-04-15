# Mixed Precision Scope

This page is the developer-facing precision map for `cuda_mixed`. It is the site version of `src/backends/cuda_mixed/mixed_precision/PRECISION_SCOPE.md`, reorganized around current implementation domains instead of plan milestones.

## Source of Truth

- The normative contract is the code under `src/backends/cuda_mixed/`.
- `claude_plan.md` is useful for rationale, audit notes, and historical intent.
- If `claude_plan.md` and code disagree, follow code.

## Path Boundaries

The paths are cumulative, but not all changes are just "more fp32". Two late-stage boundaries matter:

| Path | Main change introduced |
|---|---|
| `fp64` | Full baseline |
| `path1` | `AluScalar = float` |
| `path2` | `StoreScalar = float` |
| `path3` | `AluScalar = float`, `StoreScalar = float` |
| `path4` | `StoreScalar = float`, `PcgAuxScalar = float` |
| `path5` | `AluScalar = float`, `StoreScalar = float`, `PcgAuxScalar = float` |
| `path6` | `path5` plus `preconditioner_no_double_intermediate` |
| `path7` | `path6` plus full-PCG fp32 for `SolveScalar` and `PcgIterScalar` |
| `path8` | `path6` plus `SolveScalar = float` while keeping `PcgIterScalar = double` |

Shared energy buffers and line-search reductions use `EnergyScalar = ActivePolicy::AluScalar`.

## ALU Domain

The ALU domain covers local kernel evaluation before values are downcast into storage buffers.

| Component | Representative files | Status | Notes |
|---|---|---|---|
| IPC normal contact | `contact_system/contact_models/ipc_simplex_normal_contact.cu` | Implemented | Uses `ActivePolicy::AluScalar` for barrier evaluation before storing results |
| IPC friction and half-plane contact | `contact_system/contact_models/ipc_simplex_frictional_contact.cu`, `contact_system/contact_models/ipc_vertex_half_plane_normal_contact.cu`, `contact_system/contact_models/ipc_vertex_half_plane_frictional_contact.cu` | Implemented | Local contact algebra is typed through `ActivePolicy::AluScalar` |
| AL-IPC contact and half-plane contact | `contact_system/al_simplex_normal_contact.cu`, `contact_system/al_simplex_frictional_contact.cu`, `contact_system/al_vertex_half_plane_normal_contact.cu`, `contact_system/al_vertex_half_plane_frictional_contact.cu` | Implemented | Mirrors the same ALU policy boundary as the IPC path |
| FEM stable neo-Hookean 3D | `finite_element/constitutions/stable_neo_hookean_3d.cu` | Partial | Main ALU path is typed, but this kernel family is still watched as an active mixed-precision hotspot |
| FEM shell and rod constitutions | `finite_element/constitutions/neo_hookean_shell_2d.cu`, `finite_element/constitutions/kirchhoff_rod_bending.cu` | Implemented | Scalar inputs are narrowed with `safe_cast` and outputs are downcast before storage |
| FEM shell bending and plastic variants | `finite_element/constitutions/discrete_shell_bending.cu`, `finite_element/constitutions/plastic_discrete_shell_bending.cu`, `finite_element/constitutions/stress_plastic_discrete_shell_bending.cu` | Implemented | Mixed-precision path is active for energy, gradient, and Hessian assembly |
| FEM ARAP | `finite_element/constitutions/arap_3d.cu` | Implemented | Uses `ActivePolicy::AluScalar` for local energy and derivative evaluation |
| ABD constitutions | `affine_body/constitutions/ortho_potential.cu`, `affine_body/constitutions/arap.cu` | Implemented | Core affine-body constitutions are typed |
| ABD joints and joint limits | `affine_body/constitutions/affine_body_revolute_joint.cu`, `affine_body/constitutions/affine_body_prismatic_joint.cu`, `affine_body/constitutions/affine_body_fixed_joint.cu`, `affine_body/constitutions/affine_body_spherical_joint.cu`, `affine_body/constitutions/affine_body_revolute_joint_limit.cu`, `affine_body/constitutions/affine_body_prismatic_joint_limit.cu` | Implemented | Joint kernels follow the same ALU boundary |
| ABD driving external forces | `affine_body/affine_body_revolute_joint_external_body_force.cu`, `affine_body/affine_body_prismatic_joint_external_body_force.cu` | Implemented | Compute path is typed; host-side bridge remains separate |
| ABD BDF kinetics | `affine_body/bdf/affine_body_bdf1_kinetic.cu`, `affine_body/bdf/affine_body_bdf2_kinetic.cu` | Implemented | Time-integration energy terms follow the ALU policy |
| ABD constraints | `affine_body/constraints/soft_transform_constraint.cu`, `affine_body/constraints/external_articulation_constraint.cu` | Implemented | Constraint evaluation is in-policy before assembly |
| ABD linear-subsystem aggregation | `affine_body/abd_linear_subsystem.cu` | Implemented | Aggregation path keeps typed local results through subsystem assembly |
| ABD-FEM coupling aggregation | `coupling_system/abd_fem_linear_subsystem.cu` | Implemented | Coupling assembly downcasts from typed ALU results into storage buffers |
| ABDJacobi `J^T H J` and stack helpers | `affine_body/abd_jacobi_matrix.h`, `affine_body/abd_jacobi_matrix.cu`, `affine_body/details/abd_jacobi_matrix.inl` | Implemented | Mixed helper math is templated so joint and inter-ABD assembly can stay in ALU / store domains until the final downcast |
| External-force constraint host bridge | `affine_body/constraints/affine_body_revolute_joint_external_body_force_constraint.cu`, `affine_body/constraints/affine_body_prismatic_joint_external_body_force_constraint.cu` | Bridge-only | Host-side attribute bridge remains `Float` and is not the kernel-side ALU path |

## Energy Domain

The energy domain covers the shared interfaces used by constitutions, reporters, contact managers, and line search.

| Component | Representative files | Status | Notes |
|---|---|---|---|
| Shared line-search energy type | `line_search/line_searcher.h`, `line_search/line_searcher.cu` | Implemented | `EnergyScalar = ActivePolicy::AluScalar` is used for buffered energies and final reductions |
| Contact energy buffers and reporters | `contact_system/contact_reporter.h`, `contact_system/simplex_normal_contact.h`, `contact_system/simplex_frictional_contact.h`, `contact_system/vertex_half_plane_normal_contact.h`, `contact_system/vertex_half_plane_frictional_contact.h` | Implemented | Normal and friction contact both write into `EnergyScalar` buffers |
| ABD / FEM / inter-primitive reporters | `affine_body/abd_line_search_reporter.h`, `finite_element/fem_line_search_reporter.h`, `inter_primitive_effect_system/inter_primitive_constitution_manager.h`, `dytopo_effect_system/dytopo_effect_line_search_reporter.h` | Implemented | Shared reporter surfaces no longer force mixed kernels back through `Float` |

## Store Domain

The store domain covers the values that leave local kernels and enter global linear-system buffers.

| Component | Representative files | Status | Notes |
|---|---|---|---|
| Matrix and vector write boundary | `utils/matrix_assembler.h`, `utils/matrix_unpacker.h` | Implemented | Main downcast boundary through `downcast_gradient`, `downcast_hessian`, and `safe_cast` |
| Reporter / assembler local buffers | `affine_body/abd_linear_subsystem.h`, `finite_element/fem_linear_subsystem.h` | Implemented | Local subsystem buffers participate in the store-domain contract |
| Global triplet Hessian `triplet_A` | `linear_system/global_linear_system.h`, `linear_system/global_linear_system.cu` | Implemented | Typed on `StoreScalar` |
| Global BCOO Hessian `bcoo_A` | `linear_system/global_linear_system.h`, `linear_system/global_linear_system.cu` | Implemented | Conversion and SpMV source matrix stay in the store domain |
| Global gradient / RHS vector `b` | `linear_system/global_linear_system.h`, `linear_system/global_linear_system.cu` | Implemented | Typed on `StoreScalar` and populated from subsystem assembly |

## PCG Auxiliary Domain

The PCG auxiliary domain covers vectors and preconditioner outputs used during iterative solve.

| Component | Representative files | Status | Notes |
|---|---|---|---|
| Classic PCG auxiliary vectors | `linear_system/linear_pcg.h`, `linear_system/linear_pcg.cu` | Implemented | `r`, `z`, `p`, and `Ap` follow `PcgAuxScalar`; `path7` also changes solve and iteration scalars |
| Fused PCG auxiliary vectors | `linear_system/linear_fused_pcg.h`, `linear_system/linear_fused_pcg.cu` | Implemented | Same domain contract, with fused dot/update kernels |
| Global SpMV interface | `linear_system/global_linear_system.h`, `linear_system/global_linear_system.cu`, `linear_system/iterative_solver.h`, `linear_system/iterative_solver.cu` | Implemented | Current interface explicitly handles `StoreScalar x PcgAuxScalar x PcgIterScalar` |
| ABD diagonal preconditioner | `affine_body/abd_diag_preconditioner.cu` | Implemented | `path6` and `path7` switch the intermediate inverse/input algebra to `float` |
| FEM diagonal preconditioner | `finite_element/fem_diag_preconditioner.cu` | Implemented | Same `preconditioner_no_double_intermediate` boundary as ABD |
| FEM MAS preconditioner | `finite_element/fem_mas_preconditioner.cu`, `finite_element/mas_preconditioner_engine.h`, `finite_element/mas_preconditioner_engine.cu` | Implemented | Mixed path now supports MAS; partitioned meshes use MAS and unpartitioned vertices fall back to diagonal blocks inside the same preconditioner |

## Solve and Iteration Domain

The solve domain is intentionally conservative until `path7`.

| Component | Representative files | Status | Notes |
|---|---|---|---|
| Solve vector `x` | `linear_system/global_linear_system.h`, `linear_system/linear_pcg.cu`, `linear_system/linear_fused_pcg.cu` | Implemented | `double` for `fp64` through `path6`; `float` in `path7` and `path8` |
| Iteration scalars `rz`, `alpha`, `beta` | `linear_system/linear_pcg.h`, `linear_system/linear_pcg.cu`, `linear_system/linear_fused_pcg.h`, `linear_system/linear_fused_pcg.cu` | Implemented | `double` through `path6`, `float` in `path7`, restored to `double` in diagnostic `path8` |
| Solution export for quality checks | `linear_system/global_linear_system.cu` | Implemented | `extras/debug/dump_solution_x` dumps the current `x` for offline comparison |

## Important Caveats

- `path6` is not just a label change from `path5`. It activates `preconditioner_no_double_intermediate` in supported preconditioners.
- `path7` is the first level that moves both the solve vector and PCG iteration scalars to fp32.
- `path8` is a diagnostic split path for `path7`: solve-vector storage stays fp32 while iteration scalars return to fp64.
- The active mixed friction helper is the `codim_*` simplex friction call chain in `contact_system/contact_models/`. The older `ipc_simplex_frictional_contact_function.h` file is not the active mixed path.
- `dt`, scene config values, contact coefficients, and a few host-side bridge buffers intentionally remain `Float`; the mixed-precision patch only moves kernel-local compute and shared energy/storage interfaces.
- Mixed precision remains compile-time only. There is no runtime precision switch and no auto-fallback path.

## Insertion Points

When porting a kernel or a subsystem from `cuda` into `cuda_mixed`, first decide which precision domain the change belongs to.

| Insertion point | Meaning | Typical files |
|---|---|---|
| `A` | IPC barrier core computation | Contact kernels under `contact_system/contact_models/` |
| `B` | Contact gradient / Hessian just before storage | Contact kernels and matrix write helpers |
| `C` | Matrix assembler write interface | `utils/matrix_assembler.h`, `utils/matrix_unpacker.h` |
| `D` | FEM energy scalar evaluation | FEM constitution `.cu` files |
| `E` | FEM gradient / Hessian just before storage | FEM constitution `.cu` files |
| `F` | Local typed algebra inside templated helpers | Sym / inl helper files used by constitutions |
| `G` | SpMV and iterative-solver interfaces | `linear_system/*` |

## Maintenance Guide for Porting New `cuda` Code

When a new feature lands in `src/backends/cuda/`, port it into `cuda_mixed` with the following checklist:

1. Identify the precision domain first: ALU, store, PCG auxiliary, or solve / iteration.
2. Replace hard-coded scalar assumptions with the correct `ActivePolicy` aliases.
3. Use `safe_cast` for scalar narrowing and `downcast_gradient` / `downcast_hessian` at storage boundaries.
4. Preserve `cuda_mixed` as a separate backend. Do not backdoor mixed-precision policy into `cuda`.
5. Update this scope document if the new feature changes coverage or caveats.

Useful audit command after a rebase:

```shell
git diff ORIG_HEAD..HEAD -- src/backends/cuda/
```

Recommended follow-up checks:

1. Re-run the mixed Stage1 smoke benchmark.
2. Re-run Stage2 perf / quality for affected paths.
3. Re-run `python apps/benchmarks/mixed/uipc_assets/cli.py run ...` if the change touches solution quality or contact-heavy workloads.
