# Mixed Precision Benchmark And Code Audit 2026-04-15

Primary benchmark run:

- `output/benchmarks/mixed/uipc_assets/mixed_assets_representative_warmup_20260415_010834`

Reference documents:

- `output/benchmarks/mixed/uipc_assets/mixed_assets_representative_warmup_20260415_010834/reports/deep_analysis/analysis.md`
- `docs/development/backend_cuda/mixed_precision/precision_scope.md`
- `src/backends/cuda_mixed/mixed_precision/PRECISION_SCOPE.md`
- `src/backends/cuda_mixed/mixed_precision/policy.h`

This note has two goals:

1. Consolidate the benchmark conclusions into one engineering-facing summary.
2. Audit whether the code actually pushes the intended domains to fp32, instead of relying only on the path table in `policy.h`.

## 1. Executive Summary

The benchmark conclusion is still:

- `path8` is the current best general-purpose path for `ABD/contact/animated/solve-heavy` cases.
- `path3` is the strongest path for pure assembly speedups.
- `path6/7/8` are the paths that materially improve `Build Linear System`, `Assemble Preconditioner`, and `Solve Global Linear System`.
- `Compute Energy` and `Line Search` are not currently strong mixed-precision win zones. Their gains are small and unstable.

The code audit conclusion is:

- The path contract in `policy.h` is clear and internally consistent.
- The linear-system layer mostly follows the contract for `StoreScalar`, `PcgAuxScalar`, `SolveScalar`, and `PcgIterScalar`.
- FEM assembly is mostly wired correctly through `AluScalar` plus `downcast_gradient/downcast_hessian<StoreScalar>`.
- ABD is not fully store-domain typed yet. Several important intermediate buffers and helper math paths still stay in `Float`, and `Float` is `double` in this repo.
- Friction contact has a more serious gap: some helper functions are still hard-coded on `Float`, so the nominal `AluScalar=float` path can get promoted back to double inside the helper.
- `path8` does not restore double accumulation in SpMV. It restores some PCG iteration scalars to double, but the actual `A<float> * x<float> -> y<float>` SpMV kernel still instantiates the `float` accumulation path.

So the short answer to "the places that should be fp32, are they all fp32?" is:

- No.
- The policy boundary is defined.
- Large parts of the implementation follow it.
- But there are still several important `ABD`, `friction contact`, and `energy/reporter` paths where the effective precision is more conservative than the path table suggests.

The short answer to "are some paths already no longer worth keeping?" is:

- The benchmark is accurate for the current code.
- But it does not yet measure the fully realized design intent of every path.
- `path2` and `path4` already look weak enough that they should be treated as low-value product candidates.
- They still retain diagnostic value as ablation paths.
- `path3`, `path6`, and `path8` are still the main paths worth actively optimizing and re-measuring after coverage fixes.

## 2. Benchmark Conclusions

### 2.1 Overall ranking

From the primary representative run:

| Path | Median speedup | Mean speedup | Support | Failure rate |
| --- | ---: | ---: | ---: | ---: |
| `path8` | 1.092 | 1.212 | 12 | 14.3% |
| `path7` | 1.079 | 1.059 | 11 | 21.4% |
| `path3` | 1.069 | 1.271 | 12 | 14.3% |
| `path1` | 1.065 | 1.054 | 11 | 21.4% |
| `path5` | 1.058 | 1.223 | 12 | 14.3% |
| `path6` | 1.033 | 1.041 | 11 | 21.4% |
| `path2` | 1.008 | 0.978 | 12 | 14.3% |
| `path4` | 1.003 | 1.035 | 12 | 14.3% |

Interpretation:

- `path8` is the best current default.
- `path3` is not the best total path, but it is the strongest assembly path.
- `path2` and `path4` are diagnostic paths, not good final choices.

### 2.2 Path-to-scenario mapping

- `path1`
  - Best use: low-risk ALU-only speedup, small/clean cases, sanity baseline.
  - Evidence: `libuipc_test` family is best on `path1`.

- `path3`
  - Best use: assembly-dominated cases, especially `ABD/contact-heavy` setups where local constitutions dominate.
  - Evidence: stage leaderboard winner on `Assemble ABD`, `Assemble Contact`, `Assemble FEM`, and `Assemble Linear System`.

- `path5`
  - Best use: intermediate `contains_fem` or medium-cost mixed scenes when `path3` is not enough and solver-side work starts to matter.
  - Evidence: best pipeline path on `contains_fem` in this run.

- `path6`
  - Best use: preconditioner/solver-heavy workloads, especially FEM-side workloads.
  - Evidence: best `Build Linear System` median overall; best pipeline/solve on the limited pure-FEM slice in this run.

- `path7`
  - Best use: experimental full-fp32 PCG path.
  - Evidence: strong preconditioner results, but not clearly better than `path8` in end-to-end results and has higher failure rate.

- `path8`
  - Best use: current general-purpose production candidate.
  - Evidence: best overall pipeline, simulation, and solver results; strongest on `abd` and `animated`.

### 2.3 Where the gains really come from

The strongest stage-level gains are not uniform.

The largest gains are concentrated in:

- `Assemble Contact`
- `Assemble FEM`
- `Assemble ABD`
- `Assemble Linear System`
- `Build Linear System`
- `Assemble Preconditioner`
- `Solve Global Linear System`

The most important split is:

- `path3` wins the assembly stages.
- `path6/7/8` win the build / preconditioner / solve stages.

This is why `path8` beats `path3` overall even though `path3` wins more individual assembly leaderboards:

- `path3` accelerates "make the system".
- `path8` accelerates "make the system and then solve it".

### 2.4 Why line search did not improve much

The benchmark numbers already show the answer:

- `Compute Energy` gains are around `1.00x ~ 1.01x`.
- `Line Search` gains are around `1.00x ~ 1.03x`.

That means line search is not behaving like the assembly and solve stages.

There are three reasons:

1. `Line Search` is not just one big ALU kernel.
   It includes control flow, candidate filtering, repeated energy evaluation, and branch-heavy logic.

2. The energy/reporter interfaces still use `Float` output buffers.
   In this repo `Float = double`, so even when the local energy formula uses `AluScalar`, the public energy path still lands in double buffers.

3. Some energy helpers are still hard-coded to `Float`.
   The frictional contact helper path is the clearest example.

So the benchmark result is consistent with the code: line search is not yet a fully typed fp32 pipeline.

### 2.5 Which paths still matter

This needs to be stated carefully.

The current benchmark is not inaccurate. It measures the real behavior of the current codebase.

What is still incomplete is the mapping from "path as defined in `policy.h`" to "path as fully implemented in all relevant kernels and buffers".

So there are two different questions:

1. Which paths are good on the current implementation?
2. Which paths still deserve engineering investment because their intended coverage is not fully realized yet?

The answer is not the same for all paths.

#### `path2` and `path4`

These are close to being retired as product-facing candidates.

Why:

- Their measured end-to-end value is already weak.
- Their design is structurally weaker than the ALU-enabled paths.
- The strongest observed speedups come from ALU-heavy assembly and solver work, not from isolated store-only or store-plus-PCG-aux changes.

Recommended interpretation:

- keep them as diagnostic / ablation paths
- stop treating them as serious final-path candidates unless a future audit unexpectedly changes their behavior

#### `path1`

This path still has value.

Why:

- it is the cleanest ALU-only baseline
- it is useful as a sanity check
- it performs well on some cleaner families such as `libuipc_test`

Recommended interpretation:

- keep as a conservative baseline and regression reference

#### `path3`

This path clearly has value.

Why:

- it is the strongest assembly path
- it wins several of the most important local stages
- it remains the clearest evidence that `AluScalar=float` is worthwhile

Recommended interpretation:

- keep as a first-class optimization target

#### `path5`

This path should not be retired yet.

Why:

- it helps isolate the impact of bringing `PcgAuxScalar` into fp32
- it still behaves well on some `contains_fem` and medium-cost slices
- it remains a useful bridge between the assembly-heavy and solver-heavy paths

Recommended interpretation:

- keep as an ablation / transition path, even if it is not the final default candidate

#### `path6`

This path still matters.

Why:

- it is the first path where preconditioner internals materially change
- it shows real gains in `Build Linear System` and `Solve Global Linear System`
- its overall median alone understates its importance

Recommended interpretation:

- keep as a first-class optimization target

#### `path7`

This path is losing value as a production candidate, but still matters as an experiment.

Why:

- it isolates the effect of full-fp32 PCG
- it helps answer whether pushing `PcgIterScalar` to float is worth the numerical cost
- its current failure profile is worse than `path8`

Recommended interpretation:

- keep as a sensitivity / diagnostic path
- do not treat it as the preferred final path unless future fixes change the tradeoff materially

#### `path8`

This path clearly has value.

Why:

- it is currently the best overall path
- it has the best current balance between speed and robustness
- it remains the strongest default candidate after this benchmark

Recommended interpretation:

- keep as the primary product-facing path candidate

#### Practical path triage

If the current set needs to be simplified, the most practical grouping is:

- Mainline optimization and re-measurement:
  - `path3`
  - `path6`
  - `path8`
- Keep for baseline / ablation:
  - `path1`
  - `path5`
  - `path7`
- Downgrade to diagnostic/reference-only:
  - `path2`
  - `path4`

That is the correct interpretation today.

It does not mean `path2` and `path4` were measured incorrectly.

It means:

- they are already weak on the current code
- and the known implementation gaps are more likely to affect the stronger paths than to turn `path2` or `path4` into winners

#### What still needs re-measurement

Some paths are still being systematically under-realized by current implementation gaps.

The most important known gaps are:

- ABD local buffers still staying in `Float` / `double`
- `ABDJacobi` helpers still hard-coded on `Float`
- friction contact helpers still hard-coded on `Float`
- energy / line-search output paths still landing in `Float` buffers
- `path8` restoring some PCG scalar precision without restoring double accumulation inside the actual `A<float> * x<float> -> y<float>` SpMV kernel

So the right conclusion is:

- some paths already look weak enough to downgrade today, especially `path2` and `path4`
- but the stronger paths still need continued measurement after code-coverage fixes
- the current ranking is good enough for triage, but not the final word on each path's ceiling

## 3. Asset Coverage Conclusion

The current representative set is useful, but not comprehensive.

### 3.1 What is covered well

- `ABD`
- `contact-heavy`
- `rigid_ipc`-style scenes
- some `animated`/motor-driven cases

### 3.2 What is under-covered

- pure `FEM`
- `ABD-FEM coupling`
- `particle-only`
- friction sweeps
- degeneracy/stability families
- more solver-dominated large systems

### 3.3 What is missing from the representative set

The representative set is still missing several families from the catalog, including:

- `rigid_ipc_unit_tests_erleben`
- `rigid_ipc_unit_tests_rotation`
- `rigid_ipc_unit_tests_tunnel`
- `rigid_ipc_unit_tests_tessellated_plane`
- `rigid_ipc_compactor`
- `rigid_ipc_friction`
- `rigid_ipc_friction_arch`
- `rigid_ipc_friction_rolling`
- `rigid_ipc_mechanisms`
- `rigid_ipc_octopus`

The benchmark conclusions should therefore be interpreted as:

- strong for current ABD/contact representative scenes
- only directional for FEM/coupling/particle
- incomplete for stability-stress and friction families

## 4. Code Audit: What Is Correctly Typed

### 4.1 Policy layer

`src/backends/cuda_mixed/mixed_precision/policy.h` is internally consistent:

- `path1/3/5/6/7/8`: `AluScalar=float`
- `path2/3/4/5/6/7/8`: `StoreScalar=float`
- `path4/5/6/7/8`: `PcgAuxScalar=float`
- `path7/8`: `SolveScalar=float`
- `path7`: `PcgIterScalar=float`
- `path8`: `PcgIterScalar=double`

The compile-time flags are also consistent:

- `preconditioner_no_double_intermediate` is enabled on `path6/7/8`
- `full_pcg_fp32` is enabled only on `path7`

### 4.2 Linear-system path

These parts are largely correct:

- `src/backends/cuda_mixed/linear_system/global_linear_system.h`
- `src/backends/cuda_mixed/linear_system/global_linear_system.cu`
- `src/backends/cuda_mixed/linear_system/linear_pcg.h`
- `src/backends/cuda_mixed/linear_system/linear_pcg.cu`
- `src/backends/cuda_mixed/linear_system/linear_fused_pcg.h`
- `src/backends/cuda_mixed/linear_system/linear_fused_pcg.cu`
- `src/backends/cuda_mixed/affine_body/abd_diag_preconditioner.cu`
- `src/backends/cuda_mixed/finite_element/fem_diag_preconditioner.cu`

What is implemented correctly there:

- global matrix/vector storage is typed on `StoreScalar`
- PCG aux vectors are typed on `PcgAuxScalar`
- solve vector `x` is typed on `SolveScalar`
- alpha/beta/rz interfaces are typed on `PcgIterScalar`
- `path6/7/8` do switch preconditioner intermediates away from forced double

### 4.3 FEM main constitutions

The main FEM assembly path is mostly correctly wired.

Representative good files:

- `src/backends/cuda_mixed/finite_element/constitutions/stable_neo_hookean_3d.cu`
- `src/backends/cuda_mixed/finite_element/constitutions/arap_3d.cu`
- `src/backends/cuda_mixed/finite_element/constitutions/discrete_shell_bending.cu`
- `src/backends/cuda_mixed/finite_element/constitutions/neo_hookean_shell_2d.cu`
- `src/backends/cuda_mixed/finite_element/constitutions/kirchhoff_rod_bending.cu`
- `src/backends/cuda_mixed/inter_primitive_effect_system/constitutions/soft_vertex_stitch.cu`

Pattern:

- local math is evaluated in `ActivePolicy::AluScalar`
- outputs are written through `downcast_gradient<Store>` and `downcast_hessian<Store>`
- FEM subsystem reporter/kinetic buffers are already typed on `StoreScalar`

This is why the FEM-side benchmark behavior is more aligned with the intended path design than ABD.

## 5. Code Audit: Where fp32 Coverage Is Still Incomplete

### 5.1 `Float` is `double`

This matters for every audit judgment.

In `include/uipc/common/type_define.h`:

- `using Float = double;`

So any mixed-backend path that still writes or computes in `Float` is still in double precision, not in an abstract "default scalar".

### 5.2 ABD subsystem local buffers are still double

This is the clearest store-domain gap.

In `src/backends/cuda_mixed/affine_body/abd_linear_subsystem.h`:

- `ComputeGradientHessianInfo` uses `muda::BufferView<Vector12>` and `muda::BufferView<Matrix12x12>`
- `body_id_to_shape_hessian`
- `body_id_to_shape_gradient`
- `body_id_to_kinetic_hessian`
- `body_id_to_kinetic_gradient`
- `diag_hessian`

All of these are still `Vector12` / `Matrix12x12`, which are `Float`-typed, therefore `double`.

Impact:

- ABD constitutions and ABD kinetic do not hand gradients/Hessians directly to `StoreScalar` buffers.
- They first land in double intermediate buffers.
- Only later does `ABDLinearSubsystem` cast them back to `AluScalar` and then downcast to `StoreScalar`.

This means the current ABD path is not a clean store-domain fp32 pipeline.

It also means the documentation line "local subsystem buffers participate in the store-domain contract" is not fully true for ABD yet.

### 5.3 Several ABD constitutions still rebuild gradient/Hessian in `Float`

These files still use explicit `Float`-typed gradient/Hessian temporaries:

- `src/backends/cuda_mixed/affine_body/constitutions/ortho_potential.cu`
- `src/backends/cuda_mixed/affine_body/constitutions/arap.cu`
- `src/backends/cuda_mixed/affine_body/constitutions/affine_body_fixed_joint.cu`
- `src/backends/cuda_mixed/affine_body/constitutions/affine_body_spherical_joint.cu`
- `src/backends/cuda_mixed/affine_body/constitutions/affine_body_prismatic_joint.cu`
- `src/backends/cuda_mixed/affine_body/constitutions/affine_body_revolute_joint.cu`
- `src/backends/cuda_mixed/affine_body/constitutions/affine_body_prismatic_joint_limit.cu`
- `src/backends/cuda_mixed/affine_body/constitutions/affine_body_revolute_joint_limit.cu`

Typical pattern:

- evaluate core formula in `AluScalar`
- then cast into `Vector<Float, ...>` / `Matrix<Float, ...>`
- then write or pass onward

That is better than all-double formula evaluation, but it is still not "the entire intended domain is fp32".

It inserts a double-precision rebound in the middle of the path.

### 5.4 `ABDJacobi` and `ABDJacobiStack` are still hard-coded on `Float`

Files:

- `src/backends/cuda_mixed/affine_body/abd_jacobi_matrix.h`
- `src/backends/cuda_mixed/affine_body/abd_jacobi_matrix.cu`
- `src/backends/cuda_mixed/affine_body/details/abd_jacobi_matrix.inl`

Examples:

- `Vector<Float, 3 * N>`
- `Matrix<Float, 3 * N, 12>`
- `Matrix3x12`
- `Matrix12x12`

This is exactly consistent with the old scope note that `ABDJacobi` is still partial / transitional.

Impact:

- contact-to-ABD projection and some ABD-side helper algebra are still pinned to double
- this weakens the real mixed-precision coverage of ABD contact assembly

### 5.5 ABD dytopo path still bounces through `Float`

In `src/backends/cuda_mixed/affine_body/abd_linear_subsystem.cu` the dy-topo gradient path does:

- `G3.template cast<Float>()`
- multiply with `J_i.T()`
- then cast back to `Alu`
- then downcast to `StoreScalar`

That means the middle transform still runs through double, not through the intended typed helper path.

### 5.6 Friction contact helper is still hard-coded to `Float`

This is a stronger issue than the ABD temporary-buffer issue.

File:

- `src/backends/cuda_mixed/contact_system/contact_models/ipc_simplex_frictional_contact_function.h`

The helper functions use:

- `Float`
- `Vector3`
- `Vector12`
- `Matrix12x12`
- `Eigen::Matrix<Float, ...>`

The call sites in:

- `src/backends/cuda_mixed/contact_system/contact_models/ipc_simplex_frictional_contact.cu`

do build `Vec3A = Eigen::Matrix<Alu, 3, 1>`, but they pass those values into helper signatures that are still double-typed.

Impact:

- on `path1/3/5/6/7/8`, the nominal `AluScalar=float` friction path can still get promoted to double inside the helper
- so friction contact is not actually as "implemented" as the scope doc currently implies

This is one of the best code-level explanations for why line-search / friction-heavy gains are not obvious.

### 5.7 `path8` does not restore double SpMV accumulation

This is subtle but important.

Files:

- `src/backends/cuda_mixed/linear_system/global_linear_system.cu`
- `src/backends/cuda_mixed/linear_system/spmv.h`
- `src/backends/cuda_mixed/linear_system/spmv.cu`

What happens:

- `GlobalLinearSystem::Impl::spmv()` takes scalar parameters of type `ActivePolicy::PcgIterScalar`
- for `path8`, that means `a` and `b` are `double`
- but `x` and `y` are still `PcgAuxScalar=float`
- overload resolution then selects:
  - `rbk_sym_spmv(double, CBCOOMatrixView<float,3>, CDenseVectorView<float>, double, DenseVectorView<float>)`
- that overload instantiates:
  - `detail::rbk_sym_spmv_impl<float, float>(...)`

So `path8` still uses float accumulation inside the SpMV kernel.

What `path8` really restores to double:

- the PCG iteration scalars at the API level
- some scalar algebra in `linear_pcg.cu`

What `path8` does not restore to double:

- the actual `A<float> * x<float> -> y<float>` SpMV accumulation kernel

This is not necessarily wrong, but it is narrower than the conceptual description "path8 restores iteration scalar precision" may suggest.

### 5.8 Energy/reporter interfaces are still double by design

This is widespread and probably intentional, but it is still a major reason why `Compute Energy` does not show strong speedups.

Representative files:

- `src/backends/cuda_mixed/contact_system/contact_reporter.h`
- `src/backends/cuda_mixed/contact_system/simplex_normal_contact.h`
- `src/backends/cuda_mixed/contact_system/simplex_frictional_contact.h`
- `src/backends/cuda_mixed/finite_element/fem_3d_constitution.h`
- `src/backends/cuda_mixed/finite_element/finite_element_elastics.h`
- `src/backends/cuda_mixed/affine_body/abd_line_search_reporter.h`
- `src/backends/cuda_mixed/affine_body/affine_body_constitution.h`
- `src/backends/cuda_mixed/affine_body/inter_affine_body_constitution_manager.h`
- `src/backends/cuda_mixed/inter_primitive_effect_system/inter_primitive_constitution_manager.h`

Common pattern:

- `muda::BufferView<Float> energies`
- constitutions compute in `AluScalar`
- then write `safe_cast<Float>(...)`

That means:

- the energy path still converges to double buffers
- line-search energy accumulation is not a fully fp32 pipeline
- benchmark expectations for `Compute Energy` should stay conservative until this changes

This also matches the observed benchmark result: `Compute Energy` and `Line Search` do not scale like assembly/solve.

## 6. What Is Not A Bug

Not every `Float` in `cuda_mixed` is a problem.

The following categories are expected to remain in `Float` unless the design changes:

- scene attributes and host-side configuration (`dt`, coefficients, thresholds)
- quality/debug dump paths
- state vectors that intentionally remain in baseline precision
- host-side bridge data for external force / articulation integration
- final energy outputs if line-search quality is intentionally kept conservative

This audit only flags places where:

- the path table suggests a typed mixed-precision domain
- but the implementation still inserts double-typed intermediates inside that domain

## 7. Prioritized Follow-Up Work

### Priority 1: fix ABD store-domain gaps

Target:

- `ABDLinearSubsystem::ComputeGradientHessianInfo`
- `body_id_to_shape_gradient`
- `body_id_to_shape_hessian`
- `body_id_to_kinetic_gradient`
- `body_id_to_kinetic_hessian`
- `diag_hessian`

Direction:

- change these to `StoreScalar`-typed or explicitly domain-typed buffers
- stop routing ABD constitutions through `Vector12/Matrix12x12` double intermediates

Expected payoff:

- clearer realization of `path2/3/5/6/7/8`
- better ABD assembly/build scaling
- more faithful store-domain benchmark behavior

### Priority 2: make `ABDJacobi` typed

Target:

- `abd_jacobi_matrix.h/.cu`
- `details/abd_jacobi_matrix.inl`

Expected payoff:

- cleaner ABD contact assembly path
- fewer hidden double promotions

### Priority 3: template the friction helper path

Target:

- `ipc_simplex_frictional_contact_function.h`

Direction:

- replace `Float`/`Vector3`/`Vector12`/`Matrix12x12` helper signatures with scalar-templated forms
- keep call sites on `AluScalar`

Expected payoff:

- friction contact finally follows the same ALU policy as normal contact
- better chance of visible gains in friction-heavy and line-search-heavy scenes

### Priority 4: decide whether `Compute Energy` should remain double

This is a design question, not just an implementation bug.

Options:

- keep energy outputs in double for robustness
- or introduce a separate policy for line-search energy buffers / reductions

Without making this decision, `Compute Energy` will likely continue to show weak speedups.

### Priority 5: clarify the `path8` contract

The code should explicitly state what `path8` restores and what it does not restore.

At minimum the docs should say:

- `path8` restores `alpha/beta/rz`-level scalar precision
- but not necessarily double accumulation inside every solver kernel

## 8. Bottom Line

The benchmark story and the code story are aligned:

- `path8` is currently the best overall path.
- `path3` is the best assembly path.
- `Build + Solve` gains are real.
- `Line Search / Compute Energy` gains are weak because those pipelines are not fully mixed-precision yet.

And the code audit answers the practical question directly:

- no, not every place that should already be fp32 is really fp32 in effect
- the biggest remaining gaps are in:
  - ABD local buffers
  - ABDJacobi helpers
  - friction contact helper code
  - energy/reporter interfaces
  - the exact interpretation of `path8` inside SpMV

That is why some benchmark improvements are smaller than the path table alone would lead you to expect.
