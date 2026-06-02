# RCC Adhesion Acceleration

This is the focused subsystem note for replacing stable RCC point-triangle adhesion pairs with bonded virtual tetrahedra. The roadmap says what is current; this page says what the subsystem is supposed to do.

## Scope

In scope:

- RCC adhesive point-triangle pairs only.
- Stable pairs whose beta and motion history make contact topology unlikely to change during the next step.
- CUDA backend trajectory filtering, RCC beta persistence, and complement-energy assembly.
- Dynamic runtime state owned by the backend.

Out of scope for the first production milestone:

- PE, PP, or EE adhesion locking.
- Vertex-half-plane adhesion locking.
- Frontend `SoftVertexTriangleStitch` geometry rebuilds for transient pairs.
- Default-on behavior before correctness and benchmark gates pass.

## Algorithm Summary

```text
For each end-of-step RCC PT pair:
  compute stable membership key
  evaluate lock gate
  if lock gate passes:
    build oriented virtual-tet topology
    build conditioned rest shape using SVTS-compatible logic
    store beta, age, Dm_inv, rest volume

For each next-step PT candidate in simplex filters:
  compute membership key
  if key is locked:
    skip PT CCD broadphase and active-pair emission
  else:
    keep current contact pipeline behavior

During assembly:
  contact/RCC sees only unlocked pairs
  bonded reporter assembles one high-kappa ABD-style complement energy per locked topology

During release/update:
  evaluate release gate
  record release reason
  carry beta back to RCC persistence
```

## Pair Keys

Use two representations:

| Representation | Contents | Use |
| --- | --- | --- |
| Membership key | Point id kept distinct, triangle global vertex ids sorted, then hashed exactly like RCC PT beta persistence | Filter lookup and beta matching |
| Oriented topology | `(point, tri0, tri1, tri2)` global vertex ids | Virtual-tet rest shape and energy assembly |

The membership key can answer "is this four-vertex set locked?" It cannot answer "what oriented tet should be assembled?" Therefore every lock stores both.

## Lock Gate

A candidate can lock only if every condition passes.

| Gate | Data Needed | Rejection Counter |
| --- | --- | --- |
| Feature enabled | Config | `rcc_bonded_pt_disabled_count` |
| PT adhesion scope | Current RCC PT pair | `rcc_bonded_pt_rejected_type_count` |
| Beta threshold | `beta >= rcc_bonded_pt_beta_lock_threshold` | `rcc_bonded_pt_rejected_beta_count` |
| Minimum age | `locked_age` or candidate age | `rcc_bonded_pt_rejected_age_count` |
| Sticky-side consistency | Lagged normals and `rcc_sticky_sign` | `rcc_bonded_pt_rejected_sticky_count` |
| Normal gap band | End-of-step positions | `rcc_bonded_pt_rejected_gap_count` |
| Tangential slip band | Lagged closest coordinates/basis | `rcc_bonded_pt_rejected_slip_count` |
| Rest-shape quality | Triangle area, normal, `det(Dm)`, rest volume | `rcc_bonded_pt_rejected_degenerate_count` |
| Contact policy | Contact tabular enable/disable | `rcc_bonded_pt_rejected_policy_count` |

The conservative rule is: reject on missing data. A missing sticky-side normal, invalid triangle normal, or unavailable beta carry is not a reason to guess.

## Rest Shape

The rest-shape construction follows the existing `SoftVertexTriangleStitch` convention:

1. Use rest positions for point `x0` and triangle vertices `x1`, `x2`, `x3`.
2. Compute triangle normal and signed point-plane distance.
3. If the distance magnitude is below `rcc_bonded_pt_min_separate_distance`, offset the rest point along the triangle normal to create finite thickness.
4. Build `Dm = [x1 - x0, x2 - x0, x3 - x0]`.
5. Reject if triangle area, `abs(det(Dm))`, or rest volume is below threshold.
6. Store `Dm_inv` and positive `rest_volume`.

`min_separate_distance` is a numerical thickness, not a cosmetic gap. It must scale with scene/mesh units and should not be set arbitrarily close to zero.

## Energy Model

The runtime bonded energy is not the same thing as `SoftVertexTriangleStitch` energy. SVTS supplies the PT rest-shape convention above; production bonded PT acceleration must use an ABD-style high-stiffness shape energy.

For one locked topology `(x0, x1, x2, x3)`:

$$
D_s = [x_1 - x_0, x_2 - x_0, x_3 - x_0], \qquad F = D_s D_m^{-1}.
$$

`F` is treated as the affine transform of a virtual tetrahedron whose DOFs are still the four real vertex positions. The target default model is ABD OrthoPotential:

$$
E_{\text{abd\_ortho}} = \kappa \, V_0 \, \Delta t^2 \, \|F F^T - I\|_F^2.
$$

`V0` is `rest_volume`, and `kappa` comes from `rcc_bonded_pt_kappa`. The first production gates should use `kappa >= 1e8` in scene units. `abd_arap` may be added as an alternate model:

$$
E_{\text{abd\_arap}} = \kappa \, V_0 \, \Delta t^2 \, \|F - R\|_F^2,
$$

where `R` is the rotation from the polar decomposition of `F`.

The reporter should assemble this ABD-style energy directly into the 12 vertex-position DOFs through `dF/dx`. It should not create transient frontend ABD objects or extra affine DOFs. The current Stable Neo-Hookean reporter/oracle is a prototype path only; it proves GPU/CPU consistency of that prototype but does not satisfy the production requirement for skipping CCD.

## Release Gate

A locked pair releases when any condition fails.

| Release Reason | Condition | Required Follow-Up |
| --- | --- | --- |
| `strain` | Stretch, shear, or volume distortion exceeds threshold | Return to RCC/contact next detection |
| `gap` | Predicted normal gap exceeds release distance | Carry beta and emit release counter |
| `slip` | Tangential slip exceeds release distance | Carry beta and let RCC tangential adhesion handle sliding |
| `flip` | Triangle orientation or virtual tet quality becomes invalid | Release before assembly if detected early |
| `sticky_side` | Sticky-side gate no longer passes | Return to normal contact/RCC policy |
| `policy` | Contact tabular or scene policy disables the pair | Remove lock and report policy release |

Release is not failure. It is the intended fallback when the bonded approximation stops matching contact-like adhesion.

## Assembly Contract

The bonded virtual-tet reporter:

- Reports complement energy, not contact energy.
- Assembles energy, gradient, and Hessian over global vertex ids.
- Uses ABD-style high-kappa shape energy over `F = Ds Dm_inv`; default target model is `abd_ortho`.
- Scales by `rest_volume * dt^2`, matching the backend reporter convention.
- Applies SPD projection in the Hessian path.
- Emits report fields for assembled count and timing.

The reporter must not:

- Update beta.
- Run broadphase or CCD.
- Reinsert locked pairs into `PTs()` or `friction_PTs()`.
- Count as contact for adaptive contact-parameter strategies unless a separate diagnostic explicitly asks for that comparison.

## Implementation Targets

| Area | Target Files | Notes |
| --- | --- | --- |
| Host state owner | `include/uipc/core/rcc_bonded_pt_state.h`, `src/core/core/rcc_bonded_pt_state.cpp` | Minimum host contract, counters, release flags, `Dm_inv`, rest volume, and deterministic fixtures |
| CUDA state bridge | `src/backends/cuda/contact_system/rcc_bonded_pt_state_bridge.*` | Owns device buffers for locked keys, topologies, beta, age, release flags, `Dm_inv`, rest volume, and host counter snapshots; can be replaced from compact sorted device entries without host roundtrip while rest-shape payloads stay in SoA buffers |
| CUDA owner | `src/backends/cuda/contact_system/rcc_bonded_pt_system.*` | Owns the bridge at runtime, honors `rcc_bonded_pt_enabled`, feeds sorted locked keys to `SimplexTrajectoryFilter`, syncs common active-filter skip counters, and compacts RCC Phase A high-beta PTs into locked state after live rest-shape construction and degeneracy rejection |
| Filter helper | `src/backends/cuda/contact_system/rcc_bonded_pt_lookup.h` | Shared device helper for RCC PT key construction, sorted membership lower-bound, and lock lookup |
| Common active filter | `src/backends/cuda/collision_detection/simplex_trajectory_filter.*` | Can compact locked PTs out of active `PTs()` and `friction_PTs()` when supplied sorted locked keys; default is no-op |
| Filter backends | `src/backends/cuda/collision_detection/filters/*simplex_trajectory_filter.cu` | Planned: skip before PT CCD broadphase |
| Reporter | `src/backends/cuda/inter_primitive_effect_system/...` or contact-adjacent complement reporter | Dynamic, no frontend geometry rebuild; current SNH implementation is prototype and must be replaced by ABD-style high-kappa math |
| RCC integration | `ipc_simplex_rcc_adhesive_contact.cu` | Phase A beta evolution now optionally calls the bonded-PT producer with current positions for lock-time rest-shape construction; release and production ABD virtual-tet assembly remain planned |
| Tests | `apps/tests/core`, `apps/tests/backends/cuda`, `apps/tests/sim_case` | Follow the test matrix in conventions |
| Benchmarks | `scripts/bench_rcc_adhesion_acceleration.py` | Planned after timers/counters exist |

## Oracles

Required oracles before production use:

| Oracle | Input | Expected Output |
| --- | --- | --- |
| State oracle | Two PT pairs with deterministic beta/age/release flags and rest-shape payloads | Implemented by `uipc_test_core "[rcc_bonded_pt][state]"`: one lock stays active, one release is extracted, stable key/topology/beta/age/release/rest-shape permutation is preserved |
| Rest-shape oracle | Point near triangle plane with known `min_separate_distance` | Implemented by `uipc_test_core "[rcc_bonded_pt][oracle][rest_shape]"`: `Dm_inv`, positive rest volume, point offset, orientation swap, and degenerate-triangle rejection match SVTS rules |
| Prototype SNH energy oracle | Single virtual tet with deterministic deformation | Implemented by `uipc_test_core "[rcc_bonded_pt][oracle][energy]"`: CPU energy, gradient, and Hessian match center-difference checks for the prototype only |
| ABD-style production energy oracle | Single virtual tet with deterministic deformation and high stiffness | Planned: `uipc_test_core "[rcc_bonded_pt][oracle][abd_energy]"` proves `abd_ortho` E/G/H, finite differences, SPD projection, and `kappa >= 1e8` conditioning |
| CUDA state bridge oracle | Host state with pending and extracted release paths | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][backend_state]"`: device buffers preserve key/topology/beta/age/release/rest-shape alignment and counters roundtrip through upload/download |
| CUDA lookup oracle | One locked key, one triangle permutation, two misses, and an empty locked set | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][lookup]"`: lookup uses the existing RCC PT key, treats triangle permutations as the same membership key, keeps the point id distinct, and misses cleanly |
| Common active-filter oracle | Synthetic active PT list with two locked and two unlocked pairs | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][filter]"`: locked pairs are compacted out of `SimplexTrajectoryFilter::PTs()` and stay absent after `record_friction_candidates()` copies to `friction_PTs()` |
| CUDA owner oracle | Host locked state plus synthetic filter active view | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][owner]"`: owner uploads state, feeds sorted keys, syncs filter-skip counters, and downloads counters aligned with active locks |
| Beta/rest producer oracle | Existing locks plus a RCC PT beta snapshot with one refresh, one carry, one new lock, one duplicate, one low-beta reject, and one degenerate high-beta reject | Implemented by `uipc_test_backend_cuda "[rcc_bonded_pt][owner][producer]"`: device-side producer carries old locks and their rest-shape payloads, refreshes high-beta candidates, increments age, suppresses duplicate keys, builds SVTS-compatible `Dm_inv/rest_volume` for fresh locks, counts degenerate rejects, and leaves low-beta or degenerate candidates unlocked |
| ABD-style reporter oracle | One locked pair with known rest shape and high stiffness | Planned: `uipc_test_backend_cuda "[rcc_bonded_pt][reporter][abd_oracle]"` proves CUDA E/G/H matches the ABD CPU oracle |
| CUDA BVH/radix-sort regression | Bunny sanity mesh through the existing GPU sanity checker | Implemented by `uipc_test_backend_cuda "gpu_sanity_check" -c "bunny"` and `scripts/run_rcc_adhesion_acceleration_cuda_gates.py`: bonded-PT device layout changes must not destabilize `SimplicialSurfaceDistanceCheck` or `InfoStacklessBVH` |
| Pre-CCD filter ownership oracle | One locked key and one unlocked key in every concrete filter fixture | Planned: locked absent from candidate/TOI/contact views, unlocked unchanged |

## Scene Gate

The first bonded-mode real-scene gate is `pt_lift_release`: a PT-rich adhesion fixture promoted from visual demos to pass/fail assertions. The 2026-06-01 coarse cube-cube `rcc_adhesion_pick_and_lift` probe is not accepted as the seed because adhesion-on and adhesion-off runs produced matching bottom-cube heights even though RCC adhesion assembly was active. A point-dense variant works: a subdivided contact-face or subdivided tet cube can produce stable adhesion-on/off lift. Prefer a production fixture seeded from `rcc_adhesion_cloth_peel`, `python/examples/rcc_adhesive_oriented_cloth_demo.py`, or a point-dense variant of `python/examples/rcc_adhesive_pick_and_lift_demo.py`.

Current legacy RCC gates now cover the non-accelerated behavior that bonded PT must preserve:

| Gate | Command | What It Proves |
| --- | --- | --- |
| Python subdivided cube and cube-cloth fixtures | `python/.venv/bin/python -m pytest python/tests/sim_case/test_rcc_adhesive_lift_release.py -q` | Existing RCC adhesion can lift during hold and release/separate during pull in PT-rich fixtures |
| Native C++ sim-case fixtures | `build/cuda_mixed_fused_pcg/RelWithDebInfo/bin/uipc_test_sim_case "[rcc_adhesion][gate]" -r compact` | The same cube-cube and cube-cloth behaviors are available as CUDA sim-case gates |

These gates are legacy behavior baselines, not bonded-PT proof. They do not observe locked keys, filter skips, duplicate ownership, release reason flags, or beta carry through bonded state because those fields do not exist yet.

| Phase | Scene Action | Required Observation |
| --- | --- | --- |
| Adhesion-off baseline | Disable RCC adhesive contact while keeping the driver, constraints, and gravity identical | The adhered body or patch does not follow the driver beyond the declared tolerance |
| Legacy RCC baseline | Run current RCC adhesion with bonded PT disabled, including the current Python and C++ legacy scene gates | The fixture shows the reference lift/release behavior for comparison |
| Press/hold | In bonded mode, bring the driver into a PT-rich contact patch under gravity | At least one PT lock is reported, no duplicate ownership is reported |
| Lift | Raise the driver below release thresholds | Adhered geometry follows the legacy RCC baseline within the declared tolerance, locks are reused, and the ABD-style energy prevents visible penetration while CCD is skipped |
| Forced pull | Increase normal gap or tangential slip past release threshold | Release counter and reason flag are reported, beta is carried back, and the adhered geometry separates or falls |

The gate must read simulation state or report fields. Writing OBJ sequences is useful for debugging, but it is not sufficient evidence.

## Reports

The host state contract, CUDA state bridge, CUDA owner, and beta-threshold producer now carry matching `RCCBondedPTCounters` fields, and `SimplexTrajectoryFilter` has a local `rcc_bonded_pt_filter_skipped_count()` for the common active compact path. Before bonded scene gates can claim ownership correctness, the live CUDA pipeline must add release/rest-shape accounting and expose the same fields through reports or feature accessors.

Minimum backend report fields before scene gates:

| Field | Meaning |
| --- | --- |
| `rcc_bonded_pt_candidate_count` | Candidate PT pairs considered for lock |
| `rcc_bonded_pt_locked_count` | Active bonded PT locks |
| `rcc_bonded_pt_released_count` | Locks released this step |
| `rcc_bonded_pt_filter_skip_count` | PT candidates skipped before CCD/contact |
| `rcc_bonded_pt_duplicate_suppressed_count` | Duplicate ownership prevented |
| `rcc_bonded_pt_rejected_degenerate_count` | Rest-shape quality rejection |
| `rcc_bonded_pt_energy_model` | Reported model, expected production value `abd_ortho` unless the test explicitly chooses another model |
| `rcc_bonded_pt_kappa` | Reported ABD-style stiffness used by the bonded reporter |
| `rcc_bonded_pt_assembly_ms` | Bonded virtual-tet assembly time |

Scene and benchmark gates must fail if these fields are missing.

## Performance Claim Requirements

A valid speed claim must compare:

1. Baseline RCC adhesion with bonded PT disabled.
2. Bonded PT enabled with the same scene, seed, frame range, solver settings, binary, and GPU.
3. Cold setup timing for initial lock build.
4. Cache-hot timing for stable locked sets.
5. Churn timing for controlled lock/release turnover.
6. End-to-end timing plus subsystem timers.

The expected first win should appear in PT CCD/filter and contact/RCC assembly time. If only total frame time moves, the benchmark is not diagnostic enough.
