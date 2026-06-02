# Architecture

This document is the stable architecture for RCC bonded point-triangle acceleration. It describes intended ownership and data flow; date-specific evidence lives in [the journal](./development/rcc_adhesion_acceleration_journal.md).

## Motivation

RCC adhesion currently evaluates long-lived point-triangle pairs through the same active contact pipeline used for transient contact. Stable sticky pairs repeatedly pay for collision detection, CCD filtering, contact visibility, friction candidate recording, RCC beta matching, and RCC adhesion assembly even when their topology and adhesion state are unlikely to change.

The performance thesis is to classify stable RCC PT pairs, remove them from the contact/RCC hot path, and replace them with a bonded virtual tetrahedron over the same four global vertices until a release gate fails.

## Core Thesis

Stable adhesive contact should become a complement energy with explicit ownership, not a hidden modification of contact detection. The design is valid only if the locked pair is absent from contact/RCC views and present in exactly one bonded virtual-tet reporter for the same step.

## Baseline Data Flow

```text
AdvanceIPC frame
  -> record previous positions
  -> DCD emits active simplex pairs
  -> record_friction_candidates copies PTs() to friction_PTs()
  -> RCC Phase B matches friction_PTs() against m_prev_keys_PT
  -> normal contact, friction, RCC adhesion assemble from trajectory-filter views
  -> line search calls filter_toi()
  -> RCCBetaEvolutionTimeIntegrator runs Phase A beta update
  -> m_prev_keys_PT / m_prev_beta_PT are sorted for the next step
```

Relevant current anchors:

| Current Anchor | Role |
| --- | --- |
| `src/backends/cuda/engine/advance_ipc.cu` | Frame order, DCD, friction candidate recording, line-search CCD |
| `src/backends/cuda/collision_detection/simplex_trajectory_filter.cu` | `record_friction_candidates()` copies active PTs to friction PTs |
| `src/backends/cuda/contact_system/contact_models/ipc_simplex_rcc_adhesive_contact.cu` | RCC PT beta buffers, Phase A/Phase B, beta persistence |
| `src/backends/cuda/collision_detection/filters/*simplex_trajectory_filter.cu` | PT CCD broadphase and active-pair emission |
| `src/backends/cuda/inter_primitive_effect_system/constitutions/soft_vertex_triangle_stitch.cu` | Existing static vertex-triangle rest-shape and thickness reference |
| `src/backends/cuda/affine_body/constitutions/ortho_potential.cu` | Existing ABD OrthoPotential energy reference |
| `src/backends/cuda/affine_body/constitutions/arap.cu` | Existing ABD ARAP energy reference |

## Target Data Flow

```text
End of step
  -> RCC beta evolves on active PT pairs
  -> bonded-PT classifier tests beta, age, sticky side, gap, slip, rest shape
  -> RCCBondedPTState publishes sorted locked_keys and oriented locked_topos

Next detection/filtering
  -> simplex filter forms PT candidate ids
  -> shared locked-key lookup tests membership before PT CCD broadphase
  -> locked PTs are not emitted to PTs()
  -> record_friction_candidates cannot copy locked PTs to friction_PTs()

Newton assembly
  -> normal contact/friction/RCC assemble only unlocked pairs
  -> bonded virtual-tet reporter assembles high-kappa ABD-style complement energy for locked_topos, Dm_inv, and rest_volume

Release/update
  -> release gate records reason flags
  -> released pairs carry beta back into RCC persistence
  -> still-locked pairs update age and diagnostics
```

## Project Structure

```text
docs/
  roadmap.md
  architecture.md
  conventions.md
  rcc_adhesion_acceleration.md
  development/
    rcc_adhesion_acceleration_journal.md
scripts/
  run_rcc_adhesion_acceleration_all_gates.py
  run_rcc_adhesion_acceleration_gates.py
  build_docs.py
src/backends/cuda/
  collision_detection/
  contact_system/
  inter_primitive_effect_system/
apps/tests/
  core/
  backends/cuda/
  sim_case/
```

The `apps/tests` locations hold the current deterministic state, oracle, CUDA bridge, lookup, filter, owner, reporter, and scene gates. Source scans remain boundary checks, not numeric proof.

## Ownership And Boundaries

| Component | Owns | Must Not Own |
| --- | --- | --- |
| `RCCBondedPTState` | Locked membership keys, oriented topologies, beta, age, release flags, rest-shape metrics | Contact force assembly, BVH traversal, frontend geometry objects |
| Simplex trajectory filters | Candidate generation, PT CCD broadphase, active-pair output | Beta evolution, material parameters, virtual-tet energy |
| `IPCSimplexRCCAdhesiveContact` | RCC beta evolution and persistence for active/unlocked PT pairs | Dynamic tet Hessian assembly, filter-specific skip logic |
| Bonded virtual-tet reporter | Complement energy, gradient, Hessian for locked topologies | RCC beta law, broadphase decisions, contact-component accounting |
| Global dynamic topology manager | Aggregation and scattering of complement energy | Classification policy or release policy |
| `RCCBondedPTStateAccessorFeature` | Frontend snapshots of locked state and counters | Physics decisions or implicit synchronization outside explicit query points |
| Bench/report layer | Timers and counters | Physics decisions |

## Runtime Data Model

Use two keys for different jobs:

| Data | Purpose | Required Property |
| --- | --- | --- |
| `locked_keys` | Membership lookup in filters and beta matching | Sorted by key, one entry per locked PT ownership record |
| `locked_topos` | Energy assembly topology | Oriented `(point, tri0, tri1, tri2)` and permuted with `locked_keys` |
| `locked_beta` | Carry adhesion state through lock/release | Matched to RCC beta convention |
| `locked_age` | Enforce minimum stable lifetime before lock/reuse | Incremented only after accepted steps |
| `Dm_inv` | Virtual-tet rest-shape inverse | Finite, conditioned, derived from rest positions |
| `rest_volume` | ABD-style virtual-tet volume scale | Positive and above configured minimum |
| `release_flags` | Device-generated release reasons | Stable enum or bit mask for reports/tests |

The sorted membership key can match current RCC persistence behavior, but it is not enough for energy. Energy must use oriented topology because orientation controls `Dm`, normal direction, and rest volume sign handling.

## Frame Lifecycle Contract

| Stage | Locked Pair Requirement | Test Or Report |
| --- | --- | --- |
| Classification | Candidate accepted only after all lock gates pass | State fixture and candidate/rejected counters |
| DCD/PT emission | Locked PT is skipped before PT CCD broadphase | Filter contract and `filter_skip_count` |
| Friction candidate recording | Locked PT cannot enter `friction_PTs()` | Contract test reading trajectory-filter views |
| RCC Phase B | Released pairs can receive carried beta | Beta carry fixture |
| Newton assembly | Locked PT appears only in bonded reporter | Duplicate-suppressed counter and E/G/H oracle |
| End-of-step update | Release reasons and beta state are recorded | Lifecycle scene gate |

## Virtual Tet Rest Shape And Energy

The dynamic reporter uses `SoftVertexTriangleStitch` only for the PT rest-shape convention:

1. Build `Dm = [x1 - x0, x2 - x0, x3 - x0]` from rest positions.
2. If point-plane rest distance is below `min_separate_distance`, offset the rest point along the triangle normal before computing `Dm_inv`.
3. Store `Dm_inv` and positive `rest_volume`.

The production bonded energy must be ABD-style, not Stable Neo-Hookean. A locked pair is removed from CCD/contact/RCC, so the replacement energy is responsible for making the four vertices behave like a stiff bonded patch during that step.

For the same four real vertex positions, build:

$$
D_s = [x_1 - x_0, x_2 - x_0, x_3 - x_0], \qquad F = D_s D_m^{-1}.
$$

Treat `F` as the virtual affine transform. The default target model is ABD OrthoPotential:

$$
E = \kappa \, V_0 \, \Delta t^2 \, \|F F^T - I\|_F^2,
$$

where `V0 = rest_volume` and `kappa = rcc_bonded_pt_kappa`. `abd_arap` may be supported as an alternate model only if it has the same CPU and GPU E/G/H oracle coverage. The initial target stiffness is `kappa >= 1e8` in scene units, with higher values allowed when solver conditioning gates pass.

Implementation rules:

- Do not instantiate a real frontend ABD body for each lock; assemble the ABD-style energy directly into the same four vertex DOFs through `dF/dx`.
- Keep `rcc_bonded_pt_energy_model` explicit; default target model is `abd_ortho`.
- Keep bonded PT acceleration default-off until high-kappa ABD oracle, no-penetration scene, release, and benchmark gates pass.
- Apply SPD projection in the Hessian path.
- Do not reintroduce the old Stable Neo-Hookean `rcc_bonded_pt_mu/lambda` reporter as a production path; it did not justify skipping CCD and has been replaced by ABD-style energy.

## Relationship To Existing Systems

- RCC adhesion: provides beta evolution, sticky-side semantics, PT-only adhesion scope, and persistence keys.
- Simplex trajectory filters: provide the only high-performance place to skip locked PT work before CCD broadphase.
- Inter-primitive constitutions: provide complement-energy ownership and the existing SVTS rest-shape/thickness convention.
- Affine body constitutions: provide the ABD-style high-stiffness shape energy formulas that the bonded virtual tet should mirror over `F = Ds Dm_inv`.
- Dynamic topology manager: aggregates complement reporter output into global energy, gradient, and Hessian.
- Contact adaptive strategies: should not count bonded virtual tets as contact unless an explicit diagnostic adds comparison accounting.

## Failure Modes

| Failure | Symptom | Required Guard |
| --- | --- | --- |
| Late skip | Contact/RCC still pays CCD and assembly cost | Filter contract requires skip before PT CCD broadphase |
| Duplicate ownership | Pair is both contact/RCC and bonded tet | Duplicate counter and contract failure |
| Key/topology mismatch | Wrong four vertices or wrong orientation assembled | State fixture with key/topology permutation check |
| Singular rest shape | Large `Dm_inv`, unstable Hessian, solver spikes | Determinant, area, volume, and normal validity gates |
| Wrong replacement energy | Locked pair skips CCD but uses soft SNH/prototype energy | ABD-style high-kappa oracle and no-penetration scene gate |
| Release beta pop | Released pair behaves like a new contact | Beta carry fixture and scene report |
| False speed claim | Total frame time hides moved cost | Benchmark protocol requires subsystem timers |

## External References

- [RCC Adhesion specification](./specification/contact_models/rcc_adhesion.md)
- [Soft Vertex Triangle Stitch specification](./specification/constitutions/soft_vertex_triangle_stitch.md)
- [Affine Body specification](./specification/constitutions/affine_body.md)
- [CUDA backend development notes](./development/backend_cuda/index.md)
