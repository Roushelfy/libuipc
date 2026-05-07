# SOCU Mixed Structured Direct Solver

This document records the current split mixed-backend SOCU integration.
Historical checkpoint notes have been retired from this file so the documented
behavior matches the production path.

## Scope

`cuda_mixed` is the restored FullSparse/fused-PCG baseline backend.
`cuda_mixed_socu` is the SOCU-owned backend. `socu_approx` is a
structured-band direct direction solver for `cuda_mixed_socu`. Local linear
build providers assemble directly into a `StructuredAssemblySink`; the solver
does not build, filter, or redistribute a full Hessian triplet for the SOCU
path.

The assembled matrix contains the diagonal block band and the first
off-diagonal block band accepted by `socu_native`. If a scene's true Hessian is
fully represented by that band, the solve is exact for the assembled Newton
system. If contributions fall outside the band, they are recorded and dropped
from the SOCU matrix, and the solve is explicitly a structured-band
approximation.

The default structured scope is:

```text
linear_system/socu_approx/structured_scope = multi_provider
```

`single_provider` remains available for strict provider isolation checks.

## Runtime Path

The final solve path is:

1. A `StructuredChainProvider` owns the block-chain layout and old-to-chain
   mappings.
2. Local ABD, FEM, ABD-FEM coupling, joint, constraint, and contact reporters
   write Hessian and RHS contributions directly into the structured sink when
   the selected solver requests SOCU assembly.
3. The runtime uploads validated mappings once, creates the `socu_native` plan
   during solver build, and reuses the plan in Newton solves.
4. Each solve assembles the structured band, calls `socu_native`
   `factor_and_solve`, runs lightweight direction validation, and scatters the
   direction back to the global vector.

The fused PCG path in `cuda_mixed` keeps using the normal full sparse assembly
route. Shared local Hessian evaluation code may feed either sink, but the SOCU
path in `cuda_mixed_socu` writes its structured destination directly.

Runtime Hessian-based RCM reordering is optional and disabled by default. When
enabled, a matching frame first runs a graph-only structured probe, records the
atom-pair graph from structured Hessian writes, including off-band writes before
they are dropped from the structured matrix, reorders the merged graph on the
CPU, and installs the new structured runtime/plan before the final structured
assembly and solve. A runtime reorder failure leaves the previous ordering
active and is reported as a diagnostic.

Current-frame probing must be treated as a per-linear-build concern, not only a
per-frame concern. Contact sets can change after the first Newton solve in a
frame; for example, `wrecking_ball` frame 13 has only PH contacts in Newton
iteration 0, then activates EE/PP simplex contacts in Newton iteration 1. If
runtime reorder probes only once at the start of the frame, the later EE/PP
body-pair graph is absent from the ordering, strong off-band blocks are dropped,
and the SOCU direct direction can become invalid even though the current-frame
ordering report for the first solve shows `off_band_ratio = 0`.

## Contact Structured Writes

Ordinary IPC contact assembly keeps the pre-SOCU shape: gradient/Hessian kernels
write the original doublet and triplet buffers and do not capture the structured
contact sink. SOCU contact Hessian assembly is split into separate structured
translation units so the ordinary kernels do not inherit SOCU register pressure
or template instantiations.

The production structured path is deliberately direct:

```text
H = compute_contact_hessian(...)
structured_sink.write_hessian_half(stencil, H)
```

Simplex normal and frictional PT/EE/PE/PP contacts, plus vertex-half-plane
normal/frictional PH contacts, use this compute-and-write path. The structured
sink maps the current global vertex to FEM/ABD old DoFs, projects ABD blocks,
classifies diag/first-offdiag/off-band writes, and records runtime-reorder graph
edges when the runtime ordering collector is active.

The previous reorder-level vertex-slot table, contact-set write plans, planned
scatter path, and planned-direct-write debug switches are not part of the
current runtime. They can be revisited only as a separate optimization after
the direct structured path is stable on `wrecking_ball socu_rt1 --frames 20`
through `>>> Begin Frame: 16`, with any later known direction-validation NaN
tracked separately.

The ordinary `fused_pcg` path remains the contact assembly baseline. Any SOCU
contact split must keep `wrecking_ball fused_pcg --frames 20` passing through
frame 20; a fused-PCG failure before frame 20 is a refactor blocker, not an
acceptable SOCU validation limitation.

## Failure And Report Semantics

Hard failures are limited to invalid or unsupported states:

- missing or malformed ordering data
- unsupported block size
- incomplete DoF coverage or invalid mapping
- unsupported precision contract
- unavailable or failing `socu_native` runtime
- non-finite, non-descent, or residual-invalid directions
- configured line-search rejection fallback

The following quality fields are diagnostics only and do not reject a solve:

- `min_block_utilization`
- `min_near_band_ratio`
- `max_off_band_ratio`
- `max_off_band_drop_norm_ratio`

Ordering quality, low block utilization, and runtime off-band contributions are
written into the report. Runtime off-band contributions do not disable the
direction; the report status explains that the solve continued with the
in-band structured matrix.

## Configuration

Typical strict structured solve configuration:

```json
{
  "linear_system": {
    "solver": "socu_approx",
    "socu_approx": {
      "structured_scope": "multi_provider",
      "ordering_source": "init_time",
      "ordering_orderer": "rcm",
      "ordering_block_size": "64",
      "damping_shift": 0.0,
      "runtime_reorder_frame_interval": 0,
      "runtime_reorder_edge_capacity": 0,
      "runtime_reorder_graph_source": "topology",
      "contact_offband_policy": "drop"
    }
  }
}
```

`damping_shift = 0.0`, `ordering_block_size = "64"`, and
`contact_offband_policy = "drop"` are the defaults.

`ordering_source` only supports `init_time`; external ordering report mode has
been removed from the solver path. `ordering_orderer` only supports `rcm` in
the runtime solver. `generated_ordering_report` may still be set to write the
init-time ordering diagnostics for inspection.

`runtime_reorder_frame_interval = 0` means init-time ordering only. A positive
interval enables runtime probe checks on structured linear builds whose frame
satisfies `frame % interval == 0`. The probe is contact-signature aware: if the
current contact-set signature matches the last installed runtime ordering, the
probe is skipped and the previous ordering remains active; if the signature
changes later in the same frame, such as after a Newton/contact-set update, a
new probe may run and install a new ordering before the final structured
assembly. Runtime reorder should therefore be treated as a per-linear-build and
per-contact-signature mechanism, not as exactly one rebuild per frame.
`runtime_reorder_graph_source` selects the graph source: `topology` uses cached
base topology plus current contact topology with unit weights,
`contact_hessian` keeps base topology at weight 1 and weights current contacts
by accumulated absolute contact Hessian contribution, and `full_hessian` uses a
graph-only full Hessian probe for diagnostics. Experimental graph sources are
`contact_weight_approx`, `full_weight_approx`, and `full_hessian_cached`.
Approximate sources use cheap contact coefficients for ordering weights. The
cached full Hessian source replays the contact Hessian half-blocks computed by
the graph probe for the immediately following structured contact assembly.
Runtime reorder preserves the currently installed block size. The collector
capacity defaults to an automatic value; setting
`runtime_reorder_edge_capacity > 0` overrides it. Collector overflow, empty
graphs, invalid mappings, or
`socu_native` plan creation failures do not abort the solve; they are reported
and the previous ordering remains active.

The current wrecking-ball comparison script exposes baseline runtime variants
such as `socu_rt1`, diagnostics such as `socu_rt1_contact_hessian` and
`socu_rt1_full_hessian`, and the corrected opt-in stability/performance
variants used in the current journal:

```text
socu_init_topology_diag_lump
socu_init_contact_hessian_diag_lump
socu_init_full_hessian_diag_lump
socu_rt{20,25,50}_topology_diag_lump
socu_rt{20,50}_contact_hessian_diag_lump
```

Broader sweeps can still inject temporary variants from a harness, but variants
listed as current regression commands must be present in the script.
The script-level suite names are intentionally explicit: `--variant quick` is
the legacy lightweight smoke sweep, `--variant regression` is the current
diag-lump regression sweep, and `--variant all` runs every statically defined
variant in the script.

Off-band contact policies are currently split into default and opt-in behavior:

- `drop` is the default and preserves the original structured-band
  approximation: in-band scalars are written and off-band scalars are dropped.
- `diag` is an implemented opt-in diagnostic policy. It computes the exact
  contact Hessian, writes it unchanged when the whole contact stencil fits in
  the current SOCU band, and otherwise replaces the whole partial off-band
  stencil with exact per-vertex diagonal blocks.
- `diag_lump` is an implemented opt-in stability/performance policy and is the
  current best measured off-band policy on wrecking ball. It uses the same
  stencil-level decision as `diag`, but replaces partial off-band stencils with
  nonnegative physical-coordinate lumped diagonal compensation.

The default remains `drop`. `diag_lump` is a measured opt-in candidate, not a
new default. Benchmark scripts that set `contact_offband_policy` must ensure the
scene config path exists; otherwise the solver falls back to `drop`, which was
the source of an earlier invalid measurement.

Future matrix-approximation policies must remain opt-in:

1. `approx_diag`: use cheap normal/frictional contact weights to write only
   per-vertex diagonal blocks in the final structured matrix. This is a matrix
   approximation, not merely a runtime ordering graph approximation.
2. `hybrid`: use exact structured contact writes for fully in-band stencils and
   approximate diagonal writes for stencils that would otherwise be partially
   dropped.
3. Approximate cache: only consider this after profiling shows approximate
   diagonal assembly is still dominated by repeated traversal or
   classification.

These policies are intended to diagnose or avoid partial off-band contact block
dropping breaking positive semidefiniteness. Validation must compare opt-in
variants against `fused_pcg`, `socu_rt1_full_hessian_cached`, and the exact
`contact_hessian` / `full_hessian` diagnostics before any default behavior
changes.

SOCU approx now exposes only the strict structured direct solve path. The
previous assembly-only validation path has been removed; structured assembly
diagnostics are reported from the real solve path through
`linear_system/socu_approx/report`.

## Report Fields

The solve report records:

- layout size, block utilization, active and padding DoF counts
- ordering quality diagnostics and configured thresholds
- structured write counters for diagonal, first off-diagonal, and off-band
  dropped contributions
- off-band contact policy and fallback counters:
  `runtime_reorder.contact_offband_policy`,
  `contact.contact_offband_diag_fallback_count`,
  `contact.contact_offband_lump_fallback_count`, and
  `contact.structured_off_band_drop_count`
- RHS norm, residual, relative residual, descent dot, gradient norm, direction
  norm, and direction validation thresholds
- plan/timing/report counters when enabled
- line-search feedback when available
- optional `runtime_reorder` diagnostics: enabled state, interval, capacity,
  collecting frame, last applied frame, raw/unique edge counts, overflow count,
  apply status, and failure detail

The report does not include dense matrix eigen summaries or pre-factor matrix
downloads. Direction validation is the required lightweight correctness check
on the solve path.

## Ordering Lab

`experiments/socu_ordering_lab` remains available as a standalone lab target
for ordering studies. It is not part of the runtime hot path.

## Next Stage: SOCU-Native Backend And Matrix Build

The current structured sink path has proven useful as an integration and
correctness vehicle, but it should not be treated as the final performance
architecture. The fastest corrected SOCU wrecking-ball variants are already
competitive with fused PCG because direct solve time and Newton count are lower,
but the structured build remains much heavier than FullSparse build. A typical
100-frame run shows the gap:

```text
fused_pcg Build Linear System:       ~3 ms/call
fast SOCU structured Build:         ~16 ms/call
SOCU runtime full_hessian rt1 build: ~58 ms/call
```

The next optimization stage is therefore not another small
`StructuredContactAssemblySink` tuning pass. It is a SOCU-native matrix
construction pipeline whose primary destination is the SOCU block-band storage
itself.

### Motivation

SOCU and fused PCG now have different engineering needs:

- fused PCG wants the historical FullSparse contact path to stay light, stable,
  and isolated from SOCU headers, structured sinks, `socu_native`, MathDx, and
  runtime-reorder experiments.
- SOCU wants specialized structured-band storage, direct block writes,
  off-band fallback policies, compact caches, reorder descriptors, and
  potentially solver-specific compiler flags.

Keeping both paths in the same `cuda_mixed` build target makes unrelated
translation units pay for SOCU templates and headers. It also discourages
aggressive SOCU-only refactors because any change risks the fused-PCG baseline.

### Backend Split Proposal

Introduce a SOCU-specific backend target, tentatively named:

```text
cuda_mixed_socu
```

The intended ownership is:

```text
cuda_mixed
    Historical mixed backend baseline.
    Primary solver: fused_pcg.
    FullSparse contact assembly stays in the pre-SOCU shape.
    No SOCU structured contact sink, runtime reorder, MathDx, or socu-native
    code should be required to compile the fused-PCG-only path.

cuda_mixed_socu
    SOCU experimental/direct-solver backend.
    Primary solver: socu_approx or successor.
    Owns SOCU-native matrix build, structured descriptors, reorder policies,
    direct solver runtime, and SOCU-specific compile optimization.
    Fused PCG is intentionally not part of this backend.
```

The split is not meant to duplicate the whole library forever. Shared math,
geometry, contact functions, ABD utilities, and collision detection can still
live in common modules. The important boundary is the matrix build backend:
FullSparse/fused PCG and SOCU should not share the same hot write path or the
same large SOCU-specific template headers.

Recommended first split:

1. Copy the entire current `src/backends/cuda_mixed` directory to a new
   SOCU-owned backend directory, initially `src/backends/cuda_mixed_socu`.
   Milestone 1 should not use a same-directory dual-target scaffold because
   that keeps the two backends coupled through the same source layout and
   headers.
2. Restore the ordinary `cuda_mixed` FullSparse/fused-PCG contact and linear
   paths to the pre-SOCU behavior.
3. Remove or disable fused-PCG-specific code from the copied SOCU backend. The
   SOCU backend should not compile fused-PCG solver code, fused-PCG-only
   matrix-conversion paths, or benchmark/config entries that imply fused PCG is
   available there.
4. Keep current SOCU structured build in `cuda_mixed_socu` as the correctness
   baseline while the SOCU-native builder is introduced.
5. Verify:
   - `cuda_mixed fused_pcg --frames 100` remains the reference baseline.
   - `cuda_mixed_socu` reproduces the current best SOCU variants.
6. Move SOCU compile and runtime experiments only inside `cuda_mixed_socu`.

### Manager / Reporter Reuse Boundary

The manager/reporter system should remain the high-level orchestration layer.
It already owns system lifetime, frame/Newton scheduling, scene configuration,
active sets, contact sets, collision filters, material data, timers, and debug
reports. Rewriting that whole layer would add risk without directly reducing
SOCU build time.

What should be reused:

- `GlobalContactManager`, trajectory filters, active contact buffers, contact
  coefficients, thickness, `d_hat`, friction data, and positions.
- ABD/FEM managers and reporters that expose body/vertex state.
- Global timing, reporting, scene config, and failure/report conventions.
- Existing FullSparse assembly interfaces for `cuda_mixed` fused PCG.

What should not remain the SOCU hot path:

```text
manager -> generic Hessian/triplet/structured sink adapter -> SOCU matrix
```

SOCU should add a new optional assembly interface beside the existing
FullSparse path, for example:

```cpp
struct SocuNativeBuildInfo
{
    SocuNativeMatrixBuilder& builder;
    SocuOrderingView         ordering;
    SocuDescriptorViews      descriptors;
    cudaStream_t             stream;
    // existing positions/contact/ABD/material views as needed
};

class SomeSystem
{
    void do_assemble(...);                // existing FullSparse path
    void do_assemble_socu(SocuNativeBuildInfo& info); // SOCU-native path
};
```

Systems that do not yet implement `do_assemble_socu` can temporarily fall back
to the current structured sink path inside `cuda_mixed_socu`, but the production
goal is direct SOCU-native build coverage for every major Hessian/RHS provider.

### SOCU-Native Matrix Builder

The SOCU-native builder should be centered on the storage consumed by
`socu_native`, not on a generic sparse/structured sink abstraction.

Core storage:

```text
D[block]      diagonal blocks
E[block]      first off-diagonal blocks
rhs[block]    packed RHS
metadata      block size, active lanes, padding lanes, ordering epoch
```

Core write operations:

```cpp
add_diag_scalar(block, lane_i, lane_j, value)
add_offdiag_scalar(block, lane_i, lane_j, value)
add_diag_block3(block, lane_i, lane_j, H3x3)
add_offdiag_block3(block, lane_i, lane_j, H3x3)
add_projected_abd_block(...)
add_rhs(block, lane, value)
add_diag_lump(vertex_or_atom, value)
```

The builder can still use atomics initially. Later optimization can replace
high-contention atomics with compact records, block-local accumulation,
segmented reduction, or warp aggregation. The first goal is a clear direct
storage path with matrix equivalence, not immediate maximum performance.

### Native Target Model

The end-state production path should not expose the old structured sink as the
provider-facing abstraction. A provider kernel should receive a small native
target view, compute its local contribution, and write directly to SOCU storage.
The structured sink should remain as a debug/reference/fallback adapter only.

Target vocabulary:

```cpp
struct NativeDiagBlockTarget
{
    SizeT block;
    SizeT row_lane;
    SizeT col_lane;
};

struct NativeFirstOffdiagBlockTarget
{
    SizeT left_block;
    SizeT row_lane;
    SizeT col_lane;
    bool  transposed; // old pair order differs from SOCU E orientation
};

struct NativeDenseStencilTarget
{
    SocuNativeBandClass cls; // Diag, FirstOffdiag, OffBand, Skipped
    SizeT               block_or_left_block;
    SizeT               row_lane;
    SizeT               col_lane;
    bool                mirror_symmetric_entry;
};

struct NativeContactStencilTarget
{
    IndexT              contact_id;
    SocuNativeBandClass half_block_class;
    SizeT               block_or_left_block;
    SizeT               row_lane;
    SizeT               col_lane;
    IndexT              local_row_vertex;
    IndexT              local_col_vertex;
    SocuNativeDescriptorKind row_kind;
    SocuNativeDescriptorKind col_kind;
    IndexT              row_jacobian_index;
    IndexT              col_jacobian_index;
    StructuredContactOffbandPolicy fallback_policy;
};
```

These targets are addresses and policies, not cached physics values. They may
store descriptor indices, old DoFs, local vertex slots, ABD body/J indices, and
SOCU block/lane destinations. They should not store time-varying Jacobian,
position, material, friction basis, or Hessian values unless a later benchmark
proves that doing so is safe and beneficial.

The intended provider kernel shape is:

```text
load provider data
load precomputed native target(s)
compute local Hessian/RHS
if target is exact in-band:
    write native D/E/rhs directly
else if policy allows diagonal/lump fallback:
    write fallback native diagonal contribution
else:
    count skipped/off-band contribution
```

Debug and migration modes may additionally mirror the same contribution into
the legacy structured representation and diff the two matrices. Production
performance runs should use native single-write only.

The old `StructuredDeviceAssemblySink` remains useful during migration, but its
long-term role should narrow to:

- reference assembly for matrix/RHS diff tests;
- fallback for a provider family that has not received native targets yet;
- runtime-ordering graph collection when the graph path still depends on old
  DoF/atom scalar edges;
- diagnostic mirror writes behind explicit debug flags.

It should not be the performance-critical abstraction once a provider has a
validated native target table.

### Target Build Lifecycle

Native target tables are tied to an ordering epoch and, for contact, to a
contact-set signature:

```text
ordering install or runtime reorder install
    -> rebuild DoF and vertex descriptors
    -> rebuild static chain/base targets
    -> rebuild constraint/joint targets
active contact set changes
    -> rebuild contact stencil targets
linear solve begins
    -> clear native storage
    -> provider kernels compute and write through native targets
    -> optional debug mirror/diff
    -> factor/solve/scatter
```

Each target table should report both coverage and fast-path hit rates. At
minimum, reports should include:

```text
native_target_epoch
native_chain_base_fast_hit_count
native_chain_base_fast_miss_count
native_chain_base_scalar_fallback_count
native_contact_exact_count
native_contact_diag_fallback_count
native_contact_lump_fallback_count
native_contact_structured_fallback_count
native_build_time_ms
native_target_build_time_ms
```

These counters are part of the acceptance signal. A native target implementation
that is correct but rarely hit should not be treated as a completed performance
milestone.

### Ordering And Descriptor Tables

After init ordering install or runtime reorder install, SOCU should build
solver-native descriptor tables. These tables should answer the questions that
the current structured sink repeatedly answers per scalar.

Minimum vertex/atom table:

```text
global vertex -> kind(None/FEM/ABD)
global vertex -> fixed
global vertex -> old DoF
global vertex -> SOCU block
global vertex -> SOCU lane
global vertex -> ABD body
global vertex -> ABD J index
```

Static provider descriptors:

```text
chain/local Hessian contribution -> target block/lane pairs
constraint/joint contribution    -> target block/lane pairs
mass/inertia contribution        -> diagonal block/lane
RHS contribution                 -> packed RHS block/lane
```

Dynamic contact descriptors:

```text
contact stencil id
local vertex ids
half-block target class:
    diag band
    first off-diagonal band
    off-band
projection kind:
    FEM/FEM
    ABD/FEM
    FEM/ABD
    ABD/ABD
mirror/transpose policy
fallback policy:
    exact
    diag
    diag_lump
```

The descriptor table should be rebuilt when the ordering changes. Contact
descriptors should also be rebuilt when active contact stencils change. ABD
Jacobian values should not be baked into descriptors unless their lifecycle is
proven static for the build; descriptors should normally store indices and read
current `ABDJacobi` values during assembly.

### Native Build Pipeline

The target per-linear-build pipeline is:

```text
ensure native descriptors match ordering epoch
ensure provider target tables are valid for this epoch/signature
clear SOCU D/E/rhs
assemble_mass_and_inertia_socu(targets)
assemble_chain_hessian_socu(targets)
assemble_constraints_and_joints_socu(targets)
assemble_contact_socu(contact_targets)
assemble_rhs_socu(targets)
optional_debug_mirror_or_diff()
factor_and_solve_socu()
scatter_direction()
validate_direction()
```

This should replace the current SOCU hot path:

```text
assemble existing local systems
write through StructuredAssemblySink / StructuredContactAssemblySink
classify/scatter per scalar
solve SOCU band matrix
```

The direct-write definition for this plan is intentionally precise:

- provider kernels may still compute the local Hessian/RHS exactly where they do
  today;
- the write side should use precomputed native block/lane targets;
- the production write side should not classify old DoF pairs per scalar;
- debug mirror and legacy fallback may use structured sinks, but those paths
  must be disabled in performance runs.

### Provider Migration Order

The migration should be incremental. Each step must keep the current structured
sink path available as a comparison/fallback inside the SOCU backend until the
native provider is validated.

1. **Backend split and frozen baselines**
   - Establish `cuda_mixed` fused-PCG baseline.
   - Establish `cuda_mixed_socu` current structured-SOCU baseline.
   - Record 20/100-frame wrecking-ball timings for both.

2. **SOCU-native storage and clear/solve path**
   - Create `SocuNativeMatrixBuilder`.
   - Populate it from the current structured matrix as an adapter first, if
     needed, to validate storage layout and solve/scatter behavior.
   - No provider logic changes yet.

3. **Mass / inertia / diagonal regularization / RHS**
   - Move pure diagonal and RHS writes to native storage first.
   - These are low-risk and should avoid generic scalar classification.
   - Validate RHS equality and direction equality on small scenes.

4. **Chain / base Hessian**
   - Build static descriptors for chain/base Hessian providers.
   - Write directly to SOCU `D/E`.
   - Compare against structured sink matrix on small and medium scenes.
   - This provider runs every build and is a major long-term performance target.

5. **Constraints, joints, and external forces**
   - Add descriptors for fixed-size stencils.
   - Keep provider-specific kernels small; avoid including contact-heavy
     headers in constraint TUs.

6. **Contact**
   - Implement SOCU-native contact build with stencil-level descriptors.
   - Start with exact Hessian + current `diag_lump` off-band fallback.
   - Avoid per-scalar band classification in the contact Hessian kernel.
   - Compare against current structured contact matrix and fused-PCG behavior.

7. **Runtime reorder / graph build**
   - Rebuild descriptors after ordering install.
   - Separate graph-probe descriptors from final matrix descriptors.
   - Keep `topology + diag_lump` as the first performance target because it is
     currently the fastest measured runtime-reorder family on wrecking ball.

8. **Compact records and reductions**
   - If direct atomics are still slow, emit compact contribution records:

     ```text
     target block/lane
     value or 3x3 block
     provider tag
     ```

   - Reduce records by target before writing `D/E`.
   - Contact and constraints can use provider-specific compact record formats.

9. **Direct fused compute/write kernels**
   - Only after descriptor and compact-record paths are validated, try
     one-kernel `compute Hessian -> write SOCU storage` variants.
   - Gate by register count, spills, compile memory, and measured runtime.

### Contact-Specific Native Policy

The contact native path should preserve the key SPD lesson learned from the
current structured integration:

```text
if every half-block of a contact stencil fits the SOCU band:
    write exact projected SPD Hessian contribution
else:
    do not write a partial off-band stencil
    write diag or diag_lump fallback for the entire stencil
```

The current best stability/performance candidate is `diag_lump`. The exact
`diag` policy is still useful as a diagnostic because it preserves the exact
per-vertex diagonal blocks of the projected contact Hessian.

### Correctness Gates

Every native provider migration step should pass:

- `git diff --check`.
- `cuda_mixed fused_pcg --frames 20` and later `--frames 100`.
- `cuda_mixed_socu` current SOCU baseline variant, at least 20 frames, later
  100 frames.
- Existing CUDA mixed solver smoke tests.
- Matrix comparison on small deterministic scenes:
  - native SOCU matrix vs current structured sink matrix
  - same sparsity in represented band
  - values within tolerance, initially `max(1e-9 abs, 1e-10 rel)`
- CPU reference factorization for dumped SOCU problems when debug is enabled.
- Direction validation finite/nonzero/descent checks.

Provider-specific matrix equality can be relaxed only when the provider is
intentionally changing the approximation, such as `diag_lump` replacing partial
off-band contact stencils. Such changes must have their own variant name and
benchmark record.

### Unit And Component Test Plan

The SOCU-native builder must not rely only on wrecking-ball end-to-end runs.
Each new component should have focused tests that can fail close to the broken
layer. These tests should live under the CUDA mixed backend test area, with
SOCU-native-specific names so they can be run independently from long examples.

Recommended test target groups:

```text
socu_native_storage_tests
socu_native_descriptor_tests
socu_native_provider_tests
socu_native_contact_policy_tests
socu_native_reorder_tests
socu_native_solver_contract_tests
```

#### Storage Builder Tests

Goal: prove `SocuNativeMatrixBuilder` writes exactly the block-band storage
expected by `socu_native`.

Tests:

- clear initializes all `D/E/rhs` values to zero and preserves metadata.
- diagonal scalar write lands at the expected block/lane entry.
- off-diagonal scalar write lands in the expected first-offdiag block and
  transpose/mirror convention is correct.
- 3x3 FEM block write equals nine scalar writes.
- ABD projected block write equals explicit host-side `J^T H J` reference.
- repeated writes accumulate deterministically within tolerance.
- padding lanes remain zero or ignored according to solver contract.
- fixed DoFs are skipped.
- out-of-band writes are not silently accepted by native band write APIs.

Small deterministic host/device fixtures should be used first:

```text
block_size = 4 or 8
block_count = 3
known old_dof -> block/lane mapping
hand-authored 2x2 or 3x3 contributions
```

Validation:

- download `D/E/rhs`,
- compare to CPU reference arrays,
- tolerance `max(1e-12 abs, 1e-12 rel)` for fp64 unit fixtures unless atomic
  ordering requires a slightly looser tolerance.

#### Descriptor Table Tests

Goal: prove ordering-derived descriptors match scalar classification and remain
valid across reorder epochs.

Tests:

- FEM vertex descriptor maps global vertex to expected block/lane.
- ABD vertex descriptor maps global vertex to body, old DoF, block/lane, and
  current J index.
- fixed and inactive vertices produce skip descriptors.
- same-block pairs classify as diagonal band.
- adjacent-block pairs classify as first-offdiag band.
- non-adjacent-block pairs classify as off-band.
- descriptor rebuild after reorder changes block/lane as expected and bumps the
  epoch.
- stale descriptor epoch is detected in debug builds or forces rebuild before
  use.
- descriptor construction does not store stale ABD Jacobian values; tests should
  mutate J values after descriptor build and verify assembly uses the updated J.

These tests should compare descriptor classification against the current
`StructuredAssemblySink` / `StructuredContactAssemblySink` classification during
the transition period.

#### Offband Policy Tests

Goal: prove SPD-preserving contact fallback is applied at the intended stencil
granularity.

Tests:

- `drop` keeps current behavior: in-band scalars are written, off-band scalars
  are dropped and counted.
- `diag` with a fully in-band stencil writes the exact full projected Hessian.
- `diag` with any off-band half-block writes only exact per-vertex diagonal
  blocks for the whole stencil and writes no off-diagonal contact blocks.
- `diag_lump` with any off-band half-block writes only nonnegative scalar
  diagonal compensation for the whole stencil using the configured physical
  3D lumping rule.
- `diag_lump` diagonal entries are nonnegative. In the current implementation,
  ABD lumping maps physical-coordinate lump values to generalized DoFs by
  squared ABD weights; it is not a projected 12x12 Gershgorin/row-absolute-sum
  bound. If a future policy needs projected row-sum conservativeness, both the
  implementation and tests must compute row sums after ABD projection.
- duplicate vertices inside a stencil accumulate correctly.
- FEM/FEM, FEM/ABD, ABD/FEM, ABD/ABD, and same-body ABD cases are covered.
- single-vertex PH path follows the same policy when ABD projection would
  create off-band scalar entries.

Matrix-level assertions:

- for a PSD local Hessian and fully in-band descriptor, the native write equals
  the exact projected host reference;
- for off-band `diag`/`diag_lump`, no off-band scalar is written;
- for off-band `diag_lump`, diagonal-only contact contribution is positive
  semidefinite by construction.

#### Provider Unit Tests

Goal: each provider can be migrated independently and compared with the current
structured sink baseline.

For each provider family:

```text
mass / inertia / damping
RHS / gradient packing
chain / base Hessian
constraints / joints
simplex normal contact
simplex frictional contact
vertex-half-plane normal contact
vertex-half-plane frictional contact
```

Tests:

- construct a tiny deterministic scene or synthetic provider input;
- run current structured sink provider;
- run SOCU-native provider with the same ordering;
- download both SOCU band matrices and RHS vectors;
- compare represented band entries within tolerance;
- verify fixed DoFs and padding lanes match the contract;
- for approximate policy variants, compare against the policy-specific CPU
  reference rather than the exact structured sink.

Provider tests should be small enough to run as part of normal CUDA mixed test
targets. Larger scenes can be separate regression tests.

#### Runtime Reorder Tests

Goal: ordering installation, descriptor rebuild, and solver plan refresh are
correct and observable.

Tests:

- init-time ordering builds descriptor tables before first solve.
- runtime reorder interval `0` does not rebuild runtime ordering.
- intervals `1`, `2`, `5`, and a large interval allow probes only on expected
  frames and skip repeated probes when the contact-set signature is unchanged.
- if the contact-set signature changes later in the same eligible frame, a new
  probe/reorder can run before the final structured assembly.
- descriptor epoch changes after successful reorder install.
- previous ordering remains active after reorder failure.
- collector overflow is reported and does not corrupt descriptors.
- topology graph source produces expected unit-weight edges on a synthetic
  contact set.
- contact/full Hessian graph sources produce deterministic weighted edges for
  a synthetic known Hessian.
- approximate graph sources produce expected coefficient-based weights.

Where possible, these tests should run without full scene advancement by
directly constructing the graph builder inputs.

#### Native Solve Contract Tests

Goal: `SocuNativeMatrixBuilder` output can be consumed by `socu_native` and
matches CPU reference behavior.

Tests:

- SPD synthetic block-tridiagonal matrix solves to known answer.
- singular/indefinite matrix is detected and reported as a factorization
  failure.
- RHS zero produces zero direction and does not trigger nonfinite values.
- damping shift changes diagonal as expected.
- fp64 solve residual matches CPU reference threshold.
- MathDx/runtime artifact discovery failure is reported clearly when SOCU
  native is enabled.
- `UIPC_WITH_SOCU_NATIVE=OFF` build/test path compiles and reports the solver
  unavailable without leaving dangling symbols.

#### End-To-End Regression Tests

End-to-end tests remain necessary, but they should validate integration rather
than replace component tests.

Required short regressions:

```text
86_cuda_mixed_linear_solver_selection_smoke
policy contract tests
python/examples/cuda_mixed_wrecking_ball_compare.py --variant fused_pcg --frames 20
python/examples/cuda_mixed_wrecking_ball_compare.py --variant socu_init_topology_diag_lump --frames 20
python/examples/cuda_mixed_wrecking_ball_compare.py --variant socu_rt20_topology_diag_lump --frames 20
python/examples/cuda_mixed_wrecking_ball_compare.py --variant socu_rt50_topology_diag_lump --frames 20
```

After the `cuda_mixed_socu` backend exists, the same SOCU variants should also
be exposed through that backend. Until then, the current runnable commands are
the `cuda_mixed_wrecking_ball_compare.py` variants above.

Required longer performance/stability regressions before accepting a major
native builder step:

```text
python/examples/cuda_mixed_wrecking_ball_compare.py --variant fused_pcg --frames 100
python/examples/cuda_mixed_wrecking_ball_compare.py --variant socu_rt50_topology_diag_lump --frames 100
python/examples/cuda_mixed_wrecking_ball_compare.py --variant socu_rt50_contact_hessian_diag_lump --frames 100
python/examples/cuda_mixed_wrecking_ball_compare.py --variant socu_init_full_hessian_diag_lump --frames 100
```

If a long regression fails with NaN or non-descent, rerun only that failing
variant with debug dump enabled and check:

- pre-contact/base matrix factorization,
- final matrix factorization,
- diagonal sign/minimum,
- off-band counters and policy fallback counters,
- descriptor epoch and runtime-ordering report.

#### CI Strategy

The test suite should be split by cost:

- **always-on unit tests**: storage, descriptors, small policy tests, synthetic
  solver contracts;
- **CUDA backend smoke**: current solver selection smoke and short frame
  regressions;
- **nightly/manual performance**: 100-frame wrecking-ball sweeps and expensive
  matrix-dump comparisons.

Each native-provider PR should state which tier it changes and include the
exact command lines used for any manual tier.

### Performance Gates

The immediate performance target is not to beat fused PCG on every micro-step.
The staged targets are:

```text
SOCU build baseline today:             ~16 ms/call for best variants
near-term native build target:          <10 ms/call
aggressive native build target:          5-8 ms/call
fused-PCG build reference:              ~3 ms/call
```

Accept a provider migration when:

- correctness gates pass,
- total 100-frame wall time does not regress versus the structured SOCU
  baseline by more than 2%, unless the step is explicitly infrastructure-only,
- the migrated provider's timer decreases or enables a later planned
  migration, and
- compile memory/register pressure does not get worse for unrelated providers.

Rejected experiments should be reverted and summarized in the journal with
timing, compile behavior, and failure reason.

### Compile And File-Organization Goals

The SOCU-native backend should be organized to reduce nvcc rebuild blast radius:

- keep heavy contact kernels in contact-specific `.cu` files,
- keep descriptor builders in narrow `.cu` files,
- keep public headers mostly POD views and small inline accessors,
- move large template/device implementation out of headers where practical,
- avoid including SOCU contact sink headers from fused-PCG/FullSparse contact
  translation units,
- keep `matrix_converter.inl` and `fast_segmental_reduce.inl` out of SOCU
  experiments unless the experiment explicitly targets those shared utilities.

Compile-specific SOCU flags or object-level flags can be considered only after
the backend split, so they do not change fused-PCG build behavior.

### Validation Variants To Preserve

The following wrecking-ball families should remain available while the native
builder is developed. Today they are runnable through
`python/examples/cuda_mixed_wrecking_ball_compare.py`; after the backend split,
the SOCU entries should also be runnable through `cuda_mixed_socu`.

```text
fused_pcg
socu_init_topology_diag_lump
socu_rt{20,25,50}_topology_diag_lump
socu_rt{20,50}_contact_hessian_diag_lump
socu_init_full_hessian_diag_lump
```

The current fastest measured runtime-reorder candidates are topology-based
`diag_lump` variants at larger intervals. They should be the first performance
targets for the native builder. `full_hessian` runtime reordering should remain
diagnostic until its build cost is addressed.

### Milestone Plan

The SOCU-native backend work should be delivered as small, reviewable
milestones. Each milestone must leave the repository in a runnable state and
should have a clear fallback if the new path is slower, less stable, or too
expensive to compile.

Commit discipline:

- Each accepted milestone must be committed separately.
- Do not batch multiple milestones into one code commit; benchmark attribution
  must stay clear.
- If a milestone fails its acceptance criteria, revert its code changes before
  moving on. Record the rejected result in the journal with timing, compile
  behavior, and failure reason.
- Documentation-only corrections may be committed separately.
- Benchmark or script fixes that affect reproducibility must be committed before
  using their results as a new baseline.
- A commit that changes behavior should include the relevant smoke/regression
  commands in the final message or journal entry.
- Every milestone must strictly satisfy the global constraints, design
  boundaries, correctness gates, test strategy, file-organization goals, and its
  own acceptance criteria described earlier in this document. Any exception must
  be documented in the journal as a planned exception before the milestone is
  committed.

#### Milestone 0: Freeze Current Baselines

Goal: record current behavior before structural refactors.

Deliverables:

- Keep the current `cuda_mixed` structured-SOCU implementation unchanged except
  for documentation or benchmark-script fixes.
- Store the current best-known 20/100-frame benchmark commands and output paths
  in the journal.
- Confirm current best SOCU variants:

  ```text
  socu_rt{20,25,50}_topology_diag_lump
  socu_init_full_hessian_diag_lump
  socu_init_contact_hessian_diag_lump
  ```

- Confirm fused-PCG reference:

  ```text
  fused_pcg --frames 100
  ```

Acceptance:

- `git diff --check`.
- `uipc_test_sim_case_cuda_mixed_only "86_cuda_mixed_linear_solver_selection_smoke" -s`.
- `fused_pcg --frames 100` reaches frame 100.
- best SOCU topology `diag_lump` variant reaches frame 100.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- No code refactor in this milestone; if a benchmark is inconsistent, rerun and
  document variance rather than changing implementation.

#### Milestone 1: Backend Split Scaffold

Goal: create a SOCU-owned backend target without changing solver semantics.
This milestone uses a full backend directory copy. It deliberately avoids a
same-directory `cuda_mixed`/`cuda_mixed_socu` dual-target CMake setup so the
original backend can later be restored to a historical pre-SOCU commit without
dragging the SOCU backend with it.

Deliverables:

- Copy the entire current `src/backends/cuda_mixed` directory to
  `src/backends/cuda_mixed_socu`.
- Add a `cuda_mixed_socu` backend target/module registration from the copied
  directory.
- Keep copied file contents as close as possible to current `cuda_mixed` during
  this milestone; only rename backend/module identifiers, CMake target names,
  include paths, generated source definitions, and Python benchmark selection
  glue required for the new backend to load.
- Keep `cuda_mixed` fused-PCG behavior unchanged.
- Keep `cuda_mixed_socu` running the current structured-SOCU path.
- Do not restore or clean up original `cuda_mixed` yet; that is Milestone 2.
- Do not redesign SOCU matrix build yet; `cuda_mixed_socu` starts as a copied
  baseline that reproduces current SOCU behavior.
- Remove or hard-disable only the minimum fused-PCG exposure needed so
  `cuda_mixed_socu` cannot silently run `"solver": "fused_pcg"`. Larger
  deletion of fused-PCG source files from the SOCU copy may be done after the
  copied backend builds and passes the acceptance checks.
- `cuda_mixed_socu` should reject or not expose `"solver": "fused_pcg"`; fused
  PCG remains available only through `cuda_mixed`.
- Add benchmark-script support for selecting `cuda_mixed` vs
  `cuda_mixed_socu`.

Acceptance:

- Both backend modules build.
- `cuda_mixed fused_pcg --frames 20` passes.
- `cuda_mixed_socu socu_init_topology_diag_lump --frames 20` passes.
- `cuda_mixed_socu socu_rt20_topology_diag_lump --frames 20` passes.
- Attempting to request fused PCG from `cuda_mixed_socu` fails clearly at config
  or solver-selection time instead of silently using shared fused-PCG code.
- Unit tests and solver-selection smoke pass.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- If copying the full backend creates registration or symbol conflicts, keep
  the new backend disabled by default behind a CMake option while resolving the
  naming issue. Do not switch back to a same-directory dual target unless the
  full-copy approach is proven impossible.

#### Milestone 2: Restore `cuda_mixed` Baseline

Goal: restore `cuda_mixed` to the pre-SOCU fused-PCG / FullSparse backend
state. This milestone should use the last commit before SOCU integration as
the behavioral and source-shape reference, not incrementally patch the current
SOCU-enabled `cuda_mixed` into something similar.

Deliverables:

- Identify and record the exact pre-SOCU baseline commit used as the restore
  reference.
- Start this milestone only after `src/backends/cuda_mixed_socu` exists and
  preserves the current SOCU structured path. The restore should modify the
  original `src/backends/cuda_mixed` tree only; SOCU-specific code should remain
  available in the copied backend.
- Restore `cuda_mixed` files that existed before SOCU integration to that
  baseline shape, including ordinary FullSparse contact assembly, linear solver
  registration, CMake source lists, and any manager/reporter paths that were
  only added for SOCU.
- Remove SOCU-specific compile dependencies from `cuda_mixed`; this includes
  SOCU structured contact files, `socu_approx` linear-system files,
  `socu_native`/MathDx linkage, runtime ordering reports, structured contact
  sink/cache/offband-policy code that is not present in the pre-SOCU backend,
  and benchmark/config paths that imply SOCU is available through
  `cuda_mixed`.
- Keep only genuinely shared, pre-existing math/util code in `cuda_mixed`. Any
  helper introduced solely for SOCU should live in `cuda_mixed_socu` or a
  SOCU-only shared module.
- Do not use shared public algorithm files such as `matrix_converter.inl` or
  `fast_segmental_reduce.inl` to hide fused-PCG regressions; restore the
  ordinary backend path first and compare behavior against the recorded
  pre-SOCU reference.

Acceptance:

- `cuda_mixed fused_pcg --frames 100` passes and is not slower than the frozen
  baseline by more than 2% without explanation.
- A source/build scan shows `cuda_mixed` no longer compiles or links
  `socu_approx`, `socu_native`, MathDx, SOCU structured contact TUs, SOCU
  runtime ordering, SOCU report, SOCU contact cache, or SOCU off-band policy
  files.
- Ordinary `cuda_mixed` contact files match the pre-SOCU reference except for
  intentional, documented non-SOCU maintenance changes.
- `cuda_mixed_socu` build no longer compiles fused-PCG-specific solver files or
  fused-PCG-only matrix conversion paths.
- `cuda_mixed_socu` still passes the Milestone-1 SOCU 20-frame checks.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- If restoring the entire backend in one patch is too broad, split the restore
  by historical ownership boundaries but keep the same target state: first
  restore ordinary contact/FullSparse files to the pre-SOCU commit, then remove
  SOCU linear-system/build/report/config dependencies. Do not leave hidden SOCU
  fallback paths in `cuda_mixed`.

#### Milestone 3: SOCU-Native Storage Skeleton

Goal: introduce `SocuNativeMatrixBuilder` without migrating providers yet.

Deliverables:

- Define SOCU-native storage views for `D`, `E`, `rhs`, block metadata, and
  padding lanes.
- Implement clear, scalar write, 3x3 block write, RHS write, and debug download
  helpers.
- Add storage builder unit tests.
- Optionally add an adapter that copies the current structured matrix into
  native storage so solve/scatter can be tested before provider migration.

Acceptance:

- Storage unit tests pass on small synthetic fixtures.
- Native storage -> `socu_native` solve contract passes on synthetic SPD
  matrices.
- Current structured-SOCU path remains the default.
- No performance requirement beyond negligible overhead when the native
  skeleton is disabled.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- If direct integration with `socu_native` is blocked, keep builder tests
  host-validated and postpone solve integration to Milestone 4.

#### Milestone 4: Ordering And Descriptor Infrastructure

Goal: build reusable SOCU descriptors after ordering install.

Deliverables:

- Implement vertex/atom descriptor table:

  ```text
  kind, fixed, old_dof, block, lane, ABD body, ABD J index
  ```

- Implement pair/stencil classification helpers for diag/first-offdiag/off-band.
- Add descriptor epoch tracking.
- Rebuild descriptors after init ordering and runtime reorder install.
- Add descriptor unit tests and comparison against current structured
  classification.

Acceptance:

- Descriptor tests pass for FEM/FEM, FEM/ABD, ABD/FEM, ABD/ABD, fixed, padding,
  and reorder epoch cases.
- Current structured-SOCU path remains numerically unchanged.
- Reorder reports expose descriptor epoch in debug/report mode if useful.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- If full descriptor coverage is large, start with FEM-only and ABD-body-only
  fixtures, then expand before provider migration.

#### Milestone 5: Native Diagonal And RHS Providers

Goal: migrate the lowest-risk provider writes first.

Implementation boundary after the 2026-05-07 completion pass:

- M5 covers solver-owned diagonal workspace initialization
  (`damping_shift`/regularization and padding identity) plus packed RHS/gradient
  writes through `SocuNativeMatrixView`.
- FEM/ABD kinetic, inertia, and shape Hessian contributions still enter through
  the existing structured chain/base provider assembly. They are not separable
  from that provider API without the Milestone 6 native chain/base Hessian
  migration.

Deliverables:

- Native damping/regularization/padding writes.
- Native RHS/gradient packing.
- Provider tests comparing native output to current structured sink output.
- Debug option to assemble both native and structured paths and diff `D/E/rhs`.

Acceptance:

- Provider unit tests pass.
- Small scene matrix/RHS diff passes.
- 20-frame SOCU regression passes with native diagonal/RHS enabled.
- 100-frame best SOCU variant does not regress by more than 2%, or the
  milestone is marked infrastructure-only and left disabled by default.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- Keep native diagonal/RHS behind a config flag and default to structured path
  until parity is proven.

#### Milestone 6: Native Chain/Base Hessian Provider

Goal: move the always-present structured chain/base Hessian to native writes.

Implementation split:

- **M6a: parity native sink.** Keep the existing FEM/ABD provider call surface,
  route scalar chain/base writes through native DoF descriptors, and prove exact
  pre-contact `D/E/rhs` parity. This is the bridge that makes the native matrix a
  drop-in replacement, not the final performance shape.
- **M6b: fast native chain/base targets.** Move hot providers from
  scalar-by-scalar classification to block/stencil targets that already know
  `block/lane` destinations. M6b owns ABD `12x12` same-block and
  adjacent-block targets, FEM `3x3` block targets, first-offdiag target
  precomputation, and production native single-write with mirror diff kept
  debug-only.

Current implementation status (2026-05-07):

- Implemented as an opt-in descriptor-backed backend inside the existing
  structured assembly sink API. Existing FEM/ABD providers keep their current
  call surface, but chain/base matrix scalars can be routed through native DoF
  descriptors.
- `StructuredAssemblyInfo` now marks chain/base and contact phases explicitly.
  Native chain/base writes are only enabled during the chain/base phase; DyTopo
  contact remains on the legacy structured path.
- Runtime diff mode assembles the chain/base Hessian into both native and
  legacy buffers at the pre-contact checkpoint and compares `D/E/rhs`.
- Contact-enabled scene acceptance still requires a successful fallback build
  with legacy structured contact TUs. On the current machine,
  `ipc_simplex_frictional_contact_structured.cu` enters very high memory/swap
  usage during `cicc`, so the full contact 20/100-frame acceptance remains a
  pending M6 gate rather than a completed one.
- M6b currently includes ABD base Hessian fast paths for 12-DoF bodies that map
  to one native block or exactly two adjacent native blocks, even with arbitrary
  RCM lane order inside each block. Same-block pieces write directly to native
  `D`; cross-boundary adjacent-block pieces write directly to first-offdiag
  `E`; debug compare mirrors to the legacy structured representation only when
  requested. If the body spans unsupported blocks, runtime ordering is being
  collected, or native storage is disabled, the provider falls back to the
  scalar structured sink.

Detailed M6b execution plan:

1. **Instrumentation first.**
   - Add native chain/base target counters:
     `native_chain_base_same_block_dense_hit_count`,
     `native_chain_base_adjacent_dense_hit_count`,
     `native_chain_base_dense_miss_count`,
     `native_chain_base_scalar_fallback_count`, then extend the same counter
     family with FEM `3x3` and first-offdiag target counters as those target
     families land.
   - Add a replacement timer for the old `Assemble Structured Chain` signal:
     `native_chain_base_target_build_time_ms` and
     `native_chain_base_assembly_time_ms`.
   - Report counters even when debug matrix diff is disabled, because perf runs
     must be diff-off.

2. **Target API split.**
   - Introduce a narrow header for native chain/base targets, separate from the
     legacy structured contact sink headers.
   - Keep `StructuredDeviceAssemblySink` as a fallback adapter, but make new
     provider code call native target writers directly where possible.
   - The writer API should accept `SocuNativeMatrixView`, target arrays, and
     provider-local Hessian/RHS values. It should not need `old_to_chain` in the
     hot path.

3. **ABD base Hessian targets.**
   - Same-block target: implemented M6b slice. A 12-DoF ABD body fully inside
     one native block writes the upper `3x3` subblocks directly to `D` using the
     descriptor lanes as-is; lane contiguity is not required because RCM commonly
     reverses or permutes lanes inside the block.
   - Adjacent-block target: implemented M6b slice. If the 12 DoFs span exactly
     two adjacent SOCU blocks, gather the left block and per-local-DoF lanes,
     then write same-block pieces to `D` and cross-boundary pieces to
     first-offdiag `E` without falling back to scalar band classification.
   - Non-adjacent ABD bodies keep scalar native fallback.

4. **FEM `3x3` and stencil targets.**
   - Add vertex-local `3x3` diag targets for FEM kinetic/mass-style blocks.
   - Add pair targets for FEM element/report contributions that naturally know
     their local vertex pair. Each pair target should classify once at descriptor
     build time as diag, first-offdiag, off-band, or skipped.
   - For fixed vertices, use the descriptor fixed flag to choose identity or
     skip/write policy before entering the hot write loop.

5. **First-offdiag target precomputation.**
   - Store SOCU orientation explicitly: `left_block`, `row_lane`, `col_lane`,
     and whether the provider-local pair order is transposed.
   - Provider kernels must not recompute `abs(block_i - block_j)` per scalar
     once this target exists.

6. **Production single-write mode.**
   - Default production path for enabled native targets writes only native
     `D/E/rhs`.
   - Mirror diff is debug-only and must be disabled for performance gates.
   - If a provider has both a native target and scalar fallback, report the
     fallback count; a high fallback rate blocks declaring that provider
     performance-complete.

7. **M6b completion criteria.**
   - Provider unit tests cover ABD same-block and adjacent first-offdiag fast
     paths, and the remaining target families add off-band/skipped descriptor
     coverage as they land.
   - No-contact 20-frame scene passes with native chain/base targets and diff
     enabled.
   - Diff-off 100-frame no-contact performance run shows either measurable
     native chain/base assembly reduction or documents why the bottleneck has
     moved elsewhere.
   - Hit-rate report shows the optimized target family is actually exercised on
     the benchmark scene.

Deliverables:

- Static descriptors for chain/base Hessian contributions.
- Native projected ABD/FEM block writes.
- Provider-level fast targets:
  - FEM `3x3` block/lane target writes for vertex-local Hessian contributions.
  - ABD `12x12` same-block and adjacent-block arbitrary-lane writes for
    kinetic/shape/base Hessian.
  - First-offdiag target descriptors that avoid per-scalar band checks when the
    provider already knows adjacent block pairs.
- Matrix diff tests for synthetic chain fixtures.
- Scene-level diff for small deterministic scenes.

Acceptance:

- Native chain/base matrix equals structured sink represented band within
  tolerance.
- `cuda_mixed_socu` 20-frame topology `diag_lump` regression passes.
- 100-frame best SOCU variant is not slower; target improvement is measurable
  reduction in `Assemble Structured Chain` or replacement native timer.
- M6b target hit-rate counters prove that the benchmark exercises the optimized
  target family; otherwise the benchmark is not accepted as a performance gate
  for that target.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- If exact parity fails only for a specific provider type, split that provider
  back to structured fallback and continue migrating the rest.
- If a fast target's descriptor assumptions are not satisfied, keep the scalar
  native sink fallback active and record the missed fast-target condition in the
  M6 journal.

#### Milestone 7: Native Constraints / Joints / External Forces

Goal: migrate fixed-stencil non-contact effects.

Deliverables:

- Descriptor builders for constraints, joints, and external force Hessian/RHS
  contributions.
- Native write kernels with narrow includes.
- Provider tests per constraint/joint family.

Acceptance:

- Provider matrix/RHS diffs pass.
- Existing constraint policy contract tests pass.
- Short/medium scenes using joints/constraints match structured baseline.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- Keep unsupported constraint families routed through structured fallback until
  covered by tests.

#### Milestone 8: Native Contact Build V1

Goal: replace contact structured sink hot path with SOCU-native contact build.

M8 design requirements:

- Contact kernels should compute the same local IPC Hessian/RHS quantities as
  today, but the write side should consume `NativeContactStencilTarget` records
  instead of `StructuredContactAssemblySink`.
- A contact target table is valid only for the current ordering epoch and active
  contact-set signature. If either changes, rebuild before the next matrix
  assembly.
- Exact in-band writes must preserve the full projected SPD stencil. Partial
  off-band writes are not allowed because they can destroy the SPD behavior that
  motivated `diag_lump`.
- The native contact path should be family-scoped. Normal contacts can ship
  before frictional contacts if frictional compilation cost or register pressure
  is too high.

Detailed M8 execution plan:

1. **Contact target schema and policy tests.**
   - Build target records for PT, EE, PE, PP, and PH stencils.
   - For each half-block, store target class, SOCU orientation, local vertex
     pair, FEM/ABD projection kind, ABD Jacobian indices, and fallback policy.
   - Add synthetic descriptor tests for all combinations: fully in-band,
     adjacent first-offdiag, off-band, skipped/fixed, FEM/ABD, and ABD/ABD.

2. **Normal contact exact native write.**
   - Implement simplex normal and PH normal native writes first.
   - The kernel computes local normal Hessian as today and writes only through
     exact in-band targets or approved diagonal fallback.
   - Compare native matrix against structured contact matrix for fully in-band
     cases.

3. **Off-band fallback path.**
   - Implement native `diag` and `diag_lump` fallback for off-band contact
     stencils.
   - For `diag_lump`, prove the fallback reference is the current policy result,
     not the exact structured matrix, because the policy intentionally changes
     off-band behavior.
   - Report `native_contact_exact_count`, `native_contact_diag_fallback_count`,
     `native_contact_lump_fallback_count`, and
     `native_contact_structured_fallback_count`.

4. **Frictional contact migration.**
   - Add simplex frictional and PH frictional native writes after normal contact
     is stable.
   - Track register count, spills, compile time, and object size. If frictional
     native kernels become compile-time bottlenecks, split by contact family and
     keep structured fallback for the heaviest family.

5. **Runtime reorder integration.**
   - Rebuild contact targets after runtime ordering install.
   - If the contact-set signature changes inside an eligible frame, rebuild the
     contact target table for the new signature before using it.
   - Keep graph-probe targets distinct from final matrix-write targets so graph
     collection can stay cheap.

6. **M8 completion criteria.**
   - Native contact policy tests pass for every family that is enabled.
   - Fully in-band contact provider matrix diff passes against structured
     reference.
   - Off-band `diag_lump` tests pass against policy reference.
   - `socu_rt20_topology_diag_lump --frames 20` and
     `socu_rt50_topology_diag_lump --frames 100` pass with native contact
     enabled.
   - Native contact build shows reduced contact assembly timer or documents a
     clear non-contact bottleneck that now dominates.

Deliverables:

- Contact descriptor table for PT/EE/PE/PP/PH stencils.
- Native simplex normal, simplex frictional, PH normal, and PH frictional
  contact build kernels.
- Exact write for fully in-band stencils.
- `diag` and `diag_lump` fallback for any stencil with off-band half-blocks.
- Contact policy unit tests.
- Contact provider matrix diff tests for exact in-band cases and
  policy-reference tests for off-band cases.

Acceptance:

- `socu_rt20_topology_diag_lump --frames 20` passes.
- `socu_rt50_topology_diag_lump --frames 100` passes.
- No frame-16 `contact_hessian` NaN in diagnostic variants.
- Native contact build reduces contact/chain build timer or total wall time
  versus structured contact baseline.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- Keep per-contact-family fallback to structured contact path.
- If frictional contact is too heavy, ship normal contact native first and leave
  frictional structured until separately optimized.

#### Milestone 9: Runtime Reorder Native Integration

Goal: make runtime reorder descriptor rebuild and graph collection native.

Deliverables:

- Native topology graph builder from contact descriptors.
- Native contact/full Hessian graph weighting path, or documented fallback.
- Descriptor rebuild after runtime ordering install.
- Report fields for graph source, interval, descriptor epoch, fallback count,
  and reorder failure reason.

Acceptance:

- `topology + diag_lump` intervals 20/25/50 pass 100 frames.
- `contact_hessian + diag_lump` diagnostics pass 100 frames.
- Runtime reorder failure leaves previous descriptors and solver plan active.
- No stale-descriptor use after reorder.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- Keep runtime graph probe structured while final native build proceeds, but
  document the remaining cost and isolate it from fused-PCG builds.

#### Milestone 10: Compact Records And Reduction

Goal: reduce atomic contention and improve build throughput.

Deliverables:

- Compact contribution record format for one or more provider families.
- Reserve/overflow handling.
- Reduction/scatter kernel to SOCU `D/E/rhs`.
- Tests comparing compact path to direct native atomics.

Acceptance:

- Matrix equality with direct native path.
- No collector overflow on benchmark scenes, or graceful fallback if overflowed.
- At least 5% reduction in native build time for the targeted provider or a
  documented reason to reject/revert.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- Revert compact path for that provider and keep direct native writes.

#### Milestone 11: Direct Fused Compute/Write Kernels

Goal: test whether one-kernel compute-and-write can beat compact records.

Deliverables:

- Provider-specific fused kernels for the most expensive remaining provider.
- Register/spill/compile-memory measurements.
- Runtime comparison against compact/native direct path.

Acceptance:

- Correctness tests match provider reference.
- No nvcc OOM or unacceptable register spill.
- At least 5% total native build improvement for the targeted provider.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- Reject fused kernel and keep compact/native path if compile pressure or
  runtime is worse.

#### Milestone 12: Default Selection And Cleanup

Goal: choose production defaults and retire obsolete SOCU paths.

Deliverables:

- Pick default SOCU backend variant based on 100-frame and broader benchmark
  results.
- Remove or clearly mark deprecated structured-SOCU fallback code.
- Update docs, benchmark scripts, and test commands.
- Ensure `cuda_mixed` fused-PCG remains clean and stable.

Acceptance:

- `cuda_mixed fused_pcg` remains at or above frozen baseline.
- `cuda_mixed_socu` default SOCU variant passes 100-frame regressions.
- Unit/component/native provider tests are part of normal test workflow.
- Documentation matches code and benchmark names.
- All global constraints, design requirements, correctness gates, and acceptance
  rules defined above remain satisfied unless explicitly documented as a planned
  exception before commit.

Fallback:

- If native builder wins only on narrow scenes, keep structured-SOCU as a
  diagnostic fallback and do not change the default.

## Verification

Primary fp64 build:

```bash
cmake --build build/build_impl_fp64 --target libuipc_backend_cuda_mixed.so pyuipc uipc_test_sim_case_cuda_mixed_only -j 8
```

Contract smoke:

```bash
build/build_impl_fp64/Release/bin/uipc_test_sim_case_cuda_mixed_only "86_cuda_mixed_linear_solver_selection_smoke" -s
```

Recommended asset regression checks cover:

- `abd_fem_tower` with zero damping and strict structured solve residual near
  machine precision when all contributions are in band
- `abd_external_force` and `cube_ground` with default diagnostics enabled
- `fem_bouncing_cubes` long frame runs with runtime off-band reporting and no
  abort
- `fem_link_drop` with init-time off-band diagnostics and no quality gate
