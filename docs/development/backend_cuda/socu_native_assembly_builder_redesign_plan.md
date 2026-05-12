# SOCU Native Assembly Builder Redesign Plan

This document describes a large-scale redesign of the `cuda_mixed_socu`
structured matrix builder. It is intentionally separate from
`socu_mixed_solver_integration_plan.md`: the integration plan records the
current production path, while this document is the high-performance redesign
target.

The central idea is to replace the current contact-side target-table adapter
with a symbolic assembly plan and specialized numeric executors. The symbolic
state is split into a vertex side/lane plan and a contact program plan: the
side/lane plan changes only when ordering, native descriptors, mapping, fixed
flags, or projection data changes, while the contact program plan changes when
active contact topology changes or when the side/lane key changes. Per-solve
numeric assembly should only evaluate local Hessians and run regular,
preclassified matrix write microkernels.

## Goals

- Make native SOCU assembly faster than the legacy structured contact sink and
  faster than the current per-half-block native target path.
- Keep the production writer free of `old_to_chain` lookup, descriptor lookup,
  scalar pair classification, and legacy structured sink fallback.
- Represent `Drop`, `Diag`, and `DiagLump` as first-class symbolic policies.
- Avoid rebuilding symbolic contact targets in ordinary Newton solves when the
  contact topology and SOCU ordering are unchanged.
- Provide deterministic correctness gates against the legacy structured sink
  before any performance-only cutover.
- Make performance diagnosis visible through explicit plan, bucket, fallback,
  and hot-block counters.

## Non Goals

- This plan does not redesign `socu_native` factorization or solve.
- This plan does not change IPC contact Hessian formulas.
- This plan does not require removing the legacy structured contact path during
  early milestones. The legacy path remains the correctness oracle until the
  native builder passes all gates.
- This plan does not make runtime reordering mandatory.

## Starting Branch And Reused Baseline

Implementation starts from commit `ea8c59bc` (`Record SOCU M6 fallback contact
gates`) on a dedicated branch/worktree. That baseline is intentionally before
the abandoned per-half-block native contact target series. The redesign reuses
the stable M6 foundation and keeps the slower M8 native contact path only as an
external reference on the original branch.

Reuse directly:

- SOCU runtime creation, descriptor upload, chain ordering, and solve/report
  lifetime.
- Native diagonal/RHS workspace initialization and its diff guard.
- Native chain/base Hessian `D/E/RHS` writer, including the 3x3 exact fast path,
  scalar fallback, and validation counters.
- `StructuredContactOffbandPolicy` semantics for `Drop`, `Diag`, and
  `DiagLump`.
- Legacy structured contact assembly as the correctness oracle in fallback
  builds.
- Runtime reorder/report infrastructure and contact Hessian cache stamps where
  they are already stable.

Do not reuse as production code:

- The M8 `SocuNativeContactStencilTarget` per-half-block target table.
- Per-solve target rebuild kernels that repeat descriptor lookup and scalar
  pair classification.
- Any hot-path adapter that calls back into `StructuredContactAssemblySink`.
- Host-copy contact signatures as the final cache validity mechanism.

Temporary compatibility code may be copied only behind an explicit flag and
must be deleted once the compact symbolic plan executor passes the M8 cutover
criteria.

Performance baselines must be reproducible. The abandoned per-half-block native
target baseline is `8d0ad39b` on the original `mipc` branch unless the journal
records a newer frozen baseline binary or artifact. Every performance result
that claims a speedup over the old target path must record the baseline commit,
binary path, CMake cache, GPU, driver, and scene seed.

## Current Pain Points

The abandoned M8 native contact path is a useful bridge, but it is not the target
architecture:

- `SocuNativeContactStencilTarget` stores one large record per contact
  half-block. PT and EE contacts therefore write ten target records per contact,
  each duplicating side lane data.
- Target rebuild kernels still repeat stencil and half-block classification.
- The exact writer still executes scalar-level control flow before writing
  `D/E`.
- `DiagFallback` exists, but the current native implementation is scalar
  diagonal fallback, not full diagonal-block fallback.
- The final assembly path must not depend on a hot-path host copy signature for
  contact cache correctness.
- The performance report does not yet tell whether time is spent in plan
  rebuild, Hessian evaluation, atomic scatter, or fallback work.

## Architecture Overview

The redesigned builder has two layers:

1. Symbolic builder.
   - Runs only when a symbolic key changes.
   - Consumes SOCU ordering descriptors, active contact topology, ABD/FEM
     vertex mapping, fixed flags, and `StructuredContactOffbandPolicy`.
   - Emits two cacheable layers:
     - a vertex side/lane plan that maps global vertices to native blocks,
       lanes, components, fixed/writable state, and ABD projection weights;
     - a contact program plan that maps current contact stencils to side ids,
       whole-stencil exact/diag/drop classifications, task buckets, and
       optional hot-block adjacency lists.

2. Numeric executor.
   - Runs every assembly.
   - Evaluates local Hessians and executes preclassified microprograms.
   - Writes native `D/E` directly through specialized microkernels.
   - Does not call `classify_dof_pair`, does not read `old_to_chain`, and does
     not re-enter `StructuredContactAssemblySink`.

The builder is contact-first because contact assembly is currently the largest
regression. The same pattern can later be applied to chain/base, FEM, ABD, and
constraint assembly.

## Symbolic Keys And Invalidation

The builder cache is split by the data that can change each symbolic layer.
This split is performance-critical: contact topology can churn without changing
the native block/lane projection of a vertex, and runtime reorder can change
block/lane projection even when the active contact list is unchanged.

```cpp
struct SocuVertexSidePlanKey
{
    std::uint64_t ordering_epoch = 0;
    std::uint64_t native_descriptor_epoch = 0;
    std::uint64_t fixed_mapping_epoch = 0;
    std::uint64_t vertex_projection_epoch = 0;

    SizeT horizon = 0;
    SizeT block_size = 0;

    bool operator==(const SocuVertexSidePlanKey&) const noexcept = default;
};

struct SocuContactProgramPlanKey
{
    SocuVertexSidePlanKey side_key;

    std::uint64_t contact_topology_epoch = 0;
    std::uint64_t contact_layout_hash = 0;
    std::uint64_t contact_content_hash = 0;

    StructuredContactOffbandPolicy offband_policy =
        StructuredContactOffbandPolicy::Drop;
    bool scalar_diag_fallback_compatibility = false;

    bool operator==(const SocuContactProgramPlanKey&) const noexcept = default;
};
```

The M1 implementation may keep a temporary combined `SocuAssemblyPlanKey`
because it is the first lightweight POD contract shared by reports and tests.
From M2 onward, cache behavior and report fields must still distinguish the two
logical keys above. A combined host wrapper is valid only if its contact key
contains the same `side_key` as the side/lane plan it references.

Required rule: the final assembly path always receives a valid topology epoch
and contact hashes. It must not rely on
`GlobalDyTopoEffectManager::contact_set_signature()` doing a device-to-host copy
in the hot path.

`fixed_mapping_epoch` covers FEM/ABD fixed flags, vertex ownership, old DoF
offsets, and global vertex offsets. `vertex_projection_epoch` covers ABD
projection data such as `ABDJacobi` values used to precompute side weights.

M1 is a hard prerequisite for all downstream implementation. No M2 compact side
table work may be merged until the final assembly path can build a plan key
without a host-copy contact signature and all cache-invalidation tests in M1
pass.

Epoch producers are part of the public contract:

- `ordering_epoch` is owned by the SOCU ordering/runtime reorder layer and bumps
  whenever native block order, horizon, block size, or old-to-new DoF layout can
  change.
- `native_descriptor_epoch` is owned by the native descriptor cache and bumps
  whenever descriptor records or their device layout are rebuilt.
- `contact_topology_epoch`, `contact_layout_hash`, and `contact_content_hash`
  are owned by `GlobalDyTopoEffectManager` or the dy-topology contact manager.
  The epoch bumps when active contact stencil vertex ids, reporter/source
  ownership, source order, or contact storage layout changes. The layout hash
  covers source/layout structure; the content hash covers ordered active
  stencil vertex ids. They do not bump for geometry-only numeric updates.
- `fixed_mapping_epoch` is owned by the FEM/ABD mapping and global vertex
  managers. It bumps when fixed flags, vertex ownership, old DoF offsets, global
  vertex offsets, or in-place mapping tables change.
- `vertex_projection_epoch` is owned by ABD projection/Jacobi producers. It
  bumps when `ABDJacobi::x_bar()` or any precomputed projection weight used by a
  side lane changes, even if contact topology and descriptor layout are stable.

Runtime graph probes may still compute diagnostic signatures, but probe-only
keys and final assembly keys must be isolated. A probe-built plan must never be
inserted into the final assembly cache unless it was built from the final
assembly key above.

`StructuredContactOffbandPolicy` is assumed to be constant during one normal
simulation. It still belongs in `SocuContactProgramPlanKey` because changing it
changes program structure: `Drop`, `Diag`, and `DiagLump` emit different tasks.
If a debug run or restart changes only the off-band policy while side mapping is
unchanged, the side/lane plan is reused and only the contact program plan is
rebuilt.

Invalidation matrix:

| Change | Side/lane plan | Contact program plan | Numeric Hessian |
| --- | --- | --- | --- |
| Geometry values only | cache hit | cache hit | recompute |
| Active contact vertex ids, source ownership, source order, family counts, or contact storage layout | cache hit if side key unchanged | rebuild | recompute |
| Runtime reorder install or any ordering/horizon/block-size change | rebuild | rebuild | recompute |
| Native descriptor, FEM/ABD mapping, fixed flag, old DoF offset, or global vertex offset change | rebuild | rebuild | recompute |
| ABD projection/Jacobi weight change | rebuild | rebuild | recompute |
| Off-band policy or scalar-diag compatibility change | cache hit | rebuild | recompute |

This matrix describes semantic side/program keys. It does not include side
coverage fill/refresh work. In final `global` or `demand_filled` modes,
topology churn may trigger a coverage fill for newly referenced vertices, but
that is reported separately from semantic side rebuild. In M2
`active_set_temporary` mode, topology churn may refresh the active side table;
that is a temporary coverage refresh, not the target behavior.

The old structured sink effectively recomputes target lookup and classification
inside each assembly, so it naturally follows topology changes but pays the
lookup/classification cost every time. The new native path makes that work
explicitly cacheable: frequent topology churn may still rebuild the contact
program plan often, but it should not rebuild vertex side/lane data unless the
side key changed.

Contact generation should maintain:

```cpp
struct SocuContactTopologyStamp
{
    std::uint64_t epoch = 0;
    std::uint64_t layout_hash = 0;
    std::uint64_t content_hash = 0;
    SizeT reporter_count = 0;
    SizeT source_count = 0;
    SizeT pt_count = 0;
    SizeT ee_count = 0;
    SizeT pe_count = 0;
    SizeT pp_count = 0;
    SizeT ph_count = 0;
};
```

The epoch changes when the active contact stencil vertex ids, contact family
counts, reporter ownership, or contact storage layout changes. Geometry-only
value changes do not change this stamp.

`layout_hash` must include the ordered list of contact sources, their reporter
ids, model kinds, family kinds, stencil sizes, contact counts, and contact
storage layout tokens. `content_hash` must include the active stencil vertex ids
in source order. Neither hash may include positions, distances, barrier values,
friction bases, or other numeric data that changes without changing symbolic
matrix destinations.

## Core Data Structures

### Native Matrix Target

```cpp
enum class SocuAssemblyBand : std::uint8_t
{
    Diag,
    FirstOffdiag,
};

enum class SocuAssemblySideKind : std::uint8_t
{
    None,
    Fem,
    Abd,
};

enum class SocuAssemblyWriteKind : std::uint8_t
{
    ExactFemFem,
    ExactAbdFem,
    ExactFemAbd,
    ExactAbdAbdSameBody,
    ExactAbdAbdCrossBody,
    DiagBlockFem,
    DiagBlockAbd,
    DiagScalarFem,
    DiagScalarAbd,
    LumpScalarFem,
    LumpScalarAbd,
    Drop,
    Skipped,
};

enum class SocuContactTaskFlag : std::uint8_t
{
    None = 0,
    TransposedFirstOffdiag = 1u << 0,
    MirrorDiagBlock = 1u << 1,
    SameAbdBody = 1u << 2,
    HotReduceEligible = 1u << 3,
    DebugRejected = 1u << 4,
};
```

`Diag` and `DiagLump` are different policies and must never share the same
numeric writer:

- `DiagBlock*` writes the exact local diagonal block for a vertex when that
  vertex diagonal block is representable in native `D`.
- `DiagScalar*` writes only scalar diagonal entries. It is used when the exact
  vertex diagonal block is not symbolically representable or when the selected
  compatibility mode intentionally matches the current scalar-diag native
  fallback.
- `LumpScalar*` writes row-sum absolute-value lumps from the full stencil row
  into scalar diagonal entries.
- `Drop` records that the contribution is outside the supported SOCU band and
  is intentionally omitted.

### Compact Side Table

Side records are stored once per unique active global vertex in the plan. A
half-block task references side ids instead of embedding full row and column
side arrays.

```cpp
using SocuAssemblySideId = std::uint32_t;
using SocuContactSourceId = std::uint32_t;
using SocuContactProgramId = std::uint32_t;

inline constexpr SocuAssemblySideId SocuInvalidAssemblySideId =
    std::numeric_limits<SocuAssemblySideId>::max();
inline constexpr SocuContactSourceId SocuInvalidContactSourceId =
    std::numeric_limits<SocuContactSourceId>::max();
inline constexpr SocuContactProgramId SocuInvalidContactProgramId =
    std::numeric_limits<SocuContactProgramId>::max();

struct SocuAssemblyDofLane
{
    std::uint32_t block = 0;
    std::uint16_t lane = 0;
    std::uint8_t component = 0;
    std::uint8_t flags = 0;
    Float weight = Float{0};
};

struct SocuAssemblySideRecord
{
    IndexT global_vertex = -1;
    IndexT old_dof = -1;
    IndexT dof_count = 0;
    IndexT abd_body = -1;
    IndexT abd_jacobian_index = -1;

    SocuAssemblySideKind kind = SocuAssemblySideKind::None;
    bool fixed = false;
    bool writable = false;

    std::uint32_t first_lane = 0;
    std::uint16_t lane_count = 0;
    std::uint16_t reserved = 0;
};
```

The `SocuAssemblyDofLane` array contains 3 entries for FEM sides and 12 entries
for ABD sides. It stores native block/lane/component/projection weight once per
side, not once per half-block.

Implementation note: after correctness is stable, this can be specialized into
smaller FEM and ABD side layouts. The first version should prefer clarity and
testability over byte-perfect packing.

Side/lane records are not semantically contact-topology dependent. Given the
same `SocuVertexSidePlanKey`, a global vertex maps to the same side kind,
fixed/writable state, native block/lane/component tuples, and ABD projection
weights no matter which contacts reference it. The final design therefore treats
the side/lane plan as a global or demand-filled `vertex -> side/lane` cache
whose key does not contain contact topology.

There is one important distinction:

- Semantic side rebuild: the meaning of a vertex mapping changed because
  ordering, descriptor, mapping, fixed flag, horizon, block size, or projection
  changed. This increments `native_contact_side_plan_rebuild_count`.
- Side coverage refresh/fill: the semantic mapping is unchanged, but the
  current cache does not yet contain every vertex needed by the new contact
  topology. This increments coverage/fill counters, not semantic side rebuild
  counters.

The final performance target is topology churn with no semantic side rebuild
and no active-side-set refresh: only the contact program plan rebuilds. M2 is
allowed to use a temporary active-contact-vertex side table for deterministic
correctness-first development, but that mode is explicitly not the final
two-level design. It must report `active_side_set_changed` and side coverage
refresh cost separately so M2 performance numbers cannot be mistaken for the
final topology-churn target.

### Contact Microprograms

Each contact owns a compact program. The program groups all writes for one
local Hessian evaluation so the executor can compute `H` once per contact.

```cpp
enum class SocuContactFamily : std::uint8_t
{
    PT,
    EE,
    PE,
    PP,
    PH,
};

enum class SocuContactModelKind : std::uint8_t
{
    SimplexNormal,
    SimplexFrictional,
    VertexHalfPlaneNormal,
    VertexHalfPlaneFrictional,
};

enum class SocuContactProgramKind : std::uint8_t
{
    Exact,
    Diag,
    DiagLump,
    Drop,
    Skipped,
    MixedRejectedDebugOnly,
};

enum class SocuContactProgramMapStatus : std::uint16_t
{
    Missing = 0,
    Valid = 1,
    Skipped = 2,
    Dropped = 3,
    MixedRejected = 4,
};

struct SocuContactSourceHeader
{
    SocuContactSourceId source_id = SocuInvalidContactSourceId;
    std::uint32_t reporter_id = 0;

    SocuContactModelKind model = SocuContactModelKind::SimplexNormal;
    SocuContactFamily family = SocuContactFamily::PT;
    std::uint16_t stencil_size = 0;
    std::uint16_t flags = 0;

    std::uint32_t contact_count = 0;
    std::uint32_t first_program = 0;
    std::uint32_t program_count = 0;
    std::uint32_t first_source_to_program = 0;
};

struct SocuContactSourceToProgram
{
    SocuContactProgramId program_id = SocuInvalidContactProgramId;
    SocuContactProgramMapStatus status = SocuContactProgramMapStatus::Missing;
    std::uint16_t reserved = 0;
};

struct SocuContactProgramHeader
{
    SocuContactSourceId source_id = SocuInvalidContactSourceId;
    IndexT local_contact_id = -1;
    SocuContactModelKind model = SocuContactModelKind::SimplexNormal;
    SocuContactFamily family = SocuContactFamily::PT;
    SocuContactProgramKind program_kind = SocuContactProgramKind::Skipped;

    std::uint32_t first_task = 0;
    std::uint16_t task_count = 0;
    std::uint16_t stencil_size = 0;

    SocuAssemblySideId side_ids[4] = {};
};

struct SocuContactMicroTask
{
    SocuAssemblySideId row_side = 0;
    SocuAssemblySideId col_side = 0;

    std::uint8_t local_row_vertex = 0;
    std::uint8_t local_col_vertex = 0;
    SocuAssemblyBand band = SocuAssemblyBand::Diag;
    SocuAssemblyWriteKind write_kind = SocuAssemblyWriteKind::Skipped;

    std::uint32_t block_or_left_block = 0;
    std::uint8_t flags = 0;
    std::uint8_t reserved[3] = {};
};
```

`SocuContactMicroTask` is intentionally much smaller than
`SocuNativeContactStencilTarget`. It does not duplicate lane arrays. The writer
loads lane data through `row_side` and `col_side`.

`source_id + local_contact_id` is the only valid source-contact identity. A
plain local contact id is ambiguous because normal simplex, frictional simplex,
normal PH, and frictional PH reporters can all own contact `0` in the same
linear build.

Task flags have fixed semantics:

- `TransposedFirstOffdiag`: the source row side belongs to the left native block
  and the source column side belongs to the right native block, so the `E`
  writer swaps lanes to match SOCU storage.
- `MirrorDiagBlock`: the local half-block must also accumulate the transposed
  value into the same native diagonal block.
- `SameAbdBody`: row and column sides project to the same ABD body and use the
  same-body symmetric ABD/ABD microkernel.
- `HotReduceEligible`: the task may be moved from direct scatter to owner-reduce
  when hot-block splitting is enabled.
- `DebugRejected`: debug-only marker for a task that exists only to report an
  unsupported symbolic state.

### Buckets

Programs are bucketed by model, family, program kind, and execution strategy:

```cpp
enum class SocuContactExecutionStrategy : std::uint8_t
{
    DirectScatter,
    DetectOnlyHotBlock,
    RecomputeOwnerReduce,
    CachedMicroblockOwnerReduce,
};

struct SocuContactProgramBucket
{
    SocuContactModelKind model;
    SocuContactFamily family;
    SocuContactProgramKind program_kind;
    SocuContactExecutionStrategy execution_strategy =
        SocuContactExecutionStrategy::DirectScatter;
    std::uint32_t first_program = 0;
    std::uint32_t program_count = 0;
};

enum class SocuVertexSideCoverageMode : std::uint8_t
{
    Global,
    DemandFilled,
    ActiveSetTemporary,
};

struct SocuVertexSideCoverageStamp
{
    SocuVertexSideCoverageMode mode = SocuVertexSideCoverageMode::Global;
    std::uint64_t active_side_set_hash = 0;
    SizeT covered_vertex_count = 0;
    bool complete_for_current_contacts = false;
};

struct SocuVertexSidePlan
{
    SocuVertexSidePlanKey key;
    muda::DeviceBuffer<SocuAssemblySideRecord> sides;
    muda::DeviceBuffer<SocuAssemblyDofLane> lanes;

    // Final modes use either a global dense/remap table or a demand-filled
    // device cache. M2 may temporarily use ActiveSetTemporary with the sorted
    // unique active vertex list, but that mode must report coverage refreshes.
    muda::DeviceBuffer<IndexT> sorted_side_vertices;
    SocuVertexSideCoverageStamp coverage;

    SocuContactPlanStats last_stats;
};

struct SocuContactProgramPlan
{
    SocuContactProgramPlanKey key;

    muda::DeviceBuffer<SocuContactSourceHeader> sources;
    muda::DeviceBuffer<SocuContactProgramHeader> programs;
    muda::DeviceBuffer<SocuContactMicroTask> tasks;
    muda::DeviceBuffer<SocuContactProgramBucket> buckets;
    muda::DeviceBuffer<SocuContactSourceToProgram> source_to_program;

    // Optional, enabled after hot-block detection is implemented.
    SocuHotBlockPlan hot_blocks;

    SocuContactPlanStats last_stats;
};

struct SocuContactAssemblyPlan
{
    SocuVertexSidePlan side_plan;
    SocuContactProgramPlan program_plan;
};
```

Device code consumes view-only plans:

```cpp
struct SocuContactAssemblyPlanView
{
    SocuVertexSidePlanKey side_key;
    SocuContactProgramPlanKey program_key;

    muda::CBufferView<SocuContactSourceHeader> sources;
    muda::CBufferView<SocuAssemblySideRecord> sides;
    muda::CBufferView<SocuAssemblyDofLane> lanes;
    muda::CBufferView<SocuContactProgramHeader> programs;
    muda::CBufferView<SocuContactMicroTask> tasks;
    muda::CBufferView<SocuContactProgramBucket> buckets;
    muda::CBufferView<SocuContactSourceToProgram> source_to_program;
    SocuHotBlockPlanView hot_blocks;

    MUDA_GENERIC bool valid() const noexcept;

    MUDA_DEVICE SocuContactProgramId program_for(
        SocuContactSourceId source_id,
        IndexT local_contact_id) const noexcept;

    MUDA_DEVICE muda::CBufferView<SocuAssemblyDofLane> lanes_for(
        SocuAssemblySideId side_id) const noexcept;
};
```

`program_for` must be O(1): it indexes the source header, checks
`local_contact_id`, then reads
`source_to_program[first_source_to_program + local_contact_id]`. It must not
scan `programs`.

The O(1) lookup contract requires dense source ids:

```cpp
sources.size() == source_count;
sources[source_id].source_id == source_id;
```

If an upstream reporter cannot provide dense ids, the builder must create a
compact `source_id_to_source_index` table during symbolic build and the cost
must be reported. Production milestones prefer dense ids; debug validation must
fail on duplicate ids, sparse ids without a map, or any source header whose
stored id does not match its array position.

`SocuContactPlanStats` is copied to the report after build and after numeric
execution:

```cpp
struct SocuContactPlanStats
{
    SizeT side_count = 0;
    SizeT lane_count = 0;
    SizeT program_count = 0;
    SizeT task_count = 0;
    SizeT bucket_count = 0;

    SizeT exact_program_count = 0;
    SizeT diag_program_count = 0;
    SizeT diag_lump_program_count = 0;
    SizeT drop_program_count = 0;
    SizeT skipped_program_count = 0;
    SizeT mixed_rejected_program_count = 0;
    SizeT diag_block_task_count = 0;
    SizeT diag_scalar_task_count = 0;
    SizeT lump_scalar_task_count = 0;

    SizeT hot_diag_block_count = 0;
    SizeT hot_offdiag_block_count = 0;

    SizeT side_cache_hit_count = 0;
    SizeT side_rebuild_count = 0;
    SizeT side_coverage_hit_count = 0;
    SizeT side_coverage_refresh_count = 0;
    SizeT side_coverage_fill_count = 0;
    SizeT active_side_set_changed_count = 0;
    SizeT program_cache_hit_count = 0;
    SizeT program_rebuild_count = 0;

    // Backward-compatible aggregate counters: cache_hit means both layers hit;
    // rebuild means at least one layer rebuilt.
    SizeT cache_hit_count = 0;
    SizeT rebuild_count = 0;
};
```

## Host Interfaces

### Top-Level Builder

```cpp
struct SocuNativeAssemblyBuildInput
{
    SocuVertexSidePlanKey side_key;
    SocuContactProgramPlanKey contact_key;

    SocuNativeMatrixView<ActivePolicy::SolveScalar> native_matrix;
    muda::CBufferView<SocuNativeDofDescriptor> dof_descriptors;
    muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors;
    muda::CBufferView<ABDJacobi> abd_vertex_to_J;

    SocuContactBuildInput contact;

    bool build_contact_plan = true;
    bool build_hot_block_plan = false;
    bool debug_validate_plan = false;
};

struct SocuNativeAssemblyPlanView
{
    SocuVertexSidePlanKey side_key;
    SocuContactProgramPlanKey contact_key;
    SocuContactAssemblyPlanView contact;
    SocuContactPlanStats stats;

    MUDA_GENERIC bool valid() const noexcept;
};

class SocuNativeAssemblyBuilder
{
public:
    SocuNativeAssemblyPlanView ensure_plan(
        const SocuNativeAssemblyBuildInput& input,
        cudaStream_t stream);

    void invalidate_all();
    const SocuContactPlanStats& last_contact_stats() const noexcept;

private:
    SocuVertexSidePlan m_side_plan;
    SocuContactProgramPlan m_contact_plan;
};
```

The M1 code may pass a combined `SocuAssemblyPlanKey` through this interface
while the first cache tests land. Before M2 program emission is merged, the
builder implementation must expose the split key behavior above, even if a
compatibility wrapper keeps the old type name in reports.
`ensure_plan` must debug-validate that `input.contact_key.side_key` equals
`input.side_key`; a mismatch means the contact program could reference side ids
from a different ordering/projection.

`SocuApproxSolver::prepare_structured_chain(...)` should call `ensure_plan`
after native descriptors are ready and before contact assembly begins. The
resulting plan view is installed into `StructuredAssemblyInfo`:

```cpp
info.set_native_assembly_plan(m_runtime->native_assembly_plan_view());
```

### Contact Build Input

```cpp
struct SocuContactSourceInput
{
    SocuContactSourceHeader header;

    // Exactly one stencil view is active, selected by header.family and
    // header.stencil_size.
    muda::CBufferView<Vector4i> stencil4; // PT and EE.
    muda::CBufferView<Vector3i> stencil3; // PE.
    muda::CBufferView<Vector2i> stencil2; // PP and PH.
};

struct SocuContactBuildInput
{
    SocuContactTopologyStamp topology;
    StructuredContactOffbandPolicy offband_policy =
        StructuredContactOffbandPolicy::Drop;

    span<const SocuContactSourceInput> sources;
};
```

Each reporter contributes one or more sources:

- simplex normal: PT, EE, PE, PP
- simplex frictional: PT, EE, PE, PP
- vertex-half-plane normal: PH
- vertex-half-plane frictional: PH

`reporter_id` must be stable for one linear build and included in
`layout_hash`. Source order is deterministic: reporter order first, then family
order `PT, EE, PE, PP, PH`.

The exact contact stencil source type can differ from this sketch. The required
contract is that builder kernels can read active matrix vertex ids without
running the numeric contact Hessian code. For PH, `stencil2(i)(0)` is the active
matrix vertex and `stencil2(i)(1)` is the half-plane id used later by the
numeric evaluator; only the active vertex creates a side record.

## Device Interfaces

### Compatibility Writer

Milestones M2 through M4 keep the existing contact kernels responsible for
computing `H`. They replace the per-half-block target array with a compact
program lookup:

```cpp
template <typename StoreT, typename SolveT>
struct SocuContactProgramWriter
{
    SocuNativeMatrixView<SolveT> matrix;
    SocuContactAssemblyPlanView plan;
    muda::BufferView<IndexT> counters;

    template <int StencilSize, typename HMat>
    MUDA_DEVICE void write_contact(SocuContactSourceId source_id,
                                   IndexT local_contact_id,
                                   const HMat& H) const noexcept;

    template <typename H3>
    MUDA_DEVICE void write_exact_task(const SocuContactMicroTask& task,
                                      const H3& H3x3) const noexcept;

    template <int StencilSize, typename HMat>
    MUDA_DEVICE void write_diag_program(
        const SocuContactProgramHeader& program,
        const HMat& H) const noexcept;

    template <int StencilSize, typename HMat>
    MUDA_DEVICE void write_diag_lump_program(
        const SocuContactProgramHeader& program,
        const HMat& H) const noexcept;
};
```

`write_exact_task` dispatches once at task level by `SocuAssemblyWriteKind`.
Inside a task, it runs fixed-size loops:

- FEM/FEM: 3 by 3.
- ABD/FEM: 12 by 3.
- FEM/ABD: 3 by 12.
- ABD/ABD: 12 by 12 or the same-body symmetric variant.

No scalar pair should be reclassified inside these loops.

`write_contact` first resolves `program_id = plan.program_for(source_id,
local_contact_id)`. Invalid mappings are counted as skipped or mixed-rejected
according to the source-to-program status. They must not silently return in
debug validation builds.

### Plan-Owned Executor

Milestone M5 moves from model-owned kernels to plan-owned execution. Contact
models provide typed source views and evaluator functors:

```cpp
template <typename StoreT, int StencilSize>
struct SocuContactEvaluator
{
    using Hessian = Eigen::Matrix<StoreT, StencilSize * 3, StencilSize * 3>;

    MUDA_DEVICE Hessian hessian(SocuContactSourceId source_id,
                                IndexT local_contact_id) const noexcept;
};

template <typename Evaluator, int StencilSize>
void execute_socu_contact_program_bucket(
    Evaluator evaluator,
    SocuContactAssemblyPlanView plan,
    SocuContactProgramBucket bucket,
    SocuNativeMatrixView<ActivePolicy::SolveScalar> matrix,
    muda::BufferView<IndexT> counters,
    cudaStream_t stream);
```

One thread block or warp group owns one contact program, computes `H` once, and
then executes all tasks for that program. This avoids recomputing the full local
Hessian per half-block task.

## Write Semantics

### Native Matrix Orientation

The executor writes the same native matrix layout used by `SocuNativeMatrixView`:

- `D(block, row_lane, col_lane)` stores entries whose row and column DoFs are in
  the same native block.
- `E(left_block, row_lane, col_lane)` stores first-offdiag entries between
  native blocks `left_block` and `left_block + 1`.
- In `E`, `row_lane` always belongs to the right block and `col_lane` always
  belongs to the left block.

For a first-offdiag scalar pair:

- If the source row side is in the right block and the source column side is in
  the left block, the task does not set `TransposedFirstOffdiag`.
- If the source row side is in the left block and the source column side is in
  the right block, the builder sets `TransposedFirstOffdiag` and the writer
  swaps row/column lanes before indexing `E`.
- If the two blocks are not equal and not adjacent, the task cannot be exact.
  The whole contact program must become `Diag`, `DiagLump`, `Drop`, or
  `MixedRejectedDebugOnly`.

`block_or_left_block` is the native block for `D` tasks and the left native
block for `E` tasks. The writer must trust this value after debug validation;
it must not recompute block adjacency per scalar in the production path.

### Exact Writes

Exact program tasks cover only representable same-block or first-offdiag
half-blocks:

- FEM/FEM emits a fixed 3 by 3 microkernel.
- ABD/FEM emits a fixed 12 by 3 projected microkernel.
- FEM/ABD emits a fixed 3 by 12 projected microkernel.
- ABD/ABD cross-body emits a fixed 12 by 12 projected microkernel.
- ABD/ABD same-body emits the same-body symmetric projected microkernel and
  must set `SameAbdBody`.

The writer may bounds-check in debug builds. The performance writer must not:

- read `old_to_chain`
- read native DoF descriptors
- call `classify_dof_pair`
- call legacy structured contact sink functions
- branch per scalar on diag/first-offdiag/off-band classification

### Diag Policy

`StructuredContactOffbandPolicy::Diag` is whole-stencil fallback. If any
half-block of a contact stencil is off-band, the exact off-diagonal stencil
program is not emitted. Instead, the builder emits one diagonal fallback task per
active local vertex.

For each local vertex `k`, `Diag` uses only the local diagonal block
`H.block<3, 3>(3 * k, 3 * k)`:

- If the vertex diagonal block is fully representable in native `D`, emit
  `DiagBlockFem` or `DiagBlockAbd`.
- If the full vertex diagonal block is not representable, emit `DiagScalarFem`
  or `DiagScalarAbd`.
- Compatibility mode may force scalar diagonal fallback to match the current
  native `DiagFallback` behavior. This mode must be explicit in the plan key and
  report.

`Diag` is not row-sum lumping. It preserves the signed diagonal-block values
where the native band can represent them.

### DiagLump Policy

`StructuredContactOffbandPolicy::DiagLump` is also whole-stencil fallback, but
it uses absolute row-sum lumping over the full local Hessian row for each local
vertex. For local vertex `k` and physical component `r`:

```text
lump[k][r] = sum_j sum_c abs(H(3 * k + r, 3 * j + c))
```

The numeric writer emits:

- FEM: three scalar diagonal writes, one per physical component.
- ABD: one scalar diagonal write per ABD DoF `q`, using
  `weight(q) * lump[component(q)] * weight(q)`.

`DiagLump` never emits full diagonal-block writes.

### Drop Policy

`StructuredContactOffbandPolicy::Drop` emits exact programs only when all
required half-blocks are representable. Otherwise the program is marked `Drop`
and contributes no matrix values. Dropped programs must be counted so the report
can distinguish intentional approximation from missing native coverage.

## Builder Kernel Pipeline

The symbolic builder owns two rebuild pipelines. Both are CUDA-side for the
large data: side collection, side materialization, stencil side lookup,
classification, task emission, prefix sums, bucket building, and hot-block
histograms run on the same stream as assembly preparation. The host compares
keys, collects the small ordered reporter/source metadata list, resizes
`muda::DeviceBuffer`s, launches kernels, and copies back scalar report data.
The final assembly path must not copy full contact arrays to host.

Host-side source ordering is allowed because reporters are C++ objects and the
number of source headers is small. Per-contact stencil ids, contact counts, and
program construction stay on device.

### Vertex Side/Lane Rebuild And Coverage

The final side/lane design has two operations with different performance
meaning:

1. Semantic side rebuild.
   - Runs only when `SocuVertexSidePlanKey` changes.
   - Recomputes side records and lane records because vertex mapping semantics
     changed.
   - Increments `native_contact_side_plan_rebuild_count`.

2. Coverage fill or refresh.
   - Runs when the side key hits but the current side cache does not cover all
     vertices referenced by the contact program build.
   - Does not mean the vertex mapping changed.
   - Increments `native_contact_side_coverage_refresh_count` or
     `native_contact_side_coverage_fill_count`, never
     `native_contact_side_plan_rebuild_count`.

Supported coverage modes:

- `global`: materialize all FEM/ABD vertices that can ever appear in contact
  assembly for the current side key. This has the cleanest topology-churn
  behavior and the simplest lookup, but may use more memory.
- `demand_filled`: keep a persistent device cache keyed by global vertex. When
  a new topology references an uncovered vertex, fill only the missing side
  records and update the vertex-to-side lookup. Existing side ids remain valid.
- `active_set_temporary`: M2-only correctness mode. Materialize the sorted
  unique vertex set of the current active contacts. If contacts introduce a new
  vertex set, refresh the active table and report
  `native_contact_active_side_set_changed = true`.

The side coverage pipeline is CUDA-side:

1. Required vertex discovery.
   - `global` mode scans mapping/descriptor ranges.
   - `demand_filled` and `active_set_temporary` modes read active contact
     sources on device and emit required global vertex ids.
   - PH sources emit only `PH(0)` as a matrix vertex.

2. Side materialization.
   - Convert each required global vertex id into a
     `SocuAssemblySideRecord`.
   - Fill `SocuAssemblyDofLane` entries from native vertex/dof descriptors and
     ABD projection weights.
   - Mark fixed or unmapped sides as skipped/read-only according to the plan
     representation.

3. Side lookup update.
   - `global` mode can use a dense or remap table from global vertex to side id.
   - `demand_filled` mode updates an epoch-tagged device hash/remap table.
   - `active_set_temporary` mode may use lower-bound lookup into the sorted
     unique active vertex table.

Final acceptance requires either `global` or `demand_filled` mode before native
contact performance cutover. `active_set_temporary` is allowed only through the
M2 correctness builder and must be called out in reports and benchmark tables.

### Contact Program Rebuild

Run this pipeline when `SocuContactProgramPlanKey` changes. A side key change
also changes the contact program key, because side ids, block/lane projections,
and whole-stencil classifications may change.

1. Source indexing.
   - Emit `SocuContactSourceHeader` records in deterministic source order.
   - Allocate one `source_to_program` entry per source contact.
   - Fill invalid entries with `SocuInvalidContactProgramId`.

2. Stencil side lookup.
   - For each contact, read active stencil vertex ids on device.
   - Resolve each active vertex to a side id through the side-plan lookup.
   - Store side ids in the program header; do not duplicate lane arrays.

3. Whole-stencil classification.
   - Use cached side/lane records to classify the full contact stencil once
     against the native band.
   - Decide `Exact`, `Diag`, `DiagLump`, `Drop`, `Skipped`, or
     `MixedRejectedDebugOnly`.
   - The `vertex -> side/lane` mapping itself is stable under topology changes,
     but the set of side pairs inside a contact stencil is not. Therefore this
     classification is contact-program state.

4. Program sizing.
   - Count programs, tasks, buckets, fallback counts, and optional block hit
     counts.
   - Prefix-sum sizes.

5. Program emission.
   - Emit one `SocuContactProgramHeader` per source contact.
   - Emit compact `SocuContactMicroTask` records for exact or diag block writes.
   - Fill `source_to_program[source.first_source_to_program + local_contact_id]`.
   - Emit bucket ranges sorted by model, family, program kind, and execution
     strategy.

6. Optional hot-block split.
   - Use block hit counts to identify high-contention `D/E` blocks.
   - Split hot tasks into owner-reduce adjacency lists.
   - Leave low-contention tasks on direct atomic scatter.

Rebuild scheduling rules:

- Geometry-only numeric changes run neither symbolic pipeline.
- Contact topology/content/layout changes run the contact program pipeline. In
  final `global`/`demand_filled` modes, the side semantic key should hit and
  the side coverage should already hit or fill only missing vertices. In M2
  `active_set_temporary` mode, topology changes may refresh the active side
  table; this is reported as coverage refresh, not semantic side rebuild.
- Runtime reorder install, ordering epoch changes, native descriptor changes,
  fixed/mapping changes, and ABD projection changes run both pipelines.
- Off-band policy changes run only the contact program pipeline when the side
  key is unchanged. In normal simulation this policy is expected to be constant.

## Hot-Block Owner-Reduce Strategy

The first production executor should be contact-centric atomic scatter because
it is simpler and avoids duplicate Hessian evaluation. Hot-block owner-reduce is
an optional second strategy for dense contact clusters.

```cpp
struct SocuBlockContributionRef
{
    std::uint32_t program_id = 0;
    std::uint16_t task_offset = 0;
    std::uint16_t flags = 0;
};

struct SocuHotBlockRange
{
    SocuAssemblyBand band = SocuAssemblyBand::Diag;
    std::uint32_t block_or_left_block = 0;
    std::uint32_t first_ref = 0;
    std::uint32_t ref_count = 0;
};

struct SocuHotBlockPlan
{
    muda::DeviceBuffer<SocuHotBlockRange> ranges;
    muda::DeviceBuffer<SocuBlockContributionRef> refs;
    SizeT threshold = 0;
};

struct SocuHotBlockPlanView
{
    muda::CBufferView<SocuHotBlockRange> ranges;
    muda::CBufferView<SocuBlockContributionRef> refs;
    SizeT threshold = 0;

    MUDA_GENERIC bool valid() const noexcept;
};
```

Acceptance rule: owner-reduce is enabled only when profiler evidence shows
atomic contention dominates. It must remain optional because it can recompute
contact Hessians when many hot blocks reference the same contact.

M6 is split into three substeps:

1. Detection only.
   - Build block hit histograms.
   - Report hot `D/E` block counts.
   - Keep all writes on direct scatter.

2. Direct recompute owner-reduce.
   - Owner-reduce kernels may recompute a contact Hessian for each hot block
     reference.
   - This mode is allowed only for measurements and for models where Hessian
     evaluation is known cheaper than atomic contention.

3. Cached microblock owner-reduce.
   - A preceding numeric pass writes hot-task microblock values into temporary
     storage.
   - Owner-reduce consumes those values without recomputing contact Hessians.
   - This is the only owner-reduce mode eligible for default production use if
     recomputation is measurable in Nsight Compute.

The report must state the selected strategy:

```text
native_contact_hot_reduce_strategy = "off" | "detect_only" | "recompute" | "cached_microblock"
```

## Integration Points

Planned files:

- `src/backends/cuda_mixed_socu/linear_system/socu_contact_plan_types.h`
- `src/backends/cuda_mixed_socu/linear_system/socu_native_assembly_plan.h`
- `src/backends/cuda_mixed_socu/linear_system/socu_contact_assembly_plan.h`
- `src/backends/cuda_mixed_socu/linear_system/socu_contact_plan_builder.cu`
- `src/backends/cuda_mixed_socu/linear_system/socu_contact_program_writer.h`
- `src/backends/cuda_mixed_socu/linear_system/socu_contact_program_debug_compare.h`
- `src/backends/cuda_mixed_socu/linear_system/socu_contact_executor.cu`
- `src/backends/cuda_mixed_socu/linear_system/socu_contact_plan_report.h`

Existing integration points:

- `GlobalLinearSystem::StructuredAssemblyInfo` gains a native assembly plan
  view and exposes it through `native_contact_sink()`.
- `SocuApproxSolver::prepare_structured_chain` owns the builder cache and
  installs the current plan view into `StructuredAssemblyInfo`.
- `GlobalDyTopoEffectManager` provides `SocuContactTopologyStamp` without
  copying full contact arrays to the host during final assembly.
- Contact model native TUs first call `SocuContactProgramWriter`. Later they
  can move to plan-owned executor launches.
- Existing `SocuNativeContactStencilTarget` remains behind a temporary
  compatibility flag until the compact plan is validated.

Header hygiene rules:

- `socu_contact_plan_types.h` contains only POD ids, enums, compact records, and
  view types needed by kernels.
- Builder and executor implementations live in `.cu` files. Model-specific
  evaluator glue stays in contact-model native TUs.
- `GlobalLinearSystem` and other widely included headers must not include heavy
  template builders or legacy debug comparison helpers. They may store opaque
  views or forward-declared owner types only.
- `socu_contact_program_writer.h` is production-only. It must not contain
  legacy target fields, debug compare state, or old writer helpers.
- `socu_contact_program_debug_compare.h` is compiled only in debug/full-fallback
  comparison builds and is the only compact-plan header allowed to reference
  legacy target tables for oracle comparison.

## Build Matrix

The redesign must keep three build modes healthy:

All three modes are SOCU native builder modes. They should keep the Augmented
Lagrangian IPC pipeline out of the default compile set with
`UIPC_CUDA_MIXED_SOCU_BUILD_AL_PIPELINE=OFF`; AL is a separate pipeline and its
translation units are not part of the native contact builder performance path.
If an AL scene is loaded in this build, initialization must fail explicitly
instead of relying on missing registrations or link errors.

Default iteration rule:

- Native builder development and performance milestones use
  `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=ON` and
  `UIPC_CUDA_MIXED_SOCU_BUILD_AL_PIPELINE=OFF`.
- Builder-only contract and smoke tests may also enable
  `UIPC_CUDA_MIXED_WRECKING_BALL_MINIMAL_BUILD=ON`. This existing narrow build
  removes unrelated heavy constitutions and coupling pipelines, including
  `inter_primitive_effect_system/constitutions/*.cu`, while keeping the SOCU
  chain/base/diag/RHS/native-contact development surface.
- This keeps AL, legacy structured contact fallback TUs, and unrelated
  inter-primitive stitch constitutions out of the compile queue, so M1-M5 can
  iterate on the native builder without paying for unrelated CUDA template
  instantiations.
- Full fallback builds are reference/oracle builds only. Use them for explicit
  matrix-diff gates and regression bisection, not for normal milestone work.

1. Native-only development/performance build.
   - `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=ON`.
   - `UIPC_CUDA_MIXED_SOCU_BUILD_AL_PIPELINE=OFF`.
   - `UIPC_CUDA_MIXED_WRECKING_BALL_MINIMAL_BUILD=ON` is allowed for
     builder-only M0-M5 contract tests and Wrecking Ball scene gates.
   - Legacy structured contact TUs are excluded.
   - Inter-primitive stitch constitution TUs are excluded when the minimal flag
     is enabled; this build does not validate scenes that require those
     constitutions.
   - This is the default build for M1-M5 implementation, focused unit tests,
     synthetic builder tests, and performance gates.
   - Before the compact plan covers a contact family, unsupported native-only
     coverage must fail with an explicit gate reason instead of silently
     linking the legacy sink.
   - Used to ensure native contact coverage does not depend on legacy TUs or on
     the abandoned per-half-block target adapter.

2. Full fallback reference build.
   - `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=OFF`.
   - `UIPC_CUDA_MIXED_SOCU_BUILD_AL_PIPELINE=OFF`.
   - Legacy structured contact TUs and compact plan code can both be compiled.
   - The abandoned per-half-block native contact target path is not part of
     this branch unless a short-lived comparison flag explicitly restores it.
   - Used for reference matrix diff and debug comparison only.

3. Native compact-plan performance build.
   - Adds a CMake or compile definition such as
     `UIPC_CUDA_MIXED_SOCU_CONTACT_PLAN_ONLY=ON`.
   - Keeps `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=ON`.
   - Keeps `UIPC_CUDA_MIXED_SOCU_BUILD_AL_PIPELINE=OFF`.
   - Excludes legacy structured contact TUs and any temporary compatibility
     adapter.
   - Used for final performance gates.

New CUDA translation units must be listed explicitly beside the existing native
linear-system TUs so native-only builds do not accidentally drop builder kernels:

```text
linear_system/socu_contact_plan_builder.cu
linear_system/socu_contact_executor.cu
```

Compile-resource acceptance:

- No new performance TU may include legacy structured contact sink headers.
- Each new CUDA TU must stay within the current native-only build peak RSS plus
  10%, measured with `/usr/bin/time -v`, unless the journal records an explicit
  exception.
- If a TU exceeds that budget, split by model/family before adding more template
  instantiations.
- Source-scan acceptance must inspect `build.ninja`, `ninja -n <test target>`,
  and `compile_commands.json` when available. The native-only development and
  compact-plan performance builds must not compile legacy structured contact
  TUs, abandoned per-half-block target TUs, or AL pipeline TUs.

## Report Fields

Add report fields under the existing SOCU report:

```text
native_contact_plan_enabled
native_contact_plan_executor_enabled
native_contact_hot_reduce_enabled
native_contact_scalar_diag_compat_enabled
native_contact_plan_cache_hit
native_contact_plan_rebuild_count
native_contact_plan_build_ms
native_contact_side_plan_cache_hit
native_contact_side_plan_rebuild_count
native_contact_side_plan_build_ms
native_contact_side_coverage_mode
native_contact_side_coverage_cache_hit
native_contact_side_coverage_refresh_count
native_contact_side_coverage_refresh_ms
native_contact_side_coverage_fill_count
native_contact_active_side_set_changed
native_contact_active_side_vertex_count
native_contact_program_plan_cache_hit
native_contact_program_plan_rebuild_count
native_contact_program_plan_build_ms
native_contact_numeric_ms
native_contact_hot_reduce_ms
native_contact_hot_reduce_strategy
native_contact_probe_path
native_contact_replay_path
native_contact_side_count
native_contact_lane_count
native_contact_program_count
native_contact_task_count
native_contact_bucket_count
native_contact_exact_program_count
native_contact_diag_program_count
native_contact_diag_lump_program_count
native_contact_drop_program_count
native_contact_skipped_program_count
native_contact_mixed_rejected_program_count
native_contact_diag_block_task_count
native_contact_diag_scalar_task_count
native_contact_lump_scalar_task_count
native_contact_hot_diag_block_count
native_contact_hot_offdiag_block_count
```

Path fields use fixed strings:

```text
native_contact_probe_path = "off" | "bypass" | "symbolic_plan" | "native_executor" | "legacy_structured"
native_contact_replay_path = "off" | "native_plan" | "legacy_structured"
native_contact_side_coverage_mode = "global" | "demand_filled" | "active_set_temporary"
```

JSON placement:

- `timing`: aggregate `native_contact_plan_build_ms`, split
  `native_contact_side_plan_build_ms` and
  `native_contact_program_plan_build_ms`,
  `native_contact_side_coverage_refresh_ms`, `native_contact_numeric_ms`,
  `native_contact_hot_reduce_ms`, and `native_contact_hot_reduce_strategy`.
- `contact`: all plan size, exact/fallback/drop, mixed-rejected, and hot-block
  count fields, plus split side/program cache hit and rebuild counters, side
  coverage mode, coverage hit/refresh/fill counters, and active-side-set
  counters.
- `status`: any nonzero mixed-rejected count in a production run adds a
  diagnostic detail even if the solve continues.
- `runtime_reorder`: `native_contact_probe_path`,
  `native_contact_replay_path`, and the existing runtime reorder graph-source
  fields.

`native_contact_side_plan_cache_hit` and
`native_contact_program_plan_cache_hit` are per solve.
`native_contact_plan_cache_hit` is a backward-compatible aggregate and is true
only when both layers hit. Split rebuild counts are cumulative for the runtime
and are reset when the SOCU runtime is rebuilt.
`native_contact_plan_rebuild_count` is the aggregate count of solves where at
least one layer rebuilt. Plan size and fallback counters are per solve. Plan
timings are recorded even when `SOCU_REPORT_COUNTERS=0`; they are timers, not
optional scalar counters.

`native_contact_side_coverage_refresh_count` is separate from
`native_contact_side_plan_rebuild_count`. A refresh means the side key hit but
the side cache needed more records or, in the temporary M2 mode, the active
side set changed. `native_contact_active_side_set_changed` may be true only in
`active_set_temporary` mode. Final performance gates prefer `global` or
`demand_filled` mode and reject topology-churn claims that hide active-side-set
refresh time inside semantic side rebuild time.

These counters are acceptance-critical. A performance result is not actionable
unless it states which symbolic layer was rebuilt, how many programs were
exact, how many fell back to `Diag` or `DiagLump`, which probe/replay path was
used, and whether hot-block reduction was enabled.

## Milestones

Milestone implementation order assumes the native-only development/performance
build above. Fallback reference builds are run only at the acceptance points that
explicitly compare against the legacy structured sink.

### M0: Baseline Instrumentation And Guard Rails

Deliverables:

- Add feature flags:
  - `linear_system/socu_approx/native_contact_plan=0/1`
  - `linear_system/socu_approx/native_contact_plan_executor=0/1`
  - `linear_system/socu_approx/native_contact_hot_reduce=0/1`
  - `linear_system/socu_approx/native_contact_hot_reduce_strategy=off|detect_only|recompute|cached_microblock`
  - `linear_system/socu_approx/native_contact_scalar_diag_compat=0/1`
- Add separate timers for:
  - native descriptor rebuild
  - contact plan build
  - numeric contact assembly
  - hot-block owner-reduce
- Add report fields listed above.
- Add a debug assert or counter for mixed/unhandled native contact programs.

Unit tests:

- Report serialization includes new fields with zero defaults.
- Flags default to off and do not change existing SOCU results.

Acceptance:

- Current native and legacy structured paths produce unchanged matrix diff
  results with all new flags off.
- `backend_cuda_mixed_socu_contract` still passes.

### M1: Epoch And Cache Correctness

Deliverables:

- Add `SocuContactTopologyStamp` to the dy-topology contact manager.
- Ensure final assembly receives a valid topology epoch, layout hash, and
  content hash.
- Add the M1 combined `SocuAssemblyPlanKey` as a temporary POD contract, and
  document the target split into `SocuVertexSidePlanKey` and
  `SocuContactProgramPlanKey`.
- Stop using host-copy contact signatures as the production cache key.
- Add deterministic `reporter_id` and `source_id` assignment for every contact
  source in one linear build.

Unit tests:

- Same contact count but different vertex ids changes the topology epoch/hash.
- Same contact count but different vertex ids without a runtime graph probe
  still rebuilds the final contact program plan.
- Same vertex ids with changed geometry does not change topology epoch/hash.
- Ordering epoch change invalidates both side and contact program plans.
- Off-band policy change invalidates the contact program plan only.
- Fixed flag or ABD/FEM mapping epoch change invalidates both plans.
- ABD projection epoch change invalidates both plans.
- ABD `x_bar()` or precomputed projection weight changes invalidate the plan
  even when contact topology and descriptor layout do not change.
- Reordering reporters or changing source family order changes `layout_hash`
  unless the source order is explicitly canonicalized.
- Probe-only graph signatures and bypassed probes do not pollute or refresh the
  final assembly plan cache.
- Contact topology/content changes invalidate the contact program key without
  invalidating the side key when ordering, descriptors, mapping, fixed flags,
  and projection are unchanged.
- Runtime reorder install invalidates both side and contact program keys.
- Off-band policy changes invalidate only the contact program key when the side
  key is unchanged.

Detailed M1 test specifications:

1. `socu_contact_topology_stamp.cu`: device topology hash producer tests.
   - Expose the device hash reducer as a small testable helper, for example
     `SocuContactTopologyHashWorkspace` or
     `compute_socu_contact_topology_hash_for_test(...)`, instead of only hiding
     it inside `GlobalDyTopoEffectManager`.
   - Allocate real `muda::DeviceBuffer<Vector4i>`,
     `muda::DeviceBuffer<Vector3i>`, and `muda::DeviceBuffer<Vector2i>`
     fixtures.
   - Test `same_count_different_vertex_ids`:
     - input A and B have identical counts and identical storage layout;
     - only vertex ids differ;
     - `layout_hash` stays equal;
     - `content_hash` differs;
     - a stamp cache fed with A then B bumps epoch.
   - Test `same_ids_geometry_only_stable`:
     - use the same topology buffers and change only synthetic geometry
       scalars outside the topology input;
     - `layout_hash`, `content_hash`, and epoch stay unchanged.
   - Test `empty_contact_families`:
     - empty PT/EE/PE/PP/PH buffers produce a valid zero-contact stamp;
     - repeated empty stamps do not bump epoch after the first valid stamp.
   - Test family separation:
     - the same vertex tuple inserted as PT and PE/PP/PH must not collide in
       `content_hash` because the family tag is part of the hash.
   - Test order sensitivity:
     - reordering contacts inside one source changes `content_hash`;
     - reordering source families changes `layout_hash` unless the builder
       explicitly canonicalizes source order.

2. `socu_contact_topology_stamp.cu`: full stamp cache state tests.
   - Add a lightweight host-side `SocuContactTopologyStampCache` helper or
     equivalent test-only wrapper around the production stamp state.
   - Feed stamps with:
     - same layout/content twice: cache hit, epoch unchanged;
     - same count but changed content hash: epoch increments;
     - changed layout token: epoch increments;
     - changed reporter count/source count: epoch increments.
   - Assert epoch starts from a nonzero value after the first valid stamp.

3. `socu_contact_assembly_plan.cu`: final plan key/cache tests.
   - Add `SocuContactPlanCache` tests before M2 code is allowed to depend on
     the cache.
   - `same_key_hits`: identical `SocuAssemblyPlanKey` returns cache hit and
     does not rebuild.
   - `topology_content_change_rebuilds`: identical contact count/layout but
     changed `contact_content_hash` rebuilds only the contact program layer.
   - `topology_epoch_change_rebuilds`: changed `contact_topology_epoch`
     rebuilds the contact program layer even if the hashes are accidentally
     equal.
   - `ordering_epoch_change_rebuilds` rebuilds both side and contact program
     layers.
   - `native_descriptor_epoch_change_rebuilds` rebuilds both layers.
   - `fixed_mapping_epoch_change_rebuilds` rebuilds both layers.
   - `vertex_projection_epoch_change_rebuilds` rebuilds both layers.
   - `offband_policy_change_rebuilds`, covering `Drop`, `Diag`, and
     `DiagLump`, rebuilds only the contact program layer.
   - `scalar_diag_compatibility_change_rebuilds` rebuilds only the contact
     program layer.

4. `socu_contact_assembly_plan.cu`: probe/final cache isolation tests.
   - Create separate probe-key and final-key fixtures, or use explicit cache
     domains if the implementation stores them together.
   - A probe-created graph state must not populate the final plan cache.
   - A bypassed probe with unchanged diagnostic signature must not refresh the
     final plan cache timestamp/epoch.
   - Final cache hit/rebuild counters must only change during final assembly
     cache lookup/build.

5. `policy_contract.cu` or a dedicated final-path spy test:
   - Add a fake/minimal `LinearSolver` that overrides
     `needs_contact_topology_stamp_for_final()`.
   - With the override returning `false`, structured final assembly must not
     call the topology stamp producer.
   - With the override returning `true`, structured final assembly must request
     exactly one topology stamp after solver workspace/stream setup and before
     final contact assembly.
   - The probe path may request `contact_set_signature()` only when
     `needs_contact_set_signature_for_probe(frame)` is true; this call must not
     be used as the final plan key.

6. Source id contract tests.
   - Dense source ids pass:
     `source_id == sources[source_id].source_id`.
   - Non-dense ids fail debug validation.
   - Duplicate ids fail debug validation.
   - Multiple reporters with duplicate local contact ids still map through
     `(source_id, local_contact_id)` without ambiguity.
   - If production ever supports non-dense ids, the test must require an
     explicit `source_id_to_source_index` table and cover that lookup.

M1 cannot be accepted with helper-only tests. At least one M1 test must execute
the production device hash reducer on real `muda::DeviceBuffer` input, and at
least one M1 test must exercise the final-cache hit/rebuild path once the cache
object exists.

Acceptance:

- No stale target reuse when active contacts change with unchanged counts.
- No device-to-host contact array copy is required in the final assembly hot
  path.
- Cache-hit tests distinguish final assembly keys from probe/debug keys.
- The journal records the owner and bump trigger for every epoch in
  `SocuAssemblyPlanKey`, and maps each field to the split side/program key it
  will belong to in M2.

Current implementation status:

- M1 strict acceptance is implemented in `socu-native-builder-redesign` as of
  2026-05-11.
- `StructuredAssemblyInfo` carries `SocuContactTopologyStamp`, and final
  structured assembly fills it from `GlobalDyTopoEffectManager` after solver
  workspace/stream setup when the selected solver requests a native contact
  plan stamp.
- Probe-only runtime graph signatures remain isolated on the probe path through
  `contact_set_signature()`.
- The topology producer is factored into `socu_contact_topology_stamp.{h,cu}` so
  tests and `GlobalDyTopoEffectManager` exercise the same device hash reducer.
  It hashes contact vertex-id buffers on device and copies back only scalar hash
  accumulators. It is correctness-first and may conservatively bump on
  storage-layout changes.
- `SocuContactTopologyStampCache` owns epoch bump semantics.
- `SocuVertexSidePlanKey`, `SocuContactProgramPlanKey`, and
  `SocuContactPlanCacheState` provide the M1 split-cache decision contract used
  by `SocuApproxSolver` final structured assembly.
- Contract tests now execute the production device hash reducer on real
  `muda::DeviceBuffer` inputs and verify same-count/different-vertex topology
  changes rebuild only the contact program layer while the side key hits.
- M1 is accepted for the redesign branch. Remaining M2 work: implement compact
  side/program buffers, side coverage stamps, production compact
  `reporter_id/source_id` records, and numeric executor coverage.

### M2: Compact Side Table And Program Builder

Deliverables:

- Implement `SocuAssemblySideRecord`, `SocuAssemblyDofLane`,
  `SocuContactSourceHeader`, `SocuContactProgramHeader`,
  `SocuContactMicroTask`, and `SocuContactSourceToProgram`.
- Implement `SocuVertexSidePlan` and `SocuContactProgramPlan` as separate
  cacheable owners, with a view that combines them for kernels. The M2 owner
  must already separate semantic side key hits from side coverage refreshes.
- Build side tables and exact/diag/drop program headers for PH and one simplex
  family first.
- Implement deterministic sort/unique side collection and lower-bound side id
  lookup for `active_set_temporary` coverage mode.
- Add `SocuVertexSideCoverageStamp` and report fields for coverage mode,
  coverage hit/refresh/fill, active-side-set changes, and active side vertex
  count.
- Keep side collection, materialization, stencil lookup, classification,
  program emission, and bucket construction on CUDA. Host code may prepare
  source headers and resize buffers, but must not host-copy full contact arrays
  for final assembly.
- Implement O(1) `source_id + local_contact_id -> program_id` lookup.
- Keep existing numeric writer disabled by default.

Unit tests:

- Synthetic FEM/FEM, ABD/FEM, FEM/ABD, ABD/ABD, fixed, unmapped, diag,
  first-offdiag, and off-band cases.
- Arbitrary lane order cases matching existing `socu_native_contact_targets.cu`
  coverage.
- Side deduplication: the same global vertex appears once even if used by many
  contacts.
- Multiple reporter/source tests: normal PT contact `0` and frictional PT
  contact `0` map to different programs.
- PH tests verify only `PH(0)` creates a matrix side and `PH(1)` remains
  evaluator data.
- Source-to-program invalid entries are initialized and counted.
- `source_id == sources[source_id].source_id` is validated for dense ids.
- Non-dense, duplicate, or out-of-range source ids produce a debug validation
  failure unless an explicit `source_id_to_source_index` map is built.
- Contact topology churn tests verify the contact program plan rebuilds while
  the side/lane semantic key records a cache hit when ordering/mapping/projection
  are unchanged.
- In `active_set_temporary` mode, contact topology changes that introduce a new
  active vertex set must report `active_side_set_changed` and coverage refresh.
  They must not increment semantic side rebuild counters.
- Runtime reorder or synthetic side-key changes rebuild both side/lane and
  contact program plans.
- Off-band policy changes rebuild the contact program plan but reuse the
  side/lane plan.
- Record size checks with `static_assert` budget targets.

Detailed M2 test specifications:

1. `socu_contact_assembly_plan.cu`: side/lane builder fixture.
   - Build a tiny synthetic ordering with:
     - FEM vertices mapped to contiguous `3x3` lanes;
     - ABD vertices mapped through body/Jacobi descriptors to `12x12` lanes;
     - fixed FEM and fixed ABD vertices;
     - unmapped vertices and off-band vertex pairs.
   - Allocate all contact topology inputs as `muda::DeviceBuffer` and invoke
     the production side collection/materialization kernels.
   - Copy only builder output buffers back for assertions.
   - Assert one `SocuAssemblySideRecord` per unique writable global vertex,
     independent of how many contacts reference that vertex.
   - Assert fixed/unmapped vertices are represented as skipped or read-only
     according to the production plan format, never as writable lanes.

2. `socu_contact_assembly_plan.cu`: contact program emission fixture.
   - Feed deterministic PT, EE, PE, PP, and PH inputs with known global vertex
     ids and expected side ids.
   - For each source family, assert:
     - source headers have dense `source_id`;
     - `first_program` and `program_count` cover exactly that source;
     - `source_to_program[source_id, local_contact_id]` returns O(1)
       `program_id`;
     - invalid or skipped contacts write an explicit invalid map entry and
       increment the skipped/invalid counter.
   - PH-specific assertion: only `PH(0)` becomes a writable matrix side;
     `PH(1)` remains evaluator metadata and must not create a side record.

3. `socu_contact_assembly_plan.cu`: symbolic classification fixture.
   - Construct pairs that are exactly diagonal, first-offdiag, off-band,
     fixed, unmapped, and mixed ABD/FEM.
   - Assert emitted `SocuContactProgramHeader` status and task kinds match a
     CPU oracle that uses the same descriptors but not the old sink or target
     table.
   - Assert task buffers are tightly packed: no uninitialized task is reachable
     from a valid program.

4. `socu_contact_assembly_plan.cu`: cache-layer fixture.
   - First build with a side key and contact key, then rebuild with:
     - topology-only change: side semantic cache hit, program rebuild;
     - topology-only change with new active vertex set in
       `active_set_temporary` mode: side semantic cache hit, coverage refresh,
       program rebuild;
     - off-band policy-only change: side semantic cache hit, program rebuild;
     - ordering/descriptor/mapping/projection change: both rebuild;
     - geometry-only change: both hit.
   - Assert split report counters: side rebuild/hit counters and program
     rebuild/hit counters change only for the layer that actually rebuilt.
   - Assert coverage counters are separate from semantic side counters.

5. Source scans and record budgets.
   - `static_assert` each POD is trivially copyable and below the byte budget
     recorded in this plan.
   - Source scan production builder TUs for forbidden includes:
     `structured_contact_assembly_sink.h`, legacy target headers, and debug
     compare headers.
   - Source scan production builder code for forbidden hot-path calls:
     `copy_to(` on contact topology buffers, `old_to_chain` lookup, and
     `classify_dof_pair`.

M2 cannot be accepted with CPU-only symbolic tests. At least one M2 test must
execute the production CUDA builder kernels and validate emitted device buffer
contents.

Current implementation status:

- M2 correctness is accepted in `socu-native-builder-redesign` as of
  2026-05-11, with `active_set_temporary` side coverage explicitly reported as
  the temporary M2 mode. It is not accepted as a performance two-level side
  coverage design; that belongs to M2b.
- Added `socu_contact_assembly_plan.{h,cu}` with compact POD records,
  split side/program owners, combined device view, dense O(1)
  `program_for(source_id, local_contact_id)` lookup, and an
  `active_set_temporary` CUDA builder.
- The builder covers normal and frictional simplex PT/EE/PE/PP source views,
  plus normal and frictional PH source views. Simplex sources emit exact,
  diag, diag-lump, drop, or skipped microtasks according to side banding and
  off-band policy. PH uses only `PH(0)` as the matrix side; `PH(1)` remains
  evaluator data and is not inserted into the side table.
- The builder performs contact-side collection, sort/unique, side/lane
  materialization, lower-bound side lookup, program/task emission, bucket
  marking/scan/compaction, and program/task stats counting on CUDA. Host code
  currently prepares source headers and resizes buffers.
- New `[m2]` CUDA contract tests execute the production builder kernels on
  `muda::DeviceBuffer` inputs and validate active side deduplication,
  fixed/unmapped/read-only side records, ABD/FEM lane materialization,
  PT/EE/PE/PP/PH source mapping, normal/frictional source disambiguation, PH
  half-plane omission, dense-source-id validation failures, and
  Drop/Diag/DiagLump off-band policies. They also validate exact/drop/skipped
  bucket ranges, execution strategies, and first program/task counters.
- M2 symbolic classification is checked against an independent CPU oracle that
  does not include the legacy sink, old target table, or production builder
  classification helpers. The oracle compares program kind, map status, task
  packing, side ids, local stencil ids, bands, write kinds, block ids, and task
  flags for exact, off-band, skipped, ABD/FEM mixed, and PH cases.
- Source scans guard the production builder TU against
  `structured_contact_assembly_sink.h`, legacy target/debug-table tokens,
  full contact-topology `copy_to` patterns, `old_to_chain`, and
  `classify_dof_pair`. The scans also assert that the final solver path calls
  the M2 plan builder and maps plan stats, and that the dy-topology adapter
  passes explicit M2 source spans.
- `SocuApproxReport` has helpers to map split contact plan stats into the
  report JSON fields and to treat aggregate cache-hit as side-plan hit plus
  contact-program hit. Topology-only changes therefore report an aggregate
  native contact plan cache miss even when the side layer hits.
- The real final solver path owns a persistent `SocuContactAssemblyPlan` and M2
  workspace, rebuilds through `StructuredAssemblyInfo` on split cache misses,
  and maps side/program stats, dense-source validation status, coverage mode,
  active-side-set changes, and source-to-program counters into public reports.

Acceptance:

- Compact builder emits the expected symbolic classes for all covered synthetic
  cases, verified by production CUDA builder output versus the independent CPU
  oracle.
- PT/EE-style contacts no longer duplicate full lane arrays per half-block.
- `program_for` does not scan the program table.
- Reports show separate side/program plan cache hits, rebuild counts, and build
  times.
- If M2 ships with `active_set_temporary`, reports explicitly show that mode.
  M2 correctness acceptance may pass in that mode, but M3/M5 performance claims
  must not treat active-side-set refresh as final two-level behavior.
- M2 final acceptance validation:
  - `git diff --check`: passed.
  - `cmake --build build --target uipc_test_backend_cuda_mixed_socu --parallel 12`:
    passed, no work to do in the final doc-only validation pass.
  - `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][m2]"`:
    passed, `552` assertions in `10` test cases.
  - `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract][socu_approx]"`:
    passed, `683` assertions in `22` test cases.
  - `uipc_test_backend_cuda_mixed_socu "[cuda_mixed_socu][contract]"`:
    passed, `5566` assertions in `42` test cases.

### M2b: Persistent Side Coverage Cache

M2b is the milestone that turns the split-key design into the intended
performance design. It must land before using M3 or M5 numbers as topology-churn
performance evidence.

Deliverables:

- Replace `active_set_temporary` for performance builds with either:
  - `global` side coverage, materializing every contact-addressable FEM/ABD
    vertex for the current `SocuVertexSidePlanKey`; or
  - `demand_filled` side coverage, using a persistent device lookup from global
    vertex to side id and filling only previously unseen vertices.
- Keep side ids stable for the lifetime of a side key in `demand_filled` mode.
  Contact program plans may be rebuilt on topology changes, but previously
  emitted side ids must not silently change underneath a still-valid plan view.
- Implement device-side missing-vertex detection for `demand_filled` mode:
  active contact stencils are scanned on device, missing vertices are compacted,
  side records are materialized, and the vertex-to-side lookup is updated.
- Keep host work limited to key checks, buffer growth decisions, kernel
  launches, and scalar report copies.
- Add report fields for side coverage mode, coverage hit/refresh/fill count,
  refresh time, active-side-set changes, and active side vertex count.
- Add a config/debug switch that can force `active_set_temporary` for
  correctness bisection, but performance builds must default to `global` or
  `demand_filled`.

Unit tests:

- `global` mode:
  - topology changes with arbitrary active vertex sets rebuild only contact
    programs;
  - side semantic cache hits and side coverage hits;
  - no active-side-set refresh is reported.
- `demand_filled` mode:
  - first topology using new vertices fills missing side records and reports
    coverage fill, not semantic side rebuild;
  - repeated topology over already covered vertices rebuilds only contact
    programs;
  - side ids for already covered vertices remain stable after additional fills.
- `active_set_temporary` mode:
  - topology changes that change the active vertex set report
    `active_side_set_changed`;
  - semantic side rebuild counters remain unchanged when only coverage changes.
- Runtime reorder, descriptor, mapping, fixed-flag, and projection changes reset
  coverage state because the side key changed.
- Empty contact sets do not clear a valid global/demand-filled side cache unless
  the side key changes.

Acceptance:

- Topology-churn benchmark with stable ordering/mapping/projection reports:
  - semantic side cache hit;
  - `native_contact_side_plan_rebuild_count` unchanged;
  - no `active_side_set_changed` in performance mode;
  - contact program rebuild count increments.
- `active_set_temporary` is permitted only for M2 correctness and debug
  comparison. It blocks final topology-churn performance acceptance.
- The report makes it impossible to confuse semantic side rebuild time,
  coverage fill/refresh time, and contact program rebuild time.

### M3: Compatibility Program Writer For Exact Writes

Deliverables:

- Implement `SocuContactProgramWriter`.
- Existing native contact kernels still compute `H`, then call
  `write_contact(source_id, local_contact_id, H)`.
- Implement exact FEM/FEM and ABD/FEM paths first.
- Implement debug validation in `socu_contact_program_debug_compare.h`. It
  recomputes task targets and verifies the symbolic task class before executing
  the write, but it is not included by the production writer header.

Unit tests:

- Single-contact exact matrix tests compare native compact writer against the
  legacy structured sink for FEM/FEM and ABD/FEM.
- Random small Hessian tests compare `D/E` values within mixed-precision
  tolerances.
- Debug compare mode verifies old and new writers on the same contacts.
- Production writer source-scan verifies that `socu_contact_program_writer.h`
  does not include debug compare headers, legacy target headers, or structured
  sink headers.

Detailed M3 test specifications:

1. `socu_contact_program_writer.cu`: exact writer device fixture.
   - Build plans manually and through the M2 builder for the same synthetic
     contacts.
   - Launch a small production writer kernel that calls
     `write_contact(source_id, local_contact_id, H)` with deterministic
     symmetric local Hessians.
   - Compare native `D/E` buffers against a CPU golden matrix and, in fallback
     builds, against the legacy structured sink.
   - Cover:
     - FEM/FEM same block and first-offdiag;
     - ABD/FEM and FEM/ABD orientation;
     - skipped fixed/unmapped programs;
     - invalid source/local ids produce a debug validation failure and no
       matrix write.

2. `socu_contact_program_writer.cu`: orientation and transpose fixture.
   - Use nonsymmetric local `3x3` block values inside an otherwise symmetric
     local Hessian so row/column orientation errors are visible.
   - Assert first-offdiag writes land in the correct `E` block and lane order
     for both local order and swapped local order.
   - Assert diagonal writes mirror only when the task explicitly requests a
     mirrored diagonal block.

3. `socu_contact_program_writer.cu`: debug compare isolation.
   - Build once with debug compare enabled and once disabled.
   - Enabled: compare path recomputes legacy/debug target classification and
     asserts equality before writing.
   - Disabled: production writer object/header does not include debug compare,
     structured sink, old target, `old_to_chain`, or `classify_dof_pair`.
   - Use `compile_commands.json` and source scan; do not rely only on runtime
     behavior.

4. `socu_contact_program_writer.cu`: report/counter fixture.
   - Exact writes increment exact-program/task counters.
   - Skipped writes increment skipped counters and do not increment exact
     counters.
   - Mixed-rejected status increments mixed-rejected counters and records a
     diagnostic status string when production mode continues.

M3 cannot be accepted unless a device kernel exercises the production writer
API. Calling helper functions from host tests is not enough.

Acceptance:

- No production writer path reads `old_to_chain` or calls
  `classify_dof_pair`.
- No production writer path includes legacy compare helpers or stores
  `SocuNativeContactStencilTarget` compatibility fields.
- No production writer path branches per scalar on diag/first-offdiag/off-band.
- Exact FEM/FEM and ABD/FEM results match the legacy structured sink.
- Plan build plus numeric exact writer is not slower than the current
  per-half-block target path on synthetic contact microbenchmarks.

### M4: Complete Exact, Diag, And DiagLump Policies

Deliverables:

- Complete exact FEM/ABD, ABD/FEM, ABD/ABD same-body, and ABD/ABD cross-body.
- Implement `DiagBlockFem`, `DiagBlockAbd`, `DiagScalarFem`,
  `DiagScalarAbd`, `LumpScalarFem`, and `LumpScalarAbd`.
- Cover PT, EE, PE, PP, and PH for normal and frictional contact.
- Add an explicit compatibility switch for scalar-only `Diag` fallback if
  needed to compare against the current native `DiagFallback`.

Unit tests:

- Exhaustive policy tests for `Drop`, `Diag`, and `DiagLump`.
- Diag tests verify exact vertex diagonal block when representable and scalar
  diagonal fallback when configured or required.
- Diag block fallback and scalar compatibility fallback use separate golden
  matrices. A test failure must identify which policy was expected.
- DiagLump tests verify absolute row-sum semantics against the legacy sink.
- ABD projection tests use nontrivial `ABDJacobi::x_bar()` values.
- Same-body ABD/ABD symmetry tests cover swapped local order and duplicate
  vertex edge cases.

Detailed M4 test specifications:

1. `socu_contact_program_writer.cu`: policy golden matrix suite.
   - For each policy `Drop`, `Diag`, and `DiagLump`, build a compact plan and
     write deterministic Hessians into native `D/E`.
   - Compare against separate golden matrices:
     - exact representable block golden;
     - `DiagBlock` fallback golden;
     - scalar-compatibility `DiagScalar` golden;
     - `DiagLump` absolute row-sum golden.
   - Tests must name the expected fallback mode in the assertion message.

2. `socu_contact_program_writer.cu`: full family coverage.
   - Cover PT, EE, PE, PP, PH for normal and frictional contact source
     families.
   - Use stencils with duplicate vertices and swapped local order.
   - Assert every covered family emits expected exact/diag/lump/drop program
     counts.

3. `socu_contact_program_writer.cu`: ABD projection fixture.
   - Use nontrivial ABD `x_bar()`/Jacobi weights so projected `12x12` blocks
     differ from simple identity-lane writes.
   - Assert ABD/FEM, FEM/ABD, ABD/ABD same-body, and ABD/ABD cross-body
     outputs match the golden matrix.
   - Change projection weights with stable contact topology and assert the M1
     projection epoch invalidates both side and program plans.

4. `socu_contact_program_writer.cu`: legacy parity fixture.
   - In full fallback builds, run the same synthetic policy cases through the
     legacy structured sink and compact writer.
   - Native-only builds run the CPU golden oracle instead and must not compile
     legacy structured contact TUs.

M4 cannot be accepted until `Diag` block fallback and scalar compatibility have
separate tests and separate counters. A single "diag fallback" golden is not
sufficient.

Acceptance:

- Matrix diff against legacy structured contact passes for all policies.
- `contact_offband_diag_fallback_count` and
  `contact_offband_lump_fallback_count` match legacy expectations.
- Reports distinguish `DiagBlock*` and `DiagScalar*` counts.
- Native-only builds do not need legacy structured contact TUs for exact,
  `Diag`, or `DiagLump` correctness gates.

### M5: Plan-Owned Numeric Executor

Deliverables:

- Add typed `SocuContactEvaluator` source views for normal/frictional simplex
  and vertex-half-plane contact.
- Launch executor buckets by model/family/program kind.
- Compute full local Hessian once per contact program and execute all tasks for
  that contact.
- Retain compatibility writer as a debug fallback.

Unit tests:

- Executor output equals compatibility writer output for identical plans.
- Bucket order changes do not affect results beyond atomic accumulation
  tolerance.
- Empty buckets and empty contact families are no-ops.

Detailed M5 test specifications:

1. `socu_contact_executor.cu`: deterministic evaluator fixture.
   - Add a test evaluator mode that returns deterministic Hessians from
     contact ids, source family, and local stencil ids. This isolates executor
     scheduling from IPC Hessian math.
   - Run the production executor on plans emitted by M2 and compare against the
     M3 compatibility writer using the same plan.
   - Cover all bucket kinds implemented by M4.

2. `socu_contact_executor.cu`: real evaluator smoke fixture.
   - For at least one simplex normal family and one half-plane family, run the
     actual contact evaluator on a small valid geometry/contact fixture.
   - Compare executor output against compatibility writer output within the
     active mixed-precision tolerance.
   - Empty source families must launch no invalid kernels and leave `D/E`
     unchanged.

3. `socu_contact_executor.cu`: bucket scheduling fixture.
   - Shuffle bucket order and task order within a bucket when mathematically
     safe.
   - Assert results are identical or within atomic accumulation tolerance.
   - Assert executor report counters reflect the original program/task counts,
     not launch order.

4. `socu_contact_executor.cu`: compile isolation fixture.
   - Source scan executor production TUs for forbidden includes:
     structured sink, compatibility writer debug compare, legacy target table.
   - Confirm native-only performance build compiles executor TUs without
     legacy structured contact model TUs.

M5 cannot be accepted with compatibility writer tests alone. At least one test
must call the production bucket executor with a plan emitted by the M2 builder.

Acceptance:

- Contact numeric assembly time is lower than M4 compatibility writer on
  contact-heavy synthetic scenes.
- Compile memory stays within the agreed CI and local build budget.
- Native executor TUs do not pull legacy structured sink templates into
  performance builds.

### M6: Hot-Block Detection And Owner-Reduce

Deliverables:

- Add block hit histograms during plan build.
- Split tasks into low-contention direct scatter and high-contention owner
  reduce.
- Add `SocuHotBlockPlan` and executor kernels for hot `D/E` blocks.
- Implement detection-only mode before enabling any owner-reduce writes.
- Implement both recompute and cached-microblock strategies or explicitly reject
  one with profiler evidence in the journal.

Unit tests:

- Synthetic many-contact same-block cases produce hot-block ranges.
- Owner-reduce and direct scatter produce equal matrices within tolerance.
- Threshold extremes work:
  - threshold zero sends all eligible tasks to owner-reduce.
  - threshold max sends all tasks to direct scatter.

Detailed M6 test specifications:

1. `socu_contact_hot_blocks.cu`: histogram/detection fixture.
   - Build synthetic plans where many contacts write the same diagonal block,
     many contacts write the same first-offdiag block, and contacts are evenly
     distributed.
   - Detection-only mode must emit hot ranges and refs but still execute direct
     scatter for all writes.
   - Report `native_contact_hot_diag_block_count` and
     `native_contact_hot_offdiag_block_count` separately.

2. `socu_contact_hot_blocks.cu`: threshold fixture.
   - Threshold zero sends every eligible repeated block to owner-reduce.
   - Max threshold sends everything to direct scatter.
   - A middle threshold classifies only blocks whose hit count is greater than
     or equal to the configured threshold.
   - Assert task counts are conserved:
     `direct_task_count + hot_ref_count == eligible_task_count`.

3. `socu_contact_hot_blocks.cu`: owner-reduce correctness fixture.
   - Compare direct scatter, detect-only, recompute owner-reduce, and
     cached-microblock owner-reduce on the same plan.
   - Use deterministic Hessians with values that expose missing refs and
     duplicate refs.
   - Assert all strategies produce equal `D/E` within tolerance.

4. `socu_contact_hot_blocks.cu`: strategy gating fixture.
   - `detect_only` never writes through owner-reduce kernels.
   - `recompute` is allowed only when the test config explicitly enables it.
   - `cached_microblock` must allocate/write/read the microblock cache and
     report cache element counts.

M6 cannot be accepted with performance counters alone. Correctness tests must
compare owner-reduce outputs against direct scatter for both diagonal and
first-offdiag hot blocks.

Acceptance:

- Nsight Compute shows reduced atomic contention on dense contact clusters.
- Owner-reduce is disabled automatically when it is slower than direct scatter.
- Default production mode cannot use recompute owner-reduce unless Hessian
  evaluation time is proven negligible for the selected model/family.

### M7: Runtime Reorder And Cached Replay Integration

Runtime reorder cadence does not change for the native contact builder. The
existing `linear_system/socu_approx/runtime_reorder_frame_interval` remains the
only periodic trigger: `0` means no runtime probe, and a positive value probes
on frames where `frame % runtime_reorder_frame_interval == 0`, subject to the
existing capacity and graph-source gates. A successful reorder install bumps the
SOCU ordering epoch, which invalidates the side/lane plan and therefore also
invalidates the contact program plan.

`cache` has two different meanings in this milestone and the report must keep
them separate:

- Symbolic plan cache: `native_contact_side_plan_*` and
  `native_contact_program_plan_*` describe side/lane and contact program reuse.
- Hessian replay cache: the current `full_hessian_cached` path records numeric
  contact Hessian half-block records and replays them through the legacy
  structured contact replay path unless M7 explicitly adds a native replay plan.

The current `full_hessian_cached` cache/replay is not CUDA Graph replay and is
not the same as the symbolic plan cache. It stores enough numeric half-block
data to replay contact Hessian writes, including source row/column global
vertices and mirrored diagonal-block state. If that replay continues to write
through `StructuredContactAssemblySink`, it must be reported as
`native_contact_replay_path = "legacy_structured"` and excluded from native
contact executor performance claims.

Probe paths use explicit report values:

```text
native_contact_probe_path = "off" | "bypass" | "symbolic_plan" | "native_executor" | "legacy_structured"
native_contact_replay_path = "off" | "native_plan" | "legacy_structured"
```

Deliverables:

- Keep probe-domain and final-assembly symbolic caches isolated. A probe may
  consume a symbolic plan or bypass it, but it must not refresh or populate the
  final side/program cache unless it uses the exact final assembly key and cache
  domain.
- Define whether graph-only probes consume:
  - no native plan (`bypass`);
  - a probe-domain symbolic plan (`symbolic_plan`);
  - the native contact executor (`native_executor`);
  - or the legacy structured path (`legacy_structured`).
- Define whether `full_hessian_cached` gets a native replay plan or remains a
  legacy structured replay path. If it remains legacy, the report and
  performance gates must say so explicitly.
- Add `native_contact_probe_path` and `native_contact_replay_path` report
  fields.
- Ensure runtime reorder install changes `ordering_epoch`, causing side/lane
  and contact program rebuild on the next final assembly.

Unit tests:

- Probe assembly with changing contacts invalidates the final program plan when
  the final topology key changes.
- Probe-built side/program plans do not become final-cache hits unless they are
  intentionally inserted through the final cache domain.
- A bypassed probe does not refresh final cache timestamps, hit counters, or
  rebuild counters.
- Cached replay never silently bypasses a requested native-contact performance
  measurement. If replay is legacy structured, the native executor timing is not
  counted as a native contact performance result.
- Runtime reorder install changes ordering epoch and rebuilds both side/lane
  and contact program plans.
- A topology-only change during runtime reorder probing rebuilds the contact
  program plan but reuses the side/lane plan when no reorder is installed.

Detailed M7 test specifications:

1. `socu_contact_assembly_plan.cu`: probe/final cache-domain fixture.
   - Use distinct probe and final cache domains.
   - Build a probe-domain symbolic plan, then run final assembly with the same
     key and assert it is not a final-cache hit unless the implementation
     explicitly inserts through the final domain.
   - Bypassed probes must not change final cache hit/rebuild counters or
     timestamps.

2. `socu_contact_assembly_plan.cu`: runtime reorder fixture.
   - Simulate runtime reorder install by changing ordering epoch and
     old-to-chain mapping.
   - Assert side/lane and contact program plans both rebuild on the next final
     assembly.
   - Simulate a probe with topology-only changes and no reorder install:
     final contact program rebuilds, side/lane plan hits.

3. `socu_contact_executor.cu`: replay path fixture.
   - Run final assembly with no replay, native replay if implemented, and
     legacy structured replay if retained.
   - Assert report fields:
     - `native_contact_probe_path`;
     - `native_contact_replay_path`;
     - native executor timings are zero or excluded when replay path is
       `legacy_structured`.
   - A performance test must fail or skip when it tries to count
     `legacy_structured` replay as native executor time.

4. Scene smoke tests.
   - Run the existing 20-frame runtime reorder scene gate with native contact
     plan enabled and verify final report counters:
     - probe path value;
     - replay path value;
     - ordering epoch changed on reorder frame;
     - side/program rebuild counters match the epoch changes.

M7 cannot be accepted until probe, replay, and final cache states are observable
in reports and covered by tests. Silent cache sharing between probe and final
domains is forbidden.

Acceptance:

- Runtime reorder variants pass the existing 20-frame and 100-frame gates.
- Reports distinguish native executor, compatibility writer, probe-only
  symbolic plan use, bypassed probes, native replay, and legacy structured
  cached replay.
- Native-contact performance gates reject any run where
  `native_contact_replay_path = "legacy_structured"` is counted as native
  executor time.

### M8: Cutover And Cleanup

Deliverables:

- Make compact plan writer/executor the default native contact path.
- Keep legacy structured contact only for comparison and fallback builds.
- Remove the current per-half-block `SocuNativeContactStencilTarget` path from
  performance builds after the replacement has equivalent coverage.
- Update integration docs and journal.

Unit tests:

- Native-only build passes all cuda_mixed_socu contract tests.
- Fallback/full build still passes legacy comparison tests.
- Source scan verifies production native writer does not include legacy
  structured contact sink fallback.

Detailed M8 test specifications:

1. Build matrix tests.
   - Native-only development/performance build:
     - `UIPC_CUDA_MIXED_WRECKING_BALL_MINIMAL_BUILD=ON`;
     - `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=ON`;
     - `UIPC_CUDA_MIXED_SOCU_BUILD_AL_PIPELINE=OFF`;
     - all compact-plan contract tests pass;
     - source scan confirms no legacy structured contact, abandoned target, or
       AL pipeline TUs are compiled.
   - Full fallback build:
     - `UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=OFF`;
     - legacy comparison tests pass;
     - compact writer/executor parity tests pass against legacy structured
       sink.

2. Cutover scene gates.
   - Run agreed no-contact and contact-enabled Wrecking Ball gates with:
     - compact native contact off;
     - compact native contact on;
     - debug diff off for timing;
     - counters/report fields on for one correctness run.
   - Assert no unexpected mixed-rejected programs in production gates.
   - Assert `native_contact_plan_cache_hit` approaches 100% on stable topology
     Newton solves after the cold build.

3. Performance acceptance tests.
   - Record cold rebuild, cache-hit numeric, and amortized per-Newton timings
     separately.
   - Compare against frozen old per-half-block target baseline commit/artifact
     and legacy structured contact baseline.
   - Performance claims must include:
     - exact executable path;
     - CMake flags;
     - baseline commit/artifact;
     - plan rebuild count;
     - side/program counts;
     - exact/diag/diag-lump/drop program counts;
     - hot-block strategy.

4. Cleanup/source-scan tests.
   - Production default path uses compact plan writer/executor.
   - Abandoned per-half-block target headers/TUs are not referenced by
     production build targets.
   - Legacy structured sink is reachable only through explicit fallback or
     debug compare builds.
   - `compile_commands.json`, `build.ninja`, and `ninja -n` agree on the
     excluded source set.

M8 cannot be accepted on correctness alone. It must pass both source/build
isolation and the agreed performance gates with cold/cache-hit/amortized timing
reported separately.

Acceptance:

- Native contact assembly is faster than both the legacy structured contact sink
  and the previous per-half-block native target path on the agreed scenes.
- Wrecking-ball fused PCG remains unaffected.
- SOCU direct variants pass direction validation and report no unexpected
  mixed-rejected native contact programs.

## Unit Test Matrix

Add tests under `apps/tests/backends/cuda_mixed_socu/`:

- `socu_contact_topology_stamp.cu`
- `socu_contact_assembly_plan.cu`
- `socu_contact_program_writer.cu`
- `socu_contact_executor.cu`
- `socu_contact_hot_blocks.cu`

All tests use Catch2 tags that match the existing backend test style:

```text
[cuda_mixed_socu][contract][socu_contact_plan]
[cuda_mixed_socu][contract][socu_contact_writer]
[cuda_mixed_socu][contract][socu_contact_executor]
[cuda_mixed_socu][contract][socu_contact_hot_blocks]
```

CUDA tests must skip cleanly when no CUDA device is available, following the
existing `has_cuda_device()` pattern in `apps/tests/backends/cuda_mixed_socu`.
Randomized tests must use fixed seeds and print the seed on failure.

Required test categories:

1. Pure symbolic tests.
   - FEM/FEM, ABD/FEM, FEM/ABD, ABD/ABD.
   - Same block, adjacent block, off-band, skipped, fixed, unmapped.
   - PT, EE, PE, PP, PH stencil sizes.
   - `Drop`, `Diag`, `DiagLump`.
   - Multiple sources with identical local contact ids.

2. Device builder tests.
   - Device-side program emission for each contact family.
   - Prefix-sum sizing and empty family handling.
   - Side deduplication across repeated vertices.
   - `program_for(source_id, local_contact_id)` O(1) lookup behavior.
   - PH source handling where only `PH(0)` maps to a side.

3. Numeric writer tests.
   - Exact matrix equality against legacy structured sink.
   - `Diag` equality against legacy semantics.
   - `Diag` block fallback equality and scalar compatibility equality use
     separate golden outputs.
   - `DiagLump` equality against legacy absolute row-sum semantics.
   - ABD projection with nontrivial weights.
   - First-offdiag lane orientation, including transposed and non-transposed
     local order.
   - Same-body ABD/ABD mirrored diagonal writes.

4. Cache tests.
   - Cache hit when only geometry changes.
   - Contact topology/content/layout changes rebuild the contact program plan
     and reuse the side/lane semantic plan when the side key is unchanged.
   - `global`/`demand_filled` side coverage tests verify topology churn can
     rebuild only contact programs.
   - `active_set_temporary` tests verify active-side-set refresh is reported as
     coverage refresh, not semantic side rebuild.
   - Runtime reorder, ordering, descriptor, mapping, fixed-flag, and projection
     changes rebuild both side/lane and contact program plans.
   - Off-band policy changes rebuild only the contact program plan.
   - Same contact count but different stencil vertex ids rebuilds without
     relying on a runtime graph probe.
   - ABD projection weight or `x_bar()` changes rebuild when topology is stable.
   - Probe-created or probe-skipped graph state cannot refresh or poison the
     final assembly plan cache.
   - Split side/program cache counters match the layer that actually rebuilt.
   - No host contact copy required for final assembly.

5. Randomized property tests.
   - Small random mappings and random symmetric local Hessians.
   - Compare compact plan writer with a CPU or legacy structured reference.
   - Run enough seeds to cover arbitrary lanes and off-band boundaries.

6. Performance smoke tests.
   - Synthetic low-contention contacts.
   - Synthetic high-contention contacts.
   - Empty contacts.
   - Large repeated-vertex contact sets to validate side-table reuse.
   - Topology churn with stable ordering/mapping to validate program rebuild
     cost separately from side/lane rebuild cost.
   - Topology churn with new active vertices to validate demand-filled coverage
     fill cost separately from contact program rebuild cost.

Source-scan tests:

- Production compact writer headers must not include
  `structured_contact_assembly_sink.h`.
- Production compact writer code must not reference `old_to_chain` or
  `classify_dof_pair`.
- Production compact writer code must not include
  `socu_contact_program_debug_compare.h` or legacy target headers.
- Native compact-plan performance builds must not compile legacy structured
  contact TUs. This must be checked against `build.ninja`, `ninja -n`, and
  `compile_commands.json` when the compile database is generated.

Tolerance policy:

- Double solve builds: relative tolerance `1e-10`, absolute tolerance `1e-12`
  unless existing mixed-precision policy requires looser thresholds.
- Float storage or mixed storage: relative tolerance `1e-5`, absolute tolerance
  `1e-6`.
- Atomic order differences are accepted only within these tolerances.

## Scene And Benchmark Gates

All performance gates must run with validation and diff instrumentation disabled
unless the gate explicitly says otherwise:

```text
SOCU_NATIVE_CONTACT_DIFF=0
SOCU_NATIVE_CHAIN_BASE_DIFF=0
SOCU_NATIVE_DIAG_RHS_DIFF=0
SOCU_REPORT_COUNTERS=0
```

Correctness gates:

- `backend_cuda_mixed_socu_contract`.
- Wrecking-ball fused PCG through the existing smoke frame count.
- Wrecking-ball SOCU native contact exact through 20 frames.
- Wrecking-ball SOCU native contact `Diag` and `DiagLump` policy runs through
  20 frames.
- 100-frame topology/contact gate with native contact plan enabled.

Performance gates:

- Same build, same scene, same variant, only toggling
  `linear_system/socu_approx/native_contact_plan` unless the gate explicitly
  measures a compile-time build mode.
- Report must include split side/program cache hit rates, split plan build
  times, aggregate plan build time, numeric contact time, exact/fallback/drop
  counts, probe/replay path, and hot-reduce state.
- Every performance result is split into:
  - cold rebuild timing, where the plan is forced to rebuild;
  - cache-hit numeric timing, where topology and symbolic keys are stable;
  - topology-churn timing, where contact programs rebuild but the side/lane
    plan should hit;
  - amortized per-Newton-solve timing over the full nonlinear step.
- Stable-topology Newton solves should report a cache-hit rate close to 100%.
  Any miss must name the key field that changed or the result is rejected.
- Topology-churn runs should report side/lane cache hits unless runtime reorder
  or mapping/projection changes occurred. If the side/lane plan rebuilds, the
  result must explain whether this is a temporary M2 active-vertex side-table
  limitation or a real side-key change.
- Topology-churn performance cutover requires `native_contact_side_coverage_mode`
  to be `global` or `demand_filled`. Runs in `active_set_temporary` mode are
  correctness/debug data only.
- Native contact plan must beat the current per-half-block native target path
  on contact-heavy synthetic microbenchmarks before scene-level cutover.
- Native contact plan must beat the legacy structured contact sink on the
  agreed 100-frame scene gate before deleting fallback from performance builds.

Measurement protocol:

- Run at least three repetitions per variant and use the median.
- If the median absolute deviation exceeds 5%, run five repetitions and report
  both median and best-of-five.
- Use the same binary for A/B measurements whenever possible. If a build flag
  must differ, record the exact CMake cache and executable path.
- Exclude the first frame from per-frame timing summaries when it includes
  runtime construction or cold plan allocation.
- Compare contact assembly timers separately from total frame time.
- Report cold-rebuild and cache-hit medians separately before reporting any
  amortized result.

Quantitative cutover targets:

- M3 compatibility writer: plan build plus exact numeric writer is no more than
  5% slower than the current per-half-block native target path on synthetic
  exact-contact tests.
- M5 plan-owned executor: numeric contact assembly is at least 15% faster than
  the M4 compatibility writer on synthetic contact-heavy tests.
- M8 scene gate: compact-plan native contact is at least 20% faster than the
  current per-half-block native target path on synthetic contact-heavy tests and
  at least 10% faster than the legacy structured contact sink on the agreed
  100-frame scene gate.
- If a threshold is missed but correctness passes, the journal must record the
  profiler bottleneck and the cutover is blocked.

Profiler gates:

- Capture Nsight Compute reports for at least:
  - compact plan build
  - contact numeric executor direct scatter
  - hot-block owner-reduce, when enabled
- Track:
  - DRAM bytes per contact
  - atomic throughput and serialization
  - achieved occupancy
  - register count
  - warp branch efficiency
  - L2 hit rate for side and task tables

## Cutover Checklist

Before making the compact builder default:

- All native contact families are covered: PT, EE, PE, PP, PH.
- All contact models are covered: simplex normal, simplex frictional,
  vertex-half-plane normal, vertex-half-plane frictional.
- `Drop`, `Diag`, and `DiagLump` pass reference matrix tests.
- `Diag` block fallback and scalar compatibility fallback have distinct golden
  tests and report counters.
- Final assembly cache keys use topology epochs, not hot-path host-copy
  signatures.
- Multiple contact sources are represented with stable `source_id` values, and
  duplicate local contact ids across reporters are tested.
- Dense source ids satisfy `source_id == sources[source_id].source_id`, or the
  plan explicitly carries and tests a source-id remap table.
- `program_for(source_id, local_contact_id)` is O(1) and covered by source-scan
  or unit tests.
- Native `E` orientation tests cover transposed and non-transposed first
  offdiag tasks.
- Reports show zero unexpected mixed-rejected programs in production gates.
- Production performance gates use `global` or `demand_filled` side coverage,
  not `active_set_temporary`.
- Topology-churn reports separate semantic side rebuild, side coverage
  fill/refresh, and contact program rebuild.
- Performance runs are made with debug diff and counters disabled.
- Native-only build passes without legacy structured contact TUs.
- Full fallback build still passes comparison tests.
- Performance claims name the frozen baseline commit or artifact used for the
  old per-half-block target comparison.
- Documentation and reports state whether `full_hessian_cached` is native replay
  or legacy structured replay for the tested variant. Legacy structured replay
  is never counted as native contact executor performance.

## Open Questions

- Should `Diag` default to full diagonal-block fallback whenever representable,
  or should compatibility mode preserve the current scalar-diagonal native
  behavior until all matrix tests are updated?
- What hot-block threshold best separates atomic scatter from owner-reduce for
  real contact clusters?
- Should final side coverage use `global` materialization or `demand_filled`
  cache by default? Global is simpler and gives clean topology-churn behavior;
  demand-filled may reduce memory but needs a robust device lookup/fill path.
- Can IPC normal/friction Hessian evaluators expose low-rank or block-local
  evaluation so owner-reduce does not need to compute full local Hessians?
- Should M7 implement a native `full_hessian_cached` replay plan, or keep the
  existing legacy structured replay as a graph-source mode that is explicitly
  excluded from native-contact performance claims?
