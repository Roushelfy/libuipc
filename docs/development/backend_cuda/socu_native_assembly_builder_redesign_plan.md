# SOCU Native Assembly Builder Redesign Plan

This document describes a large-scale redesign of the `cuda_mixed_socu`
structured matrix builder. It is intentionally separate from
`socu_mixed_solver_integration_plan.md`: the integration plan records the
current production path, while this document is the high-performance redesign
target.

The central idea is to replace the current contact-side target-table adapter
with a symbolic assembly plan and specialized numeric executors. The symbolic
plan is rebuilt only when ordering, topology, mapping, or off-band policy
changes. Per-solve numeric assembly should only evaluate local Hessians and run
regular, preclassified matrix write microkernels.

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
   - Emits compact side tables, per-contact microprograms, task buckets, and
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

The builder cache is keyed by all data that can change symbolic write targets:

```cpp
struct SocuAssemblyPlanKey
{
    std::uint64_t ordering_epoch = 0;
    std::uint64_t native_descriptor_epoch = 0;
    std::uint64_t contact_topology_epoch = 0;
    std::uint64_t contact_layout_hash = 0;
    std::uint64_t fixed_mapping_epoch = 0;
    std::uint64_t vertex_projection_epoch = 0;

    SizeT horizon = 0;
    SizeT block_size = 0;
    StructuredContactOffbandPolicy offband_policy =
        StructuredContactOffbandPolicy::Drop;
    bool scalar_diag_fallback_compatibility = false;

    bool operator==(const SocuAssemblyPlanKey&) const noexcept = default;
};
```

Required rule: the final assembly path always receives a valid topology epoch.
It must not rely on `GlobalDyTopoEffectManager::contact_set_signature()` doing
a device-to-host copy in the hot path.

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
- `contact_topology_epoch` and `contact_layout_hash` are owned by
  `GlobalDyTopoEffectManager` or the dy-topology contact manager. They bump when
  active contact stencil vertex ids, reporter/source ownership, source order, or
  contact storage layout changes. They do not bump for geometry-only numeric
  updates.
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

Contact generation should maintain:

```cpp
struct SocuContactTopologyStamp
{
    std::uint64_t epoch = 0;
    std::uint64_t layout_hash = 0;
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
ids, model kinds, family kinds, stencil sizes, contact counts, and the active
stencil vertex ids. It must not include positions, distances, barrier values,
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

struct SocuContactAssemblyPlan
{
    SocuAssemblyPlanKey key;

    muda::DeviceBuffer<SocuContactSourceHeader> sources;
    muda::DeviceBuffer<SocuAssemblySideRecord> sides;
    muda::DeviceBuffer<SocuAssemblyDofLane> lanes;
    muda::DeviceBuffer<SocuContactProgramHeader> programs;
    muda::DeviceBuffer<SocuContactMicroTask> tasks;
    muda::DeviceBuffer<SocuContactProgramBucket> buckets;
    muda::DeviceBuffer<SocuContactSourceToProgram> source_to_program;

    // Optional, enabled after hot-block detection is implemented.
    SocuHotBlockPlan hot_blocks;

    SocuContactPlanStats last_stats;
};
```

Device code consumes view-only plans:

```cpp
struct SocuContactAssemblyPlanView
{
    SocuAssemblyPlanKey key;

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
    SizeT cache_hit_count = 0;
    SizeT rebuild_count = 0;
};
```

## Host Interfaces

### Top-Level Builder

```cpp
struct SocuNativeAssemblyBuildInput
{
    SocuAssemblyPlanKey key;
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
    SocuAssemblyPlanKey key;
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
    SocuAssemblyPlanKey m_key;
    SocuContactAssemblyPlan m_contact_plan;
};
```

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

The symbolic builder runs the following stages on the same stream:

1. Side collection.
   - Read all active contact sources.
   - Emit global vertex ids used by contact assembly.
   - PH sources emit only `PH(0)` as a matrix vertex.
   - M2 uses sort/unique vertex ids for deterministic behavior. A later
     epoch-tagged device hash table can replace it only after matching tests and
     benchmarks.

2. Source indexing.
   - Emit `SocuContactSourceHeader` records in deterministic source order.
   - Allocate one `source_to_program` entry per source contact.
   - Fill invalid entries with `SocuInvalidContactProgramId`.

3. Side materialization.
   - Convert each unique vertex id into a `SocuAssemblySideRecord`.
   - Fill `SocuAssemblyDofLane` entries from native vertex/dof descriptors and
     ABD projection weights.
   - Mark fixed or unmapped sides as skipped.
   - Build a scratch vertex-to-side map by lower-bound lookup into the sorted
     unique vertex id table. This scratch map is not part of the final plan.

4. Stencil classification.
   - For each contact, read its side ids.
   - Classify the whole stencil once against the native band.
   - Decide `Exact`, `Diag`, `DiagLump`, `Drop`, `Skipped`, or
     `MixedRejectedDebugOnly`.

5. Program sizing.
   - Count programs, tasks, buckets, fallback counts, and optional block hit
     counts.
   - Prefix-sum sizes.

6. Program emission.
   - Emit one `SocuContactProgramHeader` per source contact.
   - Emit compact `SocuContactMicroTask` records for exact or diag block writes.
   - Fill `source_to_program[source.first_source_to_program + local_contact_id]`.
   - Emit bucket ranges sorted by model, family, and program kind.

7. Optional hot-block split.
   - Use block hit counts to identify high-contention `D/E` blocks.
   - Split hot tasks into owner-reduce adjacency lists.
   - Leave low-contention tasks on direct atomic scatter.

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
native_contact_numeric_ms
native_contact_hot_reduce_ms
native_contact_hot_reduce_strategy
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

JSON placement:

- `timing`: `native_contact_plan_build_ms`, `native_contact_numeric_ms`,
  `native_contact_hot_reduce_ms`, `native_contact_hot_reduce_strategy`.
- `contact`: all plan size, exact/fallback/drop, mixed-rejected, and hot-block
  count fields.
- `status`: any nonzero mixed-rejected count in a production run adds a
  diagnostic detail even if the solve continues.

`native_contact_plan_cache_hit` is per solve. `native_contact_plan_rebuild_count`
is cumulative for the runtime and is reset when the SOCU runtime is rebuilt.
Plan size and fallback counters are per solve. Plan timings are recorded even
when `SOCU_REPORT_COUNTERS=0`; they are timers, not optional scalar counters.

These counters are acceptance-critical. A performance result is not actionable
unless it states whether the plan was rebuilt, how many programs were exact,
how many fell back to `Diag` or `DiagLump`, and whether hot-block reduction was
enabled.

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
- Ensure final assembly receives a valid topology epoch and layout hash.
- Add `SocuAssemblyPlanKey`.
- Stop using host-copy contact signatures as the production cache key.
- Add deterministic `reporter_id` and `source_id` assignment for every contact
  source in one linear build.

Unit tests:

- Same contact count but different vertex ids changes the topology epoch/hash.
- Same contact count but different vertex ids without a runtime graph probe
  still rebuilds the final assembly plan.
- Same vertex ids with changed geometry does not change topology epoch/hash.
- Ordering epoch change invalidates the plan.
- Off-band policy change invalidates the plan.
- Fixed flag or ABD/FEM mapping epoch change invalidates the plan.
- ABD projection epoch change invalidates the plan.
- ABD `x_bar()` or precomputed projection weight changes invalidate the plan
  even when contact topology and descriptor layout do not change.
- Reordering reporters or changing source family order changes `layout_hash`
  unless the source order is explicitly canonicalized.
- Probe-only graph signatures and bypassed probes do not pollute or refresh the
  final assembly plan cache.

Acceptance:

- No stale target reuse when active contacts change with unchanged counts.
- No device-to-host contact array copy is required in the final assembly hot
  path.
- Cache-hit tests distinguish final assembly keys from probe/debug keys.
- The journal records the owner and bump trigger for every epoch in
  `SocuAssemblyPlanKey`.

### M2: Compact Side Table And Program Builder

Deliverables:

- Implement `SocuAssemblySideRecord`, `SocuAssemblyDofLane`,
  `SocuContactSourceHeader`, `SocuContactProgramHeader`,
  `SocuContactMicroTask`, and `SocuContactSourceToProgram`.
- Build side tables and exact/diag/drop program headers for PH and one simplex
  family first.
- Implement deterministic sort/unique side collection and lower-bound side id
  lookup.
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
- Record size checks with `static_assert` budget targets.

Acceptance:

- Compact builder emits the same symbolic classes as the current target builder
  for all covered synthetic cases.
- PT/EE-style contacts no longer duplicate full lane arrays per half-block.
- `program_for` does not scan the program table.

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

Acceptance:

- Nsight Compute shows reduced atomic contention on dense contact clusters.
- Owner-reduce is disabled automatically when it is slower than direct scatter.
- Default production mode cannot use recompute owner-reduce unless Hessian
  evaluation time is proven negligible for the selected model/family.

### M7: Runtime Reorder And Cached Replay Integration

Deliverables:

- Ensure graph-only probes can consume the symbolic plan or explicitly bypass
  it without corrupting final plan cache state.
- Define whether `full_hessian_cached` gets a native replay plan or remains a
  legacy structured replay path.
- Add report fields that state which replay path was used.

Unit tests:

- Probe assembly with changing contacts invalidates the final plan when needed.
- Cached replay never silently bypasses a requested native-contact performance
  measurement.
- Runtime reorder install changes ordering epoch and rebuilds the plan.

Acceptance:

- Runtime reorder variants pass the existing 20-frame and 100-frame gates.
- Reports distinguish native executor, compatibility writer, and cached replay.

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
   - Rebuild when ordering, topology, mapping, fixed flags, or off-band policy
     changes.
   - Same contact count but different stencil vertex ids rebuilds without
     relying on a runtime graph probe.
   - ABD projection weight or `x_bar()` changes rebuild when topology is stable.
   - Probe-created or probe-skipped graph state cannot refresh or poison the
     final assembly plan cache.
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
- Report must include plan cache hit rate, plan build time, numeric contact
  time, exact/fallback/drop counts, and hot-reduce state.
- Every performance result is split into:
  - cold rebuild timing, where the plan is forced to rebuild;
  - cache-hit numeric timing, where topology and symbolic keys are stable;
  - amortized per-Newton-solve timing over the full nonlinear step.
- Stable-topology Newton solves should report a cache-hit rate close to 100%.
  Any miss must name the key field that changed or the result is rejected.
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
- Performance runs are made with debug diff and counters disabled.
- Native-only build passes without legacy structured contact TUs.
- Full fallback build still passes comparison tests.
- Performance claims name the frozen baseline commit or artifact used for the
  old per-half-block target comparison.
- Documentation states whether `full_hessian_cached` is native replay or legacy
  replay for the tested variant.

## Open Questions

- Should `Diag` default to full diagonal-block fallback whenever representable,
  or should compatibility mode preserve the current scalar-diagonal native
  behavior until all matrix tests are updated?
- What hot-block threshold best separates atomic scatter from owner-reduce for
  real contact clusters?
- Should side-table construction use sort/unique or an epoch-tagged device hash
  table? Sort/unique is simpler and deterministic; a hash table may win for
  very large stable contact sets.
- Can IPC normal/friction Hessian evaluators expose low-rank or block-local
  evaluation so owner-reduce does not need to compute full local Hessians?
- Should `full_hessian_cached` grow a native replay plan, or should it remain a
  separate graph-source mode that is excluded from native-contact performance
  claims?
