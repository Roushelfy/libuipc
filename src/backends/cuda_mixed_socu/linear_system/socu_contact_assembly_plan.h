#pragma once

#include <linear_system/socu_contact_plan_types.h>
#include <linear_system/socu_native_descriptors.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/buffer_view.h>
#include <cuda_runtime_api.h>

#include <limits>

namespace uipc::backend::cuda_mixed
{
using SocuAssemblySideId    = std::uint32_t;
using SocuContactSourceId   = std::uint32_t;
using SocuContactProgramId  = std::uint32_t;

inline constexpr SocuAssemblySideId SocuInvalidAssemblySideId =
    std::numeric_limits<SocuAssemblySideId>::max();
inline constexpr SocuContactSourceId SocuInvalidContactSourceId =
    std::numeric_limits<SocuContactSourceId>::max();
inline constexpr SocuContactProgramId SocuInvalidContactProgramId =
    std::numeric_limits<SocuContactProgramId>::max();

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

UIPC_GENERIC inline std::uint8_t operator|(SocuContactTaskFlag lhs,
                                           SocuContactTaskFlag rhs) noexcept
{
    return static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs);
}

struct SocuAssemblyDofLane
{
    std::uint32_t block = 0;
    std::uint16_t lane = 0;
    std::uint8_t  component = 0;
    std::uint8_t  flags = 0;
    Float         weight = Float{0};
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

    std::uint32_t block = 0;
    std::uint16_t lane = 0;
    std::uint16_t reserved0 = 0;
    std::uint32_t first_lane = 0;
    std::uint16_t lane_count = 0;
    std::uint16_t reserved = 0;
};

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

    SocuAssemblySideId side_ids[4] = {SocuInvalidAssemblySideId,
                                      SocuInvalidAssemblySideId,
                                      SocuInvalidAssemblySideId,
                                      SocuInvalidAssemblySideId};
};

struct SocuContactMicroTask
{
    SocuAssemblySideId row_side = SocuInvalidAssemblySideId;
    SocuAssemblySideId col_side = SocuInvalidAssemblySideId;

    std::uint8_t local_row_vertex = 0;
    std::uint8_t local_col_vertex = 0;
    SocuAssemblyBand band = SocuAssemblyBand::Diag;
    SocuAssemblyWriteKind write_kind = SocuAssemblyWriteKind::Skipped;

    std::uint32_t block_or_left_block = 0;
    std::uint8_t flags = 0;
    std::uint8_t reserved[3] = {};
};

struct SocuContactProgramBucket
{
    SocuContactModelKind model = SocuContactModelKind::SimplexNormal;
    SocuContactFamily family = SocuContactFamily::PT;
    SocuContactProgramKind program_kind = SocuContactProgramKind::Skipped;
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
    SizeT active_side_set_hash = 0;
    SizeT covered_vertex_count = 0;
    bool complete_for_current_contacts = false;
};

struct SocuContactPlanStats
{
    SizeT side_cache_hit_count = 0;
    SizeT side_rebuild_count = 0;
    SizeT side_coverage_hit_count = 0;
    SizeT side_coverage_refresh_count = 0;
    SizeT side_coverage_fill_count = 0;
    SizeT active_side_set_changed_count = 0;
    SizeT active_side_vertex_count = 0;
    SizeT program_cache_hit_count = 0;
    SizeT program_rebuild_count = 0;
    SizeT side_count = 0;
    SizeT lane_count = 0;
    SizeT program_count = 0;
    SizeT task_count = 0;
    SizeT bucket_count = 0;
};

struct SocuVertexSidePlan
{
    SocuVertexSidePlanKey key;
    muda::DeviceBuffer<SocuAssemblySideRecord> sides;
    muda::DeviceBuffer<SocuAssemblyDofLane> lanes;
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
    SocuContactPlanStats last_stats;
};

struct SocuContactAssemblyPlan
{
    SocuVertexSidePlan side_plan;
    SocuContactProgramPlan program_plan;
};

struct SocuContactAssemblyPlanView
{
    SocuVertexSidePlanKey side_key;
    SocuContactProgramPlanKey program_key;

    muda::CBufferView<SocuContactSourceHeader> sources;
    muda::CBufferView<SocuAssemblySideRecord> sides;
    muda::CBufferView<SocuAssemblyDofLane> lanes;
    muda::CBufferView<IndexT> sorted_side_vertices;
    muda::CBufferView<SocuContactProgramHeader> programs;
    muda::CBufferView<SocuContactMicroTask> tasks;
    muda::CBufferView<SocuContactProgramBucket> buckets;
    muda::CBufferView<SocuContactSourceToProgram> source_to_program;

    MUDA_GENERIC bool valid() const noexcept
    {
        return sources.data() != nullptr && programs.data() != nullptr;
    }

    MUDA_GENERIC SocuContactSourceToProgram program_for(
        SocuContactSourceId source_id,
        SizeT               local_contact_id) const noexcept
    {
        if(source_id >= sources.size())
            return {};
        const auto source = sources.data()[source_id];
        if(source.source_id != source_id || local_contact_id >= source.contact_count)
            return {};
        const SizeT index = source.first_source_to_program + local_contact_id;
        if(index >= source_to_program.size())
            return {};
        return source_to_program.data()[index];
    }
};

struct SocuContactM2SourceInput
{
    SocuContactSourceId source_id = SocuInvalidContactSourceId;
    std::uint32_t reporter_id = 0;
    SocuContactModelKind model = SocuContactModelKind::SimplexNormal;
};

struct SocuContactAssemblyPlanM2BuildInput
{
    SocuVertexSidePlanKey side_key;
    SocuContactProgramPlanKey program_key;
    muda::CBufferView<SocuNativeVertexDescriptor> vertex_descriptors;

    muda::CBufferView<Vector4i> pt_contacts;
    muda::CBufferView<Vector4i> ee_contacts;
    muda::CBufferView<Vector3i> pe_contacts;
    muda::CBufferView<Vector2i> pp_contacts;
    muda::CBufferView<Vector2i> ph_contacts;

    muda::CBufferView<Vector4i> friction_pt_contacts;
    muda::CBufferView<Vector4i> friction_ee_contacts;
    muda::CBufferView<Vector3i> friction_pe_contacts;
    muda::CBufferView<Vector2i> friction_pp_contacts;
    muda::CBufferView<Vector2i> friction_ph_contacts;

    SocuContactM2SourceInput pt_source;
    SocuContactM2SourceInput ee_source;
    SocuContactM2SourceInput pe_source;
    SocuContactM2SourceInput pp_source;
    SocuContactM2SourceInput ph_source;

    SocuContactM2SourceInput friction_pt_source;
    SocuContactM2SourceInput friction_ee_source;
    SocuContactM2SourceInput friction_pe_source;
    SocuContactM2SourceInput friction_pp_source;
    SocuContactM2SourceInput friction_ph_source;

    StructuredContactOffbandPolicy offband_policy =
        StructuredContactOffbandPolicy::Drop;
    cudaStream_t stream = cudaStreamLegacy;
};

struct SocuContactAssemblyPlanM2Workspace
{
    muda::DeviceBuffer<IndexT> vertex_refs;
    muda::DeviceBuffer<IndexT> sorted_vertex_refs;
    muda::DeviceBuffer<int> unique_flags;
    muda::DeviceBuffer<int> unique_offsets;
    muda::DeviceBuffer<int> scalar_counts;
    muda::DeviceBuffer<int> scalar_offsets;
    muda::DeviceBuffer<int> scalar_total;
    muda::DeviceBuffer<int> task_cursor;
};

SocuContactAssemblyPlanView socu_contact_assembly_plan_view(
    const SocuContactAssemblyPlan& plan) noexcept;

void build_socu_contact_assembly_plan_m2_active_set_temporary(
    SocuContactAssemblyPlan&                 plan,
    SocuContactAssemblyPlanM2Workspace&      workspace,
    const SocuContactAssemblyPlanM2BuildInput& input);
}  // namespace uipc::backend::cuda_mixed
