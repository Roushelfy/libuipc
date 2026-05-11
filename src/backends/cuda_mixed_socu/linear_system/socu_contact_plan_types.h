#pragma once

#include <type_define.h>
#include <utils/structured_contact_offband_policy.h>
#include <array>
#include <cstdint>

namespace uipc::backend::cuda_mixed
{
constexpr SizeT SocuContactPlanHashOffset =
    static_cast<SizeT>(1469598103934665603ull);
constexpr SizeT SocuContactPlanHashPrime =
    static_cast<SizeT>(1099511628211ull);

UIPC_GENERIC inline SizeT socu_contact_mix_hash(SizeT hash, SizeT value) noexcept
{
    hash ^= value;
    hash *= SocuContactPlanHashPrime;
    return hash;
}

UIPC_GENERIC inline void socu_contact_mix_in_place(SizeT& hash, SizeT value) noexcept
{
    hash = socu_contact_mix_hash(hash, value);
}

enum class SocuContactSourceFamily : std::uint8_t
{
    SimplexNormalPT,
    SimplexNormalEE,
    SimplexNormalPE,
    SimplexNormalPP,
    SimplexFrictionPT,
    SimplexFrictionEE,
    SimplexFrictionPE,
    SimplexFrictionPP,
    HalfPlaneNormalPH,
    HalfPlaneFrictionPH,
    Unknown,
};

enum class SocuContactExecutionStrategy : std::uint8_t
{
    DirectScatter,
    DetectOnly,
    Recompute,
    CachedMicroblock,
};

struct SocuContactTopologyCounts
{
    SizeT simplex_normal_pt = 0;
    SizeT simplex_normal_ee = 0;
    SizeT simplex_normal_pe = 0;
    SizeT simplex_normal_pp = 0;
    SizeT simplex_friction_pt = 0;
    SizeT simplex_friction_ee = 0;
    SizeT simplex_friction_pe = 0;
    SizeT simplex_friction_pp = 0;
    SizeT half_plane_normal_ph = 0;
    SizeT half_plane_friction_ph = 0;

    bool operator==(const SocuContactTopologyCounts&) const noexcept = default;

    SizeT total_contact_count() const noexcept
    {
        return simplex_normal_pt + simplex_normal_ee + simplex_normal_pe
               + simplex_normal_pp + simplex_friction_pt + simplex_friction_ee
               + simplex_friction_pe + simplex_friction_pp + half_plane_normal_ph
               + half_plane_friction_ph;
    }
};

struct SocuContactTopologySource
{
    SizeT                   reporter_id = 0;
    SizeT                   source_id = 0;
    SocuContactSourceFamily family = SocuContactSourceFamily::Unknown;
    SizeT                   contact_count = 0;
    SizeT                   content_hash = 0;
    SizeT                   layout_token = 0;

    bool operator==(const SocuContactTopologySource&) const noexcept = default;
};

struct SocuContactTopologyStamp
{
    SizeT                     epoch = 0;
    SizeT                     layout_hash = 0;
    SizeT                     content_hash = 0;
    SizeT                     reporter_count = 0;
    SizeT                     source_count = 0;
    SocuContactTopologyCounts counts;

    bool operator==(const SocuContactTopologyStamp&) const noexcept = default;

    bool valid() const noexcept { return epoch != 0; }
};

struct SocuAssemblyPlanKey
{
    SizeT ordering_epoch = 0;
    SizeT native_descriptor_epoch = 0;
    SizeT contact_topology_epoch = 0;
    SizeT contact_layout_hash = 0;
    SizeT contact_content_hash = 0;
    SizeT fixed_mapping_epoch = 0;
    SizeT vertex_projection_epoch = 0;

    SizeT horizon = 0;
    SizeT block_size = 0;
    StructuredContactOffbandPolicy offband_policy =
        StructuredContactOffbandPolicy::Drop;
    bool scalar_diag_fallback_compatibility = false;

    bool operator==(const SocuAssemblyPlanKey&) const noexcept = default;
};

struct SocuVertexSidePlanKey
{
    SizeT ordering_epoch = 0;
    SizeT native_descriptor_epoch = 0;
    SizeT fixed_mapping_epoch = 0;
    SizeT vertex_projection_epoch = 0;

    SizeT horizon = 0;
    SizeT block_size = 0;

    bool operator==(const SocuVertexSidePlanKey&) const noexcept = default;
};

struct SocuContactProgramPlanKey
{
    SocuVertexSidePlanKey side_key;

    SizeT contact_topology_epoch = 0;
    SizeT contact_layout_hash = 0;
    SizeT contact_content_hash = 0;

    StructuredContactOffbandPolicy offband_policy =
        StructuredContactOffbandPolicy::Drop;
    bool scalar_diag_fallback_compatibility = false;

    bool operator==(const SocuContactProgramPlanKey&) const noexcept = default;
};

enum class SocuContactSourceIdValidationStatus : std::uint8_t
{
    ValidDense,
    NonDense,
    Duplicate,
    OutOfRange,
};

enum class SocuContactPlanCacheAction : std::uint8_t
{
    Hit,
    Rebuild,
};

struct SocuContactPlanCacheDecision
{
    SocuContactPlanCacheAction side_plan =
        SocuContactPlanCacheAction::Rebuild;
    SocuContactPlanCacheAction contact_program =
        SocuContactPlanCacheAction::Rebuild;
    bool cold_start = true;

    bool side_plan_hit() const noexcept
    {
        return side_plan == SocuContactPlanCacheAction::Hit;
    }

    bool contact_program_hit() const noexcept
    {
        return contact_program == SocuContactPlanCacheAction::Hit;
    }
};

inline SizeT socu_contact_source_family_value(SocuContactSourceFamily family) noexcept
{
    return static_cast<SizeT>(static_cast<std::uint8_t>(family));
}

inline SizeT socu_contact_source_layout_hash(
    const SocuContactTopologySource& source) noexcept
{
    SizeT hash = SocuContactPlanHashOffset;
    socu_contact_mix_in_place(hash, source.reporter_id);
    socu_contact_mix_in_place(hash, source.source_id);
    socu_contact_mix_in_place(hash,
                              socu_contact_source_family_value(source.family));
    socu_contact_mix_in_place(hash, source.contact_count);
    socu_contact_mix_in_place(hash, source.layout_token);
    return hash;
}

inline SizeT socu_contact_source_content_hash(
    const SocuContactTopologySource& source) noexcept
{
    SizeT hash = socu_contact_source_layout_hash(source);
    socu_contact_mix_in_place(hash, source.content_hash);
    return hash;
}

template <typename Sources>
SocuContactSourceIdValidationStatus socu_validate_contact_source_ids_dense(
    const Sources& sources) noexcept
{
    for(SizeT i = 0; i < static_cast<SizeT>(sources.size()); ++i)
    {
        if(sources[i].source_id >= static_cast<SizeT>(sources.size()))
            return SocuContactSourceIdValidationStatus::OutOfRange;
    }

    for(SizeT i = 0; i < static_cast<SizeT>(sources.size()); ++i)
    {
        for(SizeT j = i + 1; j < static_cast<SizeT>(sources.size()); ++j)
        {
            if(sources[i].source_id == sources[j].source_id)
                return SocuContactSourceIdValidationStatus::Duplicate;
        }
    }

    for(SizeT i = 0; i < static_cast<SizeT>(sources.size()); ++i)
    {
        if(sources[i].source_id != i)
            return SocuContactSourceIdValidationStatus::NonDense;
    }
    return SocuContactSourceIdValidationStatus::ValidDense;
}

template <typename Sources>
bool socu_contact_source_ids_dense(const Sources& sources) noexcept
{
    return socu_validate_contact_source_ids_dense(sources)
           == SocuContactSourceIdValidationStatus::ValidDense;
}

inline SizeT socu_contact_plan_key_hash(const SocuAssemblyPlanKey& key) noexcept
{
    SizeT hash = SocuContactPlanHashOffset;
    socu_contact_mix_in_place(hash, key.ordering_epoch);
    socu_contact_mix_in_place(hash, key.native_descriptor_epoch);
    socu_contact_mix_in_place(hash, key.contact_topology_epoch);
    socu_contact_mix_in_place(hash, key.contact_layout_hash);
    socu_contact_mix_in_place(hash, key.contact_content_hash);
    socu_contact_mix_in_place(hash, key.fixed_mapping_epoch);
    socu_contact_mix_in_place(hash, key.vertex_projection_epoch);
    socu_contact_mix_in_place(hash, key.horizon);
    socu_contact_mix_in_place(hash, key.block_size);
    socu_contact_mix_in_place(
        hash,
        static_cast<SizeT>(static_cast<std::uint8_t>(key.offband_policy)));
    socu_contact_mix_in_place(
        hash,
        key.scalar_diag_fallback_compatibility ? SizeT{1} : SizeT{0});
    return hash;
}

inline SocuVertexSidePlanKey socu_vertex_side_plan_key_from(
    const SocuAssemblyPlanKey& key) noexcept
{
    return SocuVertexSidePlanKey{key.ordering_epoch,
                                 key.native_descriptor_epoch,
                                 key.fixed_mapping_epoch,
                                 key.vertex_projection_epoch,
                                 key.horizon,
                                 key.block_size};
}

inline SocuContactProgramPlanKey socu_contact_program_plan_key_from(
    const SocuAssemblyPlanKey& key) noexcept
{
    return SocuContactProgramPlanKey{
        socu_vertex_side_plan_key_from(key),
        key.contact_topology_epoch,
        key.contact_layout_hash,
        key.contact_content_hash,
        key.offband_policy,
        key.scalar_diag_fallback_compatibility};
}

inline SizeT socu_vertex_side_plan_key_hash(
    const SocuVertexSidePlanKey& key) noexcept
{
    SizeT hash = SocuContactPlanHashOffset;
    socu_contact_mix_in_place(hash, key.ordering_epoch);
    socu_contact_mix_in_place(hash, key.native_descriptor_epoch);
    socu_contact_mix_in_place(hash, key.fixed_mapping_epoch);
    socu_contact_mix_in_place(hash, key.vertex_projection_epoch);
    socu_contact_mix_in_place(hash, key.horizon);
    socu_contact_mix_in_place(hash, key.block_size);
    return hash;
}

inline SizeT socu_contact_program_plan_key_hash(
    const SocuContactProgramPlanKey& key) noexcept
{
    SizeT hash = socu_vertex_side_plan_key_hash(key.side_key);
    socu_contact_mix_in_place(hash, key.contact_topology_epoch);
    socu_contact_mix_in_place(hash, key.contact_layout_hash);
    socu_contact_mix_in_place(hash, key.contact_content_hash);
    socu_contact_mix_in_place(
        hash,
        static_cast<SizeT>(static_cast<std::uint8_t>(key.offband_policy)));
    socu_contact_mix_in_place(
        hash,
        key.scalar_diag_fallback_compatibility ? SizeT{1} : SizeT{0});
    return hash;
}

class SocuContactPlanCacheState
{
  public:
    SocuContactPlanCacheDecision update(const SocuAssemblyPlanKey& key) noexcept
    {
        return update(socu_vertex_side_plan_key_from(key),
                      socu_contact_program_plan_key_from(key));
    }

    SocuContactPlanCacheDecision update(
        const SocuVertexSidePlanKey&     side_key,
        const SocuContactProgramPlanKey& contact_key) noexcept
    {
        SocuContactPlanCacheDecision decision;
        decision.cold_start = !m_valid;

        if(m_valid && side_key == m_side_key)
            decision.side_plan = SocuContactPlanCacheAction::Hit;
        if(m_valid && contact_key == m_contact_key)
            decision.contact_program = SocuContactPlanCacheAction::Hit;

        m_valid       = true;
        m_side_key    = side_key;
        m_contact_key = contact_key;
        return decision;
    }

    bool valid() const noexcept { return m_valid; }
    const SocuVertexSidePlanKey& side_key() const noexcept { return m_side_key; }
    const SocuContactProgramPlanKey& contact_key() const noexcept
    {
        return m_contact_key;
    }

  private:
    bool                         m_valid = false;
    SocuVertexSidePlanKey        m_side_key;
    SocuContactProgramPlanKey    m_contact_key;
};
}  // namespace uipc::backend::cuda_mixed
