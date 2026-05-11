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
bool socu_contact_source_ids_dense(const Sources& sources) noexcept
{
    for(SizeT i = 0; i < static_cast<SizeT>(sources.size()); ++i)
    {
        if(sources[i].source_id != i)
            return false;
    }
    return true;
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
}  // namespace uipc::backend::cuda_mixed
