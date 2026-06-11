#include <contact_system/rcc_bonded_pt_system.h>
#include <contact_system/rcc_bonded_pt_lookup.h>
#include <muda/cub/device/device_select.h>
#include <muda/ext/eigen/eigen_core_cxx20.h>
#include <muda/ext/eigen/inverse.h>
#include <muda/launch/parallel_for.h>
#include <sim_engine.h>
#include <utils/friction_utils.h>
#include <utils/simplex_contact_mask_utils.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(RCCBondedPTSystem);

namespace
{
struct RCCBondedPTRestShapeBuild
{
    bool      valid = false;
    Vector4i  topo = Vector4i::Zero();
    Matrix3x3 Dm_inv = Matrix3x3::Identity();
    Float     rest_volume = 0.0;
};

MUDA_GENERIC RCCBondedPTRestShapeBuild build_rest_shape(
    const Vector4i& topo,
    const Vector3& point,
    const Vector3& tri0,
    const Vector3& tri1,
    const Vector3& tri2,
    Float min_separate_distance,
    Float det_dm_min)
{
    RCCBondedPTRestShapeBuild out;
    out.topo = topo;

    Vector3 x0 = point;
    Vector3 x1 = tri0;
    Vector3 x2 = tri1;
    Vector3 x3 = tri2;

    Vector3 normal = (x2 - x1).cross(x3 - x1);
    const Float nrm = normal.norm();
    if(nrm <= 0.0)
        return out;
    normal /= nrm;

    const Float signed_dist = normal.dot(x0 - x1);
    if(std::abs(signed_dist) < min_separate_distance)
    {
        const Float sign = signed_dist >= 0.0 ? 1.0 : -1.0;
        x0 += (sign * min_separate_distance - signed_dist) * normal;
    }

    Vector3 a = x1 - x0;
    Vector3 b = x2 - x0;
    Vector3 c = x3 - x0;
    Float det = a.dot(b.cross(c));
    if(det < 0.0)
    {
        Vector3 tmp_x = x1;
        x1 = x2;
        x2 = tmp_x;
        const IndexT tmp_i = out.topo[1];
        out.topo[1] = out.topo[2];
        out.topo[2] = tmp_i;
        a = x1 - x0;
        b = x2 - x0;
        c = x3 - x0;
        det = -det;
    }

    if(det <= det_dm_min)
        return out;

    const Vector3 r0 = b.cross(c) / det;
    const Vector3 r1 = c.cross(a) / det;
    const Vector3 r2 = a.cross(b) / det;
    out.Dm_inv.row(0) = r0.transpose();
    out.Dm_inv.row(1) = r1.transpose();
    out.Dm_inv.row(2) = r2.transpose();
    out.rest_volume = det / 6.0;
    out.valid = true;
    return out;
}

template <typename PositionViewer>
MUDA_GENERIC bool rest_shape_is_valid(
    const RCCBondedPTDeviceEntry& entry,
    muda::CBufferView<Vector3> positions_view,
    PositionViewer positions,
    Float min_separate_distance,
    Float det_dm_min)
{
    const Vector4i topo = entry.topo;
    const IndexT n = static_cast<IndexT>(positions_view.size());
    if(topo[0] < 0 || topo[1] < 0 || topo[2] < 0 || topo[3] < 0
       || topo[0] >= n || topo[1] >= n || topo[2] >= n || topo[3] >= n)
        return false;

    const auto rest = build_rest_shape(topo,
                                       positions(topo[0]),
                                       positions(topo[1]),
                                       positions(topo[2]),
                                       positions(topo[3]),
                                       min_separate_distance,
                                       det_dm_min);
    return rest.valid;
}

MUDA_GENERIC bool rcc_bonded_pt_sticky_gate(IndexT sticky_P,
                                            IndexT sticky_T,
                                            const Vector3& n_P,
                                            const Vector3& n_T,
                                            const Vector3& P,
                                            const Vector3& T0,
                                            const Vector3& T1,
                                            const Vector3& T2)
{
    if(sticky_P == 0 && sticky_T == 0)
        return true;

    using namespace friction;
    Vector2 bary = Vector2::Zero();
    point_triangle_closest_point(P, T0, T1, T2, bary);
    const Vector3 closest = T0 + bary[0] * (T1 - T0) + bary[1] * (T2 - T0);
    const Vector3 v_TP = P - closest;

    if(sticky_P != 0 && v_TP.dot(Float(sticky_P) * n_P) < Float{0})
        return true;
    if(sticky_T != 0 && v_TP.dot(Float(sticky_T) * n_T) > Float{0})
        return true;
    return false;
}

MUDA_GENERIC bool rcc_bonded_pt_rcc_policy_enabled(
    const muda::CDense2D<RCCAdhesiveCoeff>& table,
    const Vector4i& cids)
{
    return table(cids[0], cids[1]).enabled
           && table(cids[0], cids[2]).enabled
           && table(cids[0], cids[3]).enabled;
}

template <typename PositionViewer,
          typename StickyViewer,
          typename NormalViewer,
          typename ContactIdViewer,
          typename SubsceneIdViewer,
          typename ContactMaskViewer,
          typename SubsceneMaskViewer,
          typename AdhesiveViewer>
MUDA_GENERIC U32 release_flags_from_current_shape(
    const Vector4i& topo,
    const Matrix3x3& dm_inv,
    Float rest_volume,
    muda::CBufferView<Vector3> positions_view,
    PositionViewer positions,
    Float det_dm_min,
    Float strain_threshold,
    Float gap_threshold,
    Float slip_threshold,
    Float kappa,
    Float dt,
    Float force_threshold,
    bool sticky_side_enabled,
    muda::CBufferView<IndexT> sticky_sign_view,
    StickyViewer sticky_sign,
    NormalViewer vertex_normal,
    bool policy_enabled,
    muda::CBufferView<IndexT> contact_ids_view,
    ContactIdViewer contact_ids,
    muda::CBufferView<IndexT> subscene_ids_view,
    SubsceneIdViewer subscene_ids,
    ContactMaskViewer contact_mask_tabular,
    SubsceneMaskViewer subscene_mask_tabular,
    AdhesiveViewer adhesive_tabular)
{
    const IndexT n = static_cast<IndexT>(positions_view.size());
    if(topo[0] < 0 || topo[1] < 0 || topo[2] < 0 || topo[3] < 0
       || topo[0] >= n || topo[1] >= n || topo[2] >= n || topo[3] >= n
       || rest_volume <= 0.0)
        return core::RCCBondedPTReleaseDegenerate;

    // Per-pair release thresholds: when the policy (adhesive tabular +
    // contact-element ids) is available, override the global scalar thresholds
    // with this contact-pair's values (averaged over the triangle's three
    // vertices, which normally share one contact element). Falls back to the
    // scalar (global) thresholds otherwise. kappa stays global.
    if(policy_enabled
       && topo[0] < static_cast<IndexT>(contact_ids_view.size())
       && topo[1] < static_cast<IndexT>(contact_ids_view.size())
       && topo[2] < static_cast<IndexT>(contact_ids_view.size())
       && topo[3] < static_cast<IndexT>(contact_ids_view.size()))
    {
        const Vector4i cids{contact_ids(topo[0]),
                            contact_ids(topo[1]),
                            contact_ids(topo[2]),
                            contact_ids(topo[3])};
        Float s = 0, g = 0, l = 0, f = 0;
        for(int j = 1; j < 4; ++j)
        {
            const RCCAdhesiveCoeff cc = adhesive_tabular(cids[0], cids[j]);
            s += cc.bonded_release_strain;
            g += cc.bonded_release_gap;
            l += cc.bonded_release_slip;
            f += cc.bonded_release_force;
        }
        strain_threshold = s / Float{3};
        gap_threshold    = g / Float{3};
        slip_threshold   = l / Float{3};
        force_threshold  = f / Float{3};
    }

    const Vector3 x0 = positions(topo[0]);
    const Vector3 x1 = positions(topo[1]);
    const Vector3 x2 = positions(topo[2]);
    const Vector3 x3 = positions(topo[3]);
    const Vector3 a  = x1 - x0;
    const Vector3 b  = x2 - x0;
    const Vector3 c  = x3 - x0;
    const Float   det = a.dot(b.cross(c));

    U32 flags = core::RCCBondedPTReleaseNone;
    if(det < 0.0)
        flags |= core::RCCBondedPTReleaseFlip;
    if(std::abs(det) <= det_dm_min)
        flags |= core::RCCBondedPTReleaseDegenerate;

    Matrix3x3 Ds;
    Ds.col(0) = a;
    Ds.col(1) = b;
    Ds.col(2) = c;
    const Matrix3x3 F = Ds * dm_inv;
    const Matrix3x3 C = F * F.transpose() - Matrix3x3::Identity();
    const Float strain = std::sqrt(C.squaredNorm());
    if(!std::isfinite(strain))
        flags |= core::RCCBondedPTReleaseDegenerate;
    else if(strain_threshold >= 0.0 && strain > strain_threshold)
        flags |= core::RCCBondedPTReleaseStrain;

    if(force_threshold >= 0.0)
    {
        // F-space restoring force of the ABD ortho bond:
        //   E = kappa * V0 * dt^2 * ||F F^T - I||^2,  dE/dF = 4 kappa V0 dt^2 C F.
        // Scaled by kappa, so it fires when the bond is overloaded even though
        // a stiff bond's own deformation (strain/gap) stays below threshold.
        const Float force = 4.0 * kappa * rest_volume * dt * dt * (C * F).norm();
        if(std::isfinite(force) && force > force_threshold)
            flags |= core::RCCBondedPTReleaseForce;
    }

    if(sticky_side_enabled)
    {
        if(topo[0] >= static_cast<IndexT>(sticky_sign_view.size())
           || topo[1] >= static_cast<IndexT>(sticky_sign_view.size()))
        {
            flags |= core::RCCBondedPTReleaseStickySide;
        }
        else if(!rcc_bonded_pt_sticky_gate(sticky_sign(topo[0]),
                                           sticky_sign(topo[1]),
                                           vertex_normal(topo[0]),
                                           vertex_normal(topo[1]),
                                           x0,
                                           x1,
                                           x2,
                                           x3))
        {
            flags |= core::RCCBondedPTReleaseStickySide;
        }
    }

    if(policy_enabled)
    {
        if(topo[0] >= static_cast<IndexT>(contact_ids_view.size())
           || topo[1] >= static_cast<IndexT>(contact_ids_view.size())
           || topo[2] >= static_cast<IndexT>(contact_ids_view.size())
           || topo[3] >= static_cast<IndexT>(contact_ids_view.size())
           || topo[0] >= static_cast<IndexT>(subscene_ids_view.size())
           || topo[1] >= static_cast<IndexT>(subscene_ids_view.size())
           || topo[2] >= static_cast<IndexT>(subscene_ids_view.size())
           || topo[3] >= static_cast<IndexT>(subscene_ids_view.size()))
        {
            flags |= core::RCCBondedPTReleasePolicy;
        }
        else
        {
            const Vector4i cids{contact_ids(topo[0]),
                                contact_ids(topo[1]),
                                contact_ids(topo[2]),
                                contact_ids(topo[3])};
            const Vector4i scids{subscene_ids(topo[0]),
                                 subscene_ids(topo[1]),
                                 subscene_ids(topo[2]),
                                 subscene_ids(topo[3])};
            if(!allow_PT_contact(contact_mask_tabular, cids)
               || !allow_PT_contact(subscene_mask_tabular, scids)
               || !rcc_bonded_pt_rcc_policy_enabled(adhesive_tabular, cids))
            {
                flags |= core::RCCBondedPTReleasePolicy;
            }
        }
    }

    if(gap_threshold >= 0.0 || slip_threshold >= 0.0)
    {
        using namespace friction;

        const Matrix3x3 Dm = muda::eigen::inverse(dm_inv);
        const Vector3 r0 = Vector3::Zero();
        const Vector3 r1 = Dm.col(0);
        const Vector3 r2 = Dm.col(1);
        const Vector3 r3 = Dm.col(2);

        const Vector3 rest_n = (r2 - r1).cross(r3 - r1);
        const Vector3 curr_n = (x2 - x1).cross(x3 - x1);
        const Float rest_nrm = rest_n.norm();
        const Float curr_nrm = curr_n.norm();
        if(rest_nrm <= det_dm_min || curr_nrm <= det_dm_min)
        {
            flags |= core::RCCBondedPTReleaseDegenerate;
        }
        else
        {
            const Float rest_dist = rest_n.dot(r0 - r1) / rest_nrm;
            const Float curr_dist = curr_n.dot(x0 - x1) / curr_nrm;
            if(!std::isfinite(rest_dist) || !std::isfinite(curr_dist))
            {
                flags |= core::RCCBondedPTReleaseDegenerate;
            }
            else if(gap_threshold >= 0.0)
            {
                const Float normal_gap =
                    std::abs(curr_dist) - std::abs(rest_dist);
                if(normal_gap > gap_threshold)
                    flags |= core::RCCBondedPTReleaseGap;
            }

            if(slip_threshold >= 0.0)
            {
                Vector2 rest_bary = Vector2::Zero();
                Vector2 curr_bary = Vector2::Zero();
                point_triangle_closest_point(r0, r1, r2, r3, rest_bary);
                point_triangle_closest_point(x0, x1, x2, x3, curr_bary);
                const Vector2 delta = curr_bary - rest_bary;
                const Vector3 slip =
                    delta[0] * (x2 - x1) + delta[1] * (x3 - x1);
                const Float slip_norm = slip.norm();
                if(!std::isfinite(slip_norm))
                    flags |= core::RCCBondedPTReleaseDegenerate;
                else if(slip_norm > slip_threshold)
                    flags |= core::RCCBondedPTReleaseSlip;
            }
        }
    }

    return flags;
}
}  // namespace

void RCCBondedPTSystem::Impl::clear()
{
    m_bridge.clear();
    m_counters = {};
    m_candidate_entries.resize(0);
    m_new_locked_entries.resize(0);
    m_valid_new_locked_entries.resize(0);
    m_new_locked_keys.resize(0);
    m_new_locked_dm_inv.resize(0);
    m_new_locked_rest_volume.resize(0);
    m_prev_entries.resize(0);
    m_released_entries.resize(0);
    m_released_keys.resize(0);
    m_released_topos.resize(0);
    m_released_beta.resize(0);
    m_released_age.resize(0);
    m_released_flags.resize(0);
    m_carry_prev_entries.resize(0);
    m_merged_entries.resize(0);
    m_merged_keys.resize(0);
    clear_filter_keys();
    m_last_synced_filter_generation = 0;
}

void RCCBondedPTSystem::Impl::upload(const core::RCCBondedPTState& state)
{
    m_bridge.upload(state);
    m_counters = state.counters();
}

void RCCBondedPTSystem::Impl::lock_from_rcc_pt_snapshot(
    muda::CBufferView<Vector4i> pairs,
    muda::CBufferView<Float> beta,
    muda::CBufferView<Vector3> positions,
    Float beta_lock_threshold)
{
    lock_from_rcc_pt_snapshot(
        pairs, beta, positions, beta_lock_threshold, RCCBondedPTReleaseContext{});
}

void RCCBondedPTSystem::Impl::lock_from_rcc_pt_snapshot(
    muda::CBufferView<Vector4i> pairs,
    muda::CBufferView<Float> beta,
    muda::CBufferView<Vector3> positions,
    Float beta_lock_threshold,
    const RCCBondedPTReleaseContext& release_context)
{
    if(!m_enabled)
        return;

    UIPC_ASSERT(pairs.size() == 0 || pairs.size() == beta.size(),
                "RCC bonded PT producer received unzipped PT/beta buffers.");

    using namespace muda;

    const SizeT n = pairs.size();
    m_counters.candidate_count += n;

    const auto prev_keys  = m_bridge.locked_keys();
    const auto prev_topos = m_bridge.locked_topos();
    const auto prev_beta  = m_bridge.locked_beta();
    const auto prev_age   = m_bridge.locked_age();
    const auto prev_flags = m_bridge.release_flags();
    const auto prev_dm_inv = m_bridge.locked_dm_inv();
    const auto prev_rest_volume = m_bridge.locked_rest_volume();
    const SizeT prev_n    = m_bridge.size();

    m_prev_entries.resize(prev_n);
    if(prev_n > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(prev_n,
                   [keys = prev_keys.viewer().name("prev_keys"),
                    topos = prev_topos.viewer().name("prev_topos"),
                    beta = prev_beta.viewer().name("prev_beta"),
                    age = prev_age.viewer().name("prev_age"),
                    flags = prev_flags.viewer().name("prev_flags"),
                    dm_inv = prev_dm_inv.viewer().name("prev_dm_inv"),
                    rest_volume =
                        prev_rest_volume.viewer().name("prev_rest_volume"),
                    positions_view = positions,
                    positions = positions.viewer().name("positions"),
                    det_dm_min = m_det_dm_min,
                    release_strain = m_release_strain_threshold,
                    release_gap = m_release_gap_threshold,
                    release_slip = m_release_slip_threshold,
                    kappa = m_kappa,
                    dt = m_dt,
                    release_force = m_release_force_threshold,
                    sticky_side_enabled = release_context.sticky_side_enabled,
                    sticky_sign_view = release_context.sticky_sign,
                    sticky_sign =
                        release_context.sticky_sign.viewer().name("sticky_sign"),
                    vertex_normal =
                        release_context.vertex_normal.viewer().name("vertex_normal"),
                    policy_enabled = release_context.policy_enabled,
                    contact_ids_view = release_context.contact_element_ids,
                    contact_ids = release_context.contact_element_ids.viewer().name("contact_ids"),
                    subscene_ids_view = release_context.subscene_element_ids,
                    subscene_ids =
                        release_context.subscene_element_ids.viewer().name("subscene_ids"),
                    contact_mask_tabular =
                        release_context.contact_mask_tabular.viewer().name("contact_mask_tabular"),
                    subscene_mask_tabular =
                        release_context.subscene_mask_tabular.viewer().name("subscene_mask_tabular"),
                    adhesive_tabular =
                        release_context.adhesive_tabular.viewer().name("adhesive_tabular"),
                    entries = m_prev_entries.view().viewer().name("prev_entries")] __device__(int i) mutable
                   {
                       U32 release_flags = flags(i);
                       if(release_flags == core::RCCBondedPTReleaseNone)
                       {
                           release_flags |= release_flags_from_current_shape(
                               topos(i),
                               dm_inv(i),
                               rest_volume(i),
                               positions_view,
                               positions,
                               det_dm_min,
                               release_strain,
                               release_gap,
                               release_slip,
                               kappa,
                               dt,
                               release_force,
                               sticky_side_enabled,
                               sticky_sign_view,
                               sticky_sign,
                               vertex_normal,
                               policy_enabled,
                               contact_ids_view,
                               contact_ids,
                               subscene_ids_view,
                               subscene_ids,
                               contact_mask_tabular,
                               subscene_mask_tabular,
                               adhesive_tabular);
                       }

                       RCCBondedPTDeviceEntry entry;
                       entry.key           = keys(i);
                       entry.topo          = topos(i);
                       entry.beta          = beta(i);
                       entry.age           = age(i) + 1;
                       entry.release_flags = release_flags;
                       entries(i)          = entry;
                   });
    }

    m_released_entries.resize(prev_n);
    if(prev_n > 0)
    {
        DeviceSelect().If(
            m_prev_entries.data(),
            m_released_entries.data(),
            m_released_count.data(),
            prev_n,
            [] CUB_RUNTIME_FUNCTION(const RCCBondedPTDeviceEntry& entry)
            { return entry.release_flags != core::RCCBondedPTReleaseNone; });
    }
    else
    {
        m_released_count = 0;
    }

    const IndexT released_count = m_released_count;
    m_released_entries.resize(released_count);
    m_released_keys.resize(released_count);
    m_released_topos.resize(released_count);
    m_released_beta.resize(released_count);
    m_released_age.resize(released_count);
    m_released_flags.resize(released_count);
    if(released_count > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(released_count,
                   [entries = m_released_entries.view().viewer().name("released_entries"),
                    keys = m_released_keys.view().viewer().name("released_keys"),
                    topos = m_released_topos.view().viewer().name("released_topos"),
                    beta = m_released_beta.view().viewer().name("released_beta"),
                    age = m_released_age.view().viewer().name("released_age"),
                    flags = m_released_flags.view().viewer().name("released_flags")] __device__(int i) mutable
                   {
                       const RCCBondedPTDeviceEntry entry = entries(i);
                       keys(i)  = entry.key;
                       topos(i) = entry.topo;
                       beta(i)  = entry.beta;
                       age(i)   = entry.age;
                       flags(i) = entry.release_flags;
                   });
        m_counters.released_count += static_cast<SizeT>(released_count);
    }

    m_candidate_entries.resize(n);
    if(n > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [pairs = pairs.viewer().name("pairs"),
                    beta  = beta.viewer().name("beta"),
                    prev_keys_view = prev_keys,
                    prev_keys = prev_keys.viewer().name("prev_keys"),
                    prev_age = prev_age.viewer().name("prev_age"),
                    entries = m_candidate_entries.view().viewer().name("candidate_entries")] __device__(int i) mutable
                   {
                       const Vector4i topo = pairs(i);
                       const U64      key  = rcc_bonded_pt_key(topo);
                       IndexT         age  = 1;
                       const IndexT   prev = rcc_bonded_pt_lower_bound(prev_keys_view, key);
                       if(prev < static_cast<IndexT>(prev_keys_view.size())
                          && prev_keys(prev) == key)
                           age = prev_age(prev) + 1;

                       RCCBondedPTDeviceEntry entry;
                       entry.key           = key;
                       entry.topo          = topo;
                       entry.beta          = beta(i);
                       entry.age           = age;
                       entry.release_flags = core::RCCBondedPTReleaseNone;
                       entries(i)          = entry;
                   });
    }

    m_new_locked_entries.resize(n);
    if(n > 0)
    {
        muda::CBufferView<U64> released_keys = m_released_keys;
        // Per-pair lock threshold: when the policy (adhesive tabular +
        // contact-element ids) is provided, this pair's bonded_lock_threshold
        // (averaged over the triangle's verts) decides locking instead of the
        // global scalar. Falls back to beta_lock_threshold otherwise.
        const bool use_tabular = release_context.policy_enabled
                                 && release_context.contact_element_ids.size() > 0;
        auto         contact_ids = release_context.contact_element_ids.viewer();
        auto         adhesive_tabular = release_context.adhesive_tabular.viewer();
        const IndexT n_cid =
            static_cast<IndexT>(release_context.contact_element_ids.size());
        DeviceSelect().If(
            m_candidate_entries.data(),
            m_new_locked_entries.data(),
            m_new_locked_count.data(),
            n,
            [beta_lock_threshold, released_keys, use_tabular, contact_ids,
             adhesive_tabular, n_cid] CUB_RUNTIME_FUNCTION(
                const RCCBondedPTDeviceEntry& entry)
            {
                Float thr = beta_lock_threshold;
                if(use_tabular)
                {
                    const Vector4i& t = entry.topo;
                    if(t[0] >= 0 && t[0] < n_cid && t[1] < n_cid && t[2] < n_cid
                       && t[3] < n_cid)
                    {
                        const Vector4i cids{contact_ids(t[0]),
                                            contact_ids(t[1]),
                                            contact_ids(t[2]),
                                            contact_ids(t[3])};
                        Float a = 0;
                        for(int j = 1; j < 4; ++j)
                            a += adhesive_tabular(cids[0], cids[j]).bonded_lock_threshold;
                        thr = a / Float{3};
                    }
                }
                return entry.beta >= thr
                       && entry.release_flags == core::RCCBondedPTReleaseNone
                       && !rcc_bonded_pt_is_locked(released_keys, entry.key);
            });
    }
    else
    {
        m_new_locked_count = 0;
    }

    IndexT new_locked_count = m_new_locked_count;
    m_new_locked_entries.resize(new_locked_count);

    m_valid_new_locked_entries.resize(new_locked_count);
    if(new_locked_count > 0)
    {
        auto positions_view = positions;
        auto positions_viewer = positions.viewer().name("positions");
        const Float min_separate_distance = m_min_separate_distance;
        const Float det_dm_min = m_det_dm_min;
        DeviceSelect().If(
            m_new_locked_entries.data(),
            m_valid_new_locked_entries.data(),
            m_valid_new_locked_count.data(),
            new_locked_count,
            [positions_view,
             positions_viewer,
             min_separate_distance,
             det_dm_min] CUB_RUNTIME_FUNCTION(
                const RCCBondedPTDeviceEntry& entry)
            {
                return rest_shape_is_valid(entry,
                                           positions_view,
                                           positions_viewer,
                                           min_separate_distance,
                                           det_dm_min);
            });
    }
    else
    {
        m_valid_new_locked_count = 0;
    }

    const IndexT valid_new_locked_count = m_valid_new_locked_count;
    m_counters.degenerate_rejected_count +=
        static_cast<SizeT>(new_locked_count - valid_new_locked_count);
    new_locked_count = valid_new_locked_count;
    m_valid_new_locked_entries.resize(new_locked_count);
    if(new_locked_count > 0)
    {
        m_new_locked_entries.resize(new_locked_count);
        m_new_locked_entries.view().copy_from(m_valid_new_locked_entries.view());
    }
    else
    {
        m_new_locked_entries.resize(0);
    }

    m_new_locked_keys.resize(new_locked_count);
    m_new_locked_dm_inv.resize(new_locked_count);
    m_new_locked_rest_volume.resize(new_locked_count);

    if(new_locked_count > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(new_locked_count,
                   [entries = m_new_locked_entries.view().viewer().name("new_entries"),
                    keys = m_new_locked_keys.view().viewer().name("new_keys")] __device__(int i) mutable
                   {
                       keys(i) = entries(i).key;
                   });

        thrust::sort_by_key(thrust::device,
                            m_new_locked_keys.data(),
                            m_new_locked_keys.data() + new_locked_count,
                            m_new_locked_entries.data());
        auto unique_end = thrust::unique_by_key(thrust::device,
                                                m_new_locked_keys.data(),
                                                m_new_locked_keys.data()
                                                    + new_locked_count,
                                                m_new_locked_entries.data());
        const IndexT unique_new_locked_count =
            static_cast<IndexT>(unique_end.first - m_new_locked_keys.data());
        m_counters.duplicate_suppressed_count +=
            static_cast<SizeT>(new_locked_count - unique_new_locked_count);
        new_locked_count = unique_new_locked_count;
        m_new_locked_entries.resize(new_locked_count);
        m_new_locked_keys.resize(new_locked_count);
        m_new_locked_dm_inv.resize(new_locked_count);
        m_new_locked_rest_volume.resize(new_locked_count);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(new_locked_count,
                   [entries = m_new_locked_entries.view().viewer().name("new_entries"),
                    positions_view = positions,
                    positions = positions.viewer().name("positions"),
                    dm_inv = m_new_locked_dm_inv.view().viewer().name("fresh_dm_inv"),
                    rest_volume =
                        m_new_locked_rest_volume.view().viewer().name("fresh_rest_volume"),
                    min_separate_distance = m_min_separate_distance,
                    det_dm_min = m_det_dm_min] __device__(int i) mutable
                   {
                       auto entry = entries(i);
                       const bool valid = rest_shape_is_valid(entry,
                                                              positions_view,
                                                              positions,
                                                              min_separate_distance,
                                                              det_dm_min);
                       if(valid)
                       {
                           const Vector4i topo = entry.topo;
                           const auto rest =
                               build_rest_shape(topo,
                                                positions(topo[0]),
                                                positions(topo[1]),
                                                positions(topo[2]),
                                                positions(topo[3]),
                                                min_separate_distance,
                                                det_dm_min);
                           entry.topo = rest.topo;
                           entries(i) = entry;
                           dm_inv(i) = rest.Dm_inv;
                           rest_volume(i) = rest.rest_volume;
                       }
                       else
                       {
                           dm_inv(i) = Matrix3x3::Identity();
                           rest_volume(i) = 0.0;
                       }
                   });
    }

    m_carry_prev_entries.resize(prev_n);
    if(prev_n > 0)
    {
        muda::CBufferView<U64> new_locked_keys = m_new_locked_keys;
        DeviceSelect().If(
            m_prev_entries.data(),
            m_carry_prev_entries.data(),
            m_carry_prev_count.data(),
            prev_n,
            [new_locked_keys] CUB_RUNTIME_FUNCTION(
                const RCCBondedPTDeviceEntry& entry)
            {
                return entry.release_flags == core::RCCBondedPTReleaseNone
                       && !rcc_bonded_pt_is_locked(new_locked_keys, entry.key);
            });
    }
    else
    {
        m_carry_prev_count = 0;
    }

    const IndexT carry_prev_count = m_carry_prev_count;
    m_carry_prev_entries.resize(carry_prev_count);

    const IndexT merged_count = carry_prev_count + new_locked_count;
    m_merged_entries.resize(merged_count);
    m_merged_keys.resize(merged_count);
    if(merged_count > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(merged_count,
                   [carry_count = carry_prev_count,
                    carry = m_carry_prev_entries.view().viewer().name("carry_prev"),
                    fresh = m_new_locked_entries.view().viewer().name("new_locked"),
                    merged = m_merged_entries.view().viewer().name("merged_entries"),
                    keys = m_merged_keys.view().viewer().name("merged_keys")] __device__(int i) mutable
                   {
                       const RCCBondedPTDeviceEntry entry =
                           i < carry_count ? carry(i) : fresh(i - carry_count);
                       merged(i) = entry;
                       keys(i)   = entry.key;
                   });

        thrust::sort_by_key(thrust::device,
                            m_merged_keys.data(),
                            m_merged_keys.data() + merged_count,
                            m_merged_entries.data());
    }

    m_counters.locked_count = static_cast<SizeT>(merged_count);
    m_bridge.replace_from_sorted_device_entries(m_merged_entries.view(),
                                                m_counters,
                                                m_new_locked_keys.view(),
                                                m_new_locked_dm_inv.view(),
                                                m_new_locked_rest_volume.view());
    feed_filter_keys();
}

core::RCCBondedPTState RCCBondedPTSystem::Impl::download() const
{
    auto state = m_bridge.download();
    state.set_counters(m_counters);
    return state;
}

SizeT RCCBondedPTSystem::Impl::size() const noexcept
{
    return m_bridge.size();
}

bool RCCBondedPTSystem::Impl::empty() const noexcept
{
    return m_bridge.empty();
}

void RCCBondedPTSystem::Impl::set_enabled(bool enabled) noexcept
{
    m_enabled = enabled;
}

void RCCBondedPTSystem::Impl::set_skip_ccd(bool enabled) noexcept
{
    m_skip_ccd = enabled;
}

bool RCCBondedPTSystem::Impl::enabled() const noexcept
{
    return m_enabled;
}

void RCCBondedPTSystem::Impl::set_rest_shape_config(Float min_separate_distance,
                                                    Float det_dm_min) noexcept
{
    m_min_separate_distance = min_separate_distance;
    m_det_dm_min = det_dm_min;
}

void RCCBondedPTSystem::Impl::set_release_config(Float strain_threshold,
                                                 Float gap_threshold,
                                                 Float slip_threshold) noexcept
{
    m_release_strain_threshold = strain_threshold;
    m_release_gap_threshold = gap_threshold;
    m_release_slip_threshold = slip_threshold;
}

void RCCBondedPTSystem::Impl::set_release_force_config(Float force_threshold,
                                                       Float kappa,
                                                       Float dt) noexcept
{
    m_release_force_threshold = force_threshold;
    m_kappa = kappa;
    m_dt = dt;
}

void RCCBondedPTSystem::Impl::bind_filter(SimplexTrajectoryFilter* filter) noexcept
{
    simplex_trajectory_filter = filter;
    m_last_synced_filter_generation =
        filter ? filter->rcc_bonded_pt_filter_generation() : 0;
    if(filter)
        filter->set_rcc_bonded_pt_skip_ccd(m_skip_ccd);
}

void RCCBondedPTSystem::Impl::feed_filter_keys() const noexcept
{
    if(!m_enabled || !simplex_trajectory_filter)
        return;
    simplex_trajectory_filter->set_rcc_bonded_pt_locked_keys(m_bridge.locked_keys());
    simplex_trajectory_filter->set_rcc_bonded_pt_skip_ccd(m_skip_ccd);
}

void RCCBondedPTSystem::Impl::clear_filter_keys() const noexcept
{
    if(simplex_trajectory_filter)
        simplex_trajectory_filter->clear_rcc_bonded_pt_locked_keys();
}

void RCCBondedPTSystem::Impl::sync_filter_skipped_count() noexcept
{
    if(!m_enabled || !simplex_trajectory_filter)
        return;
    const SizeT generation =
        simplex_trajectory_filter->rcc_bonded_pt_filter_generation();
    if(generation == m_last_synced_filter_generation)
        return;
    m_counters.filter_skipped_count +=
        simplex_trajectory_filter->rcc_bonded_pt_filter_skipped_count();
    m_last_synced_filter_generation = generation;
    m_counters.locked_count = size();
}

void RCCBondedPTSystem::Impl::feed_filter_keys(
    SimplexTrajectoryFilter::Impl& filter) const noexcept
{
    if(!m_enabled)
        return;
    filter.set_rcc_bonded_pt_locked_keys(m_bridge.locked_keys());
    filter.rcc_bonded_pt_skip_ccd = m_skip_ccd;
}

void RCCBondedPTSystem::Impl::sync_filter_skipped_count(
    const SimplexTrajectoryFilter::Impl& filter) noexcept
{
    if(!m_enabled)
        return;
    const SizeT generation = filter.rcc_bonded_pt_filter_generation();
    if(generation == m_last_synced_filter_generation)
        return;
    m_counters.filter_skipped_count += filter.rcc_bonded_pt_filter_skipped_count();
    m_last_synced_filter_generation = generation;
    m_counters.locked_count = size();
}

const core::RCCBondedPTCounters& RCCBondedPTSystem::Impl::counters() const noexcept
{
    return m_counters;
}

void RCCBondedPTSystem::Impl::add_lock_rejection_counts(SizeT distance_rejected,
                                                        SizeT policy_rejected) noexcept
{
    m_counters.distance_rejected_count += distance_rejected;
    m_counters.policy_rejected_count += policy_rejected;
}

muda::CBufferView<U64> RCCBondedPTSystem::Impl::released_keys() const noexcept
{
    return m_released_keys;
}

muda::CBufferView<Vector4i> RCCBondedPTSystem::Impl::released_topos() const noexcept
{
    return m_released_topos;
}

muda::CBufferView<Float> RCCBondedPTSystem::Impl::released_beta() const noexcept
{
    return m_released_beta;
}

muda::CBufferView<IndexT> RCCBondedPTSystem::Impl::released_age() const noexcept
{
    return m_released_age;
}

muda::CBufferView<U32> RCCBondedPTSystem::Impl::released_flags() const noexcept
{
    return m_released_flags;
}

RCCBondedPTStateBridge& RCCBondedPTSystem::Impl::bridge() noexcept
{
    return m_bridge;
}

const RCCBondedPTStateBridge& RCCBondedPTSystem::Impl::bridge() const noexcept
{
    return m_bridge;
}

void RCCBondedPTSystem::do_build()
{
    auto& config = world().scene().config();
    auto  enabled_attr = config.find<IndexT>("rcc_bonded_pt_enabled");
    m_impl.set_enabled(enabled_attr && enabled_attr->view()[0] != 0);
    auto min_sep_attr =
        config.find<Float>("rcc_bonded_pt_min_separate_distance");
    auto det_dm_min_attr = config.find<Float>("rcc_bonded_pt_det_dm_min");
    m_impl.set_rest_shape_config(min_sep_attr ? min_sep_attr->view()[0] : 1e-6,
                                 det_dm_min_attr ? det_dm_min_attr->view()[0] : 1e-12);
    auto release_strain_attr =
        config.find<Float>("rcc_bonded_pt_release_strain");
    auto release_gap_attr = config.find<Float>("rcc_bonded_pt_release_gap");
    auto release_slip_attr = config.find<Float>("rcc_bonded_pt_release_slip");
    m_impl.set_release_config(release_strain_attr ? release_strain_attr->view()[0]
                                                  : 1e30,
                              release_gap_attr ? release_gap_attr->view()[0] : 1e30,
                              release_slip_attr ? release_slip_attr->view()[0]
                                                : 1e30);
    auto release_force_attr = config.find<Float>("rcc_bonded_pt_release_force");
    auto kappa_attr         = config.find<Float>("rcc_bonded_pt_kappa");
    auto dt_attr            = config.find<Float>("dt");
    m_impl.set_release_force_config(
        release_force_attr ? release_force_attr->view()[0] : 1e30,
        kappa_attr ? kappa_attr->view()[0] : 1e8,
        dt_attr ? dt_attr->view()[0] : 0.01);
    auto skip_ccd_attr = config.find<IndexT>("rcc_bonded_pt_skip_ccd");
    // skip_ccd: <0 (default) = auto -> skip the CCD thickness check for locked
    // pairs whenever bonded is enabled (a locked pair is owned by the ABD
    // virtual tet; the CCD check on it is redundant and aborts on over-
    // compression). 0/1 = explicit override. Only takes effect when enabled()
    // (the filter feed/bind paths are enabled-gated).
    const IndexT skip_v = skip_ccd_attr ? skip_ccd_attr->view()[0] : IndexT{-1};
    m_impl.set_skip_ccd(skip_v < 0 ? true : (skip_v != 0));
    m_impl.global_trajectory_filter = find<GlobalTrajectoryFilter>();

    on_init_scene(
        [this]
        {
            if(!m_impl.enabled() || !m_impl.global_trajectory_filter)
                return;
            m_impl.bind_filter(
                m_impl.global_trajectory_filter->find<SimplexTrajectoryFilter>().view());
            m_impl.feed_filter_keys();
        });
    on_rebuild_scene(
        [this]
        {
            if(!m_impl.enabled())
                return;
            m_impl.feed_filter_keys();
        });
}

void RCCBondedPTSystem::clear()
{
    m_impl.clear();
}

void RCCBondedPTSystem::upload(const core::RCCBondedPTState& state)
{
    m_impl.upload(state);
    m_impl.feed_filter_keys();
}

void RCCBondedPTSystem::lock_from_rcc_pt_snapshot(
    muda::CBufferView<Vector4i> pairs,
    muda::CBufferView<Float> beta,
    muda::CBufferView<Vector3> positions,
    Float beta_lock_threshold)
{
    m_impl.lock_from_rcc_pt_snapshot(pairs, beta, positions, beta_lock_threshold);
}

void RCCBondedPTSystem::lock_from_rcc_pt_snapshot(
    muda::CBufferView<Vector4i> pairs,
    muda::CBufferView<Float> beta,
    muda::CBufferView<Vector3> positions,
    Float beta_lock_threshold,
    const RCCBondedPTReleaseContext& release_context)
{
    m_impl.lock_from_rcc_pt_snapshot(
        pairs, beta, positions, beta_lock_threshold, release_context);
}

core::RCCBondedPTState RCCBondedPTSystem::download() const
{
    return m_impl.download();
}

SizeT RCCBondedPTSystem::size() const noexcept
{
    return m_impl.size();
}

bool RCCBondedPTSystem::empty() const noexcept
{
    return m_impl.empty();
}

bool RCCBondedPTSystem::enabled() const noexcept
{
    return m_impl.enabled();
}

const core::RCCBondedPTCounters& RCCBondedPTSystem::counters() const noexcept
{
    return m_impl.counters();
}

void RCCBondedPTSystem::add_lock_rejection_counts(SizeT distance_rejected,
                                                  SizeT policy_rejected) noexcept
{
    m_impl.add_lock_rejection_counts(distance_rejected, policy_rejected);
}

muda::CBufferView<Vector4i> RCCBondedPTSystem::locked_topos() const noexcept
{
    return m_impl.bridge().locked_topos();
}

muda::CBufferView<Matrix3x3> RCCBondedPTSystem::locked_dm_inv() const noexcept
{
    return m_impl.bridge().locked_dm_inv();
}

muda::CBufferView<Float> RCCBondedPTSystem::locked_rest_volume() const noexcept
{
    return m_impl.bridge().locked_rest_volume();
}

muda::CBufferView<U64> RCCBondedPTSystem::locked_keys() const noexcept
{
    return m_impl.bridge().locked_keys();
}

muda::CBufferView<Float> RCCBondedPTSystem::locked_beta() const noexcept
{
    return m_impl.bridge().locked_beta();
}

muda::CBufferView<U64> RCCBondedPTSystem::released_keys() const noexcept
{
    return m_impl.released_keys();
}

muda::CBufferView<Float> RCCBondedPTSystem::released_beta() const noexcept
{
    return m_impl.released_beta();
}

void RCCBondedPTSystem::feed_filter_keys() const noexcept
{
    m_impl.feed_filter_keys();
}

void RCCBondedPTSystem::clear_filter_keys() const noexcept
{
    m_impl.clear_filter_keys();
}

void RCCBondedPTSystem::sync_filter_skipped_count() noexcept
{
    m_impl.sync_filter_skipped_count();
}
}  // namespace uipc::backend::cuda
