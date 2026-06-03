#include <contact_system/simplex_frictional_contact.h>
#include <contact_system/rcc_adhesive_coeff.h>
#include <contact_system/rcc_bonded_pt_beta_carry.h>
#include <contact_system/rcc_bonded_pt_system.h>
#include <contact_system/contact_models/codim_ipc_simplex_rcc_adhesive_function.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>
#include <time_integrator/time_integrator.h>
#include <utils/codim_thickness.h>
#include <utils/matrix_assembler.h>
#include <utils/make_spd.h>
#include <utils/primitive_d_hat.h>
#include <pipeline/ipc_pipeline_flag.h>
#include <kernel_cout.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_buffer_2d.h>
#include <muda/buffer/device_var.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <uipc/common/log.h>
#include <uipc/common/span.h>
#include <uipc/geometry/simplicial_complex.h>
#include <uipc/builtin/attribute_name.h>
#include <sim_engine.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <global_geometry/global_vertex_manager.h>
#include <contact_system/global_contact_manager.h>
#include <vector>

namespace uipc::backend::cuda
{
// RCC adhesive simplex contact reporter (additive on top of IPC barrier+friction).
//
// V2 implementation (PT-only):
//   - Normal adhesion E_n = Cn/(2 dHat)·β²·D, gradient/Hessian via the unflagged
//     g_PT plane-projection formula (no diagonal-bias artifact on coplanar facets).
//   - Tangential adhesion E_t = Ct/(2 dHat)·β²·‖u‖² with the same lagged tangent
//     basis as IPC friction.
//   - β EVOLVES per step via XBow's debonding/bonding rule (W, eta, bonding_rate,
//     p0 all active) — adaptive scaling factors (Cn/dHat, r_scale, W_scale,
//     η·W_scale/10) ported verbatim from XBow.
//   - β PERSISTS across steps via sorted-vertex-tuple u64 hash keys. New pairs get
//     `initial_beta` (or a single bonding kick, whichever is larger).
//
//   - EE/PE/PP adhesion DISABLED in v2 (no normal, no tangential, no β-evolution).
//   - Per-step β update runs from RCCBetaEvolutionTimeIntegrator (see bottom of
//     this file) which fires at end-of-step via TimeIntegratorManager.
//
// Disabling: if the user never calls RCCAdhesive::apply_to on the ContactTabular,
// the "Cn" attribute is absent and the reporter throws SimSystemException in
// do_build → the SimSystem framework unregisters it cleanly.
class IPCSimplexRCCAdhesiveContact final : public SimplexFrictionalContact
{
  public:
    using SimplexFrictionalContact::SimplexFrictionalContact;

    // (N x N) tabular over contact-element pairs.
    muda::DeviceBuffer2D<RCCAdhesiveCoeff> m_adhesive_tabular;
    IndexT                                 m_N = 0;

    // Per-VT-primitive β (Phase 6): aligned with friction_VTs(), one β per
    // vertex-triangle primitive regardless of which closest feature (PT/PE/PP)
    // is active this step. Name kept as `m_beta_PT` for persistence/anchor
    // continuity; it now spans the full VT list, not just face-interior PT.
    muda::DeviceBuffer<Float> m_beta_PT;
    // Face-interior (flag==4) β compacted from m_beta_PT, aligned with
    // friction_PTs(), fed to the bonded producer so bonding stays PT-only
    // (Step 5 will let the producer consume edge/corner VTs too).
    muda::DeviceBuffer<Float>  m_beta_PT_face;
    muda::DeviceVar<IndexT>    m_beta_PT_face_count;
    muda::DeviceBuffer<IndexT> m_vt_face_flags;

    // ---- v2 state ----
    // β + sorted keys snapshotted at end of last step (Phase A output;
    // Phase B input at start of next step).
    muda::DeviceBuffer<U64>     m_prev_keys_PT;
    muda::DeviceBuffer<Float>   m_prev_beta_PT;

    // ---- cross-layer occlusion gate ----
    // Per-PT-pair flag: 1 ⇔ an intervening surface triangle sits between
    // P and T at frame-open positions, so this pair cannot physically bond.
    // Aligned with m_beta_PT / friction_PTs(). Re-evaluated every frame at
    // Phase B (both for newly-created pairs and for pairs matched from
    // m_prev_keys_PT); threaded into PT_beta_evolve_existing in Phase A so
    // β stays pinned at 0 for blocked pairs (β=0 is not absorbing in the
    // evolution rule — the bonding_term can lift it from zero without this
    // gate). Not snapshotted across steps: the next frame's Phase B re-runs
    // the test against the current geometry, so persistence is unnecessary.
    muda::DeviceBuffer<IndexT>  m_blocked_PT;

    // Positions at the START of the current step. Phase A uses these to compute
    // the lagged tangent basis and tangential displacement `u` over the step
    // that just ended.
    muda::DeviceBuffer<Vector3> m_pos_at_step_begin;

    // Step-boundary detection: Phase B runs once per frame, the rest of the
    // Newton iterations within the same frame just read m_beta_PT.
    SizeT m_last_seen_frame = ~SizeT(0);
    bool  m_first_step      = true;

    // Set true by set_prev_pt_state() (called from
    // RCCAdhesionStateAccessorFeature::load_pt_state). When true, the first
    // step's Phase B enters _phase_b_match_or_init against the loaded
    // m_prev_keys_PT instead of _phase_b_init_all_new. Cleared after the
    // first frame consumes it.
    bool  m_has_loaded_prev_state = false;

    // ---- v3 state: single-sided adhesion (oriented shells) ----
    // Per-(global) vertex sticky sign. Length = total vertex count. Filled
    // lazily on the first do_compute_energy from per-SC `rcc_sticky_sign`
    // attributes. Verts whose SC has no attribute stay at 0 (= double-sided,
    // identical to v2 behaviour).
    muda::DeviceBuffer<IndexT>   m_sticky_sign;

    // Triangle topology + v→tri CSR adjacency, used to compute shell vertex
    // normals from begin-of-step positions. Both filled at the same time as
    // m_sticky_sign.
    //
    //   m_shell_triangles  : Vector3i triangles (global vertex indices),
    //                        one per surface triangle of every SC that
    //                        opted into v3.
    //   m_v2t_offsets      : CSR row pointers; m_v2t_offsets.size() ==
    //                        n_total_verts + 1.
    //   m_v2t_tri_indices  : CSR column entries; index into
    //                        m_shell_triangles.
    muda::DeviceBuffer<Vector3i> m_shell_triangles;
    muda::DeviceBuffer<IndexT>   m_v2t_offsets;
    muda::DeviceBuffer<IndexT>   m_v2t_tri_indices;

    // Per-(global) vertex lagged shell normal n̂_P. Recomputed once per step
    // from `m_pos_at_step_begin` (i.e. begin-of-frame positions) and held
    // constant through that frame's Newton iterations — same convention as
    // friction's lagged tangent basis.
    muda::DeviceBuffer<Vector3> m_vertex_normal;

    // Set true once the v3 topology buffers are populated (lazy: needs the
    // global_vertex_offset attributes which are only filled after vertex
    // reporters run).
    bool m_v3_built = false;

    // Set true iff at least one vertex in the scene has a non-zero
    // rcc_sticky_sign. Lets the per-pair gate cheaply short-circuit when v3
    // is unused.
    bool m_has_sticky = false;

    virtual void do_build(BuildInfo& info) override
    {
        require<IPCPipelineFlag>();

        // The frontend has only registered adhesion attributes if the user called
        // RCCAdhesive::apply_to(tabular). If "Cn" is absent, throw to unregister.
        auto contact_models = world().scene().contact_tabular().contact_models();
        auto attr_Cn        = contact_models.find<Float>("Cn");
        if(!attr_Cn)
            throw SimSystemException("RCC adhesion is not configured "
                                     "(call RCCAdhesive::apply_to to enable).");

        on_init_scene([this] { _rebuild_adhesive_tabular(); });
        on_rebuild_scene([this] { _rebuild_adhesive_tabular(); });
    }

    void _rebuild_adhesive_tabular()
    {
        auto contact_models = world().scene().contact_tabular().contact_models();

        auto attr_topo = contact_models.find<Vector2i>("topo");
        auto attr_Cn   = contact_models.find<Float>("Cn");
        auto attr_Ct   = contact_models.find<Float>("Ct");
        auto attr_W    = contact_models.find<Float>("W");
        auto attr_eta  = contact_models.find<Float>("eta");
        auto attr_br   = contact_models.find<Float>("bonding_rate");
        auto attr_p0   = contact_models.find<Float>("p0");
        auto attr_ib   = contact_models.find<Float>("initial_beta");
        auto attr_en   = contact_models.find<IndexT>("adhesion_enabled");

        UIPC_ASSERT(attr_topo && attr_Cn && attr_Ct && attr_W && attr_eta && attr_br
                        && attr_p0 && attr_ib && attr_en,
                    "RCCAdhesive attributes are not fully present on the ContactTabular. "
                    "Did the frontend call RCCAdhesive::apply_to(tabular)?");

        auto topo_view = attr_topo->view();
        auto Cn_view   = attr_Cn->view();
        auto Ct_view   = attr_Ct->view();
        auto W_view    = attr_W->view();
        auto eta_view  = attr_eta->view();
        auto br_view   = attr_br->view();
        auto p0_view   = attr_p0->view();
        auto ib_view   = attr_ib->view();
        auto en_view   = attr_en->view();

        m_N = static_cast<IndexT>(world().scene().contact_tabular().element_count());

        RCCAdhesiveCoeff default_coeff;
        default_coeff.Cn           = Cn_view[0];
        default_coeff.Ct           = Ct_view[0];
        default_coeff.W            = W_view[0];
        default_coeff.eta          = eta_view[0];
        default_coeff.bonding_rate = br_view[0];
        default_coeff.p0           = p0_view[0];
        default_coeff.initial_beta = ib_view[0];
        default_coeff.enabled      = en_view[0];

        std::vector<RCCAdhesiveCoeff> host(m_N * m_N, default_coeff);

        for(SizeT row = 0; row < topo_view.size(); ++row)
        {
            const auto&      ids = topo_view[row];
            RCCAdhesiveCoeff c;
            c.Cn           = Cn_view[row];
            c.Ct           = Ct_view[row];
            c.W            = W_view[row];
            c.eta          = eta_view[row];
            c.bonding_rate = br_view[row];
            c.p0           = p0_view[row];
            c.initial_beta = ib_view[row];
            c.enabled      = en_view[row];

            host[ids.x() * m_N + ids.y()] = c;
            host[ids.y() * m_N + ids.x()] = c;
        }

        m_adhesive_tabular.resize(muda::Extent2D{static_cast<size_t>(m_N),
                                                 static_cast<size_t>(m_N)});
        m_adhesive_tabular.view().copy_from(host.data());
    }

    // ---- v3 helpers: oriented-adhesion topology + vertex normals ----

    // Build m_sticky_sign + m_v2t_* + m_shell_triangles from scene SCs that
    // carry an `rcc_sticky_sign` per-vertex attribute. Skipped if the user
    // never called RCCAdhesive::set_sticky_side — in that case m_sticky_sign
    // is sized to total vertex count and zero-filled, so the gate always
    // returns true (= v2 double-sided behaviour).
    void _build_sticky_topology(SizeT n_total_verts)
    {
        std::vector<IndexT>   h_sticky(n_total_verts, 0);
        std::vector<Vector3i> h_tris;
        std::vector<std::vector<IndexT>> h_v2t(n_total_verts);

        m_has_sticky = false;

        auto geo_slots = world().scene().geometries();
        for(auto& geo_slot : geo_slots)
        {
            auto* sc = geo_slot->geometry().as<geometry::SimplicialComplex>();
            if(!sc)
                continue;

            auto attr_sign = sc->vertices().find<IndexT>("rcc_sticky_sign");
            if(!attr_sign)
                continue;  // SC opted out of v3 → its verts stay at 0.

            auto gvo = sc->meta().find<IndexT>(builtin::global_vertex_offset);
            UIPC_ASSERT(gvo,
                        "Geometry has rcc_sticky_sign attribute but no "
                        "global_vertex_offset. The vertex layout may not be "
                        "built yet at the time of this call.");
            IndexT offset = gvo->view()[0];

            auto sign_view = attr_sign->view();
            for(SizeT v = 0; v < sign_view.size(); ++v)
            {
                IndexT s         = sign_view[v];
                h_sticky[offset + v] = s;
                if(s != 0)
                    m_has_sticky = true;
            }

            // Collect every triangle of the SC (works for codim shells and
            // closed bodies alike — for a closed body the average of incident
            // face normals at a corner is the inward/outward bisector, also
            // geometrically sensible).
            auto tri_view = sc->triangles().topo().view();
            for(SizeT t = 0; t < tri_view.size(); ++t)
            {
                const Vector3i& local = tri_view[t];
                Vector3i        global_tri{local[0] + offset,
                                           local[1] + offset,
                                           local[2] + offset};
                IndexT tri_idx = IndexT(h_tris.size());
                h_tris.push_back(global_tri);
                h_v2t[global_tri[0]].push_back(tri_idx);
                h_v2t[global_tri[1]].push_back(tri_idx);
                h_v2t[global_tri[2]].push_back(tri_idx);
            }
        }

        // Pack v→tri into CSR.
        std::vector<IndexT> h_offsets(n_total_verts + 1, 0);
        for(SizeT v = 0; v < n_total_verts; ++v)
            h_offsets[v + 1] = h_offsets[v] + IndexT(h_v2t[v].size());
        std::vector<IndexT> h_csr(h_offsets.back());
        for(SizeT v = 0; v < n_total_verts; ++v)
            std::copy(h_v2t[v].begin(),
                      h_v2t[v].end(),
                      h_csr.begin() + h_offsets[v]);

        // Upload (resize-then-copy; muda's BufferView::copy_from needs a
        // non-empty source for some backends, so guard with size checks).
        m_sticky_sign.resize(n_total_verts);
        if(n_total_verts > 0)
            m_sticky_sign.view().copy_from(h_sticky.data());

        m_v2t_offsets.resize(n_total_verts + 1);
        if(n_total_verts > 0)
            m_v2t_offsets.view().copy_from(h_offsets.data());

        m_v2t_tri_indices.resize(h_csr.size());
        if(!h_csr.empty())
            m_v2t_tri_indices.view().copy_from(h_csr.data());

        m_shell_triangles.resize(h_tris.size());
        if(!h_tris.empty())
            m_shell_triangles.view().copy_from(h_tris.data());

        m_vertex_normal.resize(n_total_verts);
        if(n_total_verts > 0)
            m_vertex_normal.fill(Vector3::Zero());
    }

    // Area-weighted vertex normal recompute. Iterates the v→tri CSR; each
    // thread sums `(B-A)×(C-A)` (unnormalized → area-weighted) across its
    // incident triangles, then normalizes. Verts with no incident triangle
    // (non-shell verts) get a zero normal — they're never read because their
    // sticky_sign is 0 and the gate short-circuits.
    void _recompute_vertex_normals(muda::CBufferView<Vector3> positions)
    {
        if(!m_has_sticky)
            return;  // gate is short-circuited by sticky_sign==0; normals unused.

        using namespace muda;
        auto n_verts = positions.size();
        if(n_verts == 0)
            return;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n_verts,
                   [Ps      = positions.viewer().name("Ps"),
                    offsets = m_v2t_offsets.cviewer().name("v2t_offsets"),
                    v2t     = m_v2t_tri_indices.cviewer().name("v2t_tris"),
                    tris    = m_shell_triangles.cviewer().name("shell_tris"),
                    normals = m_vertex_normal.view().viewer().name("normals")] __device__(int v) mutable
                   {
                       IndexT  begin = offsets(v);
                       IndexT  end   = offsets(v + 1);
                       Vector3 sum   = Vector3::Zero();
                       for(IndexT k = begin; k < end; ++k)
                       {
                           const Vector3i& t = tris(v2t(k));
                           Vector3 A = Ps(t[0]);
                           Vector3 B = Ps(t[1]);
                           Vector3 C = Ps(t[2]);
                           sum += (B - A).cross(C - A);
                       }
                       Float   nrm = sum.norm();
                       Vector3 out = Vector3::Zero();
                       if(nrm > Float{0})
                           out = sum / nrm;
                       normals(v) = out;
                   });
    }

    // ---- v2 helper kernels (PT only) ----

    // Compute sorted-vertex u64 keys for the current VT primitive list.
    // The key is a pure function of the 4 topology vertices (PT_pair_key), so
    // it is identical for any VT primitive regardless of its closest feature —
    // beta therefore persists across PT<->PE<->PP transitions.
    void _compute_curr_keys_PT(muda::DeviceBuffer<U64>&    dst,
                               muda::CBufferView<ActiveVT> pairs) noexcept
    {
        using namespace muda;
        auto n = pairs.size();
        dst.resize(n);
        if(n == 0)
            return;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [pairs = pairs.viewer().name("VTs"),
                    dst   = dst.view().viewer().name("curr_keys")] __device__(int i) mutable
                   {
                       using namespace sym::codim_ipc_rcc_adhesive;
                       const auto& P = pairs(i).topo;
                       dst(i)        = PT_pair_key(P[0], P[1], P[2], P[3]);
                   });
    }

    // Phase B (initial step) — every pair is treated as new.
    // β = max(new-pair bonding kick, coeff.initial_beta).
    void _phase_b_init_all_new(muda::CBufferView<ActiveVT>      pairs,
                               muda::CBufferView<IndexT>        contact_ids,
                               muda::CBuffer2DView<ContactCoeff> barrier_table,
                               muda::CBufferView<Float>          d_hats,
                               muda::CBufferView<Vector3>        positions,
                               Float                             dt) noexcept
    {
        using namespace muda;
        auto n = pairs.size();
        m_beta_PT.resize(n);
        m_blocked_PT.resize(n);
        if(n == 0)
            return;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [pairs       = pairs.viewer().name("PTs"),
                    contact_ids = contact_ids.viewer().name("contact_ids"),
                    rcc_table   = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                    bar_table   = barrier_table.viewer().name("barrier_tabular"),
                    d_hats      = d_hats.viewer().name("d_hats"),
                    Ps          = positions.viewer().name("Ps"),
                    beta_dst    = m_beta_PT.view().viewer().name("beta_PT"),
                    blocked_dst = m_blocked_PT.view().viewer().name("blocked_PT"),
                    sticky_sign = m_sticky_sign.cviewer().name("sticky_sign"),
                    vert_normal = m_vertex_normal.cviewer().name("vert_normal"),
                    shell_tris  = m_shell_triangles.cviewer().name("shell_tris"),
                    v2t_offsets = m_v2t_offsets.cviewer().name("v2t_offsets"),
                    v2t_tris    = m_v2t_tri_indices.cviewer().name("v2t_tris"),
                    n_tris      = (IndexT)m_shell_triangles.size(),
                    dt] __device__(int i) mutable
                   {
                       using namespace sym::codim_ipc_rcc_adhesive;
                       using namespace sym::codim_ipc_contact;
                       blocked_dst(i)      = 0;
                       const auto&     vt   = pairs(i);
                       const Vector4i& PT   = vt.topo;
                       Vector4i        cids = {contact_ids(PT[0]), contact_ids(PT[1]),
                                               contact_ids(PT[2]), contact_ids(PT[3])};
                       auto            rcc  = PT_rcc_coeff(rcc_table, cids);
                       if(!rcc.enabled)
                       {
                           beta_dst(i) = 0;
                           return;
                       }
                       Float kappa = PT_contact_coeff(bar_table, cids).kappa;
                       Float d_hat = PT_d_hat(d_hats(PT[0]), d_hats(PT[1]),
                                              d_hats(PT[2]), d_hats(PT[3]));
                       Vector3 P  = Ps(PT[0]);
                       Vector3 T0 = Ps(PT[1]);
                       Vector3 T1 = Ps(PT[2]);
                       Vector3 T2 = Ps(PT[3]);

                       // v3 gate: a new pair on the non-sticky side starts at
                       // β = 0 (no initial bonding kick, no initial_beta
                       // override). Without this, initial_beta=1 would
                       // force-bond a wrong-side new pair.
                       if(!PT_sticky_gate(sticky_sign(PT[0]),
                                          sticky_sign(PT[1]),
                                          vert_normal(PT[0]),
                                          vert_normal(PT[1]),
                                          P, T0, T1, T2))
                       {
                           beta_dst(i) = 0;
                           return;
                       }

                       // Cross-layer occlusion gate: cast a segment from
                       // centroid(T) toward P at frame-open positions; if
                       // any other shell triangle blocks it, this pair is
                       // separated by an intervening layer and must not bond.
                       // Normal-side test: only fire when T faces P
                       // (otherwise IPC barrier already keeps them apart and
                       // there's no adhesion semantics on the wrong side).
                       bool blocked = false;
                       {
                           Vector3 N    = (T1 - T0).cross(T2 - T0);
                           Vector3 Cen  = (T0 + T1 + T2) * Float{1.0 / 3.0};
                           Vector3 sdir = P - Cen;
                           if(N.dot(sdir) > Float{0})
                           {
                               constexpr Float TMIN = Float{1e-5};
                               constexpr Float TMAX = Float{1} - Float{1e-5};
                               for(IndexT j = 0; j < n_tris; ++j)
                               {
                                   const Vector3i& tri = shell_tris(j);
                                   if(tri[0] == PT[0] || tri[1] == PT[0] || tri[2] == PT[0]) continue;
                                   if(tri[0] == PT[1] || tri[1] == PT[1] || tri[2] == PT[1]) continue;
                                   if(tri[0] == PT[2] || tri[1] == PT[2] || tri[2] == PT[2]) continue;
                                   if(tri[0] == PT[3] || tri[1] == PT[3] || tri[2] == PT[3]) continue;
                                   Vector3 A = Ps(tri[0]);
                                   Vector3 B = Ps(tri[1]);
                                   Vector3 Cv = Ps(tri[2]);
                                   if(segment_triangle_hit(Cen, sdir, A, B, Cv, TMIN, TMAX))
                                   { blocked = true; break; }
                               }
                           }
                       }
                       blocked_dst(i) = blocked ? IndexT{1} : IndexT{0};
                       if(blocked)
                       {
                           beta_dst(i) = 0;
                           return;
                       }

                       Float   D;
                       distance::point_triangle_distance2(vt.flag, P, T0, T1, T2, D);
                       beta_dst(i) = PT_beta_init_new(rcc.initial_beta, kappa, rcc.Cn,
                                                     rcc.bonding_rate, d_hat, dt, D);
                   });
    }

    // Phase B (subsequent step) — match curr keys against m_prev_keys_PT.
    // If found: carry prev β. If not found: new-pair bonding kick.
    //
    // The cross-layer occlusion test is re-evaluated for EVERY pair on EVERY
    // frame (not cached via m_prev_blocked_PT). That way:
    //  • Pairs that newly become occluded (geometry shifted) immediately
    //    flip β=0 in the same frame, not 1 frame later.
    //  • Pairs that newly become unoccluded (an intervening layer slid
    //    away) immediately become eligible for bonding.
    //  • Asset-load (set_prev_pt_state, which lacks blocked info) is
    //    self-correcting: the first frame after load re-evaluates.
    void _phase_b_match_or_init(muda::CBufferView<U64>            curr_keys,
                                muda::CBufferView<ActiveVT>       pairs,
                                muda::CBufferView<IndexT>         contact_ids,
                                muda::CBuffer2DView<ContactCoeff> barrier_table,
                                muda::CBufferView<Float>          d_hats,
                                muda::CBufferView<Vector3>        positions,
                                Float                             dt) noexcept
    {
        using namespace muda;
        auto n = curr_keys.size();
        m_beta_PT.resize(n);
        m_blocked_PT.resize(n);
        if(n == 0)
            return;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [curr_keys  = curr_keys.viewer().name("curr_keys"),
                    prev_keys  = m_prev_keys_PT.cviewer().name("prev_keys"),
                    prev_beta  = m_prev_beta_PT.cviewer().name("prev_beta"),
                    n_prev     = (IndexT)m_prev_keys_PT.size(),
                    pairs      = pairs.viewer().name("PTs"),
                    contact_ids= contact_ids.viewer().name("contact_ids"),
                    rcc_table  = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                    bar_table  = barrier_table.viewer().name("barrier_tabular"),
                    d_hats     = d_hats.viewer().name("d_hats"),
                    Ps         = positions.viewer().name("Ps"),
                    beta_dst   = m_beta_PT.view().viewer().name("beta_PT"),
                    blocked_dst= m_blocked_PT.view().viewer().name("blocked_PT"),
                    sticky_sign= m_sticky_sign.cviewer().name("sticky_sign"),
                    vert_normal= m_vertex_normal.cviewer().name("vert_normal"),
                    shell_tris = m_shell_triangles.cviewer().name("shell_tris"),
                    v2t_offsets= m_v2t_offsets.cviewer().name("v2t_offsets"),
                    v2t_tris   = m_v2t_tri_indices.cviewer().name("v2t_tris"),
                    n_tris     = (IndexT)m_shell_triangles.size(),
                    dt] __device__(int i) mutable
                   {
                       using namespace sym::codim_ipc_rcc_adhesive;
                       using namespace sym::codim_ipc_contact;
                       blocked_dst(i) = 0;
                       U64 key = curr_keys(i);
                       // Binary-search in prev_keys (already sorted ascending).
                       IndexT lo = 0, hi = n_prev;
                       while(lo < hi)
                       {
                           IndexT mid = (lo + hi) >> 1;
                           if(prev_keys(mid) < key) lo = mid + 1;
                           else                     hi = mid;
                       }
                       bool found = (lo < n_prev) && (prev_keys(lo) == key);

                       const auto&     vt   = pairs(i);
                       const Vector4i& PT   = vt.topo;
                       Vector4i        cids = {contact_ids(PT[0]), contact_ids(PT[1]),
                                               contact_ids(PT[2]), contact_ids(PT[3])};
                       auto            rcc  = PT_rcc_coeff(rcc_table, cids);
                       if(!rcc.enabled)
                       {
                           beta_dst(i) = 0;
                           return;
                       }

                       Vector3 P  = Ps(PT[0]);
                       Vector3 T0 = Ps(PT[1]);
                       Vector3 T1 = Ps(PT[2]);
                       Vector3 T2 = Ps(PT[3]);

                       // v3 sticky-side gate: applies to ALL pairs (found or
                       // new) — non-sticky-side pairs never bond.
                       if(!PT_sticky_gate(sticky_sign(PT[0]),
                                          sticky_sign(PT[1]),
                                          vert_normal(PT[0]),
                                          vert_normal(PT[1]),
                                          P, T0, T1, T2))
                       {
                           beta_dst(i) = 0;
                           return;
                       }

                       // Cross-layer occlusion gate (same logic as
                       // _phase_b_init_all_new). Runs for found pairs too so
                       // geometry changes flip the blocked state immediately.
                       bool blocked = false;
                       {
                           Vector3 N    = (T1 - T0).cross(T2 - T0);
                           Vector3 Cen  = (T0 + T1 + T2) * Float{1.0 / 3.0};
                           Vector3 sdir = P - Cen;
                           if(N.dot(sdir) > Float{0})
                           {
                               constexpr Float TMIN = Float{1e-5};
                               constexpr Float TMAX = Float{1} - Float{1e-5};
                               for(IndexT j = 0; j < n_tris; ++j)
                               {
                                   const Vector3i& tri = shell_tris(j);
                                   if(tri[0] == PT[0] || tri[1] == PT[0] || tri[2] == PT[0]) continue;
                                   if(tri[0] == PT[1] || tri[1] == PT[1] || tri[2] == PT[1]) continue;
                                   if(tri[0] == PT[2] || tri[1] == PT[2] || tri[2] == PT[2]) continue;
                                   if(tri[0] == PT[3] || tri[1] == PT[3] || tri[2] == PT[3]) continue;
                                   Vector3 A = Ps(tri[0]);
                                   Vector3 B = Ps(tri[1]);
                                   Vector3 Cv = Ps(tri[2]);
                                   if(segment_triangle_hit(Cen, sdir, A, B, Cv, TMIN, TMAX))
                                   { blocked = true; break; }
                               }
                           }
                       }
                       blocked_dst(i) = blocked ? IndexT{1} : IndexT{0};
                       if(blocked)
                       {
                           beta_dst(i) = 0;
                           return;
                       }

                       if(found)
                       {
                           beta_dst(i) = prev_beta(lo);
                       }
                       else
                       {
                           // new pair → bonding kick
                           Float kappa = PT_contact_coeff(bar_table, cids).kappa;
                           Float d_hat = PT_d_hat(d_hats(PT[0]), d_hats(PT[1]),
                                                  d_hats(PT[2]), d_hats(PT[3]));
                           Float   D;
                           distance::point_triangle_distance2(vt.flag, P, T0, T1, T2, D);
                           beta_dst(i) = PT_beta_init_new(rcc.initial_beta, kappa, rcc.Cn,
                                                         rcc.bonding_rate, d_hat, dt, D);
                       }
                   });
    }

    // (Per-feature EE/PE/PP β buffers and their zero-fill were retired in
    // Phase 6: adhesion now runs over a single per-VT-primitive β.)

    // v3: lazy-init the oriented-adhesion topology + lagged vertex normals
    // on the first compute call. Needs total vertex count + global_vertex_offset
    // attributes, both of which only exist after vertex reporters run — too
    // late for on_init_scene.
    template <typename Info>
    void _ensure_v3_state(Info& info)
    {
        auto positions = info.positions();
        if(!m_v3_built)
        {
            _build_sticky_topology(positions.size());
            m_v3_built = true;
        }
        // Initialize m_pos_at_step_begin from begin-of-frame positions on
        // every first-frame call. Previously this was gated by m_has_sticky;
        // dropped because the cross-layer occlusion gate (Phase B) needs
        // begin-of-frame positions even when single-sided adhesion v3 is
        // unused. _recompute_vertex_normals is internally guarded by
        // m_has_sticky so no extra work is done for non-v3 scenes.
        if(m_pos_at_step_begin.size() != positions.size())
        {
            m_pos_at_step_begin.resize(positions.size());
            m_pos_at_step_begin.view().copy_from(positions);
            _recompute_vertex_normals(m_pos_at_step_begin.view());
        }
    }

    // Phase B entry: run once per frame at the top of do_compute_energy
    // (and idempotent if called from do_assemble too).
    void _phase_b_if_new_frame(EnergyInfo& info)
    {
        _ensure_v3_state(info);

        SizeT cur = engine().frame();
        if(cur == m_last_seen_frame)
            return;
        m_last_seen_frame = cur;

        // (Phase 6: a single per-VT-primitive beta now covers all features.)

        auto pairs       = info.friction_VTs();
        auto contact_ids = info.contact_element_ids();
        auto barrier_tab = info.contact_tabular();
        auto d_hats      = info.d_hats();
        auto positions   = info.positions();
        auto dt          = info.dt();

        // Diagnostic: empty friction-PT list at the start of any frame
        // means adhesion is silently OFF for that whole frame. Most
        // common cause: frame 1 right after world.init() — the
        // SimplexTrajectoryFilter's friction candidates are recorded
        // from the *previous* step's DCD output, and at frame 1 there
        // is no previous step. Visible at INFO log level.
        logger::debug("RCC Phase B (frame={}): friction_PT={} pairs, prev_keys={} (loaded={})",
                     cur, pairs.size(), m_prev_keys_PT.size(),
                     m_has_loaded_prev_state ? "yes" : "no");
        if(pairs.size() == 0)
        {
            m_beta_PT.resize(0);
            return;
        }

        muda::DeviceBuffer<U64> curr_keys;
        _compute_curr_keys_PT(curr_keys, pairs);

        // Branch on whether we already have prev-state to match against.
        // m_prev_keys_PT is filled either by Phase A (end of previous step)
        // or by set_prev_pt_state() (asset load).
        if(m_prev_keys_PT.size() == 0)
        {
            _phase_b_init_all_new(pairs, contact_ids, barrier_tab, d_hats, positions, dt);
        }
        else
        {
            _phase_b_match_or_init(curr_keys.view(), pairs, contact_ids,
                                   barrier_tab, d_hats, positions, dt);
        }
        m_first_step = false;
        m_has_loaded_prev_state = false;  // one-shot, consumed by this frame
    }

    // Same Phase B trigger but takes a ContactInfo (for do_assemble path).
    void _phase_b_if_new_frame(ContactInfo& info)
    {
        _ensure_v3_state(info);

        SizeT cur = engine().frame();
        if(cur == m_last_seen_frame)
            return;
        m_last_seen_frame = cur;

        // (Phase 6: a single per-VT-primitive beta now covers all features.)

        auto pairs       = info.friction_VTs();
        auto contact_ids = info.contact_element_ids();
        auto barrier_tab = info.contact_tabular();
        auto d_hats      = info.d_hats();
        auto positions   = info.positions();
        auto dt          = info.dt();

        // Same diagnostic as the EnergyInfo overload. Both should
        // fire per frame; if only one logs, the other branch is
        // taking the early-return path.
        logger::debug("RCC Phase B [ContactInfo] (frame={}): friction_PT={} pairs, "
                     "prev_keys={} (loaded={})",
                     cur, pairs.size(), m_prev_keys_PT.size(),
                     m_has_loaded_prev_state ? "yes" : "no");
        if(pairs.size() == 0)
        {
            m_beta_PT.resize(0);
            return;
        }

        muda::DeviceBuffer<U64> curr_keys;
        _compute_curr_keys_PT(curr_keys, pairs);

        // Branch on whether we already have prev-state to match against.
        // m_prev_keys_PT is filled either by Phase A (end of previous step)
        // or by set_prev_pt_state() (asset load).
        if(m_prev_keys_PT.size() == 0)
        {
            _phase_b_init_all_new(pairs, contact_ids, barrier_tab, d_hats, positions, dt);
        }
        else
        {
            _phase_b_match_or_init(curr_keys.view(), pairs, contact_ids,
                                   barrier_tab, d_hats, positions, dt);
        }
        m_first_step = false;
        m_has_loaded_prev_state = false;  // one-shot, consumed by this frame
    }

    // Phase 6: all VT primitives assemble as 12-DOF (4-vertex) blocks, so the
    // reporter puts the whole VT list in the PT slot and zeroes EE/PE/PP.
    void friction_pair_counts(SizeT& pt, SizeT& ee, SizeT& pe, SizeT& pp) const override
    {
        pt = m_stf_for_phase_a ? m_stf_for_phase_a->friction_VTs().size() : 0;
        ee = 0;
        pe = 0;
        pp = 0;
    }

    virtual void do_compute_energy(EnergyInfo& info) override
    {
        using namespace muda;

        // Once per frame: Phase B match-or-init over the VT primitive list.
        _phase_b_if_new_frame(info);

        auto vt_count = info.friction_VTs().size();

        if(vt_count > 0)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(vt_count,
                       [table = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_ids"),
                        VTs         = info.friction_VTs().viewer().name("VTs"),
                        Es          = info.friction_PT_energies().viewer().name("Es"),
                        Ps          = info.positions().viewer().name("Ps"),
                        prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                        d_hats      = info.d_hats().viewer().name("d_hats"),
                        beta_buf    = m_beta_PT.cviewer().name("beta_VT"),
                        sticky_sign = m_sticky_sign.cviewer().name("sticky_sign"),
                        vert_normal = m_vertex_normal.cviewer().name("vert_normal"),
                        dt          = info.dt()] __device__(int i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           const auto&     vt   = VTs(i);
                           const Vector4i& PT   = vt.topo;
                           const Vector4i& flag = vt.flag;
                           Float           beta = beta_buf(i);
                           if(beta <= 0)
                           {
                               Es(i) = 0;
                               return;
                           }
                           Vector4i cids  = {contact_ids(PT[0]), contact_ids(PT[1]),
                                             contact_ids(PT[2]), contact_ids(PT[3])};
                           auto     coeff = PT_rcc_coeff(table, cids);
                           if(!coeff.enabled)
                           {
                               Es(i) = 0;
                               return;
                           }
                           Float d_hat = PT_d_hat(d_hats(PT[0]), d_hats(PT[1]),
                                                  d_hats(PT[2]), d_hats(PT[3]));

                           const auto& P  = Ps(PT[0]);
                           const auto& T0 = Ps(PT[1]);
                           const auto& T1 = Ps(PT[2]);
                           const auto& T2 = Ps(PT[3]);

                           // v3 single-sided gate (full triangle, feature-
                           // independent): adhesion fires iff P's or T's
                           // sticky face is the contact side.
                           if(!PT_sticky_gate(sticky_sign(PT[0]),
                                              sticky_sign(PT[1]),
                                              vert_normal(PT[0]),
                                              vert_normal(PT[1]),
                                              P, T0, T1, T2))
                           {
                               Es(i) = 0;
                               return;
                           }

                           const auto& pP  = prev_Ps(PT[0]);
                           const auto& pT0 = prev_Ps(PT[1]);
                           const auto& pT1 = prev_Ps(PT[2]);
                           const auto& pT2 = prev_Ps(PT[3]);

                           Float En = VT_normal_adhesion_energy(
                               coeff.Cn, beta, d_hat, dt, flag, P, T0, T1, T2);
                           Float Et = VT_tangential_adhesion_energy(
                               coeff.Ct, beta, d_hat, dt, flag,
                               pP, pT0, pT1, pT2, P, T0, T1, T2);
                           Es(i) = En + Et;
                       });
        }

    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace muda;

        _phase_b_if_new_frame(info);

        auto vt_count = (IndexT)info.friction_VTs().size();

        if(vt_count > 0)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(vt_count,
                       [gradient_only = info.gradient_only(),
                        table       = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_ids"),
                        Ps          = info.positions().viewer().name("Ps"),
                        prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                        d_hats      = info.d_hats().viewer().name("d_hats"),
                        dt          = info.dt(),
                        VTs         = info.friction_VTs().viewer().name("VTs"),
                        PT_Gs       = info.friction_PT_gradients().viewer().name("PT_Gs"),
                        PT_Hs       = info.friction_PT_hessians().viewer().name("PT_Hs"),
                        beta_buf    = m_beta_PT.cviewer().name("beta_VT"),
                        sticky_sign = m_sticky_sign.cviewer().name("sticky_sign"),
                        vert_normal = m_vertex_normal.cviewer().name("vert_normal")] __device__(IndexT i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           const auto&     vt   = VTs(i);
                           const Vector4i& PT   = vt.topo;
                           const Vector4i& flag = vt.flag;
                           Float           beta = beta_buf(i);

                           Vector12    G = Vector12::Zero();
                           Matrix12x12 H = Matrix12x12::Zero();

                           if(beta > 0)
                           {
                               Vector4i cids = {contact_ids(PT[0]), contact_ids(PT[1]),
                                                contact_ids(PT[2]), contact_ids(PT[3])};
                               auto     coeff = PT_rcc_coeff(table, cids);
                               if(coeff.enabled)
                               {
                                   Float d_hat = PT_d_hat(d_hats(PT[0]), d_hats(PT[1]),
                                                          d_hats(PT[2]), d_hats(PT[3]));
                                   const auto& P  = Ps(PT[0]);
                                   const auto& T0 = Ps(PT[1]);
                                   const auto& T1 = Ps(PT[2]);
                                   const auto& T2 = Ps(PT[3]);

                                   // v3 single-sided gate (P-side OR T-side),
                                   // feature-independent (full triangle).
                                   bool gate_ok = PT_sticky_gate(
                                       sticky_sign(PT[0]), sticky_sign(PT[1]),
                                       vert_normal(PT[0]), vert_normal(PT[1]),
                                       P, T0, T1, T2);

                                   if(gate_ok)
                                   {
                                       const auto& pP  = prev_Ps(PT[0]);
                                       const auto& pT0 = prev_Ps(PT[1]);
                                       const auto& pT1 = prev_Ps(PT[2]);
                                       const auto& pT2 = prev_Ps(PT[3]);

                                       Vector12    Gn = Vector12::Zero(), Gt = Vector12::Zero();
                                       Matrix12x12 Hn = Matrix12x12::Zero(), Ht = Matrix12x12::Zero();
                                       if(gradient_only)
                                       {
                                           VT_normal_adhesion_gradient(
                                               Gn, coeff.Cn, beta, d_hat, dt, flag, P, T0, T1, T2);
                                           if(coeff.Ct > 0)
                                               VT_tangential_adhesion_gradient(
                                                   Gt, coeff.Ct, beta, d_hat, dt, flag,
                                                   pP, pT0, pT1, pT2, P, T0, T1, T2);
                                           G = Gn + Gt;
                                       }
                                       else
                                       {
                                           VT_normal_adhesion_gradient_hessian(
                                               Gn, Hn, coeff.Cn, beta, d_hat, dt, flag, P, T0, T1, T2);
                                           if(coeff.Ct > 0)
                                               VT_tangential_adhesion_gradient_hessian(
                                                   Gt, Ht, coeff.Ct, beta, d_hat, dt, flag,
                                                   pP, pT0, pT1, pT2, P, T0, T1, T2);
                                           G = Gn + Gt;
                                           cuda::make_spd(Hn);  // normal block; tangential J^T J is PSD
                                           H = Hn + Ht;
                                       }
                                   }
                               }
                           }

                           DoubletVectorAssembler DVA{PT_Gs};
                           DVA.segment<4>(i * 4).write(PT, G);
                           if(!gradient_only)
                           {
                               TripletMatrixAssembler TMA{PT_Hs};
                               TMA.half_block<4>(i * PTHalfHessianSize).write(PT, H);
                           }
                       });
        }

    }

    // ====================================================================
    // Phase A — β evolution at end of step.
    // Called by RCCBetaEvolutionTimeIntegrator via do_update_state.
    // ====================================================================

    // Slots populated in do_build for use outside the do_compute_energy /
    // do_assemble lifecycle (i.e. from Phase A which runs at end-of-step).
    SimSystemSlot<GlobalContactManager>      m_gcm_for_phase_a;
    SimSystemSlot<GlobalVertexManager>       m_gvm_for_phase_a;
    SimSystemSlot<SimplexTrajectoryFilter>   m_stf_for_phase_a;
    SimSystemSlot<RCCBondedPTSystem>         m_bonded_pt_system_for_phase_a;
    RCCBondedPTBetaCarryScratch              m_bonded_pt_beta_carry;
    Float                                    m_bonded_pt_beta_lock_threshold = 1.0;

    void _evolve_beta_step_at_end(Float dt)
    {
        using namespace muda;
        auto pairs       = m_stf_for_phase_a->friction_VTs();
        auto n           = pairs.size();
        auto positions   = m_gvm_for_phase_a->positions();

        // Make sure m_pos_at_step_begin is sized to the vertex count (first time only).
        if(m_pos_at_step_begin.size() != positions.size())
        {
            m_pos_at_step_begin.resize(positions.size());
            // First step: no displacement signal yet. Just snapshot and skip evolve.
            m_pos_at_step_begin.view().copy_from(positions);
            return;
        }

        if(n > 0)
        {
            auto contact_ids = m_gvm_for_phase_a->contact_element_ids();
            auto barrier_tab = m_gcm_for_phase_a->contact_tabular();
            auto d_hats      = m_gvm_for_phase_a->d_hats();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(n,
                       [pairs       = pairs.viewer().name("VTs"),
                        contact_ids = contact_ids.viewer().name("contact_ids"),
                        rcc_table   = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        bar_table   = barrier_tab.viewer().name("barrier_tabular"),
                        d_hats      = d_hats.viewer().name("d_hats"),
                        Ps          = positions.viewer().name("Ps"),
                        Ps_begin    = m_pos_at_step_begin.cviewer().name("Ps_begin"),
                        beta_buf    = m_beta_PT.view().viewer().name("beta_VT"),
                        blocked_buf = m_blocked_PT.cviewer().name("blocked_VT"),
                        n_blocked   = (IndexT)m_blocked_PT.size(),
                        sticky_sign = m_sticky_sign.cviewer().name("sticky_sign"),
                        vert_normal = m_vertex_normal.cviewer().name("vert_normal"),
                        dt] __device__(int i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           using namespace sym::codim_ipc_contact;
                           using namespace distance;
                           using namespace friction;

                           const auto&     vt   = pairs(i);
                           const Vector4i& PT   = vt.topo;
                           const Vector4i& flag = vt.flag;
                           Vector4i cids = {contact_ids(PT[0]), contact_ids(PT[1]),
                                            contact_ids(PT[2]), contact_ids(PT[3])};
                           auto rcc = PT_rcc_coeff(rcc_table, cids);
                           if(!rcc.enabled) return;

                           Float kappa = PT_contact_coeff(bar_table, cids).kappa;
                           Float d_hat = PT_d_hat(d_hats(PT[0]), d_hats(PT[1]),
                                                  d_hats(PT[2]), d_hats(PT[3]));

                           Vector3 P  = Ps(PT[0]);
                           Vector3 T0 = Ps(PT[1]);
                           Vector3 T1 = Ps(PT[2]);
                           Vector3 T2 = Ps(PT[3]);
                           Vector3 P0  = Ps_begin(PT[0]);
                           Vector3 T00 = Ps_begin(PT[1]);
                           Vector3 T10 = Ps_begin(PT[2]);
                           Vector3 T20 = Ps_begin(PT[3]);

                           // v3 single-sided gate (same gate the energy/grad
                           // kernels use). If neither end's sticky face is on
                           // the contact side, leave β unchanged (it'll
                           // naturally stay 0 since the new-pair-init kernel
                           // also gates it).
                           if(!PT_sticky_gate(sticky_sign(PT[0]),
                                              sticky_sign(PT[1]),
                                              vert_normal(PT[0]),
                                              vert_normal(PT[1]),
                                              P, T0, T1, T2))
                               return;

                           // True closest-feature distance (PT->plane,
                           // PE->point-edge, PP->point-point) for this VT
                           // primitive's lagged feature classification.
                           Float D;
                           point_triangle_distance2(flag, P, T0, T1, T2, D);

                           // lagged tangential displacement over the step, using
                           // the matching feature basis (start-of-step positions).
                           Float u_sq = VT_tangential_rel_dx_sq(
                               flag, P0, T00, T10, T20, P, T0, T1, T2);

                           // Cross-layer occlusion gate: Phase B writes this
                           // every frame. When set, PT_beta_evolve_existing
                           // short-circuits to β=0 (the bonding_term at
                           // p_k>0 would otherwise re-ignite β from zero).
                           bool blocked = (i < n_blocked) && (blocked_buf(i) != 0);

                           beta_buf(i) = PT_beta_evolve_existing(
                               beta_buf(i), kappa,
                               rcc.Cn, rcc.Ct, rcc.W, rcc.eta,
                               rcc.bonding_rate, rcc.p0,
                               d_hat, dt, D, u_sq, blocked);
                       });

            // Compute keys for the just-evolved pairs and snapshot.
            muda::DeviceBuffer<U64> curr_keys;
            _compute_curr_keys_PT(curr_keys, pairs);
            m_prev_keys_PT.resize(n);
            m_prev_beta_PT.resize(n);
            m_prev_keys_PT.view().copy_from(curr_keys.view());
            m_prev_beta_PT.view().copy_from(m_beta_PT.view());
            // sort (keys, beta) for next step's binary search
            thrust::sort_by_key(thrust::device,
                                m_prev_keys_PT.view().data(),
                                m_prev_keys_PT.view().data() + n,
                                m_prev_beta_PT.view().data());
        }
        // else: pairs.size() == 0 on this step. PREVIOUSLY we wiped
        // m_prev_keys_PT/m_prev_beta_PT here so the next step would
        // start fresh. That's wrong for two cases:
        //   1) Frame 1 after world.init() — libuipc's friction
        //      candidate list (which we use as `pairs` here) is
        //      sourced from the *previous* step's DCD output. At
        //      frame 1 there is no previous step → `pairs.size()` is
        //      always 0. Wiping kills the β state we just loaded
        //      from an asset BEFORE it has a chance to be used in
        //      frame 2.
        //   2) Any frame where the tape briefly loses all PT contacts
        //      but the same pairs will form again next step.
        // Keep the prev snapshot intact so the next step's
        // match_or_init can re-bond returning pairs to their saved
        // β values. Stale prev entries are harmless: binary search
        // misses never trigger a match.

        // Snapshot positions for next step's u-signal.
        m_pos_at_step_begin.view().copy_from(positions);

        // v3: refresh lagged vertex normals from the new begin-of-next-step
        // positions. Held constant through the next frame's Newton iters.
        _recompute_vertex_normals(m_pos_at_step_begin.view());

        if(m_bonded_pt_system_for_phase_a
           && m_bonded_pt_system_for_phase_a->enabled())
        {
            RCCBondedPTReleaseContext release_context;
            release_context.sticky_side_enabled = m_has_sticky;
            release_context.sticky_sign = m_sticky_sign;
            release_context.vertex_normal = m_vertex_normal;
            release_context.policy_enabled = true;
            release_context.contact_element_ids =
                m_gvm_for_phase_a->contact_element_ids();
            release_context.subscene_element_ids =
                m_gvm_for_phase_a->subscene_element_ids();
            release_context.contact_mask_tabular =
                m_gcm_for_phase_a->contact_mask_tabular();
            release_context.subscene_mask_tabular =
                m_gcm_for_phase_a->subscene_mask_tabular();
            release_context.adhesive_tabular = m_adhesive_tabular;

            // Bonding stays PT-only for this milestone: compact the
            // face-interior (degenerate dim==4) per-VT beta into an array
            // aligned with friction_PTs(). The flag==4 VTs are 1:1, in order,
            // with friction_PTs (both are dim==4 selections of the same
            // candidate set), so the compacted beta lines up with the PT list.
            // Edge/corner VTs are excluded from locking until the producer
            // learns to consume them (Step 5).
            auto   vt_pairs = m_stf_for_phase_a->friction_VTs();
            auto   pt_pairs = m_stf_for_phase_a->friction_PTs();
            IndexT vt_n     = (IndexT)vt_pairs.size();
            m_vt_face_flags.resize(vt_n);
            m_beta_PT_face.resize(vt_n);
            IndexT face_n = 0;
            if(vt_n > 0)
            {
                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(vt_n,
                           [VTs   = vt_pairs.viewer().name("VTs"),
                            flags = m_vt_face_flags.view().viewer().name("face_flags")] __device__(int i) mutable
                           {
                               Vector4i off;
                               IndexT   dim =
                                   distance::degenerate_point_triangle(VTs(i).flag, off);
                               flags(i) = (dim == 4) ? IndexT{1} : IndexT{0};
                           });
                DeviceSelect().Flagged(m_beta_PT.view().data(),
                                       m_vt_face_flags.view().data(),
                                       m_beta_PT_face.view().data(),
                                       m_beta_PT_face_count.data(),
                                       vt_n);
                face_n = (IndexT)m_beta_PT_face_count;
            }
            m_beta_PT_face.resize(face_n);

            m_bonded_pt_system_for_phase_a->lock_from_rcc_pt_snapshot(
                pt_pairs,
                m_beta_PT_face.view(),
                positions,
                m_bonded_pt_beta_lock_threshold,
                release_context);
            m_bonded_pt_beta_carry.merge_released_beta(
                m_prev_keys_PT,
                m_prev_beta_PT,
                m_bonded_pt_system_for_phase_a->released_keys(),
                m_bonded_pt_system_for_phase_a->released_beta());
        }
    }

    // ====================================================================
    // β state dump / restore — exposed via RCCAdhesionStateAccessorFeature
    // so applications (wind demo → asset .npz → unwind/drop demo) can
    // round-trip per-pair β across processes.
    //
    // Only PT is persisted: EE/PE/PP β buffers are always sized 0 in v2.
    //
    // Round-trip protocol:
    //   wind side:  after world.advance(); call dump_prev_pt_state() at
    //               any frame boundary (Phase A has run, so the snapshot
    //               buffers are filled and sorted).
    //   load side:  after world.init(scene) and BEFORE the first
    //               world.advance(); call set_prev_pt_state(). The next
    //               step's Phase B then takes the match_or_init branch
    //               against these loaded keys (instead of init_all_new).
    // ====================================================================

    void dump_prev_pt_state(vector<U64>&   out_keys,
                            vector<Float>& out_betas) const
    {
        // m_prev_keys_PT / m_prev_beta_PT are already sorted by key
        // (Phase A ran thrust::sort_by_key on them at the end of the
        // previous step). BufferView::copy_to(T*) is the pointer-based
        // device→host primitive (see buffer_view.h:87).
        const SizeT n = m_prev_keys_PT.size();
        out_keys.resize(n);
        out_betas.resize(n);
        if(n == 0)
            return;
        m_prev_keys_PT.view().copy_to(out_keys.data());
        m_prev_beta_PT.view().copy_to(out_betas.data());
    }

    void set_prev_pt_state(span<const U64>   keys,
                           span<const Float> betas)
    {
        UIPC_ASSERT(keys.size() == betas.size(),
                    "RCC adhesion: set_prev_pt_state size mismatch (keys={}, betas={}).",
                    keys.size(), betas.size());
        const SizeT n = keys.size();
        m_prev_keys_PT.resize(n);
        m_prev_beta_PT.resize(n);
        if(n > 0)
        {
            m_prev_keys_PT.view().copy_from(keys.data());
            m_prev_beta_PT.view().copy_from(betas.data());
            // Defensive: keep the same sort-by-key invariant Phase A maintains.
            // If the caller provided already-sorted input (the dump round-trip
            // case) this is a no-op cost. If unsorted, match_or_init's binary
            // search would silently miss matches without this.
            thrust::sort_by_key(thrust::device,
                                m_prev_keys_PT.view().data(),
                                m_prev_keys_PT.view().data() + n,
                                m_prev_beta_PT.view().data());
        }
        m_has_loaded_prev_state = true;
    }

    SizeT prev_pt_pair_count() const { return m_prev_keys_PT.size(); }
};

REGISTER_SIM_SYSTEM(IPCSimplexRCCAdhesiveContact);


// ========================================================================
// RCCBetaEvolutionTimeIntegrator — drives Phase A once per step at end-of-step
// via the TimeIntegratorManager. Mirrors plasticity's
// `StrainPlasticDiscreteShellBendingTimeIntegrator` pattern.
// ========================================================================
class RCCBetaEvolutionTimeIntegrator final : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    SimSystemSlot<IPCSimplexRCCAdhesiveContact> rcc;
    SimSystemSlot<GlobalContactManager>         gcm;
    SimSystemSlot<GlobalVertexManager>          gvm;
    SimSystemSlot<GlobalTrajectoryFilter>       gtf;
    SimSystemSlot<RCCBondedPTSystem>            bonded_pt;

    void do_build(BuildInfo&) override
    {
        rcc = require<IPCSimplexRCCAdhesiveContact>();
        gcm = require<GlobalContactManager>();
        gvm = require<GlobalVertexManager>();
        gtf = require<GlobalTrajectoryFilter>();
        bonded_pt = find<RCCBondedPTSystem>();

        auto& config = world().scene().config();
        auto  beta_lock_threshold =
            config.find<Float>("rcc_bonded_pt_beta_lock_threshold");
        if(beta_lock_threshold)
            rcc->m_bonded_pt_beta_lock_threshold =
                beta_lock_threshold->view()[0];

        on_init_scene(
            [this]
            {
                rcc->m_gcm_for_phase_a = gcm.view();
                rcc->m_gvm_for_phase_a = gvm.view();
                auto stf = gtf->find<SimplexTrajectoryFilter>();
                rcc->m_stf_for_phase_a = stf.view();
                rcc->m_bonded_pt_system_for_phase_a = bonded_pt.view();
            });
    }

    void do_init(InitInfo&) override {}
    void do_predict_dof(PredictDofInfo&) override {}

    void do_update_state(UpdateVelocityInfo& info) override
    {
        rcc->_evolve_beta_step_at_end(info.dt());
    }
};
REGISTER_SIM_SYSTEM(RCCBetaEvolutionTimeIntegrator);


// ========================================================================
// RCCAdhesionStateAccessorFeature — exposes the reporter's
// dump_prev_pt_state / set_prev_pt_state methods to the frontend via the
// World::features() registry. Mirrors AffineBodyStateAccessor at
// src/backends/cuda/affine_body/affine_body_state_accessor.cu.
// ========================================================================
}  // namespace uipc::backend::cuda

#include <uipc/core/rcc_adhesion_state_accessor_feature.h>

namespace uipc::backend::cuda
{
class RCCAdhesionStateAccessorOverriderImpl final
    : public core::RCCAdhesionStateAccessorFeatureOverrider
{
  public:
    explicit RCCAdhesionStateAccessorOverriderImpl(IPCSimplexRCCAdhesiveContact& reporter)
        : m_reporter{reporter}
    {
    }

    SizeT get_pt_pair_count() const override
    {
        return m_reporter.prev_pt_pair_count();
    }

    void do_dump_pt_state(vector<U64>&   out_keys,
                          vector<Float>& out_betas) const override
    {
        m_reporter.dump_prev_pt_state(out_keys, out_betas);
    }

    void do_load_pt_state(span<const U64>   keys,
                          span<const Float> betas) override
    {
        m_reporter.set_prev_pt_state(keys, betas);
    }

  private:
    IPCSimplexRCCAdhesiveContact& m_reporter;
};

class RCCAdhesionStateAccessor final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    virtual void do_build() override
    {
        auto& reporter = require<IPCSimplexRCCAdhesiveContact>();
        auto  overrider =
            std::make_shared<RCCAdhesionStateAccessorOverriderImpl>(reporter);
        auto  feature =
            std::make_shared<core::RCCAdhesionStateAccessorFeature>(overrider);
        features().insert(feature);
    }
};
REGISTER_SIM_SYSTEM(RCCAdhesionStateAccessor);

}  // namespace uipc::backend::cuda
