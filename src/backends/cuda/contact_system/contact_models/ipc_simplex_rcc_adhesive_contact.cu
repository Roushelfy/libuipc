#include <contact_system/simplex_frictional_contact.h>
#include <contact_system/rcc_adhesive_coeff.h>
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
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <uipc/common/log.h>
#include <sim_engine.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <global_geometry/global_vertex_manager.h>
#include <contact_system/global_contact_manager.h>

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

    // β buffer for current step's PT pair list (aligned with friction_PTs()).
    // EE/PE/PP β buffers are stay size 0 (adhesion disabled for those types).
    muda::DeviceBuffer<Float> m_beta_PT;
    muda::DeviceBuffer<Float> m_beta_EE;
    muda::DeviceBuffer<Float> m_beta_PE;
    muda::DeviceBuffer<Float> m_beta_PP;

    // ---- v2 state ----
    // β + sorted keys snapshotted at end of last step (Phase A output;
    // Phase B input at start of next step).
    muda::DeviceBuffer<U64>     m_prev_keys_PT;
    muda::DeviceBuffer<Float>   m_prev_beta_PT;

    // Positions at the START of the current step. Phase A uses these to compute
    // the lagged tangent basis and tangential displacement `u` over the step
    // that just ended.
    muda::DeviceBuffer<Vector3> m_pos_at_step_begin;

    // Step-boundary detection: Phase B runs once per frame, the rest of the
    // Newton iterations within the same frame just read m_beta_PT.
    SizeT m_last_seen_frame = ~SizeT(0);
    bool  m_first_step      = true;

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

    // ---- v2 helper kernels (PT only) ----

    // Compute sorted-vertex u64 keys for the current PT pair list.
    void _compute_curr_keys_PT(muda::DeviceBuffer<U64>&    dst,
                               muda::CBufferView<Vector4i> pairs) noexcept
    {
        using namespace muda;
        auto n = pairs.size();
        dst.resize(n);
        if(n == 0)
            return;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [pairs = pairs.viewer().name("PTs"),
                    dst   = dst.view().viewer().name("curr_keys")] __device__(int i) mutable
                   {
                       using namespace sym::codim_ipc_rcc_adhesive;
                       const auto& P = pairs(i);
                       dst(i)        = PT_pair_key(P[0], P[1], P[2], P[3]);
                   });
    }

    // Phase B (initial step) — every pair is treated as new.
    // β = max(new-pair bonding kick, coeff.initial_beta).
    void _phase_b_init_all_new(muda::CBufferView<Vector4i>      pairs,
                               muda::CBufferView<IndexT>        contact_ids,
                               muda::CBuffer2DView<ContactCoeff> barrier_table,
                               muda::CBufferView<Float>          d_hats,
                               muda::CBufferView<Vector3>        positions,
                               Float                             dt) noexcept
    {
        using namespace muda;
        auto n = pairs.size();
        m_beta_PT.resize(n);
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
                    dt] __device__(int i) mutable
                   {
                       using namespace sym::codim_ipc_rcc_adhesive;
                       using namespace sym::codim_ipc_contact;
                       const auto& PT   = pairs(i);
                       Vector4i    cids = {contact_ids(PT[0]), contact_ids(PT[1]),
                                           contact_ids(PT[2]), contact_ids(PT[3])};
                       auto        rcc  = PT_rcc_coeff(rcc_table, cids);
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
                       Float   D;
                       distance::point_triangle_distance2(P, T0, T1, T2, D);
                       beta_dst(i) = PT_beta_init_new(rcc.initial_beta, kappa, rcc.Cn,
                                                     rcc.bonding_rate, d_hat, dt, D);
                   });
    }

    // Phase B (subsequent step) — match curr keys against m_prev_keys_PT.
    // If found: carry prev β. If not found: new-pair bonding kick.
    void _phase_b_match_or_init(muda::CBufferView<U64>            curr_keys,
                                muda::CBufferView<Vector4i>       pairs,
                                muda::CBufferView<IndexT>         contact_ids,
                                muda::CBuffer2DView<ContactCoeff> barrier_table,
                                muda::CBufferView<Float>          d_hats,
                                muda::CBufferView<Vector3>        positions,
                                Float                             dt) noexcept
    {
        using namespace muda;
        auto n = curr_keys.size();
        m_beta_PT.resize(n);
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
                    dt] __device__(int i) mutable
                   {
                       using namespace sym::codim_ipc_rcc_adhesive;
                       using namespace sym::codim_ipc_contact;
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

                       const auto& PT   = pairs(i);
                       Vector4i    cids = {contact_ids(PT[0]), contact_ids(PT[1]),
                                           contact_ids(PT[2]), contact_ids(PT[3])};
                       auto        rcc  = PT_rcc_coeff(rcc_table, cids);
                       if(!rcc.enabled)
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
                           Vector3 P  = Ps(PT[0]);
                           Vector3 T0 = Ps(PT[1]);
                           Vector3 T1 = Ps(PT[2]);
                           Vector3 T2 = Ps(PT[3]);
                           Float   D;
                           distance::point_triangle_distance2(P, T0, T1, T2, D);
                           beta_dst(i) = PT_beta_init_new(rcc.initial_beta, kappa, rcc.Cn,
                                                         rcc.bonding_rate, d_hat, dt, D);
                       }
                   });
    }

    // Resize the disabled (EE/PE/PP) β buffers to match the current pair
    // counts and fill with zero. Called every kernel-dispatch entry point so
    // even if the pair list changes between Newton iterations (e.g. friction
    // re-records candidates), the kernels never read past the buffer.
    template <typename Info>
    void _sync_disabled_buffers(Info& info)
    {
        auto n_ee = info.friction_EEs().size();
        auto n_pe = info.friction_PEs().size();
        auto n_pp = info.friction_PPs().size();
        m_beta_EE.resize(n_ee); if(n_ee > 0) m_beta_EE.fill(Float{0});
        m_beta_PE.resize(n_pe); if(n_pe > 0) m_beta_PE.fill(Float{0});
        m_beta_PP.resize(n_pp); if(n_pp > 0) m_beta_PP.fill(Float{0});
    }

    // Phase B entry: run once per frame at the top of do_compute_energy
    // (and idempotent if called from do_assemble too).
    void _phase_b_if_new_frame(EnergyInfo& info)
    {
        SizeT cur = engine().frame();
        if(cur == m_last_seen_frame)
            return;
        m_last_seen_frame = cur;

        // (EE/PE/PP buffers are sized every call by _sync_disabled_buffers.)

        auto pairs       = info.friction_PTs();
        auto contact_ids = info.contact_element_ids();
        auto barrier_tab = info.contact_tabular();
        auto d_hats      = info.d_hats();
        auto positions   = info.positions();
        auto dt          = info.dt();

        if(pairs.size() == 0)
        {
            m_beta_PT.resize(0);
            return;
        }

        muda::DeviceBuffer<U64> curr_keys;
        _compute_curr_keys_PT(curr_keys, pairs);

        if(m_first_step || m_prev_keys_PT.size() == 0)
        {
            _phase_b_init_all_new(pairs, contact_ids, barrier_tab, d_hats, positions, dt);
            m_first_step = false;
        }
        else
        {
            _phase_b_match_or_init(curr_keys.view(), pairs, contact_ids,
                                   barrier_tab, d_hats, positions, dt);
        }
    }

    // Same Phase B trigger but takes a ContactInfo (for do_assemble path).
    void _phase_b_if_new_frame(ContactInfo& info)
    {
        SizeT cur = engine().frame();
        if(cur == m_last_seen_frame)
            return;
        m_last_seen_frame = cur;

        // (EE/PE/PP buffers are sized every call by _sync_disabled_buffers.)

        auto pairs       = info.friction_PTs();
        auto contact_ids = info.contact_element_ids();
        auto barrier_tab = info.contact_tabular();
        auto d_hats      = info.d_hats();
        auto positions   = info.positions();
        auto dt          = info.dt();

        if(pairs.size() == 0)
        {
            m_beta_PT.resize(0);
            return;
        }

        muda::DeviceBuffer<U64> curr_keys;
        _compute_curr_keys_PT(curr_keys, pairs);

        if(m_first_step || m_prev_keys_PT.size() == 0)
        {
            _phase_b_init_all_new(pairs, contact_ids, barrier_tab, d_hats, positions, dt);
            m_first_step = false;
        }
        else
        {
            _phase_b_match_or_init(curr_keys.view(), pairs, contact_ids,
                                   barrier_tab, d_hats, positions, dt);
        }
    }

    virtual void do_compute_energy(EnergyInfo& info) override
    {
        using namespace muda;

        // Every call: keep EE/PE/PP β buffers sized + zeroed (adhesion is PT-only in v2).
        _sync_disabled_buffers(info);
        // Once per frame: Phase B match-or-init for PT.
        _phase_b_if_new_frame(info);

        auto pt_count = info.friction_PTs().size();
        auto ee_count = info.friction_EEs().size();
        auto pe_count = info.friction_PEs().size();
        auto pp_count = info.friction_PPs().size();

        if(pt_count > 0)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(pt_count,
                       [table = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_ids"),
                        PTs         = info.friction_PTs().viewer().name("PTs"),
                        Es          = info.friction_PT_energies().viewer().name("Es"),
                        Ps          = info.positions().viewer().name("Ps"),
                        prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                        d_hats      = info.d_hats().viewer().name("d_hats"),
                        beta_buf    = m_beta_PT.cviewer().name("beta_PT"),
                        dt          = info.dt()] __device__(int i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           const auto& PT   = PTs(i);
                           Float       beta = beta_buf(i);
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
                           const auto& pP  = prev_Ps(PT[0]);
                           const auto& pT0 = prev_Ps(PT[1]);
                           const auto& pT1 = prev_Ps(PT[2]);
                           const auto& pT2 = prev_Ps(PT[3]);

                           Float En = PT_normal_adhesion_energy(
                               coeff.Cn, beta, d_hat, dt, P, T0, T1, T2);
                           Float Et = PT_tangential_adhesion_energy(
                               coeff.Ct, beta, d_hat, dt,
                               pP, pT0, pT1, pT2, P, T0, T1, T2);
                           Es(i) = En + Et;
                       });
        }

        if(ee_count > 0)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(ee_count,
                       [table = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_ids"),
                        EEs         = info.friction_EEs().viewer().name("EEs"),
                        Es          = info.friction_EE_energies().viewer().name("Es"),
                        Ps          = info.positions().viewer().name("Ps"),
                        prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                        d_hats      = info.d_hats().viewer().name("d_hats"),
                        beta_buf    = m_beta_EE.cviewer().name("beta_EE"),
                        dt          = info.dt()] __device__(int i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           const auto& EE   = EEs(i);
                           Float       beta = beta_buf(i);
                           if(beta <= 0)
                           {
                               Es(i) = 0;
                               return;
                           }
                           Vector4i cids  = {contact_ids(EE[0]), contact_ids(EE[1]),
                                             contact_ids(EE[2]), contact_ids(EE[3])};
                           auto     coeff = EE_rcc_coeff(table, cids);
                           if(!coeff.enabled)
                           {
                               Es(i) = 0;
                               return;
                           }
                           Float d_hat = EE_d_hat(d_hats(EE[0]), d_hats(EE[1]),
                                                  d_hats(EE[2]), d_hats(EE[3]));

                           Float En = EE_normal_adhesion_energy(
                               coeff.Cn, beta, d_hat, dt,
                               Ps(EE[0]), Ps(EE[1]), Ps(EE[2]), Ps(EE[3]));
                           Float Et = EE_tangential_adhesion_energy(
                               coeff.Ct, beta, d_hat, dt,
                               prev_Ps(EE[0]), prev_Ps(EE[1]), prev_Ps(EE[2]), prev_Ps(EE[3]),
                               Ps(EE[0]), Ps(EE[1]), Ps(EE[2]), Ps(EE[3]));
                           Es(i) = En + Et;
                       });
        }

        if(pe_count > 0)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(pe_count,
                       [table = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_ids"),
                        PEs         = info.friction_PEs().viewer().name("PEs"),
                        Es          = info.friction_PE_energies().viewer().name("Es"),
                        Ps          = info.positions().viewer().name("Ps"),
                        prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                        d_hats      = info.d_hats().viewer().name("d_hats"),
                        beta_buf    = m_beta_PE.cviewer().name("beta_PE"),
                        dt          = info.dt()] __device__(int i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           const auto& PE   = PEs(i);
                           Float       beta = beta_buf(i);
                           if(beta <= 0)
                           {
                               Es(i) = 0;
                               return;
                           }
                           Vector3i cids  = {contact_ids(PE[0]), contact_ids(PE[1]),
                                             contact_ids(PE[2])};
                           auto     coeff = PE_rcc_coeff(table, cids);
                           if(!coeff.enabled)
                           {
                               Es(i) = 0;
                               return;
                           }
                           Float d_hat = PE_d_hat(d_hats(PE[0]), d_hats(PE[1]), d_hats(PE[2]));

                           Float En = PE_normal_adhesion_energy(
                               coeff.Cn, beta, d_hat, dt, Ps(PE[0]), Ps(PE[1]), Ps(PE[2]));
                           Float Et = PE_tangential_adhesion_energy(
                               coeff.Ct, beta, d_hat, dt,
                               prev_Ps(PE[0]), prev_Ps(PE[1]), prev_Ps(PE[2]),
                               Ps(PE[0]), Ps(PE[1]), Ps(PE[2]));
                           Es(i) = En + Et;
                       });
        }

        if(pp_count > 0)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(pp_count,
                       [table = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_ids"),
                        PPs         = info.friction_PPs().viewer().name("PPs"),
                        Es          = info.friction_PP_energies().viewer().name("Es"),
                        Ps          = info.positions().viewer().name("Ps"),
                        prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                        d_hats      = info.d_hats().viewer().name("d_hats"),
                        beta_buf    = m_beta_PP.cviewer().name("beta_PP"),
                        dt          = info.dt()] __device__(int i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           const auto& PP   = PPs(i);
                           Float       beta = beta_buf(i);
                           if(beta <= 0)
                           {
                               Es(i) = 0;
                               return;
                           }
                           Vector2i cids  = {contact_ids(PP[0]), contact_ids(PP[1])};
                           auto     coeff = PP_rcc_coeff(table, cids);
                           if(!coeff.enabled)
                           {
                               Es(i) = 0;
                               return;
                           }
                           Float d_hat = PP_d_hat(d_hats(PP[0]), d_hats(PP[1]));

                           Float En = PP_normal_adhesion_energy(
                               coeff.Cn, beta, d_hat, dt, Ps(PP[0]), Ps(PP[1]));
                           Float Et = PP_tangential_adhesion_energy(
                               coeff.Ct, beta, d_hat, dt,
                               prev_Ps(PP[0]), prev_Ps(PP[1]),
                               Ps(PP[0]), Ps(PP[1]));
                           Es(i) = En + Et;
                       });
        }
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace muda;

        _sync_disabled_buffers(info);
        _phase_b_if_new_frame(info);

        auto pt_count = (IndexT)info.friction_PTs().size();
        auto ee_count = (IndexT)info.friction_EEs().size();
        auto pe_count = (IndexT)info.friction_PEs().size();
        auto pp_count = (IndexT)info.friction_PPs().size();

        if(pt_count > 0)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(pt_count,
                       [gradient_only = info.gradient_only(),
                        table       = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_ids"),
                        Ps          = info.positions().viewer().name("Ps"),
                        prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                        d_hats      = info.d_hats().viewer().name("d_hats"),
                        dt          = info.dt(),
                        PTs         = info.friction_PTs().viewer().name("PTs"),
                        PT_Gs       = info.friction_PT_gradients().viewer().name("PT_Gs"),
                        PT_Hs       = info.friction_PT_hessians().viewer().name("PT_Hs"),
                        beta_buf    = m_beta_PT.cviewer().name("beta_PT")] __device__(IndexT i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           const auto& PT   = PTs(i);
                           Float       beta = beta_buf(i);

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
                                   const auto& pP  = prev_Ps(PT[0]);
                                   const auto& pT0 = prev_Ps(PT[1]);
                                   const auto& pT1 = prev_Ps(PT[2]);
                                   const auto& pT2 = prev_Ps(PT[3]);

                                   Vector12    Gn = Vector12::Zero(), Gt = Vector12::Zero();
                                   Matrix12x12 Hn = Matrix12x12::Zero(), Ht = Matrix12x12::Zero();
                                   if(gradient_only)
                                   {
                                       PT_normal_adhesion_gradient(
                                           Gn, coeff.Cn, beta, d_hat, dt, P, T0, T1, T2);
                                       if(coeff.Ct > 0)
                                           PT_tangential_adhesion_gradient(
                                               Gt, coeff.Ct, beta, d_hat, dt,
                                               pP, pT0, pT1, pT2, P, T0, T1, T2);
                                       G = Gn + Gt;
                                   }
                                   else
                                   {
                                       PT_normal_adhesion_gradient_hessian(
                                           Gn, Hn, coeff.Cn, beta, d_hat, dt, P, T0, T1, T2);
                                       if(coeff.Ct > 0)
                                           PT_tangential_adhesion_gradient_hessian(
                                               Gt, Ht, coeff.Ct, beta, d_hat, dt,
                                               pP, pT0, pT1, pT2, P, T0, T1, T2);
                                       G = Gn + Gt;
                                       cuda::make_spd(Hn);
                                       H = Hn + Ht;
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

        if(ee_count > 0)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(ee_count,
                       [gradient_only = info.gradient_only(),
                        table       = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_ids"),
                        Ps          = info.positions().viewer().name("Ps"),
                        prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                        d_hats      = info.d_hats().viewer().name("d_hats"),
                        dt          = info.dt(),
                        EEs         = info.friction_EEs().viewer().name("EEs"),
                        EE_Gs       = info.friction_EE_gradients().viewer().name("EE_Gs"),
                        EE_Hs       = info.friction_EE_hessians().viewer().name("EE_Hs"),
                        beta_buf    = m_beta_EE.cviewer().name("beta_EE")] __device__(IndexT i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           const auto& EE   = EEs(i);
                           Float       beta = beta_buf(i);

                           Vector12    G = Vector12::Zero();
                           Matrix12x12 H = Matrix12x12::Zero();

                           if(beta > 0)
                           {
                               Vector4i cids = {contact_ids(EE[0]), contact_ids(EE[1]),
                                                contact_ids(EE[2]), contact_ids(EE[3])};
                               auto     coeff = EE_rcc_coeff(table, cids);
                               if(coeff.enabled)
                               {
                                   Float d_hat = EE_d_hat(d_hats(EE[0]), d_hats(EE[1]),
                                                          d_hats(EE[2]), d_hats(EE[3]));

                                   Vector12    Gn = Vector12::Zero(), Gt = Vector12::Zero();
                                   Matrix12x12 Hn = Matrix12x12::Zero(), Ht = Matrix12x12::Zero();
                                   if(gradient_only)
                                   {
                                       EE_normal_adhesion_gradient(
                                           Gn, coeff.Cn, beta, d_hat, dt,
                                           Ps(EE[0]), Ps(EE[1]), Ps(EE[2]), Ps(EE[3]));
                                       if(coeff.Ct > 0)
                                           EE_tangential_adhesion_gradient(
                                               Gt, coeff.Ct, beta, d_hat, dt,
                                               prev_Ps(EE[0]), prev_Ps(EE[1]), prev_Ps(EE[2]), prev_Ps(EE[3]),
                                               Ps(EE[0]), Ps(EE[1]), Ps(EE[2]), Ps(EE[3]));
                                       G = Gn + Gt;
                                   }
                                   else
                                   {
                                       EE_normal_adhesion_gradient_hessian(
                                           Gn, Hn, coeff.Cn, beta, d_hat, dt,
                                           Ps(EE[0]), Ps(EE[1]), Ps(EE[2]), Ps(EE[3]));
                                       if(coeff.Ct > 0)
                                           EE_tangential_adhesion_gradient_hessian(
                                               Gt, Ht, coeff.Ct, beta, d_hat, dt,
                                               prev_Ps(EE[0]), prev_Ps(EE[1]), prev_Ps(EE[2]), prev_Ps(EE[3]),
                                               Ps(EE[0]), Ps(EE[1]), Ps(EE[2]), Ps(EE[3]));
                                       G = Gn + Gt;
                                       cuda::make_spd(Hn);
                                       H = Hn + Ht;
                                   }
                               }
                           }

                           DoubletVectorAssembler DVA{EE_Gs};
                           DVA.segment<4>(i * 4).write(EE, G);
                           if(!gradient_only)
                           {
                               TripletMatrixAssembler TMA{EE_Hs};
                               TMA.half_block<4>(i * EEHalfHessianSize).write(EE, H);
                           }
                       });
        }

        if(pe_count > 0)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(pe_count,
                       [gradient_only = info.gradient_only(),
                        table       = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_ids"),
                        Ps          = info.positions().viewer().name("Ps"),
                        prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                        d_hats      = info.d_hats().viewer().name("d_hats"),
                        dt          = info.dt(),
                        PEs         = info.friction_PEs().viewer().name("PEs"),
                        PE_Gs       = info.friction_PE_gradients().viewer().name("PE_Gs"),
                        PE_Hs       = info.friction_PE_hessians().viewer().name("PE_Hs"),
                        beta_buf    = m_beta_PE.cviewer().name("beta_PE")] __device__(IndexT i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           const auto& PE   = PEs(i);
                           Float       beta = beta_buf(i);

                           Vector9   G = Vector9::Zero();
                           Matrix9x9 H = Matrix9x9::Zero();

                           if(beta > 0)
                           {
                               Vector3i cids = {contact_ids(PE[0]), contact_ids(PE[1]),
                                                contact_ids(PE[2])};
                               auto     coeff = PE_rcc_coeff(table, cids);
                               if(coeff.enabled)
                               {
                                   Float d_hat = PE_d_hat(d_hats(PE[0]), d_hats(PE[1]),
                                                          d_hats(PE[2]));

                                   Vector9   Gn = Vector9::Zero(), Gt = Vector9::Zero();
                                   Matrix9x9 Hn = Matrix9x9::Zero(), Ht = Matrix9x9::Zero();
                                   if(gradient_only)
                                   {
                                       PE_normal_adhesion_gradient(
                                           Gn, coeff.Cn, beta, d_hat, dt,
                                           Ps(PE[0]), Ps(PE[1]), Ps(PE[2]));
                                       if(coeff.Ct > 0)
                                           PE_tangential_adhesion_gradient(
                                               Gt, coeff.Ct, beta, d_hat, dt,
                                               prev_Ps(PE[0]), prev_Ps(PE[1]), prev_Ps(PE[2]),
                                               Ps(PE[0]), Ps(PE[1]), Ps(PE[2]));
                                       G = Gn + Gt;
                                   }
                                   else
                                   {
                                       PE_normal_adhesion_gradient_hessian(
                                           Gn, Hn, coeff.Cn, beta, d_hat, dt,
                                           Ps(PE[0]), Ps(PE[1]), Ps(PE[2]));
                                       if(coeff.Ct > 0)
                                           PE_tangential_adhesion_gradient_hessian(
                                               Gt, Ht, coeff.Ct, beta, d_hat, dt,
                                               prev_Ps(PE[0]), prev_Ps(PE[1]), prev_Ps(PE[2]),
                                               Ps(PE[0]), Ps(PE[1]), Ps(PE[2]));
                                       G = Gn + Gt;
                                       cuda::make_spd(Hn);
                                       H = Hn + Ht;
                                   }
                               }
                           }

                           DoubletVectorAssembler DVA{PE_Gs};
                           DVA.segment<3>(i * 3).write(PE, G);
                           if(!gradient_only)
                           {
                               TripletMatrixAssembler TMA{PE_Hs};
                               TMA.half_block<3>(i * PEHalfHessianSize).write(PE, H);
                           }
                       });
        }

        if(pp_count > 0)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(pp_count,
                       [gradient_only = info.gradient_only(),
                        table       = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_ids"),
                        Ps          = info.positions().viewer().name("Ps"),
                        prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                        d_hats      = info.d_hats().viewer().name("d_hats"),
                        dt          = info.dt(),
                        PPs         = info.friction_PPs().viewer().name("PPs"),
                        PP_Gs       = info.friction_PP_gradients().viewer().name("PP_Gs"),
                        PP_Hs       = info.friction_PP_hessians().viewer().name("PP_Hs"),
                        beta_buf    = m_beta_PP.cviewer().name("beta_PP")] __device__(IndexT i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           const auto& PP   = PPs(i);
                           Float       beta = beta_buf(i);

                           Vector6   G = Vector6::Zero();
                           Matrix6x6 H = Matrix6x6::Zero();

                           if(beta > 0)
                           {
                               Vector2i cids  = {contact_ids(PP[0]), contact_ids(PP[1])};
                               auto     coeff = PP_rcc_coeff(table, cids);
                               if(coeff.enabled)
                               {
                                   Float d_hat = PP_d_hat(d_hats(PP[0]), d_hats(PP[1]));

                                   Vector6   Gn = Vector6::Zero(), Gt = Vector6::Zero();
                                   Matrix6x6 Hn = Matrix6x6::Zero(), Ht = Matrix6x6::Zero();
                                   if(gradient_only)
                                   {
                                       PP_normal_adhesion_gradient(
                                           Gn, coeff.Cn, beta, d_hat, dt, Ps(PP[0]), Ps(PP[1]));
                                       if(coeff.Ct > 0)
                                           PP_tangential_adhesion_gradient(
                                               Gt, coeff.Ct, beta, d_hat, dt,
                                               prev_Ps(PP[0]), prev_Ps(PP[1]),
                                               Ps(PP[0]), Ps(PP[1]));
                                       G = Gn + Gt;
                                   }
                                   else
                                   {
                                       PP_normal_adhesion_gradient_hessian(
                                           Gn, Hn, coeff.Cn, beta, d_hat, dt,
                                           Ps(PP[0]), Ps(PP[1]));
                                       if(coeff.Ct > 0)
                                           PP_tangential_adhesion_gradient_hessian(
                                               Gt, Ht, coeff.Ct, beta, d_hat, dt,
                                               prev_Ps(PP[0]), prev_Ps(PP[1]),
                                               Ps(PP[0]), Ps(PP[1]));
                                       G = Gn + Gt;
                                       // PP normal Hessian is naturally PSD; Ht also PSD.
                                       H = Hn + Ht;
                                   }
                               }
                           }

                           DoubletVectorAssembler DVA{PP_Gs};
                           DVA.segment<2>(i * 2).write(PP, G);
                           if(!gradient_only)
                           {
                               TripletMatrixAssembler TMA{PP_Hs};
                               TMA.half_block<2>(i * PPHalfHessianSize).write(PP, H);
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

    void _evolve_beta_step_at_end(Float dt)
    {
        using namespace muda;
        auto pairs       = m_stf_for_phase_a->friction_PTs();
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
                       [pairs       = pairs.viewer().name("PTs"),
                        contact_ids = contact_ids.viewer().name("contact_ids"),
                        rcc_table   = m_adhesive_tabular.cviewer().name("rcc_tabular"),
                        bar_table   = barrier_tab.viewer().name("barrier_tabular"),
                        d_hats      = d_hats.viewer().name("d_hats"),
                        Ps          = positions.viewer().name("Ps"),
                        Ps_begin    = m_pos_at_step_begin.cviewer().name("Ps_begin"),
                        beta_buf    = m_beta_PT.view().viewer().name("beta_PT"),
                        dt] __device__(int i) mutable
                       {
                           using namespace sym::codim_ipc_rcc_adhesive;
                           using namespace sym::codim_ipc_contact;
                           using namespace distance;
                           using namespace friction;

                           const auto& PT = pairs(i);
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

                           Float D;
                           point_triangle_distance2(P, T0, T1, T2, D);

                           // lagged tangent basis from start-of-step positions
                           Vector2             bary;
                           Matrix<Float, 3, 2> basis;
                           point_triangle_closest_point(P0, T00, T10, T20, bary);
                           point_triangle_tangent_basis(P0, T00, T10, T20, basis);
                           Vector3 dP  = P  - P0;
                           Vector3 dT0 = T0 - T00;
                           Vector3 dT1 = T1 - T10;
                           Vector3 dT2 = T2 - T20;
                           Vector2 u;
                           point_triangle_tan_rel_dx(dP, dT0, dT1, dT2, basis, bary, u);
                           Float u_sq = u.squaredNorm();

                           beta_buf(i) = PT_beta_evolve_existing(
                               beta_buf(i), kappa,
                               rcc.Cn, rcc.Ct, rcc.W, rcc.eta,
                               rcc.bonding_rate, rcc.p0,
                               d_hat, dt, D, u_sq);
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
        else
        {
            // No active pairs this step — drop snapshot so next step starts fresh.
            m_prev_keys_PT.resize(0);
            m_prev_beta_PT.resize(0);
        }

        // Snapshot positions for next step's u-signal.
        m_pos_at_step_begin.view().copy_from(positions);
    }
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

    void do_build(BuildInfo&) override
    {
        rcc = require<IPCSimplexRCCAdhesiveContact>();
        gcm = require<GlobalContactManager>();
        gvm = require<GlobalVertexManager>();
        gtf = require<GlobalTrajectoryFilter>();

        on_init_scene(
            [this]
            {
                rcc->m_gcm_for_phase_a = gcm.view();
                rcc->m_gvm_for_phase_a = gvm.view();
                auto stf = gtf->find<SimplexTrajectoryFilter>();
                rcc->m_stf_for_phase_a = stf.view();
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

}  // namespace uipc::backend::cuda
