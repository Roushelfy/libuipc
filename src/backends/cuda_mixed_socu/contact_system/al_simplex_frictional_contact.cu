#include <contact_system/al_contact_function.h>
#include <contact_system/al_simplex_frictional_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>
#include <utils/matrix_assembler.h>
#include <uipc/common/log.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
void ALSimplexFrictionalContact::do_build(ContactReporter::BuildInfo& info)
{
    m_impl.global_contact_manager = require<GlobalContactManager>();
    m_impl.global_vertex_manager  = require<GlobalVertexManager>();
    m_impl.global_surf_manager    = require<GlobalSimplicialSurfaceManager>();
    m_impl.global_active_set_manager = require<GlobalActiveSetManager>();

    auto dt_attr = world().scene().config().find<Float>("dt");
    m_impl.dt    = dt_attr->view()[0];
}

void ALSimplexFrictionalContact::do_report_energy_extent(GlobalContactManager::EnergyExtentInfo& info)
{
    auto& active_set = m_impl.global_active_set_manager;

    if(!active_set->is_enabled())
    {
        info.energy_count(0);
        return;
    }

    info.energy_count(active_set->PTs_friction().size()
                      + active_set->EEs_friction().size());
}

void ALSimplexFrictionalContact::do_report_gradient_hessian_extent(
    GlobalContactManager::GradientHessianExtentInfo& info)
{
    auto& active_set = m_impl.global_active_set_manager;

    if(!active_set->is_enabled())
    {
        info.gradient_count(0);
        info.hessian_count(0);
        return;
    }

    info.gradient_count(
        4 * (active_set->PTs_friction().size() + active_set->EEs_friction().size()));
    info.hessian_count(
        10 * (active_set->PTs_friction().size() + active_set->EEs_friction().size()));
}

void ALSimplexFrictionalContact::Impl::do_compute_energy(GlobalContactManager::EnergyInfo& info)
{
    using namespace muda;
    using namespace sym::al_simplex_contact;
    using Alu = ActivePolicy::AluScalar;
    using Energy = ActivePolicy::EnergyScalar;
    using Vec3A = Eigen::Matrix<Alu, 3, 1>;
    auto& active_set = global_active_set_manager;

    if(!active_set->is_enabled())
        return;

    auto PT_size     = active_set->PTs_friction().size();
    auto EE_size     = active_set->EEs_friction().size();
    auto PT_energies = info.energies().subview(0, PT_size);
    auto EE_energies = info.energies().subview(PT_size, EE_size);

    auto x         = global_vertex_manager->positions();
    auto prev_x    = global_vertex_manager->prev_positions();
    auto PTs       = active_set->PTs_friction();
    auto EEs       = active_set->EEs_friction();
    auto PT_lambda = active_set->PT_lambda_friction();
    auto EE_lambda = active_set->EE_lambda_friction();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(PT_size,
               [table = global_contact_manager->contact_tabular().cviewer().name("contact_tabular"),
                contact_ids =
                    global_vertex_manager->contact_element_ids().cviewer().name("contact_element_ids"),
                eps_v  = global_contact_manager->eps_velocity(),
                dt     = dt,
                PTs    = PTs.cviewer().name("PTs"),
                lambda = PT_lambda.cviewer().name("lambda"),
                x      = x.cviewer().name("x"),
                prev_x = prev_x.cviewer().name("prev_x"),
                Es = PT_energies.viewer().name("Es")] __device__(int idx) mutable
               {
                   auto     PT   = PTs(idx);
                   Vector4i cids = {contact_ids(PT[0]),
                                    contact_ids(PT[1]),
                                    contact_ids(PT[2]),
                                    contact_ids(PT[3])};

                   auto coeff = sym::codim_ipc_contact::PT_contact_coeff(table, cids);
                   Alu mu = safe_cast<Alu>(coeff.mu);

                   const auto& prev_P  = prev_x(PT[0]);
                   const auto& prev_T0 = prev_x(PT[1]);
                   const auto& prev_T1 = prev_x(PT[2]);
                   const auto& prev_T2 = prev_x(PT[3]);

                   const auto& P  = x(PT[0]);
                   const auto& T0 = x(PT[1]);
                   const auto& T1 = x(PT[2]);
                   const auto& T2 = x(PT[3]);
                   Vec3A prev_P_alu  = prev_P.template cast<Alu>();
                   Vec3A prev_T0_alu = prev_T0.template cast<Alu>();
                   Vec3A prev_T1_alu = prev_T1.template cast<Alu>();
                   Vec3A prev_T2_alu = prev_T2.template cast<Alu>();
                   Vec3A P_alu       = P.template cast<Alu>();
                   Vec3A T0_alu      = T0.template cast<Alu>();
                   Vec3A T1_alu      = T1.template cast<Alu>();
                   Vec3A T2_alu      = T2.template cast<Alu>();

                   Es(idx) = safe_cast<Energy>(PT_friction_energy(
                       mu,
                       safe_cast<Alu>(eps_v * dt),
                       safe_cast<Alu>(lambda(idx)),
                       prev_P_alu,
                       prev_T0_alu,
                       prev_T1_alu,
                       prev_T2_alu,
                       P_alu,
                       T0_alu,
                       T1_alu,
                       T2_alu));
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(EE_size,
               [table = global_contact_manager->contact_tabular().cviewer().name("contact_tabular"),
                contact_ids =
                    global_vertex_manager->contact_element_ids().cviewer().name("contact_element_ids"),
                eps_v  = global_contact_manager->eps_velocity(),
                dt     = dt,
                EEs    = EEs.cviewer().name("EEs"),
                lambda = EE_lambda.cviewer().name("lambda"),
                x      = x.cviewer().name("x"),
                prev_x = prev_x.cviewer().name("prev_x"),
                Es = EE_energies.viewer().name("Es")] __device__(int idx) mutable
               {
                   auto     EE   = EEs(idx);
                   Vector4i cids = {contact_ids(EE[0]),
                                    contact_ids(EE[1]),
                                    contact_ids(EE[2]),
                                    contact_ids(EE[3])};

                   auto coeff = sym::codim_ipc_contact::EE_contact_coeff(table, cids);
                   Alu mu = safe_cast<Alu>(coeff.mu);

                   const Vector3& prev_Ea0 = prev_x(EE[0]);
                   const Vector3& prev_Ea1 = prev_x(EE[1]);
                   const Vector3& prev_Eb0 = prev_x(EE[2]);
                   const Vector3& prev_Eb1 = prev_x(EE[3]);

                   const Vector3& Ea0 = x(EE[0]);
                   const Vector3& Ea1 = x(EE[1]);
                   const Vector3& Eb0 = x(EE[2]);
                   const Vector3& Eb1 = x(EE[3]);
                   Vec3A prev_Ea0_alu = prev_Ea0.template cast<Alu>();
                   Vec3A prev_Ea1_alu = prev_Ea1.template cast<Alu>();
                   Vec3A prev_Eb0_alu = prev_Eb0.template cast<Alu>();
                   Vec3A prev_Eb1_alu = prev_Eb1.template cast<Alu>();
                   Vec3A Ea0_alu      = Ea0.template cast<Alu>();
                   Vec3A Ea1_alu      = Ea1.template cast<Alu>();
                   Vec3A Eb0_alu      = Eb0.template cast<Alu>();
                   Vec3A Eb1_alu      = Eb1.template cast<Alu>();

                   Es(idx) = safe_cast<Energy>(EE_friction_energy(
                       mu,
                       safe_cast<Alu>(eps_v * dt),
                       safe_cast<Alu>(lambda(idx)),
                       prev_Ea0_alu,
                       prev_Ea1_alu,
                       prev_Eb0_alu,
                       prev_Eb1_alu,
                       Ea0_alu,
                       Ea1_alu,
                       Eb0_alu,
                       Eb1_alu));
               });
}

void ALSimplexFrictionalContact::Impl::do_assemble(GlobalContactManager::GradientHessianInfo& info)
{
    using namespace muda;
    using namespace sym::al_simplex_contact;
    using Alu   = ActivePolicy::AluScalar;
    using Store = ActivePolicy::StoreScalar;
    using Vec3A = Eigen::Matrix<Alu, 3, 1>;
    using Vec12A = Eigen::Matrix<Alu, 12, 1>;
    using Mat12A = Eigen::Matrix<Alu, 12, 12>;
    auto& active_set = global_active_set_manager;

    if(!active_set->is_enabled())
        return;

    auto PT_size = active_set->PTs_friction().size();
    auto EE_size = active_set->EEs_friction().size();
    auto PT_grad = info.gradients().subview(0, PT_size * 4);
    auto EE_grad = info.gradients().subview(PT_size * 4, EE_size * 4);
    auto PT_hess = info.hessians().subview(0, PT_size * 10);
    auto EE_hess = info.hessians().subview(PT_size * 10, EE_size * 10);

    auto x         = global_vertex_manager->positions();
    auto prev_x    = global_vertex_manager->prev_positions();
    auto rest_x    = global_vertex_manager->rest_positions();
    auto PTs       = active_set->PTs_friction();
    auto EEs       = active_set->EEs_friction();
    auto PT_lambda = active_set->PT_lambda_friction();
    auto EE_lambda = active_set->EE_lambda_friction();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(PT_size,
               [table = global_contact_manager->contact_tabular().cviewer().name("contact_tabular"),
                contact_ids =
                    global_vertex_manager->contact_element_ids().cviewer().name("contact_element_ids"),
                eps_v  = global_contact_manager->eps_velocity(),
                dt     = dt,
                PTs    = PTs.cviewer().name("PTs"),
                lambda = PT_lambda.cviewer().name("lambda"),
                x      = x.cviewer().name("x"),
                prev_x = prev_x.cviewer().name("prev_x"),
                Gs     = PT_grad.viewer().name("Gs"),
                Hs = PT_hess.viewer().name("Hs")] __device__(int idx) mutable
               {
                   auto     PT   = PTs(idx);
                   Vector4i cids = {contact_ids(PT[0]),
                                    contact_ids(PT[1]),
                                    contact_ids(PT[2]),
                                    contact_ids(PT[3])};

                   auto coeff = sym::codim_ipc_contact::PT_contact_coeff(table, cids);
                   Alu mu = safe_cast<Alu>(coeff.mu);

                   const auto& prev_P  = prev_x(PT[0]);
                   const auto& prev_T0 = prev_x(PT[1]);
                   const auto& prev_T1 = prev_x(PT[2]);
                   const auto& prev_T2 = prev_x(PT[3]);

                   const auto& P  = x(PT[0]);
                   const auto& T0 = x(PT[1]);
                   const auto& T1 = x(PT[2]);
                   const auto& T2 = x(PT[3]);
                   Vec3A prev_P_alu  = prev_P.template cast<Alu>();
                   Vec3A prev_T0_alu = prev_T0.template cast<Alu>();
                   Vec3A prev_T1_alu = prev_T1.template cast<Alu>();
                   Vec3A prev_T2_alu = prev_T2.template cast<Alu>();
                   Vec3A P_alu       = P.template cast<Alu>();
                   Vec3A T0_alu      = T0.template cast<Alu>();
                   Vec3A T1_alu      = T1.template cast<Alu>();
                   Vec3A T2_alu      = T2.template cast<Alu>();

                   Vec12A G;
                   Mat12A H;

                   PT_friction_gradient_hessian(
                       G,
                       H,
                       mu,
                       safe_cast<Alu>(eps_v * dt),
                       safe_cast<Alu>(lambda(idx)),
                       prev_P_alu,
                       prev_T0_alu,
                       prev_T1_alu,
                       prev_T2_alu,
                       P_alu,
                       T0_alu,
                       T1_alu,
                       T2_alu);

                   DoubletVectorAssembler DVA{Gs};
                   DVA.segment<4>(idx * 4).write(PT, downcast_gradient<Store>(G));

                   TripletMatrixAssembler TMA{Hs};
                   TMA.half_block<4>(idx * 10).write(PT, downcast_hessian<Store>(H));
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(EE_size,
               [table = global_contact_manager->contact_tabular().cviewer().name("contact_tabular"),
                contact_ids =
                    global_vertex_manager->contact_element_ids().cviewer().name("contact_element_ids"),
                eps_v  = global_contact_manager->eps_velocity(),
                dt     = dt,
                EEs    = EEs.cviewer().name("EEs"),
                lambda = EE_lambda.cviewer().name("lambda"),
                x      = x.cviewer().name("x"),
                rest_x = rest_x.viewer().name("rest_x"),
                prev_x = prev_x.cviewer().name("prev_x"),
                Gs     = EE_grad.viewer().name("Gs"),
                Hs = EE_hess.viewer().name("Hs")] __device__(int idx) mutable
               {
                   auto     EE   = EEs(idx);
                   Vector4i cids = {contact_ids(EE[0]),
                                    contact_ids(EE[1]),
                                    contact_ids(EE[2]),
                                    contact_ids(EE[3])};

                   auto coeff = sym::codim_ipc_contact::EE_contact_coeff(table, cids);
                   Alu mu = safe_cast<Alu>(coeff.mu);

                   const Vector3& rest_Ea0 = rest_x(EE[0]);
                   const Vector3& rest_Ea1 = rest_x(EE[1]);
                   const Vector3& rest_Eb0 = rest_x(EE[2]);
                   const Vector3& rest_Eb1 = rest_x(EE[3]);

                   const Vector3& prev_Ea0 = prev_x(EE[0]);
                   const Vector3& prev_Ea1 = prev_x(EE[1]);
                   const Vector3& prev_Eb0 = prev_x(EE[2]);
                   const Vector3& prev_Eb1 = prev_x(EE[3]);

                   const Vector3& Ea0 = x(EE[0]);
                   const Vector3& Ea1 = x(EE[1]);
                   const Vector3& Eb0 = x(EE[2]);
                   const Vector3& Eb1 = x(EE[3]);
                   Vec3A rest_Ea0_alu = rest_Ea0.template cast<Alu>();
                   Vec3A rest_Ea1_alu = rest_Ea1.template cast<Alu>();
                   Vec3A rest_Eb0_alu = rest_Eb0.template cast<Alu>();
                   Vec3A rest_Eb1_alu = rest_Eb1.template cast<Alu>();
                   Vec3A prev_Ea0_alu = prev_Ea0.template cast<Alu>();
                   Vec3A prev_Ea1_alu = prev_Ea1.template cast<Alu>();
                   Vec3A prev_Eb0_alu = prev_Eb0.template cast<Alu>();
                   Vec3A prev_Eb1_alu = prev_Eb1.template cast<Alu>();
                   Vec3A Ea0_alu      = Ea0.template cast<Alu>();
                   Vec3A Ea1_alu      = Ea1.template cast<Alu>();
                   Vec3A Eb0_alu      = Eb0.template cast<Alu>();
                   Vec3A Eb1_alu      = Eb1.template cast<Alu>();

                   Alu eps_x;
                   distance::edge_edge_mollifier_threshold(
                       rest_Ea0_alu,
                       rest_Ea1_alu,
                       rest_Eb0_alu,
                       rest_Eb1_alu,
                       static_cast<Alu>(-1.0),
                       eps_x);
                   if(!distance::need_mollify(prev_Ea0_alu,
                                              prev_Ea1_alu,
                                              prev_Eb0_alu,
                                              prev_Eb1_alu,
                                              eps_x))
                   {
                       Vec12A G;
                       Mat12A H;

                       EE_friction_gradient_hessian(
                           G,
                           H,
                           mu,
                           safe_cast<Alu>(eps_v * dt),
                           safe_cast<Alu>(lambda(idx)),
                           prev_Ea0_alu,
                           prev_Ea1_alu,
                           prev_Eb0_alu,
                           prev_Eb1_alu,
                           Ea0_alu,
                           Ea1_alu,
                           Eb0_alu,
                           Eb1_alu);

                       DoubletVectorAssembler DVA{Gs};
                       DVA.segment<4>(idx * 4).write(EE, downcast_gradient<Store>(G));

                       TripletMatrixAssembler TMA{Hs};
                       TMA.half_block<4>(idx * 10).write(EE, downcast_hessian<Store>(H));
                   }
               });
}

void ALSimplexFrictionalContact::Impl::do_assemble_structured_hessian(
    GlobalDyTopoEffectManager::StructuredHessianInfo& info)
{
    using namespace muda;
    using namespace sym::al_simplex_contact;
    using Alu    = ActivePolicy::AluScalar;
    using Vec3A  = Eigen::Matrix<Alu, 3, 1>;
    using Vec12A = Eigen::Matrix<Alu, 12, 1>;
    using Mat12A = Eigen::Matrix<Alu, 12, 12>;
    auto& active_set = global_active_set_manager;

    if(!active_set->is_enabled())
        return;

    auto PT_size = active_set->PTs_friction().size();
    auto EE_size = active_set->EEs_friction().size();
    auto x       = global_vertex_manager->positions();
    auto prev_x  = global_vertex_manager->prev_positions();
    auto rest_x  = global_vertex_manager->rest_positions();
    auto PTs     = active_set->PTs_friction();
    auto EEs     = active_set->EEs_friction();
    auto PT_lambda = active_set->PT_lambda_friction();
    auto EE_lambda = active_set->EE_lambda_friction();
    auto structured_sink = info.contact_sink();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(PT_size,
               [table = global_contact_manager->contact_tabular().cviewer().name("contact_tabular"),
                contact_ids =
                    global_vertex_manager->contact_element_ids().cviewer().name("contact_element_ids"),
                eps_v  = global_contact_manager->eps_velocity(),
                dt     = dt,
                PTs    = PTs.cviewer().name("PTs"),
                lambda = PT_lambda.cviewer().name("lambda"),
                x      = x.cviewer().name("x"),
                prev_x = prev_x.cviewer().name("prev_x"),
                structured_sink] __device__(int idx) mutable
               {
                   auto     PT   = PTs(idx);
                   Vector4i cids = {contact_ids(PT[0]),
                                    contact_ids(PT[1]),
                                    contact_ids(PT[2]),
                                    contact_ids(PT[3])};

                   auto coeff = sym::codim_ipc_contact::PT_contact_coeff(table, cids);
                   Alu  mu    = safe_cast<Alu>(coeff.mu);

                   const auto& prev_P  = prev_x(PT[0]);
                   const auto& prev_T0 = prev_x(PT[1]);
                   const auto& prev_T1 = prev_x(PT[2]);
                   const auto& prev_T2 = prev_x(PT[3]);

                   const auto& P  = x(PT[0]);
                   const auto& T0 = x(PT[1]);
                   const auto& T1 = x(PT[2]);
                   const auto& T2 = x(PT[3]);
                   Vec3A prev_P_alu  = prev_P.template cast<Alu>();
                   Vec3A prev_T0_alu = prev_T0.template cast<Alu>();
                   Vec3A prev_T1_alu = prev_T1.template cast<Alu>();
                   Vec3A prev_T2_alu = prev_T2.template cast<Alu>();
                   Vec3A P_alu       = P.template cast<Alu>();
                   Vec3A T0_alu      = T0.template cast<Alu>();
                   Vec3A T1_alu      = T1.template cast<Alu>();
                   Vec3A T2_alu      = T2.template cast<Alu>();

                   Vec12A G;
                   Mat12A H;
                   PT_friction_gradient_hessian(G,
                                                H,
                                                mu,
                                                safe_cast<Alu>(eps_v * dt),
                                                safe_cast<Alu>(lambda(idx)),
                                                prev_P_alu,
                                                prev_T0_alu,
                                                prev_T1_alu,
                                                prev_T2_alu,
                                                P_alu,
                                                T0_alu,
                                                T1_alu,
                                                T2_alu);

                   structured_sink.template write_hessian_half<4>(PT, H);
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(EE_size,
               [table = global_contact_manager->contact_tabular().cviewer().name("contact_tabular"),
                contact_ids =
                    global_vertex_manager->contact_element_ids().cviewer().name("contact_element_ids"),
                eps_v  = global_contact_manager->eps_velocity(),
                dt     = dt,
                EEs    = EEs.cviewer().name("EEs"),
                lambda = EE_lambda.cviewer().name("lambda"),
                x      = x.cviewer().name("x"),
                rest_x = rest_x.viewer().name("rest_x"),
                prev_x = prev_x.cviewer().name("prev_x"),
                structured_sink] __device__(int idx) mutable
               {
                   auto     EE   = EEs(idx);
                   Vector4i cids = {contact_ids(EE[0]),
                                    contact_ids(EE[1]),
                                    contact_ids(EE[2]),
                                    contact_ids(EE[3])};

                   auto coeff = sym::codim_ipc_contact::EE_contact_coeff(table, cids);
                   Alu  mu    = safe_cast<Alu>(coeff.mu);

                   const Vector3& rest_Ea0 = rest_x(EE[0]);
                   const Vector3& rest_Ea1 = rest_x(EE[1]);
                   const Vector3& rest_Eb0 = rest_x(EE[2]);
                   const Vector3& rest_Eb1 = rest_x(EE[3]);

                   const Vector3& prev_Ea0 = prev_x(EE[0]);
                   const Vector3& prev_Ea1 = prev_x(EE[1]);
                   const Vector3& prev_Eb0 = prev_x(EE[2]);
                   const Vector3& prev_Eb1 = prev_x(EE[3]);

                   const Vector3& Ea0 = x(EE[0]);
                   const Vector3& Ea1 = x(EE[1]);
                   const Vector3& Eb0 = x(EE[2]);
                   const Vector3& Eb1 = x(EE[3]);
                   Vec3A rest_Ea0_alu = rest_Ea0.template cast<Alu>();
                   Vec3A rest_Ea1_alu = rest_Ea1.template cast<Alu>();
                   Vec3A rest_Eb0_alu = rest_Eb0.template cast<Alu>();
                   Vec3A rest_Eb1_alu = rest_Eb1.template cast<Alu>();
                   Vec3A prev_Ea0_alu = prev_Ea0.template cast<Alu>();
                   Vec3A prev_Ea1_alu = prev_Ea1.template cast<Alu>();
                   Vec3A prev_Eb0_alu = prev_Eb0.template cast<Alu>();
                   Vec3A prev_Eb1_alu = prev_Eb1.template cast<Alu>();
                   Vec3A Ea0_alu      = Ea0.template cast<Alu>();
                   Vec3A Ea1_alu      = Ea1.template cast<Alu>();
                   Vec3A Eb0_alu      = Eb0.template cast<Alu>();
                   Vec3A Eb1_alu      = Eb1.template cast<Alu>();

                   Alu eps_x;
                   distance::edge_edge_mollifier_threshold(
                       rest_Ea0_alu,
                       rest_Ea1_alu,
                       rest_Eb0_alu,
                       rest_Eb1_alu,
                       static_cast<Alu>(-1.0),
                       eps_x);
                   if(!distance::need_mollify(prev_Ea0_alu,
                                              prev_Ea1_alu,
                                              prev_Eb0_alu,
                                              prev_Eb1_alu,
                                              eps_x))
                   {
                       Vec12A G;
                       Mat12A H;

                       EE_friction_gradient_hessian(G,
                                                    H,
                                                    mu,
                                                    safe_cast<Alu>(eps_v * dt),
                                                    safe_cast<Alu>(lambda(idx)),
                                                    prev_Ea0_alu,
                                                    prev_Ea1_alu,
                                                    prev_Eb0_alu,
                                                    prev_Eb1_alu,
                                                    Ea0_alu,
                                                    Ea1_alu,
                                                    Eb0_alu,
                                                    Eb1_alu);

                       structured_sink.template write_hessian_half<4>(EE, H);
                   }
               });
}

void ALSimplexFrictionalContact::do_compute_energy(GlobalContactManager::EnergyInfo& info)
{
    m_impl.do_compute_energy(info);
}

void ALSimplexFrictionalContact::do_assemble(GlobalContactManager::GradientHessianInfo& info)
{
    m_impl.do_assemble(info);
}

bool ALSimplexFrictionalContact::do_supports_structured_hessian() const
{
    return true;
}

void ALSimplexFrictionalContact::do_assemble_structured_hessian(
    GlobalDyTopoEffectManager::StructuredHessianInfo& info)
{
    m_impl.do_assemble_structured_hessian(info);
}

REGISTER_SIM_SYSTEM(ALSimplexFrictionalContact);
}  // namespace uipc::backend::cuda_mixed
