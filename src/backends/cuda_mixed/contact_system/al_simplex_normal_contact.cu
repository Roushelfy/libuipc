#include <contact_system/al_simplex_normal_contact.h>
#include <contact_system/al_contact_function.h>
#include <pipeline/al_ipc_pipeline_flag.h>
#include <utils/matrix_assembler.h>
#include <uipc/common/log.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
void ALSimplexNormalContact::do_build(ContactReporter::BuildInfo& info)
{
    require<ALIPCPipelineFlag>();
    m_impl.global_contact_manager = require<GlobalContactManager>();
    m_impl.global_vertex_manager  = require<GlobalVertexManager>();
    m_impl.global_surf_manager    = require<GlobalSimplicialSurfaceManager>();
    m_impl.global_active_set_manager = require<GlobalActiveSetManager>();
}

void ALSimplexNormalContact::do_report_energy_extent(GlobalContactManager::EnergyExtentInfo& info)
{
    auto& active_set = m_impl.global_active_set_manager;

    if(!active_set->is_enabled())
    {
        info.energy_count(0);
        return;
    }

    info.energy_count(active_set->PTs().size() + active_set->EEs().size());
}

void ALSimplexNormalContact::do_report_gradient_hessian_extent(GlobalContactManager::GradientHessianExtentInfo& info)
{
    auto& active_set = m_impl.global_active_set_manager;

    if(!active_set->is_enabled())
    {
        info.gradient_count(0);
        info.hessian_count(0);
        return;
    }

    info.gradient_count(4 * (active_set->PTs().size() + active_set->EEs().size()));
    info.hessian_count(10 * (active_set->PTs().size() + active_set->EEs().size()));
}

void ALSimplexNormalContact::Impl::do_compute_energy(GlobalContactManager::EnergyInfo& info)
{
    using namespace muda;
    using namespace sym::al_simplex_contact;
    using Alu = ActivePolicy::AluScalar;
    using Vec3A = Eigen::Matrix<Alu, 3, 1>;
    using Vec12A = Eigen::Matrix<Alu, 12, 1>;
    auto& active_set = global_active_set_manager;

    if(!active_set->is_enabled())
        return;

    auto PT_size = active_set->PTs().size(), EE_size = active_set->EEs().size();
    auto PT_energies = info.energies().subview(0, PT_size);
    auto EE_energies = info.energies().subview(PT_size, EE_size);

    auto x   = global_vertex_manager->positions();
    auto PTs = active_set->PTs(), EEs = active_set->EEs();
    auto PT_d0 = active_set->PT_d0(), EE_d0 = active_set->EE_d0();
    auto PT_cnt = active_set->PT_cnt(), EE_cnt = active_set->EE_cnt();
    auto PT_d_grad = active_set->PT_d_grad();
    auto EE_d_grad = active_set->EE_d_grad();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(PT_size,
               [mu_v   = active_set->mu_vertices().cviewer().name("mu_v"),
                decay  = active_set->decay_factor(),
                PTs    = PTs.cviewer().name("PTs"),
                cnt    = PT_cnt.cviewer().name("cnt"),
                d0     = PT_d0.cviewer().name("d0"),
               d_grad = PT_d_grad.cviewer().name("d_grad"),
                x      = x.cviewer().name("x"),
                Es = PT_energies.viewer().name("Es")] __device__(int idx) mutable
               {
                   auto PT = PTs(idx);
                   Vec12A d_grad_alu = d_grad(idx).template cast<Alu>();
                   Vec3A  P0         = x(PT(0)).template cast<Alu>();
                   Vec3A  P1         = x(PT(1)).template cast<Alu>();
                   Vec3A  P2         = x(PT(2)).template cast<Alu>();
                   Vec3A  P3         = x(PT(3)).template cast<Alu>();
                   Alu mu = safe_cast<Alu>(min(min(mu_v(PT(0)), mu_v(PT(1))),
                                               min(mu_v(PT(2)), mu_v(PT(3)))));
                   auto c    = cnt(idx) >= 0 ? cnt(idx) : max(-cnt(idx) - 6, 0);
                   Alu  scale = safe_cast<Alu>(pow(safe_cast<Alu>(decay), c)) * mu;
                   Es(idx) = safe_cast<Float>(
                       penalty_energy(scale,
                                      safe_cast<Alu>(d0(idx)),
                                      d_grad_alu,
                                      P0,
                                      P1,
                                      P2,
                                      P3));
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(EE_size,
               [mu_v   = active_set->mu_vertices().cviewer().name("mu_v"),
                decay  = active_set->decay_factor(),
                EEs    = EEs.cviewer().name("EEs"),
                cnt    = EE_cnt.cviewer().name("cnt"),
                d0     = EE_d0.cviewer().name("d0"),
                d_grad = EE_d_grad.cviewer().name("d_grad"),
                x      = x.cviewer().name("x"),
                Es = EE_energies.viewer().name("Es")] __device__(int idx) mutable
               {
                   auto EE = EEs(idx);
                   Vec12A d_grad_alu = d_grad(idx).template cast<Alu>();
                   Vec3A  P0         = x(EE(0)).template cast<Alu>();
                   Vec3A  P1         = x(EE(1)).template cast<Alu>();
                   Vec3A  P2         = x(EE(2)).template cast<Alu>();
                   Vec3A  P3         = x(EE(3)).template cast<Alu>();
                   Alu mu = safe_cast<Alu>(min(min(mu_v(EE(0)), mu_v(EE(1))),
                                               min(mu_v(EE(2)), mu_v(EE(3)))));
                   auto c    = cnt(idx) >= 0 ? cnt(idx) : max(-cnt(idx) - 6, 0);
                   Alu  scale = safe_cast<Alu>(pow(safe_cast<Alu>(decay), c)) * mu;
                   Es(idx) = safe_cast<Float>(
                       penalty_energy(scale,
                                      safe_cast<Alu>(d0(idx)),
                                      d_grad_alu,
                                      P0,
                                      P1,
                                      P2,
                                      P3));
               });
}

void ALSimplexNormalContact::Impl::do_assemble(GlobalContactManager::GradientHessianInfo& info)
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

    auto PT_size = active_set->PTs().size(), EE_size = active_set->EEs().size();
    auto PT_grad = info.gradients().subview(0, PT_size * 4);
    auto EE_grad = info.gradients().subview(PT_size * 4, EE_size * 4);
    auto PT_hess = info.hessians().subview(0, PT_size * 10);
    auto EE_hess = info.hessians().subview(PT_size * 10, EE_size * 10);

    auto x   = global_vertex_manager->positions();
    auto PTs = active_set->PTs(), EEs = active_set->EEs();
    auto PT_d0 = active_set->PT_d0(), EE_d0 = active_set->EE_d0();
    auto PT_cnt = active_set->PT_cnt(), EE_cnt = active_set->EE_cnt();
    auto PT_d_grad = active_set->PT_d_grad();
    auto EE_d_grad = active_set->EE_d_grad();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(PT_size,
               [mu_v   = active_set->mu_vertices().cviewer().name("mu_v"),
                decay  = active_set->decay_factor(),
                PTs    = PTs.cviewer().name("PTs"),
                cnt    = PT_cnt.cviewer().name("cnt"),
                d0     = PT_d0.cviewer().name("d0"),
                d_grad = PT_d_grad.cviewer().name("d_grad"),
                x      = x.cviewer().name("x"),
                Gs     = PT_grad.viewer().name("Gs"),
                Hs = PT_hess.viewer().name("Hs")] __device__(int idx) mutable
               {
                   auto        PT = PTs(idx);
                   Vec12A      d_grad_alu = d_grad(idx).template cast<Alu>();
                   Vec3A       P0         = x(PT(0)).template cast<Alu>();
                   Vec3A       P1         = x(PT(1)).template cast<Alu>();
                   Vec3A       P2         = x(PT(2)).template cast<Alu>();
                   Vec3A       P3         = x(PT(3)).template cast<Alu>();
                   Alu         mu = safe_cast<Alu>(min(min(mu_v(PT(0)), mu_v(PT(1))),
                                                       min(mu_v(PT(2)), mu_v(PT(3)))));
                   Vec12A      G;
                   Mat12A      H;
                   auto c    = cnt(idx) >= 0 ? cnt(idx) : max(-cnt(idx) - 6, 0);
                   Alu  scale = safe_cast<Alu>(pow(safe_cast<Alu>(decay), c)) * mu;
                   penalty_gradient_hessian(scale,
                                            safe_cast<Alu>(d0(idx)),
                                            d_grad_alu,
                                            P0,
                                            P1,
                                            P2,
                                            P3,
                                            G,
                                            H);

                   DoubletVectorAssembler DVA{Gs};
                   DVA.segment<4>(idx * 4).write(PT, downcast_gradient<Store>(G));

                   TripletMatrixAssembler TMA{Hs};
                   TMA.half_block<4>(idx * 10).write(PT, downcast_hessian<Store>(H));
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(EE_size,
               [mu_v   = active_set->mu_vertices().cviewer().name("mu_v"),
                decay  = active_set->decay_factor(),
                EEs    = EEs.cviewer().name("EEs"),
                cnt    = EE_cnt.cviewer().name("cnt"),
                d0     = EE_d0.cviewer().name("d0"),
                d_grad = EE_d_grad.cviewer().name("d_grad"),
                x      = x.cviewer().name("x"),
                Gs     = EE_grad.viewer().name("Gs"),
                Hs = EE_hess.viewer().name("Hs")] __device__(int idx) mutable
               {
                   auto        EE = EEs(idx);
                   Vec12A      d_grad_alu = d_grad(idx).template cast<Alu>();
                   Vec3A       P0         = x(EE(0)).template cast<Alu>();
                   Vec3A       P1         = x(EE(1)).template cast<Alu>();
                   Vec3A       P2         = x(EE(2)).template cast<Alu>();
                   Vec3A       P3         = x(EE(3)).template cast<Alu>();
                   Alu         mu = safe_cast<Alu>(min(min(mu_v(EE(0)), mu_v(EE(1))),
                                                       min(mu_v(EE(2)), mu_v(EE(3)))));
                   Vec12A      G;
                   Mat12A      H;
                   auto c    = cnt(idx) >= 0 ? cnt(idx) : max(-cnt(idx) - 6, 0);
                   Alu  scale = safe_cast<Alu>(pow(safe_cast<Alu>(decay), c)) * mu;
                   penalty_gradient_hessian(scale,
                                            safe_cast<Alu>(d0(idx)),
                                            d_grad_alu,
                                            P0,
                                            P1,
                                            P2,
                                            P3,
                                            G,
                                            H);

                   DoubletVectorAssembler DVA{Gs};
                   DVA.segment<4>(idx * 4).write(EE, downcast_gradient<Store>(G));

                   TripletMatrixAssembler TMA{Hs};
                   TMA.half_block<4>(idx * 10).write(EE, downcast_hessian<Store>(H));
               });
}

void ALSimplexNormalContact::do_compute_energy(GlobalContactManager::EnergyInfo& info)
{
    m_impl.do_compute_energy(info);
}

void ALSimplexNormalContact::do_assemble(GlobalContactManager::GradientHessianInfo& info)
{
    m_impl.do_assemble(info);
}

REGISTER_SIM_SYSTEM(ALSimplexNormalContact);
}  // namespace uipc::backend::cuda_mixed
