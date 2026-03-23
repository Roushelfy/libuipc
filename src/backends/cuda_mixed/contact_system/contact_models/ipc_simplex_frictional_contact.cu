#include <contact_system/simplex_frictional_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>
#include <utils/codim_thickness.h>
#include <kernel_cout.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>
#include <utils/primitive_d_hat.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>


namespace uipc::backend::cuda_mixed
{
class IPCSimplexFrictionalContact final : public SimplexFrictionalContact
{
  public:
    using SimplexFrictionalContact::SimplexFrictionalContact;

    virtual void do_build(BuildInfo& info) override
    {
        auto constitution =
            world().scene().config().find<std::string>("contact/constitution");
        if(constitution->view()[0] != "ipc")
        {
            throw SimSystemException("Constitution is not IPC");
        }
    }

    virtual void do_compute_energy(EnergyInfo& info) override
    {
        using namespace muda;
        using namespace sym::codim_ipc_contact;
        using Alu = ActivePolicy::AluScalar;
        using Vec3A = Eigen::Matrix<Alu, 3, 1>;

        // Compute Point-Triangle energy
        auto PT_count = info.friction_PTs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PT_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PTs     = info.friction_PTs().viewer().name("PTs"),
                    Es      = info.friction_PT_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    eps_v  = info.eps_velocity(),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       const auto& PT = PTs(i);

                       Vector4i cids = {contact_ids(PT[0]),
                                        contact_ids(PT[1]),
                                        contact_ids(PT[2]),
                                        contact_ids(PT[3])};

                       auto coeff = PT_contact_coeff(table, cids);
                       Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                       Alu  mu    = safe_cast<Alu>(coeff.mu);

                       const auto& prev_P  = prev_Ps(PT[0]);
                       const auto& prev_T0 = prev_Ps(PT[1]);
                       const auto& prev_T1 = prev_Ps(PT[2]);
                       const auto& prev_T2 = prev_Ps(PT[3]);

                       const auto& P  = Ps(PT[0]);
                       const auto& T0 = Ps(PT[1]);
                       const auto& T1 = Ps(PT[2]);
                       const auto& T2 = Ps(PT[3]);
                       Vec3A       prev_P_alu  = prev_P.template cast<Alu>();
                       Vec3A       prev_T0_alu = prev_T0.template cast<Alu>();
                       Vec3A       prev_T1_alu = prev_T1.template cast<Alu>();
                       Vec3A       prev_T2_alu = prev_T2.template cast<Alu>();
                       Vec3A       P_alu       = P.template cast<Alu>();
                       Vec3A       T0_alu      = T0.template cast<Alu>();
                       Vec3A       T1_alu      = T1.template cast<Alu>();
                       Vec3A       T2_alu      = T2.template cast<Alu>();


                       Alu thickness = safe_cast<Alu>(PT_thickness(thicknesses(PT[0]),
                                                                   thicknesses(PT[1]),
                                                                   thicknesses(PT[2]),
                                                                   thicknesses(PT[3])));
                       Alu d_hat     = safe_cast<Alu>(PT_d_hat(
                           d_hats(PT[0]), d_hats(PT[1]), d_hats(PT[2]), d_hats(PT[3])));


                       Es(i) = safe_cast<Float>(PT_friction_energy(
                           kt2,
                           d_hat,
                           thickness,
                           mu,
                           safe_cast<Alu>(eps_v * dt),
                           prev_P_alu,
                           prev_T0_alu,
                           prev_T1_alu,
                           prev_T2_alu,
                           P_alu,
                           T0_alu,
                           T1_alu,
                           T2_alu));
                   });

        // Compute Edge-Edge energy
        auto EE_count = info.friction_EEs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(EE_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    EEs     = info.friction_EEs().viewer().name("EEs"),
                    Es      = info.friction_EE_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    eps_v   = info.eps_velocity(),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       const auto& EE = EEs(i);

                       Vector4i cids = {contact_ids(EE[0]),
                                        contact_ids(EE[1]),
                                        contact_ids(EE[2]),
                                        contact_ids(EE[3])};

                       auto coeff = EE_contact_coeff(table, cids);
                       Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                       Alu  mu    = safe_cast<Alu>(coeff.mu);

                       const Vector3& rest_Ea0 = rest_Ps(EE[0]);
                       const Vector3& rest_Ea1 = rest_Ps(EE[1]);
                       const Vector3& rest_Eb0 = rest_Ps(EE[2]);
                       const Vector3& rest_Eb1 = rest_Ps(EE[3]);

                       const Vector3& prev_Ea0 = prev_Ps(EE[0]);
                       const Vector3& prev_Ea1 = prev_Ps(EE[1]);
                       const Vector3& prev_Eb0 = prev_Ps(EE[2]);
                       const Vector3& prev_Eb1 = prev_Ps(EE[3]);

                       const Vector3& Ea0 = Ps(EE[0]);
                       const Vector3& Ea1 = Ps(EE[1]);
                       const Vector3& Eb0 = Ps(EE[2]);
                       const Vector3& Eb1 = Ps(EE[3]);
                       Vec3A         prev_Ea0_alu = prev_Ea0.template cast<Alu>();
                       Vec3A         prev_Ea1_alu = prev_Ea1.template cast<Alu>();
                       Vec3A         prev_Eb0_alu = prev_Eb0.template cast<Alu>();
                       Vec3A         prev_Eb1_alu = prev_Eb1.template cast<Alu>();
                       Vec3A         Ea0_alu      = Ea0.template cast<Alu>();
                       Vec3A         Ea1_alu      = Ea1.template cast<Alu>();
                       Vec3A         Eb0_alu      = Eb0.template cast<Alu>();
                       Vec3A         Eb1_alu      = Eb1.template cast<Alu>();

                       Alu thickness = safe_cast<Alu>(EE_thickness(thicknesses(EE[0]),
                                                                   thicknesses(EE[1]),
                                                                   thicknesses(EE[2]),
                                                                   thicknesses(EE[3])));

                       Alu d_hat = safe_cast<Alu>(EE_d_hat(
                           d_hats(EE[0]), d_hats(EE[1]), d_hats(EE[2]), d_hats(EE[3])));

                       Float eps_x;
                       distance::edge_edge_mollifier_threshold(rest_Ea0,
                                                               rest_Ea1,
                                                               rest_Eb0,
                                                               rest_Eb1,
                                                               static_cast<Float>(1e-3),
                                                               eps_x);
                       if(distance::need_mollify(prev_Ea0,
                                                 prev_Ea1,
                                                 prev_Eb0,
                                                 prev_Eb1,
                                                 eps_x))
                       // almost parallel, don't compute energy
                       {
                           Es(i) = 0;
                       }
                       else
                       {
                           Es(i) = safe_cast<Float>(EE_friction_energy(
                               kt2,
                               d_hat,
                               thickness,
                               mu,
                               safe_cast<Alu>(eps_v * dt),
                               prev_Ea0_alu,
                               prev_Ea1_alu,
                               prev_Eb0_alu,
                               prev_Eb1_alu,
                               Ea0_alu,
                               Ea1_alu,
                               Eb0_alu,
                               Eb1_alu));
                       }
                   });

        // Compute Point-Edge energy
        auto PE_count = info.friction_PEs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PE_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PEs     = info.friction_PEs().viewer().name("PEs"),
                    Es      = info.friction_PE_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    eps_v  = info.eps_velocity(),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       const auto& PE = PEs(i);

                       Vector3i cids = {contact_ids(PE[0]),
                                        contact_ids(PE[1]),
                                        contact_ids(PE[2])};

                       auto coeff = PE_contact_coeff(table, cids);
                       Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                       Alu  mu    = safe_cast<Alu>(coeff.mu);

                       const Vector3& prev_P  = prev_Ps(PE[0]);
                       const Vector3& prev_E0 = prev_Ps(PE[1]);
                       const Vector3& prev_E1 = prev_Ps(PE[2]);

                       const Vector3& P  = Ps(PE[0]);
                       const Vector3& E0 = Ps(PE[1]);
                       const Vector3& E1 = Ps(PE[2]);
                       Vec3A         prev_P_alu  = prev_P.template cast<Alu>();
                       Vec3A         prev_E0_alu = prev_E0.template cast<Alu>();
                       Vec3A         prev_E1_alu = prev_E1.template cast<Alu>();
                       Vec3A         P_alu       = P.template cast<Alu>();
                       Vec3A         E0_alu      = E0.template cast<Alu>();
                       Vec3A         E1_alu      = E1.template cast<Alu>();

                       Alu thickness = safe_cast<Alu>(PE_thickness(thicknesses(PE[0]),
                                                                   thicknesses(PE[1]),
                                                                   thicknesses(PE[2])));

                       Alu d_hat =
                           safe_cast<Alu>(PE_d_hat(d_hats(PE[0]), d_hats(PE[1]), d_hats(PE[2])));

                       Es(i) = safe_cast<Float>(PE_friction_energy(
                           kt2,
                           d_hat,
                           thickness,
                           mu,
                           safe_cast<Alu>(eps_v * dt),
                           prev_P_alu,
                           prev_E0_alu,
                           prev_E1_alu,
                           P_alu,
                           E0_alu,
                           E1_alu));
                   });

        // Compute Point-Point energy
        auto PP_count = info.friction_PPs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PP_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PPs     = info.friction_PPs().viewer().name("PPs"),
                    Es      = info.friction_PP_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    eps_v  = info.eps_velocity(),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       const auto& PP = PPs(i);

                       Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                       auto coeff = PP_contact_coeff(table, cids);
                       Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                       Alu  mu    = safe_cast<Alu>(coeff.mu);

                       const Vector3& prev_P0 = prev_Ps(PP[0]);
                       const Vector3& prev_P1 = prev_Ps(PP[1]);

                       const Vector3& P0 = Ps(PP[0]);
                       const Vector3& P1 = Ps(PP[1]);
                       Vec3A         prev_P0_alu = prev_P0.template cast<Alu>();
                       Vec3A         prev_P1_alu = prev_P1.template cast<Alu>();
                       Vec3A         P0_alu      = P0.template cast<Alu>();
                       Vec3A         P1_alu      = P1.template cast<Alu>();

                       Alu thickness =
                           safe_cast<Alu>(PP_thickness(thicknesses(PP[0]), thicknesses(PP[1])));

                       Alu d_hat = safe_cast<Alu>(PP_d_hat(d_hats(PP[0]), d_hats(PP[1])));

                       Es(i) = safe_cast<Float>(PP_friction_energy(
                           kt2,
                           d_hat,
                           thickness,
                           mu,
                           safe_cast<Alu>(eps_v * dt),
                           prev_P0_alu,
                           prev_P1_alu,
                           P0_alu,
                           P1_alu));
                   });
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace muda;
        using namespace sym::codim_ipc_contact;
        using Alu   = ActivePolicy::AluScalar;
        using Store = ActivePolicy::StoreScalar;
        using Vec3A = Eigen::Matrix<Alu, 3, 1>;
        using Vec12A = Eigen::Matrix<Alu, 12, 1>;
        using Mat12A = Eigen::Matrix<Alu, 12, 12>;
        using Vec9A = Eigen::Matrix<Alu, 9, 1>;
        using Mat9A = Eigen::Matrix<Alu, 9, 9>;
        using Vec6A = Eigen::Matrix<Alu, 6, 1>;
        using Mat6A = Eigen::Matrix<Alu, 6, 6>;

        // Compute Point-Triangle Gradient and Hessian
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.friction_PTs().size(),
                   [gradient_only = info.gradient_only(),
                    table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PTs     = info.friction_PTs().viewer().name("PTs"),
                    Gs      = info.friction_PT_gradients().viewer().name("Gs"),
                    Hs      = info.friction_PT_hessians().viewer().name("Hs"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    eps_v  = info.eps_velocity(),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       const auto& PT = PTs(i);

                       Vector4i cids = {contact_ids(PT[0]),
                                        contact_ids(PT[1]),
                                        contact_ids(PT[2]),
                                        contact_ids(PT[3])};

                       auto coeff = PT_contact_coeff(table, cids);
                       Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                       Alu  mu    = safe_cast<Alu>(coeff.mu);

                       const auto& prev_P  = prev_Ps(PT[0]);
                       const auto& prev_T0 = prev_Ps(PT[1]);
                       const auto& prev_T1 = prev_Ps(PT[2]);
                       const auto& prev_T2 = prev_Ps(PT[3]);

                       const auto& P  = Ps(PT[0]);
                       const auto& T0 = Ps(PT[1]);
                       const auto& T1 = Ps(PT[2]);
                       const auto& T2 = Ps(PT[3]);
                       Vec3A       prev_P_alu  = prev_P.template cast<Alu>();
                       Vec3A       prev_T0_alu = prev_T0.template cast<Alu>();
                       Vec3A       prev_T1_alu = prev_T1.template cast<Alu>();
                       Vec3A       prev_T2_alu = prev_T2.template cast<Alu>();
                       Vec3A       P_alu       = P.template cast<Alu>();
                       Vec3A       T0_alu      = T0.template cast<Alu>();
                       Vec3A       T1_alu      = T1.template cast<Alu>();
                       Vec3A       T2_alu      = T2.template cast<Alu>();


                       Alu thickness = safe_cast<Alu>(PT_thickness(thicknesses(PT[0]),
                                                                   thicknesses(PT[1]),
                                                                   thicknesses(PT[2]),
                                                                   thicknesses(PT[3])));

                       Alu d_hat = safe_cast<Alu>(PT_d_hat(
                           d_hats(PT[0]), d_hats(PT[1]), d_hats(PT[2]), d_hats(PT[3])));

                       Vec12A G;
                       if(gradient_only)
                       {
                            PT_friction_gradient(G,
                                                 kt2,
                                                 d_hat,
                                                 thickness,
                                                 mu,
                                                 safe_cast<Alu>(eps_v * dt),
                                                 prev_P_alu,
                                                 prev_T0_alu,
                                                 prev_T1_alu,
                                                 prev_T2_alu,
                                                 P_alu,
                                                 T0_alu,
                                                 T1_alu,
                                                 T2_alu);
                            DoubletVectorAssembler DVA{Gs};
                            auto G_store = downcast_gradient<Store>(G);
                            DVA.segment<4>(i * 4).write(PT, G_store);
                       }
                       else
                       {
                           Mat12A H;
                            PT_friction_gradient_hessian(G,
                                                         H,
                                                         kt2,
                                                         d_hat,
                                                         thickness,
                                                         mu,
                                                         safe_cast<Alu>(eps_v * dt),
                                                         prev_P_alu,
                                                         prev_T0_alu,
                                                         prev_T1_alu,
                                                         prev_T2_alu,
                                                         P_alu,
                                                         T0_alu,
                                                         T1_alu,
                                                         T2_alu);
                            cuda_mixed::make_spd(H);
                            DoubletVectorAssembler DVA{Gs};
                            auto G_store = downcast_gradient<Store>(G);
                            DVA.segment<4>(i * 4).write(PT, G_store);
                            TripletMatrixAssembler TMA{Hs};
                            auto H_store = downcast_hessian<Store>(H);
                            TMA.half_block<4>(i * PTHalfHessianSize).write(PT, H_store);
                       }
                   });

        // Compute Edge-Edge Gradient and Hessian
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.friction_EEs().size(),
                   [gradient_only = info.gradient_only(),
                    table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    EEs     = info.friction_EEs().viewer().name("EEs"),
                    Gs      = info.friction_EE_gradients().viewer().name("Gs"),
                    Hs      = info.friction_EE_hessians().viewer().name("Hs"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    eps_v  = info.eps_velocity(),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       const auto& EE = EEs(i);

                       Vector4i cids = {contact_ids(EE[0]),
                                        contact_ids(EE[1]),
                                        contact_ids(EE[2]),
                                        contact_ids(EE[3])};

                       auto coeff = EE_contact_coeff(table, cids);
                       Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                       Alu  mu    = safe_cast<Alu>(coeff.mu);

                       const Vector3& rest_Ea0 = rest_Ps(EE[0]);
                       const Vector3& rest_Ea1 = rest_Ps(EE[1]);
                       const Vector3& rest_Eb0 = rest_Ps(EE[2]);
                       const Vector3& rest_Eb1 = rest_Ps(EE[3]);

                       const Vector3& prev_Ea0 = prev_Ps(EE[0]);
                       const Vector3& prev_Ea1 = prev_Ps(EE[1]);
                       const Vector3& prev_Eb0 = prev_Ps(EE[2]);
                       const Vector3& prev_Eb1 = prev_Ps(EE[3]);

                       const Vector3& Ea0 = Ps(EE[0]);
                       const Vector3& Ea1 = Ps(EE[1]);
                       const Vector3& Eb0 = Ps(EE[2]);
                       const Vector3& Eb1 = Ps(EE[3]);
                       Vec3A         prev_Ea0_alu = prev_Ea0.template cast<Alu>();
                       Vec3A         prev_Ea1_alu = prev_Ea1.template cast<Alu>();
                       Vec3A         prev_Eb0_alu = prev_Eb0.template cast<Alu>();
                       Vec3A         prev_Eb1_alu = prev_Eb1.template cast<Alu>();
                       Vec3A         Ea0_alu      = Ea0.template cast<Alu>();
                       Vec3A         Ea1_alu      = Ea1.template cast<Alu>();
                       Vec3A         Eb0_alu      = Eb0.template cast<Alu>();
                       Vec3A         Eb1_alu      = Eb1.template cast<Alu>();

                       Alu thickness = safe_cast<Alu>(EE_thickness(thicknesses(EE[0]),
                                                                   thicknesses(EE[1]),
                                                                   thicknesses(EE[2]),
                                                                   thicknesses(EE[3])));

                       Alu d_hat = safe_cast<Alu>(EE_d_hat(
                           d_hats(EE[0]), d_hats(EE[1]), d_hats(EE[2]), d_hats(EE[3])));

                       Float eps_x;
                       distance::edge_edge_mollifier_threshold(rest_Ea0,
                                                               rest_Ea1,
                                                               rest_Eb0,
                                                               rest_Eb1,
                                                               static_cast<Float>(1e-3),
                                                               eps_x);

                       Vec12A G;

                       bool mollified = distance::need_mollify(
                           prev_Ea0,
                           prev_Ea1,
                           prev_Eb0,
                           prev_Eb1,
                           eps_x);

                       if(gradient_only)
                       {
                           if(mollified)
                           {
                               G.setZero();
                           }
                           else
                           {
                               EE_friction_gradient(G,
                                                    kt2,
                                                    d_hat,
                                                    thickness,
                                                    mu,
                                                    safe_cast<Alu>(eps_v * dt),
                                                    prev_Ea0_alu,
                                                    prev_Ea1_alu,
                                                    prev_Eb0_alu,
                                                    prev_Eb1_alu,
                                                    Ea0_alu,
                                                    Ea1_alu,
                                                    Eb0_alu,
                                                    Eb1_alu);
                            }
                            DoubletVectorAssembler DVA{Gs};
                            auto G_store = downcast_gradient<Store>(G);
                            DVA.segment<4>(i * 4).write(EE, G_store);
                       }
                       else
                       {
                           Mat12A H;
                           if(mollified)
                           {
                               G.setZero();
                               H.setZero();
                           }
                           else
                           {
                                EE_friction_gradient_hessian(G,
                                                             H,
                                                             kt2,
                                                             d_hat,
                                                             thickness,
                                                             mu,
                                                             safe_cast<Alu>(eps_v * dt),
                                                             prev_Ea0_alu,
                                                             prev_Ea1_alu,
                                                             prev_Eb0_alu,
                                                             prev_Eb1_alu,
                                                             Ea0_alu,
                                                             Ea1_alu,
                                                             Eb0_alu,
                                                             Eb1_alu);
                                cuda_mixed::make_spd(H);
                            }
                            DoubletVectorAssembler DVA{Gs};
                            auto G_store = downcast_gradient<Store>(G);
                            DVA.segment<4>(i * 4).write(EE, G_store);
                            TripletMatrixAssembler TMA{Hs};
                            auto H_store = downcast_hessian<Store>(H);
                            TMA.half_block<4>(i * EEHalfHessianSize).write(EE, H_store);
                       }
                   });

        // Compute Point-Edge Gradient and Hessian
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.friction_PEs().size(),
                   [gradient_only = info.gradient_only(),
                    table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PEs     = info.friction_PEs().viewer().name("PEs"),
                    Gs      = info.friction_PE_gradients().viewer().name("Gs"),
                    Hs      = info.friction_PE_hessians().viewer().name("Hs"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    eps_v  = info.eps_velocity(),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       const auto& PE = PEs(i);

                       Vector3i cids = {contact_ids(PE[0]),
                                        contact_ids(PE[1]),
                                        contact_ids(PE[2])};

                       auto coeff = PE_contact_coeff(table, cids);
                       Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                       Alu  mu    = safe_cast<Alu>(coeff.mu);

                       const Vector3& prev_P  = prev_Ps(PE[0]);
                       const Vector3& prev_E0 = prev_Ps(PE[1]);
                       const Vector3& prev_E1 = prev_Ps(PE[2]);

                       const Vector3& P  = Ps(PE[0]);
                       const Vector3& E0 = Ps(PE[1]);
                       const Vector3& E1 = Ps(PE[2]);
                       Vec3A         prev_P_alu  = prev_P.template cast<Alu>();
                       Vec3A         prev_E0_alu = prev_E0.template cast<Alu>();
                       Vec3A         prev_E1_alu = prev_E1.template cast<Alu>();
                       Vec3A         P_alu       = P.template cast<Alu>();
                       Vec3A         E0_alu      = E0.template cast<Alu>();
                       Vec3A         E1_alu      = E1.template cast<Alu>();

                       Alu thickness = safe_cast<Alu>(PE_thickness(thicknesses(PE[0]),
                                                                   thicknesses(PE[1]),
                                                                   thicknesses(PE[2])));

                       Alu d_hat =
                           safe_cast<Alu>(PE_d_hat(d_hats(PE[0]), d_hats(PE[1]), d_hats(PE[2])));

                       Vec9A G;
                       if(gradient_only)
                       {
                            PE_friction_gradient(G,
                                                 kt2,
                                                 d_hat,
                                                 thickness,
                                                 mu,
                                                 safe_cast<Alu>(eps_v * dt),
                                                 prev_P_alu,
                                                 prev_E0_alu,
                                                 prev_E1_alu,
                                                 P_alu,
                                                 E0_alu,
                                                 E1_alu);
                            DoubletVectorAssembler DVA{Gs};
                            auto G_store = downcast_gradient<Store>(G);
                            DVA.segment<3>(i * 3).write(PE, G_store);
                       }
                       else
                       {
                           Mat9A H;
                            PE_friction_gradient_hessian(G,
                                                         H,
                                                         kt2,
                                                         d_hat,
                                                         thickness,
                                                         mu,
                                                         safe_cast<Alu>(eps_v * dt),
                                                         prev_P_alu,
                                                         prev_E0_alu,
                                                         prev_E1_alu,
                                                         P_alu,
                                                         E0_alu,
                                                         E1_alu);
                            cuda_mixed::make_spd(H);
                            DoubletVectorAssembler DVA{Gs};
                            auto G_store = downcast_gradient<Store>(G);
                            DVA.segment<3>(i * 3).write(PE, G_store);
                            TripletMatrixAssembler TMA{Hs};
                            auto H_store = downcast_hessian<Store>(H);
                            TMA.half_block<3>(i * PEHalfHessianSize).write(PE, H_store);
                       }
                   });

        // Compute Point-Point Gradient and Hessian
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.friction_PPs().size(),
                   [gradient_only = info.gradient_only(),
                    table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PPs     = info.friction_PPs().viewer().name("PPs"),
                    Gs      = info.friction_PP_gradients().viewer().name("Gs"),
                    Hs      = info.friction_PP_hessians().viewer().name("Hs"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    eps_v  = info.eps_velocity(),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       const auto& PP = PPs(i);

                       Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                       auto coeff = PP_contact_coeff(table, cids);
                       Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                       Alu  mu    = safe_cast<Alu>(coeff.mu);

                       const Vector3& prev_P0 = prev_Ps(PP[0]);
                       const Vector3& prev_P1 = prev_Ps(PP[1]);

                       const Vector3& P0 = Ps(PP[0]);
                       const Vector3& P1 = Ps(PP[1]);
                       Vec3A         prev_P0_alu = prev_P0.template cast<Alu>();
                       Vec3A         prev_P1_alu = prev_P1.template cast<Alu>();
                       Vec3A         P0_alu      = P0.template cast<Alu>();
                       Vec3A         P1_alu      = P1.template cast<Alu>();

                       Alu thickness =
                           safe_cast<Alu>(PP_thickness(thicknesses(PP[0]), thicknesses(PP[1])));

                       Alu d_hat = safe_cast<Alu>(PP_d_hat(d_hats(PP[0]), d_hats(PP[1])));

                       Vec6A G;
                       if(gradient_only)
                       {
                            PP_friction_gradient(G,
                                                 kt2,
                                                 d_hat,
                                                 thickness,
                                                 mu,
                                                 safe_cast<Alu>(eps_v * dt),
                                                 prev_P0_alu,
                                                 prev_P1_alu,
                                                 P0_alu,
                                                 P1_alu);
                            DoubletVectorAssembler DVA{Gs};
                            auto G_store = downcast_gradient<Store>(G);
                            DVA.segment<2>(i * 2).write(PP, G_store);
                       }
                       else
                       {
                           Mat6A H;
                            PP_friction_gradient_hessian(G,
                                                         H,
                                                         kt2,
                                                         d_hat,
                                                         thickness,
                                                         mu,
                                                         safe_cast<Alu>(eps_v * dt),
                                                         prev_P0_alu,
                                                         prev_P1_alu,
                                                         P0_alu,
                                                         P1_alu);
                            cuda_mixed::make_spd(H);
                            DoubletVectorAssembler DVA{Gs};
                            auto G_store = downcast_gradient<Store>(G);
                            DVA.segment<2>(i * 2).write(PP, G_store);
                            TripletMatrixAssembler TMA{Hs};
                            auto H_store = downcast_hessian<Store>(H);
                            TMA.half_block<2>(i * PPHalfHessianSize).write(PP, H_store);
                        }
                    });
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexFrictionalContact);
}  // namespace uipc::backend::cuda_mixed
