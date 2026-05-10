#include <contact_system/contact_models/ipc_simplex_frictional_contact_native.h>
#include <contact_system/contact_models/ipc_simplex_normal_contact_native_common.inl>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>
#include <mixed_precision/cast.h>
#include <mixed_precision/policy.h>
#include <utils/codim_thickness.h>
#include <utils/distance/edge_edge_mollifier.h>
#include <utils/make_spd.h>
#include <utils/primitive_d_hat.h>

namespace uipc::backend::cuda_mixed
{
void assemble_ipc_simplex_frictional_contact_native_exact_EE(
    SimplexFrictionalContact::ContactInfo& info)
{
    using namespace muda;
    using namespace sym::codim_ipc_contact;
    using Alu    = ActivePolicy::AluScalar;
    using Store  = ActivePolicy::StoreScalar;
    using Vec3A  = Eigen::Matrix<Alu, 3, 1>;
    using Vec12A = Eigen::Matrix<Alu, 12, 1>;
    using Mat12A = Eigen::Matrix<Alu, 12, 12>;

    if(info.friction_EEs().size() == 0)
        return;

    const auto structured_sink = info.structured_hessian_sink();
    const auto targets         = info.friction_EE_native_contact_targets();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.friction_EEs().size(),
               [structured_sink,
                targets,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                EEs     = info.friction_EEs().viewer().name("friction_EEs"),
                Ps      = info.positions().viewer().name("Ps"),
                prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats = info.d_hats().viewer().name("d_hats"),
                eps_v  = info.eps_velocity(),
                dt     = info.dt()] __device__(int i) mutable
               {
                   const Vector4i EE = EEs(i);

                   Vector4i cids = {contact_ids(EE[0]),
                                    contact_ids(EE[1]),
                                    contact_ids(EE[2]),
                                    contact_ids(EE[3])};

                   const auto coeff = EE_contact_coeff(table, cids);
                   const Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                   const Alu  mu    = safe_cast<Alu>(coeff.mu);

                   const Vec3A rest_Ea0 = rest_Ps(EE[0]).template cast<Alu>();
                   const Vec3A rest_Ea1 = rest_Ps(EE[1]).template cast<Alu>();
                   const Vec3A rest_Eb0 = rest_Ps(EE[2]).template cast<Alu>();
                   const Vec3A rest_Eb1 = rest_Ps(EE[3]).template cast<Alu>();
                   const Vec3A prev_Ea0 = prev_Ps(EE[0]).template cast<Alu>();
                   const Vec3A prev_Ea1 = prev_Ps(EE[1]).template cast<Alu>();
                   const Vec3A prev_Eb0 = prev_Ps(EE[2]).template cast<Alu>();
                   const Vec3A prev_Eb1 = prev_Ps(EE[3]).template cast<Alu>();
                   const Vec3A Ea0      = Ps(EE[0]).template cast<Alu>();
                   const Vec3A Ea1      = Ps(EE[1]).template cast<Alu>();
                   const Vec3A Eb0      = Ps(EE[2]).template cast<Alu>();
                   const Vec3A Eb1      = Ps(EE[3]).template cast<Alu>();

                   const Alu thickness =
                       safe_cast<Alu>(EE_thickness(thicknesses(EE[0]),
                                                   thicknesses(EE[1]),
                                                   thicknesses(EE[2]),
                                                   thicknesses(EE[3])));
                   const Alu d_hat = safe_cast<Alu>(EE_d_hat(d_hats(EE[0]),
                                                             d_hats(EE[1]),
                                                             d_hats(EE[2]),
                                                             d_hats(EE[3])));

                   Alu eps_x;
                   distance::edge_edge_mollifier_threshold(rest_Ea0,
                                                           rest_Ea1,
                                                           rest_Eb0,
                                                           rest_Eb1,
                                                           static_cast<Alu>(1e-3),
                                                           eps_x);

                   Vec12A G;
                   Mat12A H;
                   if(distance::need_mollify(prev_Ea0,
                                             prev_Ea1,
                                             prev_Eb0,
                                             prev_Eb1,
                                             eps_x))
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
                                                    prev_Ea0,
                                                    prev_Ea1,
                                                    prev_Eb0,
                                                    prev_Eb1,
                                                    Ea0,
                                                    Ea1,
                                                    Eb0,
                                                    Eb1);
                       cuda_mixed::make_spd(H);
                   }

                   ipc_simplex_normal_native_detail::
                       write_exact_targets<4>(
                           structured_sink,
                           targets,
                           i,
                           EE,
                           downcast_hessian<Store>(H));
               });
}
}  // namespace uipc::backend::cuda_mixed
