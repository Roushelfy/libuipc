#include <contact_system/contact_models/ipc_simplex_normal_contact_native.h>
#include <contact_system/contact_models/ipc_simplex_normal_contact_native_common.inl>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <mixed_precision/cast.h>
#include <mixed_precision/policy.h>
#include <utils/codim_thickness.h>
#include <utils/distance/distance_flagged.h>
#include <utils/make_spd.h>
#include <utils/primitive_d_hat.h>

namespace uipc::backend::cuda_mixed
{
void assemble_ipc_simplex_normal_contact_native_exact_EE(
    SimplexNormalContact::ContactInfo& info,
    const SimplexNormalContactNativeContext& native_context)
{
    using namespace muda;
    using namespace sym::codim_ipc_simplex_contact;
    using Alu    = ActivePolicy::AluScalar;
    using Store  = ActivePolicy::StoreScalar;
    using Vec3A  = Eigen::Matrix<Alu, 3, 1>;
    using Vec12A = Eigen::Matrix<Alu, 12, 1>;
    using Mat12A = Eigen::Matrix<Alu, 12, 12>;

    if(info.EEs().size() == 0)
        return;

    const auto native_sink = native_context.sink;
    const auto targets = native_context.EE_targets;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.EEs().size(),
               [native_sink,
                targets,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                EEs         = info.EEs().viewer().name("EEs"),
                Ps          = info.positions().viewer().name("Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                rest_Ps     = info.rest_positions().viewer().name("rest_Ps"),
                d_hats      = info.d_hats().viewer().name("d_hats"),
                dt          = info.dt()] __device__(int i) mutable
               {
                   Vector4i EE = EEs(i);

                   Vector4i cids = {contact_ids(EE[0]),
                                    contact_ids(EE[1]),
                                    contact_ids(EE[2]),
                                    contact_ids(EE[3])};
                   Alu kt2 = safe_cast<Alu>(EE_kappa(table, cids) * dt * dt);

                   const Vector3& E0_f = Ps(EE[0]);
                   const Vector3& E1_f = Ps(EE[1]);
                   const Vector3& E2_f = Ps(EE[2]);
                   const Vector3& E3_f = Ps(EE[3]);
                   Vec3A         E0    = E0_f.template cast<Alu>();
                   Vec3A         E1    = E1_f.template cast<Alu>();
                   Vec3A         E2    = E2_f.template cast<Alu>();
                   Vec3A         E3    = E3_f.template cast<Alu>();

                   const Vector3& t0_Ea0_f = rest_Ps(EE[0]);
                   const Vector3& t0_Ea1_f = rest_Ps(EE[1]);
                   const Vector3& t0_Eb0_f = rest_Ps(EE[2]);
                   const Vector3& t0_Eb1_f = rest_Ps(EE[3]);
                   Vec3A t0_Ea0 = t0_Ea0_f.template cast<Alu>();
                   Vec3A t0_Ea1 = t0_Ea1_f.template cast<Alu>();
                   Vec3A t0_Eb0 = t0_Eb0_f.template cast<Alu>();
                   Vec3A t0_Eb1 = t0_Eb1_f.template cast<Alu>();

                   Alu thickness =
                       safe_cast<Alu>(EE_thickness(thicknesses(EE(0)),
                                                   thicknesses(EE(1)),
                                                   thicknesses(EE(2)),
                                                   thicknesses(EE(3))));
                   Alu d_hat = safe_cast<Alu>(EE_d_hat(d_hats(EE(0)),
                                                       d_hats(EE(1)),
                                                       d_hats(EE(2)),
                                                       d_hats(EE(3))));

                   Vector4i flag =
                       distance::edge_edge_distance_flag(E0_f, E1_f, E2_f, E3_f);

                   if constexpr(RUNTIME_CHECK)
                   {
                       Float D;
                       distance::edge_edge_distance2(flag, E0_f, E1_f, E2_f, E3_f, D);
                       Vector2 range =
                           D_range(safe_cast<Float>(thickness), safe_cast<Float>(d_hat));
                       MUDA_ASSERT(is_active_D(range, D),
                                   "EE[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                   EE(0),
                                   EE(1),
                                   EE(2),
                                   EE(3),
                                   D,
                                   range(0),
                                   range(1));
                   }

                   Vec12A G;
                   Mat12A H;
                   mollified_EE_barrier_gradient_hessian(G,
                                                         H,
                                                         flag,
                                                         kt2,
                                                         d_hat,
                                                         thickness,
                                                         t0_Ea0,
                                                         t0_Ea1,
                                                         t0_Eb0,
                                                         t0_Eb1,
                                                         E0,
                                                         E1,
                                                         E2,
                                                         E3);
                   make_spd(H);
                   const auto H_store = downcast_hessian<Store>(H);
                   ipc_simplex_normal_native_detail::
                       write_exact_targets<4>(
                           native_sink,
                           targets,
                           i,
                           EE,
                           H_store);
               });
}
}  // namespace uipc::backend::cuda_mixed
