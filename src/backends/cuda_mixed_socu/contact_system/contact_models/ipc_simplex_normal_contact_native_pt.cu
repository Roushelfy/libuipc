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
void assemble_ipc_simplex_normal_contact_native_exact_PT(
    SimplexNormalContact::ContactInfo& info)
{
    using namespace muda;
    using namespace sym::codim_ipc_simplex_contact;
    using Alu    = ActivePolicy::AluScalar;
    using Store  = ActivePolicy::StoreScalar;
    using Vec3A  = Eigen::Matrix<Alu, 3, 1>;
    using Vec12A = Eigen::Matrix<Alu, 12, 1>;
    using Mat12A = Eigen::Matrix<Alu, 12, 12>;

    if(info.PTs().size() == 0)
        return;

    const auto structured_sink = info.structured_hessian_sink();
    const auto targets         = info.PT_native_contact_targets();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.PTs().size(),
               [structured_sink,
                targets,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                PTs = info.PTs().viewer().name("PTs"),
                Ps  = info.positions().viewer().name("Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats = info.d_hats().viewer().name("d_hats"),
                dt     = info.dt()] __device__(int i) mutable
               {
                   Vector4i PT = PTs(i);

                   Vector4i cids = {contact_ids(PT[0]),
                                    contact_ids(PT[1]),
                                    contact_ids(PT[2]),
                                    contact_ids(PT[3])};
                   Alu kt2 = safe_cast<Alu>(PT_kappa(table, cids) * dt * dt);

                   const Vector3& P_f  = Ps(PT[0]);
                   const Vector3& T0_f = Ps(PT[1]);
                   const Vector3& T1_f = Ps(PT[2]);
                   const Vector3& T2_f = Ps(PT[3]);
                   Vec3A         P     = P_f.template cast<Alu>();
                   Vec3A         T0    = T0_f.template cast<Alu>();
                   Vec3A         T1    = T1_f.template cast<Alu>();
                   Vec3A         T2    = T2_f.template cast<Alu>();

                   Alu thickness =
                       safe_cast<Alu>(PT_thickness(thicknesses(PT(0)),
                                                   thicknesses(PT(1)),
                                                   thicknesses(PT(2)),
                                                   thicknesses(PT(3))));
                   Alu d_hat = safe_cast<Alu>(PT_d_hat(d_hats(PT(0)),
                                                       d_hats(PT(1)),
                                                       d_hats(PT(2)),
                                                       d_hats(PT(3))));

                   Vector4i flag =
                       distance::point_triangle_distance_flag(P_f, T0_f, T1_f, T2_f);

                   if constexpr(RUNTIME_CHECK)
                   {
                       Float D;
                       distance::point_triangle_distance2(flag, P_f, T0_f, T1_f, T2_f, D);
                       Vector2 range =
                           D_range(safe_cast<Float>(thickness), safe_cast<Float>(d_hat));
                       MUDA_ASSERT(is_active_D(range, D),
                                   "PT[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                   PT(0),
                                   PT(1),
                                   PT(2),
                                   PT(3),
                                   D,
                                   range(0),
                                   range(1));
                   }

                   Vec12A G;
                   Mat12A H;
                   PT_barrier_gradient_hessian(
                       G, H, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                   make_spd(H);
                   const auto H_store = downcast_hessian<Store>(H);
                   ipc_simplex_normal_native_detail::
                       write_exact_targets<4>(
                           structured_sink,
                           targets,
                           i,
                           PT,
                           H_store);
               });
}
}  // namespace uipc::backend::cuda_mixed
