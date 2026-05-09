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
void assemble_ipc_simplex_normal_contact_native_exact_PE(
    SimplexNormalContact::ContactInfo& info)
{
    using namespace muda;
    using namespace sym::codim_ipc_simplex_contact;
    using Alu   = ActivePolicy::AluScalar;
    using Store = ActivePolicy::StoreScalar;
    using Vec3A = Eigen::Matrix<Alu, 3, 1>;
    using Vec9A = Eigen::Matrix<Alu, 9, 1>;
    using Mat9A = Eigen::Matrix<Alu, 9, 9>;

    if(info.PEs().size() == 0)
        return;

    const auto structured_sink = info.structured_hessian_sink();
    const auto targets         = info.PE_native_contact_targets();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.PEs().size(),
               [structured_sink,
                targets,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                PEs         = info.PEs().viewer().name("PEs"),
                Ps          = info.positions().viewer().name("Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats      = info.d_hats().viewer().name("d_hats"),
                dt          = info.dt()] __device__(int i) mutable
               {
                   Vector3i PE = PEs(i);

                   Vector3i cids = {
                       contact_ids(PE[0]), contact_ids(PE[1]), contact_ids(PE[2])};
                   Alu kt2 = safe_cast<Alu>(PE_kappa(table, cids) * dt * dt);

                   const Vector3& P_f  = Ps(PE[0]);
                   const Vector3& E0_f = Ps(PE[1]);
                   const Vector3& E1_f = Ps(PE[2]);
                   Vec3A         P     = P_f.template cast<Alu>();
                   Vec3A         E0    = E0_f.template cast<Alu>();
                   Vec3A         E1    = E1_f.template cast<Alu>();

                   Alu thickness = safe_cast<Alu>(PE_thickness(
                       thicknesses(PE(0)), thicknesses(PE(1)), thicknesses(PE(2))));
                   Alu d_hat = safe_cast<Alu>(
                       PE_d_hat(d_hats(PE(0)), d_hats(PE(1)), d_hats(PE(2))));

                   Vector3i flag =
                       distance::point_edge_distance_flag(P_f, E0_f, E1_f);

                   if constexpr(RUNTIME_CHECK)
                   {
                       Float D;
                       distance::point_edge_distance2(flag, P_f, E0_f, E1_f, D);
                       Vector2 range =
                           D_range(safe_cast<Float>(thickness), safe_cast<Float>(d_hat));
                       MUDA_ASSERT(is_active_D(range, D),
                                   "PE[%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                   PE(0),
                                   PE(1),
                                   PE(2),
                                   D,
                                   range(0),
                                   range(1));
                   }

                   Vec9A G;
                   Mat9A H;
                   PE_barrier_gradient_hessian(
                       G, H, flag, kt2, d_hat, thickness, P, E0, E1);
                   make_spd(H);
                   const auto H_store = downcast_hessian<Store>(H);
                   ipc_simplex_normal_native_detail::
                       write_exact_targets_or_legacy<3>(
                           structured_sink,
                           targets,
                           i,
                           PE,
                           H_store);
               });
}
}  // namespace uipc::backend::cuda_mixed
