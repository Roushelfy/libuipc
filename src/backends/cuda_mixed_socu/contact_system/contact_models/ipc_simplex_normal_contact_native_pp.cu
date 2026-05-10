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
void assemble_ipc_simplex_normal_contact_native_exact_PP(
    SimplexNormalContact::ContactInfo& info)
{
    using namespace muda;
    using namespace sym::codim_ipc_simplex_contact;
    using Alu   = ActivePolicy::AluScalar;
    using Store = ActivePolicy::StoreScalar;
    using Vec3A = Eigen::Matrix<Alu, 3, 1>;
    using Vec6A = Eigen::Matrix<Alu, 6, 1>;
    using Mat6A = Eigen::Matrix<Alu, 6, 6>;

    if(info.PPs().size() == 0)
        return;

    const auto structured_sink = info.structured_hessian_sink();
    const auto targets         = info.PP_native_contact_targets();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.PPs().size(),
               [structured_sink,
                targets,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                PPs         = info.PPs().viewer().name("PPs"),
                Ps          = info.positions().viewer().name("Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats      = info.d_hats().viewer().name("d_hats"),
                dt          = info.dt()] __device__(int i) mutable
               {
                   const auto& PP = PPs(i);

                   Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                   Alu kt2 = safe_cast<Alu>(PP_kappa(table, cids) * dt * dt);

                   const Vector3& P0_f = Ps(PP[0]);
                   const Vector3& P1_f = Ps(PP[1]);
                   Vec3A         P0    = P0_f.template cast<Alu>();
                   Vec3A         P1    = P1_f.template cast<Alu>();

                   Alu thickness = safe_cast<Alu>(
                       PP_thickness(thicknesses(PP(0)), thicknesses(PP(1))));
                   Alu d_hat =
                       safe_cast<Alu>(PP_d_hat(d_hats(PP(0)), d_hats(PP(1))));

                   Vector2i flag = distance::point_point_distance_flag(P0_f, P1_f);

                   if constexpr(RUNTIME_CHECK)
                   {
                       Float D;
                       distance::point_point_distance2(flag, P0_f, P1_f, D);
                       Vector2 range =
                           D_range(safe_cast<Float>(thickness), safe_cast<Float>(d_hat));
                       MUDA_ASSERT(is_active_D(range, D),
                                   "PP[%d,%d] d^2(%f) out of range, (%f,%f)",
                                   PP(0),
                                   PP(1),
                                   D,
                                   range(0),
                                   range(1));
                   }

                   Vec6A G;
                   Mat6A H;
                   PP_barrier_gradient_hessian(
                       G, H, flag, kt2, d_hat, thickness, P0, P1);
                   make_spd(H);
                   const auto H_store = downcast_hessian<Store>(H);
                   ipc_simplex_normal_native_detail::
                       write_exact_targets<2>(
                           structured_sink,
                           targets,
                           i,
                           PP,
                           H_store);
               });
}
}  // namespace uipc::backend::cuda_mixed
