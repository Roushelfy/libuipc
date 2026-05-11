#include <contact_system/contact_models/ipc_simplex_frictional_contact_native.h>
#include <contact_system/contact_models/ipc_simplex_normal_contact_native_common.inl>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>
#include <mixed_precision/cast.h>
#include <mixed_precision/policy.h>
#include <utils/codim_thickness.h>
#include <utils/make_spd.h>
#include <utils/primitive_d_hat.h>

namespace uipc::backend::cuda_mixed
{
void assemble_ipc_simplex_frictional_contact_native_exact_PP(
    SimplexFrictionalContact::ContactInfo& info,
    const SimplexFrictionalContactNativeContext& native_context)
{
    using namespace muda;
    using namespace sym::codim_ipc_contact;
    using Alu   = ActivePolicy::AluScalar;
    using Store = ActivePolicy::StoreScalar;
    using Vec3A = Eigen::Matrix<Alu, 3, 1>;
    using Vec6A = Eigen::Matrix<Alu, 6, 1>;
    using Mat6A = Eigen::Matrix<Alu, 6, 6>;

    if(info.friction_PPs().size() == 0)
        return;

    const auto native_sink = native_context.sink;
    const auto targets = native_context.PP_targets;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.friction_PPs().size(),
               [native_sink,
                targets,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                PPs     = info.friction_PPs().viewer().name("friction_PPs"),
                Ps      = info.positions().viewer().name("Ps"),
                prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats = info.d_hats().viewer().name("d_hats"),
                eps_v  = info.eps_velocity(),
                dt     = info.dt()] __device__(int i) mutable
               {
                   const Vector2i PP = PPs(i);

                   Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                   const auto coeff = PP_contact_coeff(table, cids);
                   const Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                   const Alu  mu    = safe_cast<Alu>(coeff.mu);

                   const Vec3A prev_P0 = prev_Ps(PP[0]).template cast<Alu>();
                   const Vec3A prev_P1 = prev_Ps(PP[1]).template cast<Alu>();
                   const Vec3A P0      = Ps(PP[0]).template cast<Alu>();
                   const Vec3A P1      = Ps(PP[1]).template cast<Alu>();

                   const Alu thickness = safe_cast<Alu>(
                       PP_thickness(thicknesses(PP[0]), thicknesses(PP[1])));
                   const Alu d_hat =
                       safe_cast<Alu>(PP_d_hat(d_hats(PP[0]), d_hats(PP[1])));

                   Vec6A G;
                   Mat6A H;
                   PP_friction_gradient_hessian(G,
                                                H,
                                                kt2,
                                                d_hat,
                                                thickness,
                                                mu,
                                                safe_cast<Alu>(eps_v * dt),
                                                prev_P0,
                                                prev_P1,
                                                P0,
                                                P1);
                   cuda_mixed::make_spd(H);

                   ipc_simplex_normal_native_detail::
                       write_exact_targets<2>(
                           native_sink,
                           targets,
                           i,
                           PP,
                           downcast_hessian<Store>(H));
               });
}
}  // namespace uipc::backend::cuda_mixed
