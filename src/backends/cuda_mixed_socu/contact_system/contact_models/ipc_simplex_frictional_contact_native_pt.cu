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
void assemble_ipc_simplex_frictional_contact_native_exact_PT(
    SimplexFrictionalContact::ContactInfo& info,
    const SimplexFrictionalContactNativeContext& native_context)
{
    using namespace muda;
    using namespace sym::codim_ipc_contact;
    using Alu    = ActivePolicy::AluScalar;
    using Store  = ActivePolicy::StoreScalar;
    using Vec3A  = Eigen::Matrix<Alu, 3, 1>;
    using Vec12A = Eigen::Matrix<Alu, 12, 1>;
    using Mat12A = Eigen::Matrix<Alu, 12, 12>;

    if(info.friction_PTs().size() == 0)
        return;

    const auto native_sink = native_context.sink;
    const auto targets = native_context.PT_targets;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.friction_PTs().size(),
               [native_sink,
                targets,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                PTs     = info.friction_PTs().viewer().name("friction_PTs"),
                Ps      = info.positions().viewer().name("Ps"),
                prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats = info.d_hats().viewer().name("d_hats"),
                eps_v  = info.eps_velocity(),
                dt     = info.dt()] __device__(int i) mutable
               {
                   const Vector4i PT = PTs(i);

                   Vector4i cids = {contact_ids(PT[0]),
                                    contact_ids(PT[1]),
                                    contact_ids(PT[2]),
                                    contact_ids(PT[3])};

                   const auto coeff = PT_contact_coeff(table, cids);
                   const Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                   const Alu  mu    = safe_cast<Alu>(coeff.mu);

                   const Vec3A prev_P  = prev_Ps(PT[0]).template cast<Alu>();
                   const Vec3A prev_T0 = prev_Ps(PT[1]).template cast<Alu>();
                   const Vec3A prev_T1 = prev_Ps(PT[2]).template cast<Alu>();
                   const Vec3A prev_T2 = prev_Ps(PT[3]).template cast<Alu>();
                   const Vec3A P       = Ps(PT[0]).template cast<Alu>();
                   const Vec3A T0      = Ps(PT[1]).template cast<Alu>();
                   const Vec3A T1      = Ps(PT[2]).template cast<Alu>();
                   const Vec3A T2      = Ps(PT[3]).template cast<Alu>();

                   const Alu thickness =
                       safe_cast<Alu>(PT_thickness(thicknesses(PT[0]),
                                                   thicknesses(PT[1]),
                                                   thicknesses(PT[2]),
                                                   thicknesses(PT[3])));
                   const Alu d_hat = safe_cast<Alu>(PT_d_hat(d_hats(PT[0]),
                                                             d_hats(PT[1]),
                                                             d_hats(PT[2]),
                                                             d_hats(PT[3])));

                   Vec12A G;
                   Mat12A H;
                   PT_friction_gradient_hessian(G,
                                                H,
                                                kt2,
                                                d_hat,
                                                thickness,
                                                mu,
                                                safe_cast<Alu>(eps_v * dt),
                                                prev_P,
                                                prev_T0,
                                                prev_T1,
                                                prev_T2,
                                                P,
                                                T0,
                                                T1,
                                                T2);
                   cuda_mixed::make_spd(H);

                   ipc_simplex_normal_native_detail::
                       write_exact_targets<4>(
                           native_sink,
                           targets,
                           i,
                           PT,
                           downcast_hessian<Store>(H));
               });
}
}  // namespace uipc::backend::cuda_mixed
