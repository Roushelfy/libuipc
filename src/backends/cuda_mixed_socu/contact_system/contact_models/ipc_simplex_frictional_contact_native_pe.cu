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
void assemble_ipc_simplex_frictional_contact_native_exact_PE(
    SimplexFrictionalContact::ContactInfo& info)
{
    using namespace muda;
    using namespace sym::codim_ipc_contact;
    using Alu   = ActivePolicy::AluScalar;
    using Store = ActivePolicy::StoreScalar;
    using Vec3A = Eigen::Matrix<Alu, 3, 1>;
    using Vec9A = Eigen::Matrix<Alu, 9, 1>;
    using Mat9A = Eigen::Matrix<Alu, 9, 9>;

    if(info.friction_PEs().size() == 0)
        return;

    const auto structured_sink = info.structured_hessian_sink();
    const auto targets         = info.friction_PE_native_contact_targets();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.friction_PEs().size(),
               [structured_sink,
                targets,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                PEs     = info.friction_PEs().viewer().name("friction_PEs"),
                Ps      = info.positions().viewer().name("Ps"),
                prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats = info.d_hats().viewer().name("d_hats"),
                eps_v  = info.eps_velocity(),
                dt     = info.dt()] __device__(int i) mutable
               {
                   const Vector3i PE = PEs(i);

                   Vector3i cids = {contact_ids(PE[0]),
                                    contact_ids(PE[1]),
                                    contact_ids(PE[2])};

                   const auto coeff = PE_contact_coeff(table, cids);
                   const Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                   const Alu  mu    = safe_cast<Alu>(coeff.mu);

                   const Vec3A prev_P  = prev_Ps(PE[0]).template cast<Alu>();
                   const Vec3A prev_E0 = prev_Ps(PE[1]).template cast<Alu>();
                   const Vec3A prev_E1 = prev_Ps(PE[2]).template cast<Alu>();
                   const Vec3A P       = Ps(PE[0]).template cast<Alu>();
                   const Vec3A E0      = Ps(PE[1]).template cast<Alu>();
                   const Vec3A E1      = Ps(PE[2]).template cast<Alu>();

                   const Alu thickness = safe_cast<Alu>(PE_thickness(
                       thicknesses(PE[0]), thicknesses(PE[1]), thicknesses(PE[2])));
                   const Alu d_hat = safe_cast<Alu>(
                       PE_d_hat(d_hats(PE[0]), d_hats(PE[1]), d_hats(PE[2])));

                   Vec9A G;
                   Mat9A H;
                   PE_friction_gradient_hessian(G,
                                                H,
                                                kt2,
                                                d_hat,
                                                thickness,
                                                mu,
                                                safe_cast<Alu>(eps_v * dt),
                                                prev_P,
                                                prev_E0,
                                                prev_E1,
                                                P,
                                                E0,
                                                E1);
                   cuda_mixed::make_spd(H);

                   ipc_simplex_normal_native_detail::
                       write_exact_targets<3>(
                           structured_sink,
                           targets,
                           i,
                           PE,
                           downcast_hessian<Store>(H));
               });
}
}  // namespace uipc::backend::cuda_mixed
