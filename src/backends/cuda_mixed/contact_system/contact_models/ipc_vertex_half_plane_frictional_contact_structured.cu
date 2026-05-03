#include <contact_system/vertex_half_plane_frictional_contact.h>
#include <implicit_geometry/half_plane.h>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <utils/make_spd.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
void assemble_ipc_vertex_half_plane_frictional_contact_structured(
    VertexHalfPlaneFrictionalContact::ContactInfo& info,
    const HalfPlane&                               half_plane)
{
    using namespace muda;
    using namespace sym::ipc_vertex_half_contact;
    using Alu   = ActivePolicy::AluScalar;
    using Store = ActivePolicy::StoreScalar;

    if(info.friction_PHs().size() == 0)
        return;

    const auto structured_sink = info.structured_hessian_sink();
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.friction_PHs().size(),
               [structured_sink,
                PHs = info.friction_PHs().viewer().name("PHs"),
                plane_positions = half_plane.positions().viewer().name("plane_positions"),
                plane_normals = half_plane.normals().viewer().name("plane_normals"),
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                Ps      = info.positions().viewer().name("Ps"),
                prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                eps_v  = info.eps_velocity(),
                d_hats = info.d_hats().viewer().name("d_hats"),
                half_plane_vertex_offset = info.half_plane_vertex_offset(),
                dt = info.dt()] __device__(int I) mutable
               {
                   Vector2i PH = PHs(I);

                   IndexT vI = PH(0);
                   IndexT HI = PH(1);

                   Eigen::Matrix<Alu, 3, 1> v = Ps(vI).template cast<Alu>();
                   Eigen::Matrix<Alu, 3, 1> prev_v =
                       prev_Ps(vI).template cast<Alu>();
                   Eigen::Matrix<Alu, 3, 1> P =
                       plane_positions(HI).template cast<Alu>();
                   Eigen::Matrix<Alu, 3, 1> N =
                       plane_normals(HI).template cast<Alu>();

                   const Alu d_hat = safe_cast<Alu>(d_hats(vI));

                   ContactCoeff coeff =
                       table(contact_ids(vI), contact_ids(HI + half_plane_vertex_offset));
                   const Alu kt2 = safe_cast<Alu>(coeff.kappa * dt * dt);
                   const Alu mu  = safe_cast<Alu>(coeff.mu);

                   const Alu thickness = safe_cast<Alu>(thicknesses(vI));
                   const Alu epsvdt    = safe_cast<Alu>(eps_v * dt);

                   Eigen::Matrix<Alu, 3, 1> G_alu;
                   Eigen::Matrix<Alu, 3, 3> H_alu;
                   PH_friction_gradient_hessian(G_alu,
                                                H_alu,
                                                kt2,
                                                d_hat,
                                                thickness,
                                                mu,
                                                epsvdt,
                                                prev_v,
                                                v,
                                                P,
                                                N);
                   cuda_mixed::make_spd(H_alu);
                   structured_sink.write_hessian(
                       vI,
                       downcast_hessian<Store>(H_alu));
               });
}
}  // namespace uipc::backend::cuda_mixed
