#include <contact_system/contact_models/ipc_vertex_half_plane_normal_contact_native.h>

#include <contact_system/contact_models/ipc_simplex_normal_contact_native_common.inl>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <implicit_geometry/half_plane.h>
#include <mixed_precision/cast.h>
#include <mixed_precision/policy.h>

namespace uipc::backend::cuda_mixed
{
void assemble_ipc_vertex_half_plane_normal_contact_native_exact(
    VertexHalfPlaneNormalContact::ContactInfo& info,
    const HalfPlane&                           half_plane)
{
    using namespace muda;
    using Alu = ActivePolicy::AluScalar;
    using Store = ActivePolicy::StoreScalar;

    if(info.PHs().size() == 0)
        return;

    const auto structured_sink = info.structured_hessian_sink();
    const auto targets         = info.PH_native_contact_targets();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.PHs().size(),
               [structured_sink,
                targets,
                PHs = info.PHs().viewer().name("PHs"),
                plane_positions = half_plane.positions().viewer().name("plane_positions"),
                plane_normals = half_plane.normals().viewer().name("plane_normals"),
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids =
                    info.contact_element_ids().viewer().name("contact_element_ids"),
                Ps          = info.positions().viewer().name("Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats      = info.d_hats().viewer().name("d_hats"),
                half_plane_vertex_offset = info.half_plane_vertex_offset(),
                dt = info.dt()] __device__(int I) mutable
               {
                   const Vector2i PH = PHs(I);

                   const IndexT vI = PH(0);
                   const IndexT HI = PH(1);

                   const Eigen::Matrix<Alu, 3, 1> v =
                       Ps(vI).template cast<Alu>();
                   const Eigen::Matrix<Alu, 3, 1> P =
                       plane_positions(HI).template cast<Alu>();
                   const Eigen::Matrix<Alu, 3, 1> N =
                       plane_normals(HI).template cast<Alu>();

                   const Alu d_hat = safe_cast<Alu>(d_hats(vI));

                   const Alu kt2 = safe_cast<Alu>(
                       table(contact_ids(vI), contact_ids(HI + half_plane_vertex_offset))
                           .kappa
                       * dt * dt);

                   const Alu thickness = safe_cast<Alu>(thicknesses(vI));

                   Eigen::Matrix<Alu, 3, 1> G_alu;
                   Eigen::Matrix<Alu, 3, 3> H_alu;
                   sym::ipc_vertex_half_contact::PH_barrier_gradient_hessian(
                       G_alu,
                       H_alu,
                       kt2,
                       d_hat,
                       thickness,
                       v,
                       P,
                       N);

                   Eigen::Matrix<IndexT, 1, 1> indices;
                   indices(0) = vI;
                   ipc_simplex_normal_native_detail::
                       write_exact_targets_or_legacy<1>(
                           structured_sink,
                           targets,
                           I,
                           indices,
                           downcast_hessian<Store>(H_alu));
               });
}
}  // namespace uipc::backend::cuda_mixed
