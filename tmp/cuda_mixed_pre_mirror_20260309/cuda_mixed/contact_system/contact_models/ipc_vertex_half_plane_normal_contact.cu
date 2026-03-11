#include <contact_system/vertex_half_plane_normal_contact.h>
#include <implicit_geometry/half_plane.h>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <kernel_cout.h>
#include <utils/make_spd.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
class IPCVertexHalfPlaneNormalContact final : public VertexHalfPlaneNormalContact
{
  public:
    using VertexHalfPlaneNormalContact::VertexHalfPlaneNormalContact;

    virtual void do_build(BuildInfo& info) override
    {
        auto constitution =
            world().scene().config().find<std::string>("contact/constitution");
        if(constitution->view()[0] != "ipc")
        {
            throw SimSystemException("Constitution is not IPC");
        }

        half_plane = &require<HalfPlane>();
    }

    virtual void do_compute_energy(EnergyInfo& info)
    {
        using namespace muda;
        using Alu = ActivePolicy::AluScalar;
        using Store = ActivePolicy::StoreScalar;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.PHs().size(),
                   [Es  = info.energies().viewer().name("Es"),
                    PHs = info.PHs().viewer().name("PHs"),
                    plane_positions = half_plane->positions().viewer().name("plane_positions"),
                    plane_normals = half_plane->normals().viewer().name("plane_normals"),
                    table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    subscene_ids = info.subscene_element_ids().viewer().name("subscene_element_ids"),
                    Ps = info.positions().viewer().name("Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v                    = info.eps_velocity(),
                    half_plane_vertex_offset = info.half_plane_vertex_offset(),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    dt     = info.dt()] __device__(int I) mutable
                   {
                       Vector2i PH = PHs(I);

                       IndexT vI = PH(0);
                       IndexT HI = PH(1);

                       const Alu d_hat = safe_cast<Alu>(d_hats(vI));

                       Eigen::Matrix<Alu, 3, 1> v = Ps(vI).template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 1> P =
                           plane_positions(HI).template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 1> N = plane_normals(HI).template cast<Alu>();

                       const Alu kt2 = safe_cast<Alu>(
                           table(contact_ids(vI), contact_ids(HI + half_plane_vertex_offset))
                               .kappa
                           * dt * dt);

                       const Alu thickness = safe_cast<Alu>(thicknesses(vI));

                       Es(I) = safe_cast<Store>(
                           sym::ipc_vertex_half_contact::PH_barrier_energy(
                               kt2, d_hat, thickness, v, P, N));
                   });
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace muda;
        using Alu = ActivePolicy::AluScalar;
        using Store = ActivePolicy::StoreScalar;

        if(info.PHs().size())
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(info.PHs().size(),
                       [gradient_only = info.gradient_only(),
                        Grad = info.gradients().viewer().name("Grad"),
                        Hess = info.hessians().viewer().name("Hess"),
                        PHs  = info.PHs().viewer().name("PHs"),
                        plane_positions = half_plane->positions().viewer().name("plane_positions"),
                        plane_normals = half_plane->normals().viewer().name("plane_normals"),
                        table = info.contact_tabular().viewer().name("contact_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                        Ps = info.positions().viewer().name("Ps"),
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
                           Eigen::Matrix<Alu, 3, 1> P =
                               plane_positions(HI).template cast<Alu>();
                           Eigen::Matrix<Alu, 3, 1> N =
                               plane_normals(HI).template cast<Alu>();

                           const Alu d_hat = safe_cast<Alu>(d_hats(vI));

                           const Alu kt2 = safe_cast<Alu>(
                               table(contact_ids(vI), contact_ids(HI + half_plane_vertex_offset))
                                   .kappa
                               * dt * dt);

                           const Alu thickness = safe_cast<Alu>(thicknesses(vI));

                           Eigen::Matrix<Alu, 3, 1>   G_alu;
                           if(gradient_only)
                           {
                               sym::ipc_vertex_half_contact::PH_barrier_gradient(
                                   G_alu, kt2, d_hat, thickness, v, P, N);
                               Grad(I).write(vI, downcast_gradient<Store>(G_alu));
                           }
                           else
                           {
                               Eigen::Matrix<Alu, 3, 3> H_alu;
                               sym::ipc_vertex_half_contact::PH_barrier_gradient_hessian(
                                   G_alu, H_alu, kt2, d_hat, thickness, v, P, N);
                               Grad(I).write(vI, downcast_gradient<Store>(G_alu));
                               Hess(I).write(vI, vI, downcast_hessian<Store>(H_alu));
                           }
                       });
        }
    }

    HalfPlane* half_plane = nullptr;
};

REGISTER_SIM_SYSTEM(IPCVertexHalfPlaneNormalContact);
}  // namespace uipc::backend::cuda_mixed
