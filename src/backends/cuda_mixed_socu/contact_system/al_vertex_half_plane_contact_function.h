#pragma once
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <utils/friction_utils.h>
#include <type_define.h>

namespace uipc::backend::cuda_mixed
{
namespace sym::al_vertex_half_plane_contact
{
    template <typename T>
    inline UIPC_GENERIC T half_plane_penalty_energy(T                             scale,
                                                    T                             d0,
                                                    const Eigen::Matrix<T, 3, 1>& d_grad,
                                                    const Eigen::Matrix<T, 3, 1>& P0)
    {
        T d = d0;
        d += d_grad.dot(P0);
        return static_cast<T>(0.5) * scale * d * d;
    }

    template <typename T>
    inline UIPC_GENERIC void half_plane_penalty_gradient_hessian(
        T                             scale,
        T                             d0,
        const Eigen::Matrix<T, 3, 1>& d_grad,
        const Eigen::Matrix<T, 3, 1>& P,
        Eigen::Matrix<T, 3, 1>&       G,
        Eigen::Matrix<T, 3, 3>&       H)
    {
        T d = d0;
        d += d_grad.dot(P);
        G = scale * d * d_grad;
        H = scale * d_grad * d_grad.transpose();
    }

    template <typename T>
    inline UIPC_DEVICE T half_plane_frictional_energy(T                             mu,
                                                      T                             eps_vh,
                                                      T                             normal_force,
                                                      const Eigen::Matrix<T, 3, 1>& v,
                                                      const Eigen::Matrix<T, 3, 1>& prev_v,
                                                      const Eigen::Matrix<T, 3, 1>& N)
    {
        using namespace codim_ipc_contact;
        using namespace ipc_vertex_half_contact;

        Eigen::Matrix<T, 3, 1> e1, e2;
        compute_tan_basis(e1, e2, N);

        Eigen::Matrix<T, 2, 1> tan_dV;
        TR(tan_dV, v, prev_v, e1, e2);

        return friction_energy(mu, normal_force, eps_vh, tan_dV);
    }

    template <typename T>
    inline UIPC_DEVICE void half_plane_frictional_gradient_hessian(
        Eigen::Matrix<T, 3, 1>&       G,
        Eigen::Matrix<T, 3, 3>&       H,
        T                             mu,
        T                             eps_vh,
        T                             normal_force,
        const Eigen::Matrix<T, 3, 1>& v,
        const Eigen::Matrix<T, 3, 1>& prev_v,
        const Eigen::Matrix<T, 3, 1>& N)
    {
        using namespace codim_ipc_contact;
        using namespace ipc_vertex_half_contact;

        Eigen::Matrix<T, 3, 1> e1, e2;
        compute_tan_basis(e1, e2, N);

        Eigen::Matrix<T, 2, 1> tan_dV;
        TR(tan_dV, v, prev_v, e1, e2);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, normal_force, eps_vh, tan_dV);

        Eigen::Matrix<T, 2, 3> J;
        dTRdx(J, v, prev_v, e1, e2);

        G = J.transpose() * G2;

        Eigen::Matrix<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, normal_force, eps_vh, tan_dV);

        H = J.transpose() * H2x2 * J;
    }

}  // namespace sym::al_vertex_half_plane_contact
}  // namespace uipc::backend::cuda_mixed
