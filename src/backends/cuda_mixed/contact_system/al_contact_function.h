#pragma once
#include <contact_system/contact_models/codim_ipc_contact_function.h>
#include <utils/friction_utils.h>
#include <type_define.h>

namespace uipc::backend::cuda_mixed
{
namespace sym::al_simplex_contact
{
    template <typename T>
    inline UIPC_GENERIC T penalty_energy(T                              scale,
                                         T                              d0,
                                         const Eigen::Matrix<T, 12, 1>& d_grad,
                                         const Eigen::Matrix<T, 3, 1>&  P0,
                                         const Eigen::Matrix<T, 3, 1>&  P1,
                                         const Eigen::Matrix<T, 3, 1>&  P2,
                                         const Eigen::Matrix<T, 3, 1>&  P3)
    {
        T d = d0;
        d += d_grad.template segment<3>(0).dot(P0);
        d += d_grad.template segment<3>(3).dot(P1);
        d += d_grad.template segment<3>(6).dot(P2);
        d += d_grad.template segment<3>(9).dot(P3);
        return static_cast<T>(0.5) * scale * d * d;
    }

    template <typename T>
    inline UIPC_GENERIC void penalty_gradient_hessian(
        T                              scale,
        T                              d0,
        const Eigen::Matrix<T, 12, 1>& d_grad,
        const Eigen::Matrix<T, 3, 1>&  P0,
        const Eigen::Matrix<T, 3, 1>&  P1,
        const Eigen::Matrix<T, 3, 1>&  P2,
        const Eigen::Matrix<T, 3, 1>&  P3,
        Eigen::Matrix<T, 12, 1>&       G,
        Eigen::Matrix<T, 12, 12>&      H)
    {
        T d = d0;
        d += d_grad.template segment<3>(0).dot(P0);
        d += d_grad.template segment<3>(3).dot(P1);
        d += d_grad.template segment<3>(6).dot(P2);
        d += d_grad.template segment<3>(9).dot(P3);
        G = scale * d * d_grad;
        H = scale * d_grad * d_grad.transpose();
    }

    template <typename T>
    inline UIPC_DEVICE T PT_friction_energy(T                             mu,
                                            T                             eps_vh,
                                            T                             normal_force,
                                            const Eigen::Matrix<T, 3, 1>& prev_P,
                                            const Eigen::Matrix<T, 3, 1>& prev_T0,
                                            const Eigen::Matrix<T, 3, 1>& prev_T1,
                                            const Eigen::Matrix<T, 3, 1>& prev_T2,
                                            const Eigen::Matrix<T, 3, 1>& P,
                                            const Eigen::Matrix<T, 3, 1>& T0,
                                            const Eigen::Matrix<T, 3, 1>& T1,
                                            const Eigen::Matrix<T, 3, 1>& T2)
    {
        using namespace friction;
        using namespace codim_ipc_contact;

        Eigen::Matrix<T, 2, 1> beta;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;

        point_triangle_closest_point(prev_P, prev_T0, prev_T1, prev_T2, beta);
        point_triangle_tangent_basis(prev_P, prev_T0, prev_T1, prev_T2, basis);

        Eigen::Matrix<T, 3, 1> dP  = P - prev_P;
        Eigen::Matrix<T, 3, 1> dT0 = T0 - prev_T0;
        Eigen::Matrix<T, 3, 1> dT1 = T1 - prev_T1;
        Eigen::Matrix<T, 3, 1> dT2 = T2 - prev_T2;
        point_triangle_tan_rel_dx(dP, dT0, dT1, dT2, basis, beta, tan_rel_dx);

        T E = friction_energy(mu, normal_force, eps_vh, tan_rel_dx);
        return E;
    }

    template <typename T>
    inline UIPC_DEVICE void PT_friction_gradient_hessian(
        Eigen::Matrix<T, 12, 1>&       G,
        Eigen::Matrix<T, 12, 12>&      H,
        T                              mu,
        T                              eps_vh,
        T                              normal_force,
        const Eigen::Matrix<T, 3, 1>&  prev_P,
        const Eigen::Matrix<T, 3, 1>&  prev_T0,
        const Eigen::Matrix<T, 3, 1>&  prev_T1,
        const Eigen::Matrix<T, 3, 1>&  prev_T2,
        const Eigen::Matrix<T, 3, 1>&  P,
        const Eigen::Matrix<T, 3, 1>&  T0,
        const Eigen::Matrix<T, 3, 1>&  T1,
        const Eigen::Matrix<T, 3, 1>&  T2)
    {
        using namespace friction;
        using namespace codim_ipc_contact;

        Eigen::Matrix<T, 2, 1> beta;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;

        point_triangle_closest_point(prev_P, prev_T0, prev_T1, prev_T2, beta);
        point_triangle_tangent_basis(prev_P, prev_T0, prev_T1, prev_T2, basis);

        Eigen::Matrix<T, 3, 1> dP  = P - prev_P;
        Eigen::Matrix<T, 3, 1> dT0 = T0 - prev_T0;
        Eigen::Matrix<T, 3, 1> dT1 = T1 - prev_T1;
        Eigen::Matrix<T, 3, 1> dT2 = T2 - prev_T2;
        point_triangle_tan_rel_dx(dP, dT0, dT1, dT2, basis, beta, tan_rel_dx);

        Eigen::Matrix<T, 2, 12> J;
        point_triangle_jacobi(basis, beta, J);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, normal_force, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;

        Eigen::Matrix<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, normal_force, eps_vh, tan_rel_dx);
        H = J.transpose() * H2x2 * J;
    }

    template <typename T>
    inline UIPC_DEVICE T EE_friction_energy(T                             mu,
                                            T                             eps_vh,
                                            T                             normal_force,
                                            const Eigen::Matrix<T, 3, 1>& prev_Ea0,
                                            const Eigen::Matrix<T, 3, 1>& prev_Ea1,
                                            const Eigen::Matrix<T, 3, 1>& prev_Eb0,
                                            const Eigen::Matrix<T, 3, 1>& prev_Eb1,
                                            const Eigen::Matrix<T, 3, 1>& Ea0,
                                            const Eigen::Matrix<T, 3, 1>& Ea1,
                                            const Eigen::Matrix<T, 3, 1>& Eb0,
                                            const Eigen::Matrix<T, 3, 1>& Eb1)
    {
        using namespace friction;
        using namespace codim_ipc_contact;

        Eigen::Matrix<T, 2, 1> gamma;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;

        edge_edge_closest_point(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, gamma);
        edge_edge_tangent_basis(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, basis);

        Eigen::Matrix<T, 3, 1> dEa0 = Ea0 - prev_Ea0;
        Eigen::Matrix<T, 3, 1> dEa1 = Ea1 - prev_Ea1;
        Eigen::Matrix<T, 3, 1> dEb0 = Eb0 - prev_Eb0;
        Eigen::Matrix<T, 3, 1> dEb1 = Eb1 - prev_Eb1;
        edge_edge_tan_rel_dx(dEa0, dEa1, dEb0, dEb1, basis, gamma, tan_rel_dx);

        T E = friction_energy(mu, normal_force, eps_vh, tan_rel_dx);
        return E;
    }

    template <typename T>
    inline UIPC_DEVICE void EE_friction_gradient_hessian(
        Eigen::Matrix<T, 12, 1>&       G,
        Eigen::Matrix<T, 12, 12>&      H,
        T                              mu,
        T                              eps_vh,
        T                              normal_force,
        const Eigen::Matrix<T, 3, 1>&  prev_Ea0,
        const Eigen::Matrix<T, 3, 1>&  prev_Ea1,
        const Eigen::Matrix<T, 3, 1>&  prev_Eb0,
        const Eigen::Matrix<T, 3, 1>&  prev_Eb1,
        const Eigen::Matrix<T, 3, 1>&  Ea0,
        const Eigen::Matrix<T, 3, 1>&  Ea1,
        const Eigen::Matrix<T, 3, 1>&  Eb0,
        const Eigen::Matrix<T, 3, 1>&  Eb1)
    {
        using namespace friction;
        using namespace codim_ipc_contact;

        Eigen::Matrix<T, 2, 1> gamma;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;

        edge_edge_closest_point(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, gamma);
        edge_edge_tangent_basis(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, basis);

        Eigen::Matrix<T, 3, 1> dEa0 = Ea0 - prev_Ea0;
        Eigen::Matrix<T, 3, 1> dEa1 = Ea1 - prev_Ea1;
        Eigen::Matrix<T, 3, 1> dEb0 = Eb0 - prev_Eb0;
        Eigen::Matrix<T, 3, 1> dEb1 = Eb1 - prev_Eb1;
        edge_edge_tan_rel_dx(dEa0, dEa1, dEb0, dEb1, basis, gamma, tan_rel_dx);

        Eigen::Matrix<T, 2, 12> J;
        edge_edge_jacobi(basis, gamma, J);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, normal_force, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;

        Eigen::Matrix<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, normal_force, eps_vh, tan_rel_dx);
        H = J.transpose() * H2x2 * J;
    }
}  // namespace sym::al_simplex_contact
}  // namespace uipc::backend::cuda_mixed
