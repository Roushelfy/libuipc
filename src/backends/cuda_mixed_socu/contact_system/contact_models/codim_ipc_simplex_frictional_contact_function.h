#pragma once
#include <type_define.h>
#include <contact_system/contact_coeff.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>
#include <utils/friction_utils.h>

namespace uipc::backend::cuda_mixed
{
namespace sym::codim_ipc_contact
{
    inline __device__ ContactCoeff PT_contact_coeff(const muda::CDense2D<ContactCoeff>& table,
                                                    const Vector4i& cids)
    {
        Float kappa = 0.0;
        Float mu    = 0.0;
        for(int j = 1; j < 4; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
            mu += coeff.mu;
        }
        return {kappa / 3.0, mu / 3.0};
    }

    inline __device__ ContactCoeff EE_contact_coeff(const muda::CDense2D<ContactCoeff>& table,
                                                    const Vector4i& cids)
    {
        Float kappa = 0.0;
        Float mu    = 0.0;
        for(int j = 0; j < 2; ++j)
        {
            for(int k = 2; k < 4; ++k)
            {
                ContactCoeff coeff = table(cids[j], cids[k]);
                kappa += coeff.kappa;
                mu += coeff.mu;
            }
        }
        return {kappa / 4.0, mu / 4.0};
    }

    inline __device__ ContactCoeff PE_contact_coeff(const muda::CDense2D<ContactCoeff>& table,
                                                    const Vector3i& cids)
    {
        Float kappa = 0.0;
        Float mu    = 0.0;
        for(int j = 1; j < 3; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
            mu += coeff.mu;
        }
        return {kappa / 2.0, mu / 2.0};
    }

    inline __device__ ContactCoeff PP_contact_coeff(const muda::CDense2D<ContactCoeff>& table,
                                                    const Vector2i& cids)
    {
        return table(cids[0], cids[1]);
    }

    template <typename T>
    inline __device__ void PT_friction_basis(
        T&                            f,
        Eigen::Matrix<T, 2, 1>&       beta,
        Eigen::Matrix<T, 3, 2>&       basis,
        Eigen::Matrix<T, 2, 1>&       tan_rel_dx,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        const Eigen::Matrix<T, 3, 1>& prev_P,
        const Eigen::Matrix<T, 3, 1>& prev_T0,
        const Eigen::Matrix<T, 3, 1>& prev_T1,
        const Eigen::Matrix<T, 3, 1>& prev_T2,
        const Eigen::Matrix<T, 3, 1>& P,
        const Eigen::Matrix<T, 3, 1>& T0,
        const Eigen::Matrix<T, 3, 1>& T1,
        const Eigen::Matrix<T, 3, 1>& T2)
    {
        using namespace distance;
        using namespace friction;

        point_triangle_closest_point(prev_P, prev_T0, prev_T1, prev_T2, beta);
        point_triangle_tangent_basis(prev_P, prev_T0, prev_T1, prev_T2, basis);

        T prev_D;
        point_triangle_distance2(prev_P, prev_T0, prev_T1, prev_T2, prev_D);

        f = normal_force(kappa, d_hat, thickness, prev_D);

        Eigen::Matrix<T, 3, 1> dP  = P - prev_P;
        Eigen::Matrix<T, 3, 1> dT0 = T0 - prev_T0;
        Eigen::Matrix<T, 3, 1> dT1 = T1 - prev_T1;
        Eigen::Matrix<T, 3, 1> dT2 = T2 - prev_T2;

        point_triangle_tan_rel_dx(dP, dT0, dT1, dT2, basis, beta, tan_rel_dx);
    }

    template <typename T>
    inline __device__ T PT_friction_energy(T                             kappa,
                                           T                             d_hat,
                                           T                             thickness,
                                           T                             mu,
                                           T                             eps_vh,
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

        T                       f;
        Eigen::Matrix<T, 2, 1>  beta;
        Eigen::Matrix<T, 3, 2>  basis;
        Eigen::Matrix<T, 2, 1>  tan_rel_dx;

        PT_friction_basis(
            f, beta, basis, tan_rel_dx, kappa, d_hat, thickness, prev_P, prev_T0, prev_T1, prev_T2, P, T0, T1, T2);

        return friction_energy(mu, f, eps_vh, tan_rel_dx);
    }

    template <typename T>
    inline __device__ void PT_friction_gradient_hessian(
        Eigen::Matrix<T, 12, 1>&      G,
        Eigen::Matrix<T, 12, 12>&     H,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        T                             mu,
        T                             eps_vh,
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

        Eigen::Matrix<T, 2, 1> beta;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;
        T                      f;

        PT_friction_basis(
            f, beta, basis, tan_rel_dx, kappa, d_hat, thickness, prev_P, prev_T0, prev_T1, prev_T2, P, T0, T1, T2);

        Eigen::Matrix<T, 2, 12> J;
        point_triangle_jacobi(basis, beta, J);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;

        Eigen::Matrix<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, f, eps_vh, tan_rel_dx);
        H = J.transpose() * H2x2 * J;
    }

    template <typename T>
    inline __device__ void PT_friction_gradient(
        Eigen::Matrix<T, 12, 1>&      G,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        T                             mu,
        T                             eps_vh,
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

        Eigen::Matrix<T, 2, 1> beta;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;
        T                      f;

        PT_friction_basis(
            f, beta, basis, tan_rel_dx, kappa, d_hat, thickness, prev_P, prev_T0, prev_T1, prev_T2, P, T0, T1, T2);

        Eigen::Matrix<T, 2, 12> J;
        point_triangle_jacobi(basis, beta, J);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;
    }

    template <typename T>
    inline __device__ void EE_friction_basis(
        T&                            f,
        Eigen::Matrix<T, 2, 1>&       gamma,
        Eigen::Matrix<T, 3, 2>&       basis,
        Eigen::Matrix<T, 2, 1>&       tan_rel_dx,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        const Eigen::Matrix<T, 3, 1>& prev_Ea0,
        const Eigen::Matrix<T, 3, 1>& prev_Ea1,
        const Eigen::Matrix<T, 3, 1>& prev_Eb0,
        const Eigen::Matrix<T, 3, 1>& prev_Eb1,
        const Eigen::Matrix<T, 3, 1>& Ea0,
        const Eigen::Matrix<T, 3, 1>& Ea1,
        const Eigen::Matrix<T, 3, 1>& Eb0,
        const Eigen::Matrix<T, 3, 1>& Eb1)
    {
        using namespace distance;
        using namespace friction;

        edge_edge_closest_point(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, gamma);
        edge_edge_tangent_basis(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, basis);

        T prev_D;
        edge_edge_distance2(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, prev_D);

        f = normal_force(kappa, d_hat, thickness, prev_D);

        Eigen::Matrix<T, 3, 1> dEa0 = Ea0 - prev_Ea0;
        Eigen::Matrix<T, 3, 1> dEa1 = Ea1 - prev_Ea1;
        Eigen::Matrix<T, 3, 1> dEb0 = Eb0 - prev_Eb0;
        Eigen::Matrix<T, 3, 1> dEb1 = Eb1 - prev_Eb1;

        edge_edge_tan_rel_dx(dEa0, dEa1, dEb0, dEb1, basis, gamma, tan_rel_dx);
    }

    template <typename T>
    inline __device__ T EE_friction_energy(T                             kappa,
                                           T                             d_hat,
                                           T                             thickness,
                                           T                             mu,
                                           T                             eps_vh,
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

        Eigen::Matrix<T, 2, 1> gamma;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;
        T                      f;

        EE_friction_basis(
            f, gamma, basis, tan_rel_dx, kappa, d_hat, thickness, prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, Ea0, Ea1, Eb0, Eb1);

        return friction_energy(mu, f, eps_vh, tan_rel_dx);
    }

    template <typename T>
    inline __device__ void EE_friction_gradient_hessian(
        Eigen::Matrix<T, 12, 1>&      G,
        Eigen::Matrix<T, 12, 12>&     H,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        T                             mu,
        T                             eps_vh,
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

        Eigen::Matrix<T, 2, 1> gamma;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;
        T                      f;

        EE_friction_basis(
            f, gamma, basis, tan_rel_dx, kappa, d_hat, thickness, prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, Ea0, Ea1, Eb0, Eb1);

        Eigen::Matrix<T, 2, 12> J;
        edge_edge_jacobi(basis, gamma, J);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;

        Eigen::Matrix<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, f, eps_vh, tan_rel_dx);
        H = J.transpose() * H2x2 * J;
    }

    template <typename T>
    inline __device__ void EE_friction_gradient(
        Eigen::Matrix<T, 12, 1>&      G,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        T                             mu,
        T                             eps_vh,
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

        Eigen::Matrix<T, 2, 1> gamma;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;
        T                      f;

        EE_friction_basis(
            f, gamma, basis, tan_rel_dx, kappa, d_hat, thickness, prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, Ea0, Ea1, Eb0, Eb1);

        Eigen::Matrix<T, 2, 12> J;
        edge_edge_jacobi(basis, gamma, J);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;
    }

    template <typename T>
    inline __device__ void PE_friction_basis(
        T&                            f,
        T&                            eta,
        Eigen::Matrix<T, 3, 2>&       basis,
        Eigen::Matrix<T, 2, 1>&       tan_rel_dx,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        const Eigen::Matrix<T, 3, 1>& prev_P,
        const Eigen::Matrix<T, 3, 1>& prev_E0,
        const Eigen::Matrix<T, 3, 1>& prev_E1,
        const Eigen::Matrix<T, 3, 1>& P,
        const Eigen::Matrix<T, 3, 1>& E0,
        const Eigen::Matrix<T, 3, 1>& E1)
    {
        using namespace distance;
        using namespace friction;

        point_edge_closest_point(prev_P, prev_E0, prev_E1, eta);
        point_edge_tangent_basis(prev_P, prev_E0, prev_E1, basis);

        T prev_D;
        point_edge_distance2(prev_P, prev_E0, prev_E1, prev_D);

        f = normal_force(kappa, d_hat, thickness, prev_D);

        Eigen::Matrix<T, 3, 1> dP  = P - prev_P;
        Eigen::Matrix<T, 3, 1> dE0 = E0 - prev_E0;
        Eigen::Matrix<T, 3, 1> dE1 = E1 - prev_E1;

        point_edge_tan_rel_dx(dP, dE0, dE1, basis, eta, tan_rel_dx);
    }

    template <typename T>
    inline __device__ T PE_friction_energy(T                             kappa,
                                           T                             d_hat,
                                           T                             thickness,
                                           T                             mu,
                                           T                             eps_vh,
                                           const Eigen::Matrix<T, 3, 1>& prev_P,
                                           const Eigen::Matrix<T, 3, 1>& prev_E0,
                                           const Eigen::Matrix<T, 3, 1>& prev_E1,
                                           const Eigen::Matrix<T, 3, 1>& P,
                                           const Eigen::Matrix<T, 3, 1>& E0,
                                           const Eigen::Matrix<T, 3, 1>& E1)
    {
        using namespace friction;

        T                      eta;
        T                      f;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;

        PE_friction_basis(
            f, eta, basis, tan_rel_dx, kappa, d_hat, thickness, prev_P, prev_E0, prev_E1, P, E0, E1);

        return friction_energy(mu, f, eps_vh, tan_rel_dx);
    }

    template <typename T>
    inline __device__ void PE_friction_gradient_hessian(
        Eigen::Matrix<T, 9, 1>&       G,
        Eigen::Matrix<T, 9, 9>&       H,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        T                             mu,
        T                             eps_vh,
        const Eigen::Matrix<T, 3, 1>& prev_P,
        const Eigen::Matrix<T, 3, 1>& prev_E0,
        const Eigen::Matrix<T, 3, 1>& prev_E1,
        const Eigen::Matrix<T, 3, 1>& P,
        const Eigen::Matrix<T, 3, 1>& E0,
        const Eigen::Matrix<T, 3, 1>& E1)
    {
        using namespace friction;

        T                      eta;
        T                      f;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;

        PE_friction_basis(
            f, eta, basis, tan_rel_dx, kappa, d_hat, thickness, prev_P, prev_E0, prev_E1, P, E0, E1);

        Eigen::Matrix<T, 2, 9> J;
        point_edge_jacobi(basis, eta, J);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;

        Eigen::Matrix<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, f, eps_vh, tan_rel_dx);
        H = J.transpose() * H2x2 * J;
    }

    template <typename T>
    inline __device__ void PE_friction_gradient(
        Eigen::Matrix<T, 9, 1>&       G,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        T                             mu,
        T                             eps_vh,
        const Eigen::Matrix<T, 3, 1>& prev_P,
        const Eigen::Matrix<T, 3, 1>& prev_E0,
        const Eigen::Matrix<T, 3, 1>& prev_E1,
        const Eigen::Matrix<T, 3, 1>& P,
        const Eigen::Matrix<T, 3, 1>& E0,
        const Eigen::Matrix<T, 3, 1>& E1)
    {
        using namespace friction;

        T                      eta;
        T                      f;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;

        PE_friction_basis(
            f, eta, basis, tan_rel_dx, kappa, d_hat, thickness, prev_P, prev_E0, prev_E1, P, E0, E1);

        Eigen::Matrix<T, 2, 9> J;
        point_edge_jacobi(basis, eta, J);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;
    }

    template <typename T>
    inline __device__ void PP_friction_basis(
        T&                            f,
        Eigen::Matrix<T, 3, 2>&       basis,
        Eigen::Matrix<T, 2, 1>&       tan_rel_dx,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        const Eigen::Matrix<T, 3, 1>& prev_P0,
        const Eigen::Matrix<T, 3, 1>& prev_P1,
        const Eigen::Matrix<T, 3, 1>& P0,
        const Eigen::Matrix<T, 3, 1>& P1)
    {
        using namespace distance;
        using namespace friction;

        point_point_tangent_basis(prev_P0, prev_P1, basis);

        T prev_D;
        point_point_distance2(prev_P0, prev_P1, prev_D);

        f = normal_force(kappa, d_hat, thickness, prev_D);

        Eigen::Matrix<T, 3, 1> dP0 = P0 - prev_P0;
        Eigen::Matrix<T, 3, 1> dP1 = P1 - prev_P1;

        point_point_tan_rel_dx(dP0, dP1, basis, tan_rel_dx);
    }

    template <typename T>
    inline __device__ T PP_friction_energy(T                             kappa,
                                           T                             d_hat,
                                           T                             thickness,
                                           T                             mu,
                                           T                             eps_vh,
                                           const Eigen::Matrix<T, 3, 1>& prev_P0,
                                           const Eigen::Matrix<T, 3, 1>& prev_P1,
                                           const Eigen::Matrix<T, 3, 1>& P0,
                                           const Eigen::Matrix<T, 3, 1>& P1)
    {
        using namespace friction;

        T                      f;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;

        PP_friction_basis(
            f, basis, tan_rel_dx, kappa, d_hat, thickness, prev_P0, prev_P1, P0, P1);

        return friction_energy(mu, f, eps_vh, tan_rel_dx);
    }

    template <typename T>
    inline __device__ void PP_friction_gradient_hessian(
        Eigen::Matrix<T, 6, 1>&       G,
        Eigen::Matrix<T, 6, 6>&       H,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        T                             mu,
        T                             eps_vh,
        const Eigen::Matrix<T, 3, 1>& prev_P0,
        const Eigen::Matrix<T, 3, 1>& prev_P1,
        const Eigen::Matrix<T, 3, 1>& P0,
        const Eigen::Matrix<T, 3, 1>& P1)
    {
        using namespace friction;

        T                      f;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;

        PP_friction_basis(
            f, basis, tan_rel_dx, kappa, d_hat, thickness, prev_P0, prev_P1, P0, P1);

        Eigen::Matrix<T, 2, 6> J;
        point_point_jacobi(basis, J);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;

        Eigen::Matrix<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, f, eps_vh, tan_rel_dx);
        H = J.transpose() * H2x2 * J;
    }

    template <typename T>
    inline __device__ void PP_friction_gradient(
        Eigen::Matrix<T, 6, 1>&       G,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        T                             mu,
        T                             eps_vh,
        const Eigen::Matrix<T, 3, 1>& prev_P0,
        const Eigen::Matrix<T, 3, 1>& prev_P1,
        const Eigen::Matrix<T, 3, 1>& P0,
        const Eigen::Matrix<T, 3, 1>& P1)
    {
        using namespace friction;

        T                      f;
        Eigen::Matrix<T, 3, 2> basis;
        Eigen::Matrix<T, 2, 1> tan_rel_dx;

        PP_friction_basis(
            f, basis, tan_rel_dx, kappa, d_hat, thickness, prev_P0, prev_P1, P0, P1);

        Eigen::Matrix<T, 2, 6> J;
        point_point_jacobi(basis, J);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_rel_dx);
        G = J.transpose() * G2;
    }
}  // namespace sym::codim_ipc_contact
}  // namespace uipc::backend::cuda_mixed
