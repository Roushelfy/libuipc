#pragma once
#include <type_define.h>
#include <contact_system/contact_coeff.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>

namespace uipc::backend::cuda_mixed
{
namespace sym::codim_ipc_simplex_contact
{
    inline __device__ Float PT_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector4i&                     cids)
    {
        Float kappa = 0.0;
        for(int j = 1; j < 4; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
        }
        return kappa / 3.0;
    }

    inline __device__ Float EE_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector4i&                     cids)
    {
        Float kappa = 0.0;
        for(int j = 0; j < 2; ++j)
        {
            for(int k = 2; k < 4; ++k)
            {
                ContactCoeff coeff = table(cids[j], cids[k]);
                kappa += coeff.kappa;
            }
        }
        return kappa / 4.0;
    }

    inline __device__ Float PE_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector3i&                     cids)
    {
        Float kappa = 0.0;
        for(int j = 1; j < 3; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
        }
        return kappa / 2.0;
    }

    inline __device__ Float PP_kappa(const muda::CDense2D<ContactCoeff>& table,
                                     const Vector2i&                     cids)
    {
        ContactCoeff coeff = table(cids[0], cids[1]);
        return coeff.kappa;
    }

    template <typename T>
    inline __device__ T PT_barrier_energy(T                             kappa,
                                          T                             d_hat,
                                          T                             thickness,
                                          const Eigen::Matrix<T, 3, 1>& P,
                                          const Eigen::Matrix<T, 3, 1>& T0,
                                          const Eigen::Matrix<T, 3, 1>& T1,
                                          const Eigen::Matrix<T, 3, 1>& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        T D;
        point_triangle_distance2(P, T0, T1, T2, D);
        T B;
        KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    template <typename T>
    inline __device__ T PT_barrier_energy(const Vector4i&               flag,
                                          T                             kappa,
                                          T                             d_hat,
                                          T                             thickness,
                                          const Eigen::Matrix<T, 3, 1>& P,
                                          const Eigen::Matrix<T, 3, 1>& T0,
                                          const Eigen::Matrix<T, 3, 1>& T1,
                                          const Eigen::Matrix<T, 3, 1>& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        T D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);
        T B;
        KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    template <typename T>
    inline __device__ void PT_barrier_gradient_hessian(
        Eigen::Matrix<T, 12, 1>&      G,
        Eigen::Matrix<T, 12, 12>&     H,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        const Eigen::Matrix<T, 3, 1>& P,
        const Eigen::Matrix<T, 3, 1>& T0,
        const Eigen::Matrix<T, 3, 1>& T1,
        const Eigen::Matrix<T, 3, 1>& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        T D;
        point_triangle_distance2(P, T0, T1, T2, D);

        Eigen::Matrix<T, 12, 1> GradD;
        point_triangle_distance2_gradient(P, T0, T1, T2, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;

        T ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Eigen::Matrix<T, 12, 12> HessD;
        point_triangle_distance2_hessian(P, T0, T1, T2, HessD);

        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    template <typename T>
    inline __device__ void PT_barrier_gradient_hessian(
        Eigen::Matrix<T, 12, 1>&      G,
        Eigen::Matrix<T, 12, 12>&     H,
        const Vector4i&               flag,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        const Eigen::Matrix<T, 3, 1>& P,
        const Eigen::Matrix<T, 3, 1>& T0,
        const Eigen::Matrix<T, 3, 1>& T1,
        const Eigen::Matrix<T, 3, 1>& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        T D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        Eigen::Matrix<T, 12, 1> GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;

        T ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Eigen::Matrix<T, 12, 12> HessD;
        point_triangle_distance2_hessian(flag, P, T0, T1, T2, HessD);

        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    template <typename T>
    inline __device__ void PT_barrier_gradient(Eigen::Matrix<T, 12, 1>&      G,
                                               const Vector4i&               flag,
                                               T                             kappa,
                                               T                             d_hat,
                                               T                             thickness,
                                               const Eigen::Matrix<T, 3, 1>& P,
                                               const Eigen::Matrix<T, 3, 1>& T0,
                                               const Eigen::Matrix<T, 3, 1>& T1,
                                               const Eigen::Matrix<T, 3, 1>& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        T D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        Eigen::Matrix<T, 12, 1> GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }

    template <typename T>
    inline __device__ T mollified_EE_barrier_energy(
        const Vector4i&               flag,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        const Eigen::Matrix<T, 3, 1>& t0_Ea0,
        const Eigen::Matrix<T, 3, 1>& t0_Ea1,
        const Eigen::Matrix<T, 3, 1>& t0_Eb0,
        const Eigen::Matrix<T, 3, 1>& t0_Eb1,
        const Eigen::Matrix<T, 3, 1>& Ea0,
        const Eigen::Matrix<T, 3, 1>& Ea1,
        const Eigen::Matrix<T, 3, 1>& Eb0,
        const Eigen::Matrix<T, 3, 1>& Eb1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        T D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);
        T B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        T eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        T ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        return ek * B;
    }

    template <typename T>
    inline __device__ void mollified_EE_barrier_gradient_hessian(
        Eigen::Matrix<T, 12, 1>&      G,
        Eigen::Matrix<T, 12, 12>&     H,
        const Vector4i&               flag,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        const Eigen::Matrix<T, 3, 1>& t0_Ea0,
        const Eigen::Matrix<T, 3, 1>& t0_Ea1,
        const Eigen::Matrix<T, 3, 1>& t0_Eb0,
        const Eigen::Matrix<T, 3, 1>& t0_Eb1,
        const Eigen::Matrix<T, 3, 1>& Ea0,
        const Eigen::Matrix<T, 3, 1>& Ea1,
        const Eigen::Matrix<T, 3, 1>& Eb0,
        const Eigen::Matrix<T, 3, 1>& Eb1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        T D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        Eigen::Matrix<T, 12, 1> GradD;
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        Eigen::Matrix<T, 12, 12> HessD;
        edge_edge_distance2_hessian(flag, Ea0, Ea1, Eb0, Eb1, HessD);

        T B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        T ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Eigen::Matrix<T, 12, 1> GradB = dBdD * GradD;
        Eigen::Matrix<T, 12, 12> HessB =
            ddBddD * GradD * GradD.transpose() + dBdD * HessD;

        T eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        T ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        Eigen::Matrix<T, 12, 1> Gradek;
        edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);

        Eigen::Matrix<T, 12, 12> Hessek;
        edge_edge_mollifier_hessian(Ea0, Ea1, Eb0, Eb1, eps_x, Hessek);

        G = Gradek * B + ek * GradB;
        H = Hessek * B + Gradek * GradB.transpose() + GradB * Gradek.transpose()
            + ek * HessB;
    }

    template <typename T>
    inline __device__ void mollified_EE_barrier_gradient(
        Eigen::Matrix<T, 12, 1>&      G,
        const Vector4i&               flag,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        const Eigen::Matrix<T, 3, 1>& t0_Ea0,
        const Eigen::Matrix<T, 3, 1>& t0_Ea1,
        const Eigen::Matrix<T, 3, 1>& t0_Eb0,
        const Eigen::Matrix<T, 3, 1>& t0_Eb1,
        const Eigen::Matrix<T, 3, 1>& Ea0,
        const Eigen::Matrix<T, 3, 1>& Ea1,
        const Eigen::Matrix<T, 3, 1>& Eb0,
        const Eigen::Matrix<T, 3, 1>& Eb1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        T D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        Eigen::Matrix<T, 12, 1> GradD;
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        T B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Eigen::Matrix<T, 12, 1> GradB = dBdD * GradD;

        T eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        T ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        Eigen::Matrix<T, 12, 1> Gradek;
        edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);

        G = Gradek * B + ek * GradB;
    }

    template <typename T>
    inline __device__ T PE_barrier_energy(const Vector3i&               flag,
                                          T                             kappa,
                                          T                             d_hat,
                                          T                             thickness,
                                          const Eigen::Matrix<T, 3, 1>& P,
                                          const Eigen::Matrix<T, 3, 1>& E0,
                                          const Eigen::Matrix<T, 3, 1>& E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        T D = static_cast<T>(0.0);
        point_edge_distance2(flag, P, E0, E1, D);
        T E = static_cast<T>(0.0);
        KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    template <typename T>
    inline __device__ void PE_barrier_gradient_hessian(
        Eigen::Matrix<T, 9, 1>&       G,
        Eigen::Matrix<T, 9, 9>&       H,
        const Vector3i&               flag,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        const Eigen::Matrix<T, 3, 1>& P,
        const Eigen::Matrix<T, 3, 1>& E0,
        const Eigen::Matrix<T, 3, 1>& E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        T D = static_cast<T>(0.0);
        point_edge_distance2(flag, P, E0, E1, D);

        Eigen::Matrix<T, 9, 1> GradD;
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        Eigen::Matrix<T, 9, 9> HessD;
        point_edge_distance2_hessian(flag, P, E0, E1, HessD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;

        T ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    template <typename T>
    inline __device__ void PE_barrier_gradient(Eigen::Matrix<T, 9, 1>&       G,
                                               const Vector3i&               flag,
                                               T                             kappa,
                                               T                             d_hat,
                                               T                             thickness,
                                               const Eigen::Matrix<T, 3, 1>& P,
                                               const Eigen::Matrix<T, 3, 1>& E0,
                                               const Eigen::Matrix<T, 3, 1>& E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        T D = static_cast<T>(0.0);
        point_edge_distance2(flag, P, E0, E1, D);

        Eigen::Matrix<T, 9, 1> GradD;
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;
    }

    template <typename T>
    inline __device__ T PP_barrier_energy(const Vector2i&               flag,
                                          T                             kappa,
                                          T                             d_hat,
                                          T                             thickness,
                                          const Eigen::Matrix<T, 3, 1>& P0,
                                          const Eigen::Matrix<T, 3, 1>& P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        T D = static_cast<T>(0.0);
        point_point_distance2(flag, P0, P1, D);
        T E = static_cast<T>(0.0);
        KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    template <typename T>
    inline __device__ void PP_barrier_gradient_hessian(
        Eigen::Matrix<T, 6, 1>&       G,
        Eigen::Matrix<T, 6, 6>&       H,
        const Vector2i&               flag,
        T                             kappa,
        T                             d_hat,
        T                             thickness,
        const Eigen::Matrix<T, 3, 1>& P0,
        const Eigen::Matrix<T, 3, 1>& P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        T D = static_cast<T>(0.0);
        point_point_distance2(flag, P0, P1, D);

        Eigen::Matrix<T, 6, 1> GradD;
        point_point_distance2_gradient(flag, P0, P1, GradD);

        Eigen::Matrix<T, 6, 6> HessD;
        point_point_distance2_hessian(flag, P0, P1, HessD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;

        T ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    template <typename T>
    inline __device__ void PP_barrier_gradient(Eigen::Matrix<T, 6, 1>&       G,
                                               const Vector2i&               flag,
                                               T                             kappa,
                                               T                             d_hat,
                                               T                             thickness,
                                               const Eigen::Matrix<T, 3, 1>& P0,
                                               const Eigen::Matrix<T, 3, 1>& P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        T D = static_cast<T>(0.0);
        point_point_distance2(flag, P0, P1, D);

        Eigen::Matrix<T, 6, 1> GradD;
        point_point_distance2_gradient(flag, P0, P1, GradD);

        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;
    }
}  // namespace sym::codim_ipc_simplex_contact
}  // namespace uipc::backend::cuda_mixed
