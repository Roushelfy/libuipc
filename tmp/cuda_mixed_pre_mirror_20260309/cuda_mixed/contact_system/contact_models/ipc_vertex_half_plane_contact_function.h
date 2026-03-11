#pragma once
#include <type_define.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>

namespace uipc::backend::cuda_mixed
{
namespace sym::ipc_vertex_half_contact
{
#include "sym/vertex_half_plane_distance.inl"

    template <typename T>
    inline __device__ T PH_barrier_energy(T                              kappa,
                                          T                              d_hat,
                                          T                              thickness,
                                          const Eigen::Matrix<T, 3, 1>& v,
                                          const Eigen::Matrix<T, 3, 1>& P,
                                          const Eigen::Matrix<T, 3, 1>& N)
    {
        using namespace codim_ipc_contact;

        T D;
        HalfPlaneD(D, v, P, N);
        T E = T{0};
        KappaBarrier(E, kappa, D, d_hat, thickness);

        return E;
    }

    template <typename T>
    inline __device__ void PH_barrier_gradient_hessian(Eigen::Matrix<T, 3, 1>&       G,
                                                       Eigen::Matrix<T, 3, 3>&       H,
                                                       T                              kappa,
                                                       T                              d_hat,
                                                       T                              thickness,
                                                       const Eigen::Matrix<T, 3, 1>& v,
                                                       const Eigen::Matrix<T, 3, 1>& P,
                                                       const Eigen::Matrix<T, 3, 1>& N)
    {
        using namespace codim_ipc_contact;

        T D;
        HalfPlaneD(D, v, P, N);

        T dBdD = T{0};
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Eigen::Matrix<T, 3, 1> dDdx;
        dHalfPlaneDdx(dDdx, v, P, N);

        G = dBdD * dDdx;

        T ddBddD = T{0};
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Eigen::Matrix<T, 3, 3> ddDddx;
        ddHalfPlaneDddx(ddDddx, v, P, N);

        H = ddBddD * dDdx * dDdx.transpose() + dBdD * ddDddx;
    }

    template <typename T>
    inline __device__ void PH_barrier_gradient(Eigen::Matrix<T, 3, 1>&       G,
                                               T                              kappa,
                                               T                              d_hat,
                                               T                              thickness,
                                               const Eigen::Matrix<T, 3, 1>& v,
                                               const Eigen::Matrix<T, 3, 1>& P,
                                               const Eigen::Matrix<T, 3, 1>& N)
    {
        using namespace codim_ipc_contact;

        T D;
        HalfPlaneD(D, v, P, N);

        T dBdD = T{0};
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Eigen::Matrix<T, 3, 1> dDdx;
        dHalfPlaneDdx(dDdx, v, P, N);

        G = dBdD * dDdx;
    }

    template <typename T>
    inline __device__ void compute_tan_basis(Eigen::Matrix<T, 3, 1>&       e1,
                                             Eigen::Matrix<T, 3, 1>&       e2,
                                             const Eigen::Matrix<T, 3, 1>& N)
    {
        using namespace codim_ipc_contact;

        Eigen::Matrix<T, 3, 1> trial = Eigen::Matrix<T, 3, 1>::UnitX();
        if(N.dot(trial) > T{0.9})
        {
            trial = Eigen::Matrix<T, 3, 1>::UnitZ();
            e1    = trial.cross(N).normalized();
        }
        else
        {
            e1 = trial.cross(N).normalized();
        }
        e2 = N.cross(e1);
    }


    template <typename T>
    inline __device__ T PH_friction_energy(T                              kappa,
                                           T                              d_hat,
                                           T                              thickness,
                                           T                              mu,
                                           T                              eps_vh,
                                           const Eigen::Matrix<T, 3, 1>& prev_v,
                                           const Eigen::Matrix<T, 3, 1>& v,
                                           const Eigen::Matrix<T, 3, 1>& P,
                                           const Eigen::Matrix<T, 3, 1>& N)
    {
        using namespace codim_ipc_contact;

        T prev_D;
        HalfPlaneD(prev_D, prev_v, P, N);
        T f = normal_force(kappa, d_hat, thickness, prev_D);

        Eigen::Matrix<T, 3, 1> e1, e2;
        compute_tan_basis(e1, e2, N);

        Eigen::Matrix<T, 2, 1> tan_dV;

        TR(tan_dV, v, prev_v, e1, e2);

        return friction_energy(mu, f, eps_vh, tan_dV);
    }

    template <typename T>
    inline __device__ void PH_friction_gradient_hessian(Eigen::Matrix<T, 3, 1>&       G,
                                                        Eigen::Matrix<T, 3, 3>&       H,
                                                        T                              kappa,
                                                        T                              d_hat,
                                                        T                              thickness,
                                                        T                              mu,
                                                        T                              eps_vh,
                                                        const Eigen::Matrix<T, 3, 1>& prev_v,
                                                        const Eigen::Matrix<T, 3, 1>& v,
                                                        const Eigen::Matrix<T, 3, 1>& P,
                                                        const Eigen::Matrix<T, 3, 1>& N)
    {
        using namespace codim_ipc_contact;

        T prev_D;
        HalfPlaneD(prev_D, prev_v, P, N);
        T f = normal_force(kappa, d_hat, thickness, prev_D);

        Eigen::Matrix<T, 3, 1> e1, e2;
        compute_tan_basis(e1, e2, N);

        Eigen::Matrix<T, 2, 1> tan_dV;

        TR(tan_dV, v, prev_v, e1, e2);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_dV);

        Eigen::Matrix<T, 2, 3> J;
        dTRdx(J, v, prev_v, e1, e2);

        G = J.transpose() * G2;

        Eigen::Matrix<T, 2, 2> H2x2;
        friction_hessian(H2x2, mu, f, eps_vh, tan_dV);

        H = J.transpose() * H2x2 * J;
    }

    template <typename T>
    inline __device__ void PH_friction_gradient(Eigen::Matrix<T, 3, 1>&       G,
                                                T                              kappa,
                                                T                              d_hat,
                                                T                              thickness,
                                                T                              mu,
                                                T                              eps_vh,
                                                const Eigen::Matrix<T, 3, 1>& prev_v,
                                                const Eigen::Matrix<T, 3, 1>& v,
                                                const Eigen::Matrix<T, 3, 1>& P,
                                                const Eigen::Matrix<T, 3, 1>& N)
    {
        using namespace codim_ipc_contact;

        T prev_D;
        HalfPlaneD(prev_D, prev_v, P, N);
        T f = normal_force(kappa, d_hat, thickness, prev_D);

        Eigen::Matrix<T, 3, 1> e1, e2;
        compute_tan_basis(e1, e2, N);

        Eigen::Matrix<T, 2, 1> tan_dV;
        TR(tan_dV, v, prev_v, e1, e2);

        Eigen::Matrix<T, 2, 1> G2;
        friction_gradient(G2, mu, f, eps_vh, tan_dV);

        Eigen::Matrix<T, 2, 3> J;
        dTRdx(J, v, prev_v, e1, e2);

        G = J.transpose() * G2;
    }
}  // namespace sym::ipc_vertex_half_contact
}  // namespace uipc::backend::cuda_mixed
