#pragma once
#include <type_define.h>
#include <muda/ext/eigen/evd.h>
#include <utils/distance.h>
#include <utils/distance/distance_flagged.h>
#include <muda/ext/eigen/evd.h>
#include <utils/friction_utils.h>

namespace uipc::backend::cuda_mixed
{
namespace sym::codim_ipc_contact
{
#include "sym/codim_ipc_contact.inl"

    // C1 clamping
    template <typename T>
    inline __device__ void f0(T x2, T epsvh, T& f0)
    {
        if(x2 >= epsvh * epsvh)
        {
            //tex: $$y$$
            f0 = std::sqrt(x2);
        }
        else
        {
            //tex: $$\frac{y^{2}}{\epsilon_{x}} + \frac{1}{3 \epsilon_{x}} - \frac{y^{3}}{3 \epsilon_{x}^{2}}$$
            f0 = x2 * (-std::sqrt(x2) / 3.0 + epsvh) / (epsvh * epsvh) + epsvh / 3.0;
        }
    }

    template <typename T>
    inline __device__ void f1_div_rel_dx_norm(T x2, T epsvh, T& result)
    {
        if(x2 >= epsvh * epsvh)
        {
            //tex: $$ \frac{1}{y}$$
            result = 1 / std::sqrt(x2);
        }
        else
        {
            //tex: $$ \frac{2 \epsilon_{x} - y}{ \epsilon_{x}^{2}}$$
            result = (-std::sqrt(x2) + 2.0 * epsvh) / (epsvh * epsvh);
        }
    }

    template <typename T>
    inline __device__ void f2_term(T x2, T epsvh, T& term)
    {
        term = -1 / (epsvh * epsvh);
        // same for x2 >= epsvh * epsvh for C1 clamped friction
    }


    template <typename T>
    inline __device__ T friction_energy(T mu,
                                        T lambda,
                                        T eps_vh,
                                        const Eigen::Matrix<T, 2, 1>& tan_rel_x)
    {
        T f0_val;
        f0(tan_rel_x.squaredNorm(), eps_vh, f0_val);
        return mu * lambda * f0_val;
    }

    template <typename T>
    inline __device__ void friction_gradient(Eigen::Matrix<T, 2, 1>&       G2,
                                             T                              mu,
                                             T                              lambda,
                                             T                              eps_vh,
                                             const Eigen::Matrix<T, 2, 1>& tan_rel_x)
    {
        T f1_val;
        f1_div_rel_dx_norm(tan_rel_x.squaredNorm(), eps_vh, f1_val);
        G2 = mu * lambda * f1_val * tan_rel_x;
    }

    template <typename T>
    inline __device__ void make_spd(Eigen::Matrix<T, 2, 2>& mat)
    {
        Eigen::Matrix<T, 2, 1> eigen_values;
        Eigen::Matrix<T, 2, 2> eigen_vectors;
        muda::eigen::template evd(mat, eigen_values, eigen_vectors);
#pragma unroll
        for(int i = 0; i < 2; ++i)
        {
            auto& v = eigen_values(i);
            v       = v < 0.0 ? 0.0 : v;
        }
        mat = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
    };

    template <typename T>
    inline __device__ void friction_hessian(Eigen::Matrix<T, 2, 2>&       H2x2,
                                            T                              mu,
                                            T                              lambda,
                                            T                              eps_vh,
                                            const Eigen::Matrix<T, 2, 1>& tan_rel_x)
    {
        T sq_norm = tan_rel_x.squaredNorm();
        T epsvh2  = eps_vh * eps_vh;
        T f1_div_rel_dx_norm_val;
        f1_div_rel_dx_norm(tan_rel_x.squaredNorm(), eps_vh, f1_div_rel_dx_norm_val);

        if(sq_norm >= epsvh2)
        {

            // no SPD projection needed
            Eigen::Matrix<T, 2, 1> ubar(-tan_rel_x[1], tan_rel_x[0]);
            H2x2 = ubar * (mu * lambda * f1_div_rel_dx_norm_val / sq_norm)
                   * ubar.transpose();
        }
        else
        {
            if(sq_norm == 0.0)
            {
                // no SPD projection needed
                H2x2 = (mu * lambda * f1_div_rel_dx_norm_val)
                       * Eigen::Matrix<T, 2, 2>::Identity();
            }
            else
            {
                T f2_term_val;
                f2_term(sq_norm, eps_vh, f2_term_val);

                T relDXNorm = std::sqrt(sq_norm);

                // only need to project the inner 2x2 matrix to SPD
                H2x2 = tan_rel_x * (f2_term_val / relDXNorm) * tan_rel_x.transpose();
                H2x2.diagonal().array() += f1_div_rel_dx_norm_val;

                make_spd(H2x2);

                H2x2 *= mu * lambda;
            }
        }
    }

    template <typename T>
    inline __device__ T normal_force(T kappa, T d_hat, T thickness, T D)
    {
        using namespace distance;
        T d = std::sqrt(D);
        T dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        T dBdd = dBdD * T{2} * d;
        return -dBdd;  // > 0
    }
}  // namespace sym::codim_ipc_contact
}  // namespace uipc::backend::cuda_mixed
