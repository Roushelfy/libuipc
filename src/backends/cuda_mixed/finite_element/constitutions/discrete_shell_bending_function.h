#pragma once
#include <type_define.h>
#include <utils/dihedral_angle.h>
//ref: https://www.cs.columbia.edu/cg/pdfs/10_ds.pdf
namespace uipc::backend::cuda_mixed
{
namespace sym::discrete_shell_bending
{
#include "sym/discrete_shell_bending.inl"

    template <typename T>
    inline UIPC_GENERIC void compute_constants(T&                             L0,
                                               T&                             h_bar,
                                               T&                             theta_bar,
                                               T&                             V_bar,
                                               const Eigen::Matrix<T, 3, 1>& x0_bar,
                                               const Eigen::Matrix<T, 3, 1>& x1_bar,
                                               const Eigen::Matrix<T, 3, 1>& x2_bar,
                                               const Eigen::Matrix<T, 3, 1>& x3_bar,
                                               T                              thickness0,
                                               T                              thickness1,
                                               T                              thickness2,
                                               T                              thickness3)

    {
        L0         = (x2_bar - x1_bar).norm();
        Eigen::Matrix<T, 3, 1> n1 = (x1_bar - x0_bar).cross(x2_bar - x0_bar);
        Eigen::Matrix<T, 3, 1> n2 = (x2_bar - x3_bar).cross(x1_bar - x3_bar);
        T                      A  = (n1.norm() + n2.norm()) / T{2};
        h_bar      = A / 3.0 / L0;
        dihedral_angle(x0_bar, x1_bar, x2_bar, x3_bar, theta_bar);

        T thickness = (thickness0 + thickness1 + thickness2 + thickness3) / T{4};
        V_bar = A * thickness;
    }

    template <typename T>
    inline UIPC_GENERIC T E(const Eigen::Matrix<T, 3, 1>& x0,
                            const Eigen::Matrix<T, 3, 1>& x1,
                            const Eigen::Matrix<T, 3, 1>& x2,
                            const Eigen::Matrix<T, 3, 1>& x3,
                            T                              L0,
                            T                              h_bar,
                            T                              theta_bar,
                            T                              kappa)
    {

        namespace DSB = sym::discrete_shell_bending;
        T theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        T R;
        DSB::E(R, kappa, theta, theta_bar, L0, h_bar);

        return R;
    }

    template <typename T>
    inline UIPC_GENERIC void dEdx(Eigen::Matrix<T, 12, 1>&      G,
                                  const Eigen::Matrix<T, 3, 1>& x0,
                                  const Eigen::Matrix<T, 3, 1>& x1,
                                  const Eigen::Matrix<T, 3, 1>& x2,
                                  const Eigen::Matrix<T, 3, 1>& x3,
                                  T                              L0,
                                  T                              h_bar,
                                  T                              theta_bar,
                                  T                              kappa)
    {
        namespace DSB = sym::discrete_shell_bending;
        T theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        T dEdtheta;
        DSB::dEdtheta(dEdtheta, kappa, theta, theta_bar, L0, h_bar);

        Eigen::Matrix<T, 12, 1> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        G = dEdtheta * dthetadx;
    }

    template <typename T>
    inline UIPC_GENERIC void ddEddx(Eigen::Matrix<T, 12, 12>&   H,
                                    const Eigen::Matrix<T, 3, 1>& x0,
                                    const Eigen::Matrix<T, 3, 1>& x1,
                                    const Eigen::Matrix<T, 3, 1>& x2,
                                    const Eigen::Matrix<T, 3, 1>& x3,
                                    T                              L0,
                                    T                              h_bar,
                                    T                              theta_bar,
                                    T                              kappa)
    {
        namespace DSB = sym::discrete_shell_bending;
        T theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        T dEdtheta;
        DSB::dEdtheta(dEdtheta, kappa, theta, theta_bar, L0, h_bar);

        T ddEddtheta;
        DSB::ddEddtheta(ddEddtheta, kappa, theta, theta_bar, L0, h_bar);

        Eigen::Matrix<T, 12, 1> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        Eigen::Matrix<T, 12, 12> ddthetaddx;
        dihedral_angle_hessian(x0, x1, x2, x3, ddthetaddx);


        H = dthetadx * ddEddtheta * dthetadx.transpose() + dEdtheta * ddthetaddx;
    }

}  // namespace sym::discrete_shell_bending
}  // namespace uipc::backend::cuda_mixed
