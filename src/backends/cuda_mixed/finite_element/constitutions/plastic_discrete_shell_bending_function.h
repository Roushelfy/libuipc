#pragma once
#include <type_define.h>
#include <utils/dihedral_angle.h>
#include <cmath>

namespace uipc::backend::cuda_mixed
{
namespace sym::plastic_discrete_shell_bending
{
#include "sym/discrete_shell_bending.inl"

    template <typename T>
    inline UIPC_GENERIC constexpr T pi()
    {
        return static_cast<T>(3.14159265358979323846264338327950288);
    }

    template <typename T>
    inline UIPC_GENERIC constexpr T plasticity_write_threshold()
    {
        return static_cast<T>(1e-6);
    }

    template <typename T>
    inline UIPC_GENERIC constexpr T dihedral_guard_eps()
    {
        return static_cast<T>(1e-12);
    }

    template <typename T>
    inline UIPC_GENERIC bool is_finite_scalar(T v)
    {
        return isfinite(v);
    }

    template <typename T>
    inline UIPC_GENERIC T abs_value(T v)
    {
        return v < T(0) ? -v : v;
    }

    template <typename T>
    inline UIPC_GENERIC bool is_finite_vec3(const Eigen::Matrix<T, 3, 1>& v)
    {
        return is_finite_scalar(v[0]) && is_finite_scalar(v[1]) && is_finite_scalar(v[2]);
    }

    template <typename T>
    inline UIPC_GENERIC T wrap_angle(T angle)
    {
        constexpr T Pi    = pi<T>();
        constexpr T TwoPi = static_cast<T>(2.0) * Pi;

        while(angle > Pi)
            angle -= TwoPi;

        while(angle < -Pi)
            angle += TwoPi;

        return angle;
    }

    template <typename T>
    inline UIPC_GENERIC T angle_delta(T theta, T theta_bar)
    {
        return wrap_angle(theta - theta_bar);
    }

    template <typename T>
    inline UIPC_GENERIC bool safe_dihedral_angle(const Eigen::Matrix<T, 3, 1>& v0,
                                                 const Eigen::Matrix<T, 3, 1>& v1,
                                                 const Eigen::Matrix<T, 3, 1>& v2,
                                                 const Eigen::Matrix<T, 3, 1>& v3,
                                                 T&                            theta)
    {
        if(!is_finite_vec3(v0) || !is_finite_vec3(v1) || !is_finite_vec3(v2)
           || !is_finite_vec3(v3))
            return false;

        const Eigen::Matrix<T, 3, 1> n1 = (v1 - v0).cross(v2 - v0);
        const Eigen::Matrix<T, 3, 1> n2 = (v2 - v3).cross(v1 - v3);

        const T n1_sq = n1.squaredNorm();
        const T n2_sq = n2.squaredNorm();
        const T eps   = dihedral_guard_eps<T>();

        if(!is_finite_scalar(n1_sq) || !is_finite_scalar(n2_sq) || n1_sq <= eps || n2_sq <= eps)
            return false;

        const T denom = sqrt(n1_sq * n2_sq);
        if(!is_finite_scalar(denom) || denom <= eps)
            return false;

        T cos_theta = n1.dot(n2) / denom;
        if(!is_finite_scalar(cos_theta))
            return false;

        cos_theta = cos_theta < T(-1) ? T(-1) : cos_theta;
        cos_theta = cos_theta > T(1) ? T(1) : cos_theta;
        theta     = acos(cos_theta);
        if(!is_finite_scalar(theta))
            return false;

        if(n2.cross(n1).dot(v1 - v2) < 0)
            theta = -theta;

        return is_finite_scalar(theta);
    }

    template <typename T>
    inline UIPC_GENERIC bool update_plastic_state(T  theta,
                                                  T& theta_bar,
                                                  T& yield_threshold,
                                                  T  hardening_modulus)
    {
        if(!is_finite_scalar(theta) || !is_finite_scalar(theta_bar)
           || !is_finite_scalar(yield_threshold) || !is_finite_scalar(hardening_modulus)
           || yield_threshold < T(0) || hardening_modulus < T(0))
            return false;

        const T delta          = angle_delta(theta, theta_bar);
        const T plastic_excess = abs_value(delta) - yield_threshold;

        if(!is_finite_scalar(plastic_excess)
           || plastic_excess <= plasticity_write_threshold<T>())
            return false;

        const T direction = delta >= T(0) ? T(1) : T(-1);

        theta_bar       = wrap_angle(theta_bar + direction * plastic_excess);
        yield_threshold = yield_threshold + hardening_modulus * plastic_excess;

        return is_finite_scalar(theta_bar) && is_finite_scalar(yield_threshold)
               && yield_threshold >= T(0);
    }

    template <typename T>
    inline UIPC_GENERIC bool try_angle_delta(const Eigen::Matrix<T, 3, 1>& x0,
                                             const Eigen::Matrix<T, 3, 1>& x1,
                                             const Eigen::Matrix<T, 3, 1>& x2,
                                             const Eigen::Matrix<T, 3, 1>& x3,
                                             T                            theta_bar,
                                             T&                           theta,
                                             T&                           delta)
    {
        if(!is_finite_scalar(theta_bar))
            return false;

        if(!safe_dihedral_angle(x0, x1, x2, x3, theta))
            return false;

        delta = angle_delta(theta, theta_bar);
        return is_finite_scalar(delta);
    }

    template <typename T>
    inline UIPC_GENERIC void compute_constants(T&                            L0,
                                               T&                            h_bar,
                                               T&                            theta_bar,
                                               T&                            V_bar,
                                               const Eigen::Matrix<T, 3, 1>& x0_bar,
                                               const Eigen::Matrix<T, 3, 1>& x1_bar,
                                               const Eigen::Matrix<T, 3, 1>& x2_bar,
                                               const Eigen::Matrix<T, 3, 1>& x3_bar,
                                               T                             thickness0,
                                               T                             thickness1,
                                               T                             thickness2,
                                               T                             thickness3)
    {
        L0         = (x2_bar - x1_bar).norm();
        Eigen::Matrix<T, 3, 1> n1 = (x1_bar - x0_bar).cross(x2_bar - x0_bar);
        Eigen::Matrix<T, 3, 1> n2 = (x2_bar - x3_bar).cross(x1_bar - x3_bar);
        T                      A  = (n1.norm() + n2.norm()) / static_cast<T>(2.0);
        h_bar                     = A / static_cast<T>(3.0) / L0;
        dihedral_angle(x0_bar, x1_bar, x2_bar, x3_bar, theta_bar);

        T thickness =
            (thickness0 + thickness1 + thickness2 + thickness3) / static_cast<T>(4.0);
        V_bar = A * thickness;
    }

    template <typename T>
    inline UIPC_GENERIC T E(const Eigen::Matrix<T, 3, 1>& x0,
                            const Eigen::Matrix<T, 3, 1>& x1,
                            const Eigen::Matrix<T, 3, 1>& x2,
                            const Eigen::Matrix<T, 3, 1>& x3,
                            T                             L0,
                            T                             h_bar,
                            T                             theta_bar,
                            T                             kappa)
    {
        namespace PDSB = sym::plastic_discrete_shell_bending;
        T theta = static_cast<T>(0.0);
        T delta = static_cast<T>(0.0);
        if(!PDSB::try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta))
            return static_cast<T>(0.0);

        T R;
        PDSB::E(R, kappa, theta_bar + delta, theta_bar, L0, h_bar);
        return R;
    }

    template <typename T>
    inline UIPC_GENERIC void dEdx(Eigen::Matrix<T, 12, 1>&      G,
                                  const Eigen::Matrix<T, 3, 1>& x0,
                                  const Eigen::Matrix<T, 3, 1>& x1,
                                  const Eigen::Matrix<T, 3, 1>& x2,
                                  const Eigen::Matrix<T, 3, 1>& x3,
                                  T                             L0,
                                  T                             h_bar,
                                  T                             theta_bar,
                                  T                             kappa)
    {
        namespace PDSB = sym::plastic_discrete_shell_bending;
        T theta = static_cast<T>(0.0);
        T delta = static_cast<T>(0.0);
        if(!PDSB::try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta))
        {
            G.setZero();
            return;
        }

        T dEdtheta;
        PDSB::dEdtheta(dEdtheta, kappa, theta_bar + delta, theta_bar, L0, h_bar);

        Eigen::Matrix<T, 12, 1> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        G = dEdtheta * dthetadx;
    }

    template <typename T>
    inline UIPC_GENERIC void ddEddx(Eigen::Matrix<T, 12, 12>&    H,
                                    const Eigen::Matrix<T, 3, 1>& x0,
                                    const Eigen::Matrix<T, 3, 1>& x1,
                                    const Eigen::Matrix<T, 3, 1>& x2,
                                    const Eigen::Matrix<T, 3, 1>& x3,
                                    T                              L0,
                                    T                              h_bar,
                                    T                              theta_bar,
                                    T                              kappa)
    {
        namespace PDSB = sym::plastic_discrete_shell_bending;
        T theta = static_cast<T>(0.0);
        T delta = static_cast<T>(0.0);
        if(!PDSB::try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta))
        {
            H.setZero();
            return;
        }

        T dEdtheta;
        PDSB::dEdtheta(dEdtheta, kappa, theta_bar + delta, theta_bar, L0, h_bar);

        T ddEddtheta;
        PDSB::ddEddtheta(ddEddtheta, kappa, theta_bar + delta, theta_bar, L0, h_bar);

        Eigen::Matrix<T, 12, 1> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        Eigen::Matrix<T, 12, 12> ddthetaddx;
        dihedral_angle_hessian(x0, x1, x2, x3, ddthetaddx);

        H = dthetadx * ddEddtheta * dthetadx.transpose() + dEdtheta * ddthetaddx;
    }
}  // namespace sym::plastic_discrete_shell_bending
}  // namespace uipc::backend::cuda_mixed
