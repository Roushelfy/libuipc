#pragma once

#include <type_define.h>
#include <utils/dihedral_angle.h>
#include <finite_element/constitutions/plastic_discrete_shell_bending_function.h>
#include <cmath>

namespace uipc::backend::cuda_mixed
{
namespace sym::stress_plastic_discrete_shell_bending
{
    template <typename T>
    inline UIPC_GENERIC constexpr T pi()
    {
        return sym::plastic_discrete_shell_bending::pi<T>();
    }

    template <typename T>
    inline UIPC_GENERIC constexpr T plasticity_write_threshold()
    {
        return sym::plastic_discrete_shell_bending::plasticity_write_threshold<T>();
    }

    template <typename T>
    inline UIPC_GENERIC constexpr T dihedral_guard_eps()
    {
        return sym::plastic_discrete_shell_bending::dihedral_guard_eps<T>();
    }

    template <typename T>
    inline UIPC_GENERIC bool is_finite_scalar(T v)
    {
        return sym::plastic_discrete_shell_bending::is_finite_scalar(v);
    }

    template <typename T>
    inline UIPC_GENERIC T abs_value(T v)
    {
        return sym::plastic_discrete_shell_bending::abs_value(v);
    }

    template <typename T>
    inline UIPC_GENERIC T max_value(T a, T b)
    {
        return a < b ? b : a;
    }

    template <typename T>
    inline UIPC_GENERIC T sign_value(T v)
    {
        return v >= T(0) ? T(1) : T(-1);
    }

    template <typename T>
    inline UIPC_GENERIC bool is_finite_vec3(const Eigen::Matrix<T, 3, 1>& v)
    {
        return sym::plastic_discrete_shell_bending::is_finite_vec3(v);
    }

    template <typename T>
    inline UIPC_GENERIC T wrap_angle(T angle)
    {
        return sym::plastic_discrete_shell_bending::wrap_angle(angle);
    }

    template <typename T>
    inline UIPC_GENERIC T angle_delta(T theta, T theta_bar)
    {
        return sym::plastic_discrete_shell_bending::angle_delta(theta, theta_bar);
    }

    template <typename T>
    inline UIPC_GENERIC bool safe_dihedral_angle(const Eigen::Matrix<T, 3, 1>& v0,
                                                 const Eigen::Matrix<T, 3, 1>& v1,
                                                 const Eigen::Matrix<T, 3, 1>& v2,
                                                 const Eigen::Matrix<T, 3, 1>& v3,
                                                 T&                            theta)
    {
        return sym::plastic_discrete_shell_bending::safe_dihedral_angle(
            v0, v1, v2, v3, theta);
    }

    template <typename T>
    inline UIPC_GENERIC bool try_trial_state(T  theta,
                                             T  theta_bar,
                                             T  kappa,
                                             T  L0,
                                             T  h_bar,
                                             T  yield_stress,
                                             T& delta,
                                             T& tau_trial,
                                             T& theta_y,
                                             T& delta_gamma)
    {
        if(!is_finite_scalar(theta) || !is_finite_scalar(theta_bar) || !is_finite_scalar(kappa)
           || !is_finite_scalar(L0) || !is_finite_scalar(h_bar)
           || !is_finite_scalar(yield_stress) || yield_stress < T(0) || L0 <= T(0)
           || h_bar <= T(0))
            return false;

        const T elastic_slope = static_cast<T>(2.0) * kappa * L0 / h_bar;
        if(!is_finite_scalar(elastic_slope) || elastic_slope <= T(0))
            return false;

        delta       = angle_delta(theta, theta_bar);
        tau_trial   = elastic_slope * delta;
        theta_y     = yield_stress / elastic_slope;
        delta_gamma = max_value(abs_value(delta) - theta_y, T(0));

        return is_finite_scalar(delta) && is_finite_scalar(tau_trial)
               && is_finite_scalar(theta_y) && theta_y >= T(0)
               && is_finite_scalar(delta_gamma) && delta_gamma >= T(0);
    }

    template <typename T>
    inline UIPC_GENERIC bool augmented_response_from_angle_delta(T  delta,
                                                                 T  kappa,
                                                                 T  L0,
                                                                 T  h_bar,
                                                                 T  yield_stress,
                                                                 T& energy,
                                                                 T& dEdtheta,
                                                                 T& ddEddtheta)
    {
        energy     = T(0);
        dEdtheta   = T(0);
        ddEddtheta = T(0);

        if(!is_finite_scalar(delta) || !is_finite_scalar(kappa) || !is_finite_scalar(L0)
           || !is_finite_scalar(h_bar) || !is_finite_scalar(yield_stress) || yield_stress < T(0)
           || L0 <= T(0) || h_bar <= T(0))
            return false;

        const T elastic_scale = kappa * L0 / h_bar;
        const T elastic_slope = static_cast<T>(2.0) * elastic_scale;
        if(!is_finite_scalar(elastic_scale) || !is_finite_scalar(elastic_slope)
           || elastic_slope <= T(0))
            return false;

        const T theta_y = yield_stress / elastic_slope;
        if(!is_finite_scalar(theta_y) || theta_y < T(0))
            return false;

        if(abs_value(delta) <= theta_y)
        {
            energy     = elastic_scale * delta * delta;
            dEdtheta   = elastic_slope * delta;
            ddEddtheta = elastic_slope;
        }
        else
        {
            const T delta_gamma = abs_value(delta) - theta_y;
            energy              = elastic_scale * theta_y * theta_y + yield_stress * delta_gamma;
            dEdtheta            = sign_value(delta) * yield_stress;
            ddEddtheta          = T(0);
        }

        return is_finite_scalar(energy) && is_finite_scalar(dEdtheta)
               && is_finite_scalar(ddEddtheta);
    }

    template <typename T>
    inline UIPC_GENERIC bool update_plastic_state(T  theta,
                                                  T& theta_bar,
                                                  T& yield_stress,
                                                  T  hardening_modulus,
                                                  T  kappa,
                                                  T  L0,
                                                  T  h_bar)
    {
        if(!is_finite_scalar(theta) || !is_finite_scalar(theta_bar)
           || !is_finite_scalar(yield_stress) || !is_finite_scalar(hardening_modulus)
           || hardening_modulus < T(0))
            return false;

        T delta       = T(0);
        T tau_trial   = T(0);
        T theta_y     = T(0);
        T delta_gamma = T(0);
        if(!try_trial_state(
               theta, theta_bar, kappa, L0, h_bar, yield_stress, delta, tau_trial, theta_y, delta_gamma))
            return false;

        if(delta_gamma <= plasticity_write_threshold<T>())
            return false;

        theta_bar    = wrap_angle(theta_bar + sign_value(delta) * delta_gamma);
        yield_stress = yield_stress + hardening_modulus * delta_gamma;

        return is_finite_scalar(theta_bar) && is_finite_scalar(yield_stress)
               && yield_stress >= T(0);
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
        auto n1    = (x1_bar - x0_bar).cross(x2_bar - x0_bar);
        auto n2    = (x2_bar - x3_bar).cross(x1_bar - x3_bar);
        T    A     = (n1.norm() + n2.norm()) / static_cast<T>(2.0);
        h_bar      = A / static_cast<T>(3.0) / L0;
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
                            T                             kappa,
                            T                             yield_stress)
    {
        T theta = T(0);
        T delta = T(0);
        if(!try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta))
            return T(0);

        T energy     = T(0);
        T dEdtheta   = T(0);
        T ddEddtheta = T(0);
        if(!augmented_response_from_angle_delta(
               delta, kappa, L0, h_bar, yield_stress, energy, dEdtheta, ddEddtheta))
            return T(0);

        return energy;
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
                                  T                             kappa,
                                  T                             yield_stress)
    {
        T theta = T(0);
        T delta = T(0);
        if(!try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta))
        {
            G.setZero();
            return;
        }

        T energy     = T(0);
        T dEdtheta   = T(0);
        T ddEddtheta = T(0);
        if(!augmented_response_from_angle_delta(
               delta, kappa, L0, h_bar, yield_stress, energy, dEdtheta, ddEddtheta))
        {
            G.setZero();
            return;
        }

        Eigen::Matrix<T, 12, 1> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);
        G = dEdtheta * dthetadx;
    }

    template <typename T>
    inline UIPC_GENERIC void ddEddx(Eigen::Matrix<T, 12, 12>&     H,
                                    const Eigen::Matrix<T, 3, 1>& x0,
                                    const Eigen::Matrix<T, 3, 1>& x1,
                                    const Eigen::Matrix<T, 3, 1>& x2,
                                    const Eigen::Matrix<T, 3, 1>& x3,
                                    T                             L0,
                                    T                             h_bar,
                                    T                             theta_bar,
                                    T                             kappa,
                                    T                             yield_stress)
    {
        T theta = T(0);
        T delta = T(0);
        if(!try_angle_delta(x0, x1, x2, x3, theta_bar, theta, delta))
        {
            H.setZero();
            return;
        }

        T energy     = T(0);
        T dEdtheta   = T(0);
        T ddEddtheta = T(0);
        if(!augmented_response_from_angle_delta(
               delta, kappa, L0, h_bar, yield_stress, energy, dEdtheta, ddEddtheta))
        {
            H.setZero();
            return;
        }

        Eigen::Matrix<T, 12, 1> dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        Eigen::Matrix<T, 12, 12> ddthetaddx;
        dihedral_angle_hessian(x0, x1, x2, x3, ddthetaddx);

        H = dthetadx * ddEddtheta * dthetadx.transpose() + dEdtheta * ddthetaddx;
    }
}  // namespace sym::stress_plastic_discrete_shell_bending
}  // namespace uipc::backend::cuda_mixed
