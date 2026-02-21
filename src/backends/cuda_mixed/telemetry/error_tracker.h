#pragma once
#include <muda/muda.h>
#include <muda/buffer.h>
#include <muda/cub/device/device_reduce.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace uipc::backend::cuda_mixed
{
struct ErrorMetrics
{
    double rel_l2       = 0.0;
    double abs_linf     = 0.0;
    bool   nan_inf_flag = false;
};

template <typename ScalarA, typename ScalarB>
inline ErrorMetrics compute_error_metrics(muda::CDenseVectorView<ScalarA> v_test,
                                          muda::CDenseVectorView<ScalarB> v_ref,
                                          double eps = 1e-15)
{
    ErrorMetrics metrics{};
    const auto   n = v_test.size();
    if(n == 0 || n != v_ref.size())
    {
        metrics.nan_inf_flag = (n != v_ref.size());
        return metrics;
    }

    muda::DeviceBuffer<double> sq_diff;
    muda::DeviceBuffer<double> sq_ref;
    muda::DeviceBuffer<double> abs_diff;
    sq_diff.resize(n);
    sq_ref.resize(n);
    abs_diff.resize(n);

    muda::ParallelFor()
        .kernel_name("error_metrics")
        .apply(n,
               [test = v_test.cviewer().name("test"),
                ref = v_ref.cviewer().name("ref"),
                sqd = sq_diff.viewer().name("sq_diff"),
                sqr = sq_ref.viewer().name("sq_ref"),
                abd = abs_diff.viewer().name("abs_diff")] __device__(int i) mutable
               {
                   const double t = static_cast<double>(test(i));
                   const double r = static_cast<double>(ref(i));
                   const double d = t - r;
                   sqd(i)         = d * d;
                   sqr(i)         = r * r;
                   abd(i)         = ::fabs(d);
               });

    muda::DeviceVar<double> sum_sq_diff{0.0};
    muda::DeviceVar<double> sum_sq_ref{0.0};
    muda::DeviceVar<double> max_abs_diff{0.0};

    muda::DeviceReduce()
        .Sum(sq_diff.data(), sum_sq_diff.data(), static_cast<int>(n))
        .Sum(sq_ref.data(), sum_sq_ref.data(), static_cast<int>(n))
        .Max(abs_diff.data(), max_abs_diff.data(), static_cast<int>(n));

    const double h_sum_sq_diff = sum_sq_diff;
    const double h_sum_sq_ref  = sum_sq_ref;
    const double h_abs_linf    = max_abs_diff;

    const double denom = std::max(std::sqrt(h_sum_sq_ref), eps);
    metrics.rel_l2       = std::sqrt(h_sum_sq_diff) / denom;
    metrics.abs_linf     = h_abs_linf;
    metrics.nan_inf_flag = !(std::isfinite(metrics.rel_l2) && std::isfinite(metrics.abs_linf));
    return metrics;
}
}  // namespace uipc::backend::cuda_mixed

