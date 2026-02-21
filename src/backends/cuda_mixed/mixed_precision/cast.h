#pragma once
#include <Eigen/Core>
#include <muda/muda.h>
#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

namespace uipc::backend::cuda_mixed
{
template <typename To, typename From>
MUDA_INLINE MUDA_GENERIC To safe_cast(From v) noexcept
{
#ifndef NDEBUG
    if constexpr(std::is_floating_point_v<From> && std::is_floating_point_v<To>
                 && (sizeof(To) < sizeof(From)))
    {
        const double dv = static_cast<double>(v);
        assert(::isfinite(dv) && "safe_cast: input contains NaN/Inf");
        const double lim = static_cast<double>(std::numeric_limits<To>::max());
        assert(::fabs(dv) <= lim && "safe_cast: narrowing overflow");
    }
#endif
    if constexpr(std::is_same_v<To, From>)
        return v;
    else
        return static_cast<To>(v);
}

template <typename To, typename FromMatrix>
MUDA_INLINE MUDA_GENERIC auto downcast_gradient(const FromMatrix& G) noexcept
{
    using From = typename FromMatrix::Scalar;
    constexpr int N = FromMatrix::RowsAtCompileTime;
    static_assert(N != Eigen::Dynamic, "Only fixed-size vectors are supported.");

    if constexpr(std::is_same_v<To, From>)
    {
        return G;
    }
    else
    {
#ifndef NDEBUG
#if !defined(__CUDA_ARCH__)
        assert(G.allFinite() && "downcast_gradient: source contains NaN/Inf");
#endif
#endif
        Eigen::Matrix<To, N, 1> out = G.template cast<To>();
#ifndef NDEBUG
#if !defined(__CUDA_ARCH__)
        assert(out.allFinite() && "downcast_gradient: narrowing produced NaN/Inf");
#endif
#endif
        return out;
    }
}

template <typename To, typename FromMatrix>
MUDA_INLINE MUDA_GENERIC auto downcast_hessian(const FromMatrix& H) noexcept
{
    using From = typename FromMatrix::Scalar;
    constexpr int R = FromMatrix::RowsAtCompileTime;
    constexpr int C = FromMatrix::ColsAtCompileTime;
    static_assert(R != Eigen::Dynamic && C != Eigen::Dynamic,
                  "Only fixed-size matrices are supported.");

    if constexpr(std::is_same_v<To, From>)
    {
        return H;
    }
    else
    {
#ifndef NDEBUG
#if !defined(__CUDA_ARCH__)
        assert(H.allFinite() && "downcast_hessian: source contains NaN/Inf");
#endif
#endif
        Eigen::Matrix<To, R, C> out = H.template cast<To>();
#ifndef NDEBUG
#if !defined(__CUDA_ARCH__)
        assert(out.allFinite() && "downcast_hessian: narrowing produced NaN/Inf");
#endif
#endif
        return out;
    }
}
}  // namespace uipc::backend::cuda_mixed

