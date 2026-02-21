#pragma once
#include <type_define.h>
#include <muda/ext/eigen/evd.h>

namespace uipc::backend::cuda_mixed
{
template <typename T, int N>
UIPC_GENERIC void make_spd(Matrix<T, N, N>& H)
{
    Vector<T, N>    eigen_values;
    Matrix<T, N, N> eigen_vectors;
    muda::eigen::template evd<T, N>(H, eigen_values, eigen_vectors);
    for(int i = 0; i < N; ++i)
    {
        auto& v = eigen_values(i);
        v       = v < T{0} ? T{0} : v;
    }
    H = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
}
}
