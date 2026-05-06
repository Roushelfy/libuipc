#pragma once
#include <finite_element/matrix_utils.h>
#include <Eigen/Core>
namespace uipc::backend::cuda_mixed::fem
{
UIPC_GENERIC Float invariant2(const Matrix3x3& F);
UIPC_GENERIC Float invariant2(const Vector3& Sigma);
UIPC_GENERIC Float invariant3(const Matrix3x3& F);
UIPC_GENERIC Float invariant3(const Vector3& Sigma);
UIPC_GENERIC Float invariant4(const Matrix3x3& F, const Vector3& a);
UIPC_GENERIC Float invariant5(const Matrix3x3& F, const Vector3& a);

//tex: $\frac{\partial \det(\mathbf{F})}{\partial F}$
UIPC_GENERIC Matrix3x3 dJdF(const Matrix3x3& F);

// inverse material coordinates
UIPC_GENERIC Matrix3x3 Dm_inv(const Vector3& X0, const Vector3& X1, const Vector3& X2, const Vector3& X3);
UIPC_GENERIC Matrix3x3 Ds(const Vector3& X0, const Vector3& X1, const Vector3& X2, const Vector3& X3);
UIPC_GENERIC Matrix9x12 dFdx(const Matrix3x3& DmInv);
UIPC_GENERIC Matrix3x3    F(const Vector3& x0,
                          const Vector3& x1,
                          const Vector3& x2,
                          const Vector3& x3,
                          const Matrix3x3& DmInv);

template <typename T>
UIPC_GENERIC Eigen::Matrix<T, 3, 3> Ds(const Eigen::Matrix<T, 3, 1>& x0,
                                        const Eigen::Matrix<T, 3, 1>& x1,
                                        const Eigen::Matrix<T, 3, 1>& x2,
                                        const Eigen::Matrix<T, 3, 1>& x3)
{
    Eigen::Matrix<T, 3, 3> Ds;
    Ds.col(0) = x1 - x0;
    Ds.col(1) = x2 - x0;
    Ds.col(2) = x3 - x0;
    return Ds;
}

template <typename T>
UIPC_GENERIC Eigen::Matrix<T, 3, 3> F(const Eigen::Matrix<T, 3, 1>& x0,
                                       const Eigen::Matrix<T, 3, 1>& x1,
                                       const Eigen::Matrix<T, 3, 1>& x2,
                                       const Eigen::Matrix<T, 3, 1>& x3,
                                       const Eigen::Matrix<T, 3, 3>& DmInv)
{
    auto ds = Ds<T>(x0, x1, x2, x3);
    return ds * DmInv;
}

template <typename T>
UIPC_GENERIC Eigen::Matrix<T, 9, 12> dFdx(const Eigen::Matrix<T, 3, 3>& DmInv)
{
    const T m = DmInv(0, 0);
    const T n = DmInv(0, 1);
    const T o = DmInv(0, 2);
    const T p = DmInv(1, 0);
    const T q = DmInv(1, 1);
    const T r = DmInv(1, 2);
    const T s = DmInv(2, 0);
    const T t = DmInv(2, 1);
    const T u = DmInv(2, 2);

    const T t1 = -m - p - s;
    const T t2 = -n - q - t;
    const T t3 = -o - r - u;

    Eigen::Matrix<T, 9, 12> PFPu = Eigen::Matrix<T, 9, 12>::Zero();
    PFPu(0, 0)                   = t1;
    PFPu(0, 3)                   = m;
    PFPu(0, 6)                   = p;
    PFPu(0, 9)                   = s;
    PFPu(1, 1)                   = t1;
    PFPu(1, 4)                   = m;
    PFPu(1, 7)                   = p;
    PFPu(1, 10)                  = s;
    PFPu(2, 2)                   = t1;
    PFPu(2, 5)                   = m;
    PFPu(2, 8)                   = p;
    PFPu(2, 11)                  = s;
    PFPu(3, 0)                   = t2;
    PFPu(3, 3)                   = n;
    PFPu(3, 6)                   = q;
    PFPu(3, 9)                   = t;
    PFPu(4, 1)                   = t2;
    PFPu(4, 4)                   = n;
    PFPu(4, 7)                   = q;
    PFPu(4, 10)                  = t;
    PFPu(5, 2)                   = t2;
    PFPu(5, 5)                   = n;
    PFPu(5, 8)                   = q;
    PFPu(5, 11)                  = t;
    PFPu(6, 0)                   = t3;
    PFPu(6, 3)                   = o;
    PFPu(6, 6)                   = r;
    PFPu(6, 9)                   = u;
    PFPu(7, 1)                   = t3;
    PFPu(7, 4)                   = o;
    PFPu(7, 7)                   = r;
    PFPu(7, 10)                  = u;
    PFPu(8, 2)                   = t3;
    PFPu(8, 5)                   = o;
    PFPu(8, 8)                   = r;
    PFPu(8, 11)                  = u;
    return PFPu;
}
}  // namespace uipc::backend::cuda_mixed
