#include <muda/ext/eigen/atomic.h>
#include <mixed_precision/cast.h>
namespace uipc::backend::cuda_mixed
{
template <typename Scalar>
MUDA_GENERIC Vector<Scalar, 3> ABDJacobi::point_x_t(const Vector<Scalar, 12>& q) const
{
    const auto& t  = q.segment<3>(0);
    const auto& a1 = q.segment<3>(3);
    const auto& a2 = q.segment<3>(6);
    const auto& a3 = q.segment<3>(9);
    auto        x  = m_x_bar.template cast<Scalar>();
    return Vector<Scalar, 3>{x.dot(a1), x.dot(a2), x.dot(a3)} + t;
}

template <typename Scalar>
MUDA_GENERIC Vector<Scalar, 3> ABDJacobi::vec_x_t(const Vector<Scalar, 12>& q) const
{
    const auto& a1 = q.segment<3>(3);
    const auto& a2 = q.segment<3>(6);
    const auto& a3 = q.segment<3>(9);
    auto        x  = m_x_bar.template cast<Scalar>();
    return Vector<Scalar, 3>{x.dot(a1), x.dot(a2), x.dot(a3)};
}

template <typename Scalar>
MUDA_GENERIC Matrix<Scalar, 3, 12> ABDJacobi::to_mat_t() const
{
    Matrix<Scalar, 3, 12> ret = Matrix<Scalar, 3, 12>::Zero();
    auto                  x   = m_x_bar.template cast<Scalar>();
    ret(0, 0)            = 1;
    ret(1, 1)            = 1;
    ret(2, 2)            = 1;
    ret.template block<1, 3>(0, 3) = x.transpose();
    ret.template block<1, 3>(1, 6) = x.transpose();
    ret.template block<1, 3>(2, 9) = x.transpose();
    return ret;
}

template <typename Scalar>
MUDA_GENERIC Vector<Scalar, 12> ABDJacobi::ABDJacobiT::mul(
    const Vector<Scalar, 3>& g) const
{
    Vector<Scalar, 12> g12;
    auto               x = m_j.m_x_bar.template cast<Scalar>();
    g12.template segment<3>(0) = g;
    g12.template segment<3>(3) = x * g.x();
    g12.template segment<3>(6) = x * g.y();
    g12.template segment<3>(9) = x * g.z();
    return g12;
}

template <typename Scalar>
MUDA_GENERIC Matrix<Scalar, 12, 12> ABDJacobi::JT_H_J_t(
    const ABDJacobiT&               lhs_J_T,
    const Matrix<Scalar, 3, 3>&     Hessian,
    const ABDJacobi&                rhs_J)
{
    Matrix<Scalar, 12, 12> ret = Matrix<Scalar, 12, 12>::Zero();
    auto                   x   = lhs_J_T.J().x_bar().template cast<Scalar>();
    auto                   y   = rhs_J.x_bar().template cast<Scalar>();
    ret.template block<3, 3>(0, 0) = Hessian;

    ret.template block<3, 3>(0, 3) = Hessian.template block<3, 1>(0, 0) * y.transpose();
    ret.template block<3, 3>(0, 6) = Hessian.template block<3, 1>(0, 1) * y.transpose();
    ret.template block<3, 3>(0, 9) = Hessian.template block<3, 1>(0, 2) * y.transpose();

    ret.template block<3, 3>(3, 0) = x * Hessian.template block<1, 3>(0, 0);
    ret.template block<3, 3>(6, 0) = x * Hessian.template block<1, 3>(1, 0);
    ret.template block<3, 3>(9, 0) = x * Hessian.template block<1, 3>(2, 0);

    Matrix<Scalar, 3, 3> x_y = x * y.transpose();

    ret.template block<3, 3>(3, 3) = x_y * Hessian(0, 0);
    ret.template block<3, 3>(3, 6) = x_y * Hessian(0, 1);
    ret.template block<3, 3>(3, 9) = x_y * Hessian(0, 2);

    ret.template block<3, 3>(6, 3) = x_y * Hessian(1, 0);
    ret.template block<3, 3>(6, 6) = x_y * Hessian(1, 1);
    ret.template block<3, 3>(6, 9) = x_y * Hessian(1, 2);

    ret.template block<3, 3>(9, 3) = x_y * Hessian(2, 0);
    ret.template block<3, 3>(9, 6) = x_y * Hessian(2, 1);
    ret.template block<3, 3>(9, 9) = x_y * Hessian(2, 2);

    return ret;
}

template <size_t N>
template <typename Scalar>
MUDA_GENERIC Vector<Scalar, 3 * N> ABDJacobiStack<N>::mul(
    const Vector<Scalar, 12>& q) const
{
    Vector<Scalar, 3 * N> ret;
#pragma unroll
    for(size_t i = 0; i < N; ++i)
    {
        ret.template segment<3>(3 * i) = m_jacobis[i]->point_x_t(q);
    }
    return ret;
}

template <size_t N>
MUDA_GENERIC Vector<Float, 3 * N> ABDJacobiStack<N>::operator*(const Vector12& q) const
{
    return mul<Float>(q);
}

template <size_t N>
template <typename Scalar>
MUDA_GENERIC Matrix<Scalar, 3 * N, 12> ABDJacobiStack<N>::to_mat_t() const
{
    Matrix<Scalar, 3 * N, 12> ret;
    for(size_t i = 0; i < N; ++i)
    {
        ret.template block<3, 12>(3 * i, 0) =
            m_jacobis[i]->template to_mat_t<Scalar>();
    }
    return ret;
}

template <size_t N>
MUDA_GENERIC Matrix<Float, 3 * N, 12> ABDJacobiStack<N>::to_mat() const
{
    return to_mat_t<Float>();
}

template <size_t N>
template <typename Scalar>
MUDA_GENERIC Vector<Scalar, 12>
ABDJacobiStack<N>::ABDJacobiStackT::mul(const Vector<Scalar, 3 * N>& g) const
{
    Vector<Scalar, 12> ret = Vector<Scalar, 12>::Zero();
#pragma unroll
    for(size_t i = 0; i < N; ++i)
    {
        const ABDJacobi* jacobi = m_origin.m_jacobis[i];
        ret += jacobi->T().template mul<Scalar>(g.template segment<3>(3 * i));
    }
    return ret;
}

template <size_t N>
MUDA_GENERIC Vector12
ABDJacobiStack<N>::ABDJacobiStackT::operator*(const Vector<Float, 3 * N>& g) const
{
    return mul<Float>(g);
}

template <typename Scalar>
MUDA_GENERIC void ABDJacobiDyadicMass::add_to_t(Matrix<Scalar, 12, 12>& h) const
{
    const Scalar m = safe_cast<Scalar>(m_mass);
    const auto   x = m_mass_times_x_bar.template cast<Scalar>();
    const auto   D = m_mass_times_dyadic_x_bar.template cast<Scalar>();

    h(0, 0) += m;
    h.template block<1, 3>(0, 3) += x.transpose();
    h.template block<3, 1>(3, 0) += x;

    h(1, 1) += m;
    h.template block<1, 3>(1, 6) += x.transpose();
    h.template block<3, 1>(6, 1) += x;

    h(2, 2) += m;
    h.template block<1, 3>(2, 9) += x.transpose();
    h.template block<3, 1>(9, 2) += x;

    h.template block<3, 3>(3, 3) += D;
    h.template block<3, 3>(6, 6) += D;
    h.template block<3, 3>(9, 9) += D;
}

template <typename Scalar>
MUDA_GENERIC Matrix<Scalar, 12, 12> ABDJacobiDyadicMass::to_mat_t() const
{
    Matrix<Scalar, 12, 12> h = Matrix<Scalar, 12, 12>::Zero();
    add_to_t(h);
    return h;
}
}  // namespace uipc::backend::cuda_mixed
