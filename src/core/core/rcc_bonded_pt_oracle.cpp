#include <uipc/core/rcc_bonded_pt_oracle.h>

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <cmath>
#include <utility>

namespace uipc::core
{
namespace
{
using Matrix9x12 = Matrix<Float, 9, 12>;

Vector9 flatten(const Matrix3x3& A)
{
    Vector9 v;
    IndexT  k = 0;
    for(IndexT j = 0; j < 3; ++j)
        for(IndexT i = 0; i < 3; ++i, ++k)
            v[k] = A(i, j);
    return v;
}

Matrix3x3 ds(const Vector3& x0,
             const Vector3& x1,
             const Vector3& x2,
             const Vector3& x3)
{
    Matrix3x3 D;
    D.col(0) = x1 - x0;
    D.col(1) = x2 - x0;
    D.col(2) = x3 - x0;
    return D;
}

Matrix3x3 deformation_gradient(const RCCBondedPTVirtualTetInput& input)
{
    return ds(input.x0, input.x1, input.x2, input.x3) * input.Dm_inv;
}

Matrix9x12 dFdx(const Matrix3x3& Dm_inv)
{
    const Float m = Dm_inv(0, 0);
    const Float n = Dm_inv(0, 1);
    const Float o = Dm_inv(0, 2);
    const Float p = Dm_inv(1, 0);
    const Float q = Dm_inv(1, 1);
    const Float r = Dm_inv(1, 2);
    const Float s = Dm_inv(2, 0);
    const Float t = Dm_inv(2, 1);
    const Float u = Dm_inv(2, 2);

    const Float t1 = -m - p - s;
    const Float t2 = -n - q - t;
    const Float t3 = -o - r - u;

    Matrix9x12 P = Matrix9x12::Zero();
    P(0, 0)      = t1;
    P(0, 3)      = m;
    P(0, 6)      = p;
    P(0, 9)      = s;
    P(1, 1)      = t1;
    P(1, 4)      = m;
    P(1, 7)      = p;
    P(1, 10)     = s;
    P(2, 2)      = t1;
    P(2, 5)      = m;
    P(2, 8)      = p;
    P(2, 11)     = s;
    P(3, 0)      = t2;
    P(3, 3)      = n;
    P(3, 6)      = q;
    P(3, 9)      = t;
    P(4, 1)      = t2;
    P(4, 4)      = n;
    P(4, 7)      = q;
    P(4, 10)     = t;
    P(5, 2)      = t2;
    P(5, 5)      = n;
    P(5, 8)      = q;
    P(5, 11)     = t;
    P(6, 0)      = t3;
    P(6, 3)      = o;
    P(6, 6)      = r;
    P(6, 9)      = u;
    P(7, 1)      = t3;
    P(7, 4)      = o;
    P(7, 7)      = r;
    P(7, 10)     = u;
    P(8, 2)      = t3;
    P(8, 5)      = o;
    P(8, 8)      = r;
    P(8, 11)     = u;
    return P;
}

Matrix9x9 project_spd(const Matrix9x9& H)
{
    Matrix9x9 sym = 0.5 * (H + H.transpose());
    Eigen::SelfAdjointEigenSolver<Matrix9x9> solver(sym);
    Vector9 values = solver.eigenvalues();
    for(IndexT i = 0; i < values.size(); ++i)
        values[i] = values[i] > 0.0 ? values[i] : 0.0;
    return solver.eigenvectors() * values.asDiagonal() * solver.eigenvectors().transpose();
}

Matrix3x3 abd_ortho_gradient_F(const Matrix3x3& F, Float kappa)
{
    const Matrix3x3 C = F * F.transpose() - Matrix3x3::Identity();
    return 4.0 * kappa * C * F;
}

Matrix9x9 abd_ortho_hessian_F(const Matrix3x3& F, Float kappa)
{
    Matrix9x9 H = Matrix9x9::Zero();
    const Matrix3x3 FFT = F * F.transpose();

    for(IndexT col = 0; col < 3; ++col)
    {
        for(IndexT row = 0; row < 3; ++row)
        {
            Matrix3x3 dF = Matrix3x3::Zero();
            dF(row, col) = 1.0;

            const Matrix3x3 dG =
                4.0 * kappa
                * (dF * F.transpose() * F + F * dF.transpose() * F
                   + FFT * dF - dF);

            H.col(col * 3 + row) = flatten(dG);
        }
    }

    return 0.5 * (H + H.transpose());
}

// StableNeoHookean F-space energy/gradient/Hessian, reimplemented on the host as
// an INDEPENDENT reference (the GPU reporter calls sym::stable_neo_hookean_3d::*;
// the oracle mirrors the closed form from detail/stable_neo_hookean_3d.inl so the
// test is a genuine cross-check, not the same code path).
Float snh_energy_F(const Matrix3x3& F, Float mu, Float lambda)
{
    const Float J     = F.determinant();
    const Float Ic    = F.squaredNorm();
    const Float alpha = 1.0 + 0.75 * mu / lambda;
    return 0.5 * lambda * (J - alpha) * (J - alpha) + 0.5 * mu * (Ic - 3.0)
           - 0.5 * mu * std::log(Ic + 1.0);
}

Matrix3x3 snh_gradient_F(const Matrix3x3& F, Float mu, Float lambda)
{
    const Float J  = F.determinant();
    const Float Ic = F.squaredNorm();
    Matrix3x3   pJpF;
    pJpF(0, 0) = F(1, 1) * F(2, 2) - F(1, 2) * F(2, 1);
    pJpF(0, 1) = F(1, 2) * F(2, 0) - F(1, 0) * F(2, 2);
    pJpF(0, 2) = F(1, 0) * F(2, 1) - F(1, 1) * F(2, 0);
    pJpF(1, 0) = F(2, 1) * F(0, 2) - F(2, 2) * F(0, 1);
    pJpF(1, 1) = F(2, 2) * F(0, 0) - F(2, 0) * F(0, 2);
    pJpF(1, 2) = F(2, 0) * F(0, 1) - F(2, 1) * F(0, 0);
    pJpF(2, 0) = F(0, 1) * F(1, 2) - F(1, 1) * F(0, 2);
    pJpF(2, 1) = F(0, 2) * F(1, 0) - F(0, 0) * F(1, 2);
    pJpF(2, 2) = F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0);
    return mu * (1.0 - 1.0 / (Ic + 1.0)) * F
           + (lambda * (J - 1.0 - 0.75 * mu / lambda)) * pJpF;
}

Matrix9x9 snh_hessian_F(const Matrix3x3& F, Float mu, Float lambda)
{
    const Float J  = F.determinant();
    const Float Ic = F.squaredNorm();

    Matrix9x9 H1 = 2.0 * Matrix9x9::Identity();

    Vector9 g1;
    g1.segment<3>(0) = 2.0 * F.col(0);
    g1.segment<3>(3) = 2.0 * F.col(1);
    g1.segment<3>(6) = 2.0 * F.col(2);

    Vector9 gJ;
    gJ.segment<3>(0) = F.col(1).cross(F.col(2));
    gJ.segment<3>(3) = F.col(2).cross(F.col(0));
    gJ.segment<3>(6) = F.col(0).cross(F.col(1));

    Matrix3x3 f0hat, f1hat, f2hat;
    f0hat << 0, -F(2, 0), F(1, 0), F(2, 0), 0, -F(0, 0), -F(1, 0), F(0, 0), 0;
    f1hat << 0, -F(2, 1), F(1, 1), F(2, 1), 0, -F(0, 1), -F(1, 1), F(0, 1), 0;
    f2hat << 0, -F(2, 2), F(1, 2), F(2, 2), 0, -F(0, 2), -F(1, 2), F(0, 2), 0;

    Matrix9x9 HJ;
    HJ.block<3, 3>(0, 0) = Matrix3x3::Zero();
    HJ.block<3, 3>(0, 3) = -f2hat;
    HJ.block<3, 3>(0, 6) = f1hat;
    HJ.block<3, 3>(3, 0) = f2hat;
    HJ.block<3, 3>(3, 3) = Matrix3x3::Zero();
    HJ.block<3, 3>(3, 6) = -f0hat;
    HJ.block<3, 3>(6, 0) = -f1hat;
    HJ.block<3, 3>(6, 3) = f0hat;
    HJ.block<3, 3>(6, 6) = Matrix3x3::Zero();

    return (Ic * mu) / (2.0 * (Ic + 1.0)) * H1
           + lambda * (J - 1.0 - (3.0 * mu) / (4.0 * lambda)) * HJ
           + (mu / (2.0 * (Ic + 1.0) * (Ic + 1.0))) * g1 * g1.transpose()
           + lambda * gJ * gJ.transpose();
}
}  // namespace

RCCBondedPTRestShape
build_rcc_bonded_pt_rest_shape_svts(const RCCBondedPTRestShapeInput& input)
{
    RCCBondedPTRestShape out;
    out.oriented_topo      = input.topo;
    out.conditioned_point  = input.point;

    Vector3 x0 = input.point;
    Vector3 x1 = input.tri0;
    Vector3 x2 = input.tri1;
    Vector3 x3 = input.tri2;

    Vector3 e1     = x2 - x1;
    Vector3 e2     = x3 - x1;
    Vector3 normal = e1.cross(e2);
    Float   nrm    = normal.norm();
    if(nrm < input.triangle_degeneracy_tol)
        return out;

    normal /= nrm;
    out.normal = normal;

    Float signed_dist = normal.dot(x0 - x1);
    out.signed_distance = signed_dist;

    // Band-edge rest (rest_height_target > 0): the rest point is placed at
    // exactly the target height on its current side, so the bond's
    // equilibrium gap is the target (normally xi + d_hat) regardless of the
    // creation-time distance. Otherwise the legacy clamp only guards
    // degeneracy.
    Float sign = (signed_dist >= 0) ? 1.0 : -1.0;
    if(input.rest_height_target > 0.0)
    {
        x0 = x0 + (sign * input.rest_height_target - signed_dist) * normal;
    }
    else if(std::abs(signed_dist) < input.min_separate_distance)
    {
        x0 = x0 + (sign * input.min_separate_distance - signed_dist) * normal;
    }

    out.conditioned_point = x0;
    out.conditioned_signed_distance = normal.dot(x0 - x1);

    Matrix3x3 Dm;
    Dm.col(0) = x1 - x0;
    Dm.col(1) = x2 - x0;
    Dm.col(2) = x3 - x0;

    Float det = Dm.determinant();
    if(det < 0)
    {
        std::swap(x1, x2);
        std::swap(out.oriented_topo[1], out.oriented_topo[2]);
        Dm.col(0) = x1 - x0;
        Dm.col(1) = x2 - x0;
        det = -det;
    }

    out.Dm          = Dm;
    out.Dm_inv      = Dm.inverse();
    out.rest_volume = (1.0 / 6.0) * det;
    out.valid       = true;
    return out;
}

RCCBondedPTVirtualTetOracle
build_rcc_bonded_pt_virtual_tet_oracle(const RCCBondedPTVirtualTetInput& input)
{
    RCCBondedPTVirtualTetOracle out;
    if(input.rest_volume <= 0.0 || input.dt == 0.0)
        return out;

    out.F = deformation_gradient(input);
    const Matrix3x3& F = out.F;
    const Float      Vdt2 = input.rest_volume * input.dt * input.dt;

    Float     psi = 0.0;
    Matrix3x3 dpsi_dF_mat = Matrix3x3::Zero();
    Matrix9x9 ddpsi_ddF = Matrix9x9::Zero();

    switch(input.energy_model)
    {
        case RCCBondedPTVirtualTetEnergyModel::ABDOrtho:
        {
            if(input.kappa <= 0.0)
                return out;
            const Matrix3x3 C = F * F.transpose() - Matrix3x3::Identity();
            psi         = input.kappa * C.squaredNorm();
            dpsi_dF_mat = abd_ortho_gradient_F(F, input.kappa);
            ddpsi_ddF   = abd_ortho_hessian_F(F, input.kappa);
            break;
        }
        case RCCBondedPTVirtualTetEnergyModel::StableNeoHookean:
        {
            if(input.mu <= 0.0 || input.lambda <= 0.0)
                return out;
            psi         = snh_energy_F(F, input.mu, input.lambda);
            dpsi_dF_mat = snh_gradient_F(F, input.mu, input.lambda);
            ddpsi_ddF   = snh_hessian_F(F, input.mu, input.lambda);
            break;
        }
        default:
            return out;
    }

    if(input.project_hessian_to_spd)
        ddpsi_ddF = project_spd(ddpsi_ddF);

    Matrix9x12 PFPx = dFdx(input.Dm_inv);

    out.energy   = psi * Vdt2;
    out.dpsi_dF  = flatten(dpsi_dF_mat);
    out.ddpsi_ddF = ddpsi_ddF;
    out.gradient = PFPx.transpose() * (out.dpsi_dF * Vdt2);
    out.hessian  = PFPx.transpose() * (out.ddpsi_ddF * Vdt2) * PFPx;
    out.valid    = true;
    return out;
}
}  // namespace uipc::core
