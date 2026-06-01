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

Matrix3x3 dJdF(const Matrix3x3& F)
{
    Matrix3x3 G;
    G.col(0) = F.col(1).cross(F.col(2));
    G.col(1) = F.col(2).cross(F.col(0));
    G.col(2) = F.col(0).cross(F.col(1));
    return G;
}

Matrix3x3 skew(const Vector3& v)
{
    Matrix3x3 H;
    H << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return H;
}

Matrix9x9 hessian_J(const Matrix3x3& F)
{
    const Matrix3x3 f0hat = skew(F.col(0));
    const Matrix3x3 f1hat = skew(F.col(1));
    const Matrix3x3 f2hat = skew(F.col(2));

    Matrix9x9 H = Matrix9x9::Zero();
    H.block<3, 3>(0, 3) = -f2hat;
    H.block<3, 3>(0, 6) = f1hat;
    H.block<3, 3>(3, 0) = f2hat;
    H.block<3, 3>(3, 6) = -f0hat;
    H.block<3, 3>(6, 0) = -f1hat;
    H.block<3, 3>(6, 3) = f0hat;
    return H;
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

    if(std::abs(signed_dist) < input.min_separate_distance)
    {
        Float sign = (signed_dist >= 0) ? 1.0 : -1.0;
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
    if(input.rest_volume <= 0.0 || input.lambda == 0.0 || input.dt == 0.0)
        return out;

    out.F = deformation_gradient(input);
    const Matrix3x3& F = out.F;
    const Float      J = F.determinant();
    const Float      Jm1 = J - 1.0;
    const Float      Vdt2 = input.rest_volume * input.dt * input.dt;

    const Matrix3x3 cof = dJdF(F);
    const Vector9   gJ  = flatten(cof);

    const Float psi = 0.5 * input.lambda * Jm1 * Jm1 - input.mu * Jm1
                      + 0.5 * input.mu * (F.squaredNorm() - 3.0)
                      + (input.mu * input.mu) / (input.lambda * input.lambda);

    Matrix3x3 dpsi_dF_mat =
        input.mu * F + (input.lambda * Jm1 - input.mu) * cof;

    Matrix9x9 ddpsi_ddF =
        input.mu * Matrix9x9::Identity()
        + input.lambda * (gJ * gJ.transpose())
        + (input.lambda * Jm1 - input.mu) * hessian_J(F);

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
