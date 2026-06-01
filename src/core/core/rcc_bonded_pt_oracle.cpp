#include <uipc/core/rcc_bonded_pt_oracle.h>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include <utility>

namespace uipc::core
{
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
}  // namespace uipc::core
