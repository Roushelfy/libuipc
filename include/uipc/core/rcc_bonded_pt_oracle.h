#pragma once
#include <uipc/common/dllexport.h>
#include <uipc/common/type_define.h>

namespace uipc::core
{
struct UIPC_CORE_API RCCBondedPTRestShapeInput
{
    Vector4i topo = Vector4i::Zero();
    Vector3  point = Vector3::Zero();
    Vector3  tri0 = Vector3::Zero();
    Vector3  tri1 = Vector3::Zero();
    Vector3  tri2 = Vector3::Zero();
    Float    min_separate_distance = 0.001;
    Float    triangle_degeneracy_tol = 1e-12;
};

struct UIPC_CORE_API RCCBondedPTRestShape
{
    bool     valid = false;
    Vector4i oriented_topo = Vector4i::Zero();
    Vector3  normal = Vector3::Zero();
    Vector3  conditioned_point = Vector3::Zero();
    Float    signed_distance = 0.0;
    Float    conditioned_signed_distance = 0.0;
    Matrix3x3 Dm = Matrix3x3::Zero();
    Matrix3x3 Dm_inv = Matrix3x3::Zero();
    Float     rest_volume = 0.0;
};

UIPC_CORE_API RCCBondedPTRestShape
build_rcc_bonded_pt_rest_shape_svts(const RCCBondedPTRestShapeInput& input);
}  // namespace uipc::core
