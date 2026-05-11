#pragma once

#include <contact_system/vertex_half_plane_normal_contact.h>
#include <linear_system/socu_native_contact_assembly_sink.h>
#include <linear_system/socu_native_contact_targets.h>

namespace uipc::backend::cuda_mixed
{
class HalfPlane;

struct VertexHalfPlaneNormalContactNativeContext
{
    SocuNativeContactAssemblySink<VertexHalfPlaneNormalContact::StoreScalar,
                                  ActivePolicy::SolveScalar>
        sink;
    muda::CBufferView<SocuNativeContactStencilTarget> PH_targets;
};

void assemble_ipc_vertex_half_plane_normal_contact_native_exact(
    VertexHalfPlaneNormalContact::ContactInfo&          info,
    const HalfPlane&                                    half_plane,
    const VertexHalfPlaneNormalContactNativeContext& native_context);
}  // namespace uipc::backend::cuda_mixed
