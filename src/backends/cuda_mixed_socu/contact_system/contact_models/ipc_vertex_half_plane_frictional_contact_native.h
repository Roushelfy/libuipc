#pragma once

#include <contact_system/vertex_half_plane_frictional_contact.h>

namespace uipc::backend::cuda_mixed
{
class HalfPlane;

void assemble_ipc_vertex_half_plane_frictional_contact_native_exact(
    VertexHalfPlaneFrictionalContact::ContactInfo& info,
    const HalfPlane&                               half_plane);
}  // namespace uipc::backend::cuda_mixed
