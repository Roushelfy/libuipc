#pragma once

#include <contact_system/contact_coeff.h>
#include <mixed_precision/policy.h>
#include <utils/structured_contact_assembly_sink.h>

namespace uipc::backend::cuda_mixed
{
class SimplexNormalContactStructuredInfo
{
  public:
    using StoreScalar = ActivePolicy::StoreScalar;
    using SolveScalar = ActivePolicy::SolveScalar;

    muda::CBuffer2DView<ContactCoeff> contact_tabular() const noexcept
    {
        return contact_tabular_view;
    }
    muda::CBufferView<Vector4i> PTs() const noexcept { return PT_view; }
    muda::CBufferView<Vector4i> EEs() const noexcept { return EE_view; }
    muda::CBufferView<Vector3i> PEs() const noexcept { return PE_view; }
    muda::CBufferView<Vector2i> PPs() const noexcept { return PP_view; }
    muda::CBufferView<Vector3> positions() const noexcept { return positions_view; }
    muda::CBufferView<Vector3> rest_positions() const noexcept
    {
        return rest_positions_view;
    }
    muda::CBufferView<Float> thicknesses() const noexcept { return thicknesses_view; }
    muda::CBufferView<IndexT> contact_element_ids() const noexcept
    {
        return contact_element_ids_view;
    }
    muda::CBufferView<Float> d_hats() const noexcept { return d_hats_view; }
    Float dt() const noexcept { return dt_value; }
    auto structured_hessian_sink() const noexcept { return structured_sink; }

    muda::CBuffer2DView<ContactCoeff> contact_tabular_view;
    muda::CBufferView<Vector4i>       PT_view;
    muda::CBufferView<Vector4i>       EE_view;
    muda::CBufferView<Vector3i>       PE_view;
    muda::CBufferView<Vector2i>       PP_view;
    muda::CBufferView<Vector3>        positions_view;
    muda::CBufferView<Vector3>        rest_positions_view;
    muda::CBufferView<Float>          thicknesses_view;
    muda::CBufferView<IndexT>         contact_element_ids_view;
    muda::CBufferView<Float>          d_hats_view;
    Float                             dt_value = 0;
    StructuredContactAssemblySink<StoreScalar, SolveScalar> structured_sink;
};

class SimplexFrictionalContactStructuredInfo
{
  public:
    using StoreScalar = ActivePolicy::StoreScalar;
    using SolveScalar = ActivePolicy::SolveScalar;

    muda::CBuffer2DView<ContactCoeff> contact_tabular() const noexcept
    {
        return contact_tabular_view;
    }
    muda::CBufferView<Vector4i> friction_PTs() const noexcept
    {
        return friction_PT_view;
    }
    muda::CBufferView<Vector4i> friction_EEs() const noexcept
    {
        return friction_EE_view;
    }
    muda::CBufferView<Vector3i> friction_PEs() const noexcept
    {
        return friction_PE_view;
    }
    muda::CBufferView<Vector2i> friction_PPs() const noexcept
    {
        return friction_PP_view;
    }
    muda::CBufferView<Vector3> positions() const noexcept { return positions_view; }
    muda::CBufferView<Vector3> prev_positions() const noexcept
    {
        return prev_positions_view;
    }
    muda::CBufferView<Vector3> rest_positions() const noexcept
    {
        return rest_positions_view;
    }
    muda::CBufferView<Float> thicknesses() const noexcept { return thicknesses_view; }
    muda::CBufferView<IndexT> contact_element_ids() const noexcept
    {
        return contact_element_ids_view;
    }
    muda::CBufferView<Float> d_hats() const noexcept { return d_hats_view; }
    Float dt() const noexcept { return dt_value; }
    Float eps_velocity() const noexcept { return eps_velocity_value; }
    auto structured_hessian_sink() const noexcept { return structured_sink; }

    muda::CBuffer2DView<ContactCoeff> contact_tabular_view;
    muda::CBufferView<Vector4i>       friction_PT_view;
    muda::CBufferView<Vector4i>       friction_EE_view;
    muda::CBufferView<Vector3i>       friction_PE_view;
    muda::CBufferView<Vector2i>       friction_PP_view;
    muda::CBufferView<Vector3>        positions_view;
    muda::CBufferView<Vector3>        prev_positions_view;
    muda::CBufferView<Vector3>        rest_positions_view;
    muda::CBufferView<Float>          thicknesses_view;
    muda::CBufferView<IndexT>         contact_element_ids_view;
    muda::CBufferView<Float>          d_hats_view;
    Float                             dt_value = 0;
    Float                             eps_velocity_value = 0;
    StructuredContactAssemblySink<StoreScalar, SolveScalar> structured_sink;
};

class VertexHalfPlaneNormalContactStructuredInfo
{
  public:
    using StoreScalar = ActivePolicy::StoreScalar;
    using SolveScalar = ActivePolicy::SolveScalar;

    muda::CBuffer2DView<ContactCoeff> contact_tabular() const noexcept
    {
        return contact_tabular_view;
    }
    muda::CBufferView<Vector2i> PHs() const noexcept { return PH_view; }
    muda::CBufferView<Vector3> positions() const noexcept { return positions_view; }
    muda::CBufferView<Float> thicknesses() const noexcept { return thicknesses_view; }
    muda::CBufferView<IndexT> contact_element_ids() const noexcept
    {
        return contact_element_ids_view;
    }
    muda::CBufferView<Float> d_hats() const noexcept { return d_hats_view; }
    IndexT half_plane_vertex_offset() const noexcept
    {
        return half_plane_vertex_offset_value;
    }
    Float dt() const noexcept { return dt_value; }
    auto structured_hessian_sink() const noexcept { return structured_sink; }

    muda::CBuffer2DView<ContactCoeff> contact_tabular_view;
    muda::CBufferView<Vector2i>       PH_view;
    muda::CBufferView<Vector3>        positions_view;
    muda::CBufferView<Float>          thicknesses_view;
    muda::CBufferView<IndexT>         contact_element_ids_view;
    muda::CBufferView<Float>          d_hats_view;
    IndexT                            half_plane_vertex_offset_value = 0;
    Float                             dt_value = 0;
    StructuredContactAssemblySink<StoreScalar, SolveScalar> structured_sink;
};

class VertexHalfPlaneFrictionalContactStructuredInfo
{
  public:
    using StoreScalar = ActivePolicy::StoreScalar;
    using SolveScalar = ActivePolicy::SolveScalar;

    muda::CBuffer2DView<ContactCoeff> contact_tabular() const noexcept
    {
        return contact_tabular_view;
    }
    muda::CBufferView<Vector2i> friction_PHs() const noexcept
    {
        return friction_PH_view;
    }
    muda::CBufferView<Vector3> positions() const noexcept { return positions_view; }
    muda::CBufferView<Vector3> prev_positions() const noexcept
    {
        return prev_positions_view;
    }
    muda::CBufferView<Float> thicknesses() const noexcept { return thicknesses_view; }
    muda::CBufferView<IndexT> contact_element_ids() const noexcept
    {
        return contact_element_ids_view;
    }
    muda::CBufferView<Float> d_hats() const noexcept { return d_hats_view; }
    IndexT half_plane_vertex_offset() const noexcept
    {
        return half_plane_vertex_offset_value;
    }
    Float dt() const noexcept { return dt_value; }
    Float eps_velocity() const noexcept { return eps_velocity_value; }
    auto structured_hessian_sink() const noexcept { return structured_sink; }

    muda::CBuffer2DView<ContactCoeff> contact_tabular_view;
    muda::CBufferView<Vector2i>       friction_PH_view;
    muda::CBufferView<Vector3>        positions_view;
    muda::CBufferView<Vector3>        prev_positions_view;
    muda::CBufferView<Float>          thicknesses_view;
    muda::CBufferView<IndexT>         contact_element_ids_view;
    muda::CBufferView<Float>          d_hats_view;
    IndexT                            half_plane_vertex_offset_value = 0;
    Float                             dt_value = 0;
    Float                             eps_velocity_value = 0;
    StructuredContactAssemblySink<StoreScalar, SolveScalar> structured_sink;
};
}  // namespace uipc::backend::cuda_mixed
