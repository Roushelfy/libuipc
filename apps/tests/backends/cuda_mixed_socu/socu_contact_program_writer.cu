#include <app/app.h>
#include <affine_body/abd_jacobi_matrix.h>
#include <linear_system/socu_contact_program_debug_compare.h>
#include <linear_system/socu_contact_program_writer.h>
#include <utils/structured_contact_assembly_sink.h>

#include <cuda_runtime.h>
#include <Eigen/Core>
#include <muda/buffer/device_buffer.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using namespace uipc::backend::cuda_mixed;
using uipc::Float;
using uipc::IndexT;
using uipc::SizeT;
using uipc::Vector2i;
using uipc::Vector3;

bool has_cuda_device()
{
    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        return false;
    }
    return true;
}

template <typename Scalar, int Rows, int Cols = Rows>
struct DeviceMatrix
{
    Scalar values[Rows * Cols];

    MUDA_GENERIC Scalar operator()(IndexT row, IndexT col) const noexcept
    {
        return values[row * Cols + col];
    }

    template <typename OtherScalar>
    MUDA_GENERIC Eigen::Matrix<OtherScalar, Rows, Cols> cast() const noexcept
    {
        Eigen::Matrix<OtherScalar, Rows, Cols> out;
        for(IndexT row = 0; row < Rows; ++row)
        {
            for(IndexT col = 0; col < Cols; ++col)
                out(row, col) = static_cast<OtherScalar>((*this)(row, col));
        }
        return out;
    }

    template <int BlockRows, int BlockCols>
    MUDA_GENERIC DeviceMatrix<Scalar, BlockRows, BlockCols> block(
        IndexT row_offset,
        IndexT col_offset) const noexcept
    {
        DeviceMatrix<Scalar, BlockRows, BlockCols> out{};
        for(IndexT row = 0; row < BlockRows; ++row)
        {
            for(IndexT col = 0; col < BlockCols; ++col)
                out.values[row * BlockCols + col] =
                    (*this)(row_offset + row, col_offset + col);
        }
        return out;
    }
};

template <typename Scalar, int Rows, int Cols = Rows>
DeviceMatrix<Scalar, Rows, Cols> make_matrix(Scalar base)
{
    DeviceMatrix<Scalar, Rows, Cols> matrix{};
    for(IndexT row = 0; row < Rows; ++row)
    {
        for(IndexT col = 0; col < Cols; ++col)
            matrix.values[row * Cols + col] =
                static_cast<Scalar>(base + Scalar{10} * row + col);
    }
    return matrix;
}

template <typename Scalar, int Rows, int Cols = Rows>
DeviceMatrix<Scalar, Rows, Cols> make_signed_matrix(Scalar base)
{
    auto matrix = make_matrix<Scalar, Rows, Cols>(base);
    for(IndexT row = 0; row < Rows; ++row)
    {
        for(IndexT col = 0; col < Cols; ++col)
        {
            if(((row + col) & 1) != 0)
                matrix.values[row * Cols + col] =
                    -matrix.values[row * Cols + col];
        }
    }
    return matrix;
}

template <typename Scalar>
void symmetrize_block3(DeviceMatrix<Scalar, 6>& matrix, IndexT offset)
{
    for(IndexT row = 0; row < 3; ++row)
    {
        for(IndexT col = row + 1; col < 3; ++col)
        {
            const auto value =
                static_cast<Scalar>((matrix(offset + row, offset + col)
                                     + matrix(offset + col, offset + row))
                                    / Scalar{2});
            matrix.values[(offset + row) * 6 + offset + col] = value;
            matrix.values[(offset + col) * 6 + offset + row] = value;
        }
    }
}

template <typename Scalar>
MUDA_GENERIC DeviceMatrix<Scalar, 3> subblock3(
    const DeviceMatrix<Scalar, 6>& matrix,
    IndexT                          row_offset,
    IndexT                          col_offset)
{
    DeviceMatrix<Scalar, 3> out{};
    for(IndexT row = 0; row < 3; ++row)
    {
        for(IndexT col = 0; col < 3; ++col)
            out.values[row * 3 + col] =
                matrix(row_offset + row, col_offset + col);
    }
    return out;
}

SocuNativeVertexDescriptor make_vertex(SocuNativeDescriptorKind kind,
                                       IndexT old_dof,
                                       IndexT dof_count,
                                       SizeT block,
                                       SizeT lane,
                                       IndexT abd_body = -1,
                                       IndexT abd_j_index = -1)
{
    SocuNativeVertexDescriptor out;
    out.kind = kind;
    out.fixed = false;
    out.old_dof = old_dof;
    out.dof_count = dof_count;
    out.block = block;
    out.lane = lane;
    out.abd_body = abd_body;
    out.abd_j_index = abd_j_index;
    out.epoch = 13;
    out.active = true;
    return out;
}

SocuContactAssemblyPlanM2BuildInput make_fem_pp_input(
    const muda::DeviceBuffer<SocuNativeVertexDescriptor>& vertices,
    const muda::DeviceBuffer<Vector2i>& pps,
    SizeT horizon = 2,
    StructuredContactOffbandPolicy policy = StructuredContactOffbandPolicy::Drop)
{
    SocuVertexSidePlanKey side_key;
    side_key.ordering_epoch = 1;
    side_key.native_descriptor_epoch = 13;
    side_key.fixed_mapping_epoch = 3;
    side_key.vertex_projection_epoch = 5;
    side_key.horizon = horizon;
    side_key.block_size = 16;

    SocuContactProgramPlanKey program_key;
    program_key.side_key = side_key;
    program_key.contact_topology_epoch = 17;
    program_key.contact_layout_hash = 19;
    program_key.contact_content_hash = 23;
    program_key.offband_policy = policy;

    SocuContactAssemblyPlanM2BuildInput input;
    input.side_key = side_key;
    input.program_key = program_key;
    input.vertex_descriptors = vertices.view();
    input.pp_contacts = pps.view();
    input.pp_source =
        SocuContactM2SourceInput{0, 10, SocuContactModelKind::SimplexNormal};
    input.side_coverage_mode = SocuVertexSideCoverageMode::Global;
    input.offband_policy = policy;
    return input;
}

template <typename T>
void zero_device_buffer(muda::DeviceBuffer<T>& buffer, cudaStream_t stream = nullptr)
{
    if(buffer.size() == 0)
        return;
    REQUIRE(cudaMemsetAsync(buffer.data(),
                            0,
                            buffer.size() * sizeof(T),
                            stream)
            == cudaSuccess);
}

template <typename Scalar>
void require_vectors_close(const std::vector<Scalar>& lhs,
                           const std::vector<Scalar>& rhs,
                           double tolerance)
{
    REQUIRE(lhs.size() == rhs.size());
    for(std::size_t i = 0; i < lhs.size(); ++i)
    {
        CAPTURE(i);
        REQUIRE(std::isfinite(static_cast<double>(lhs[i])));
        CHECK(static_cast<double>(lhs[i])
              == Catch::Approx(static_cast<double>(rhs[i])).margin(tolerance));
    }
}

template <typename Store, typename Solve>
__global__ void write_fem_pp_with_program_writer_kernel(
    SocuContactProgramWriter<Store, Solve> writer,
    DeviceMatrix<Store, 6> same_block_H,
    DeviceMatrix<Store, 6> first_offdiag_H)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    writer.write_contact(0, 0, same_block_H);
    writer.write_contact(0, 1, first_offdiag_H);
}

template <typename Store, typename Solve>
__global__ void write_invalid_contact_with_program_writer_kernel(
    SocuContactProgramWriter<Store, Solve> writer,
    DeviceMatrix<Store, 6> H)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;
    writer.write_contact(7, 0, H);
    writer.write_contact(0, 99, H);
}

template <typename Store, typename Solve>
__global__ void write_fem_pp_with_legacy_sink_kernel(
    StructuredContactAssemblySink<Store, Solve> sink,
    DeviceMatrix<Store, 6> same_block_H,
    DeviceMatrix<Store, 6> first_offdiag_H)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    sink.write_contact_half_block(0, 0, subblock3(same_block_H, 0, 0), false);
    sink.write_contact_half_block(0, 1, subblock3(same_block_H, 0, 3), true);
    sink.write_contact_half_block(1, 1, subblock3(same_block_H, 3, 3), false);

    sink.write_contact_half_block(0, 0, subblock3(first_offdiag_H, 0, 0), false);
    sink.write_contact_half_block(0, 2, subblock3(first_offdiag_H, 0, 3), false);
    sink.write_contact_half_block(2, 2, subblock3(first_offdiag_H, 3, 3), false);
}

template <typename Store, typename Solve>
__global__ void write_abd_fem_with_program_writer_kernel(
    SocuContactProgramWriter<Store, Solve> writer,
    DeviceMatrix<Store, 6> H)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;
    writer.write_contact(0, 0, H);
}

template <typename Store, typename Solve>
__global__ void write_abd_fem_with_legacy_sink_kernel(
    StructuredContactAssemblySink<Store, Solve> sink,
    DeviceMatrix<Store, 3> H)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;
    sink.write_contact_half_block(0, 5, H, false);
}

template <typename Store, typename Solve>
__global__ void write_fem_abd_with_legacy_sink_kernel(
    StructuredContactAssemblySink<Store, Solve> sink,
    DeviceMatrix<Store, 3> H)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;
    sink.write_contact_half_block(5, 0, H, false);
}

template <typename Store, typename Solve>
__global__ void write_pair_half_with_legacy_sink_kernel(
    StructuredContactAssemblySink<Store, Solve> sink,
    Vector2i indices,
    DeviceMatrix<Store, 6> H)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;
    sink.template write_hessian_half<2>(indices, H);
}

template <typename Store, typename Solve>
__global__ void write_contact_block_with_legacy_sink_kernel(
    StructuredContactAssemblySink<Store, Solve> sink,
    IndexT global_i,
    IndexT global_j,
    DeviceMatrix<Store, 3> H,
    bool mirror_diag_block)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;
    sink.write_contact_half_block(global_i, global_j, H, mirror_diag_block);
}

template <typename Store, typename Solve>
__global__ void write_status_contacts_with_program_writer_kernel(
    SocuContactProgramWriter<Store, Solve> writer,
    DeviceMatrix<Store, 6> H)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;
    writer.write_contact(0, 0, H);
    writer.write_contact(0, 1, H);
    writer.write_contact(0, 2, H);
    writer.write_contact(0, 3, H);
    writer.write_contact(0, 4, H);
}

__global__ void debug_compare_matches_kernel(
    SocuContactProgramDebugCompare debug_compare,
    muda::BufferView<int> results)
{
    if(threadIdx.x != 0 || blockIdx.x != 0 || results.size() < 3)
        return;

    results.data()[0] =
        debug_compare.matches(0, 0, {SocuContactProgramKind::Exact, 3}) ? 1 : 0;
    results.data()[1] =
        debug_compare.matches(0, 1, {SocuContactProgramKind::Exact, 3}) ? 1 : 0;
    results.data()[2] =
        debug_compare.matches(0, 1, {SocuContactProgramKind::Skipped, 0}) ? 1 : 0;
}

template <typename Store, typename Solve>
SocuContactProgramWriter<Store, Solve> make_writer(
    SocuContactAssemblyPlan& plan,
    SocuNativeMatrixBuilder<Solve>& matrix,
    muda::DeviceBuffer<IndexT>& counters)
{
    return SocuContactProgramWriter<Store, Solve>{
        socu_contact_assembly_plan_view(plan),
        matrix.view(),
        counters.view()};
}

template <typename Store, typename Solve>
StructuredContactAssemblySink<Store, Solve> make_fem_legacy_sink(
    muda::DeviceBuffer<Solve>& diag,
    muda::DeviceBuffer<Solve>& offdiag,
    muda::DeviceBuffer<IndexT>& old_to_chain,
    muda::DeviceBuffer<IndexT>& fem_fixed,
    SizeT horizon = 2,
    StructuredContactOffbandPolicy policy = StructuredContactOffbandPolicy::Drop)
{
    StructuredContactAssemblySink<Store, Solve> sink;
    sink.sink = StructuredDeviceAssemblySink<Store, Solve>{
        diag.view(),
        offdiag.view(),
        old_to_chain.view(),
        horizon,
        16,
        {},
        {}};
    sink.fem_vertex_offset = 0;
    sink.fem_vertex_count = static_cast<IndexT>(fem_fixed.size());
    sink.fem_old_dof_offset = 0;
    sink.fem_vertex_is_fixed = fem_fixed.view();
    sink.offband_policy = policy;
    return sink;
}

template <typename Store, typename Solve>
StructuredContactAssemblySink<Store, Solve> make_abd_fem_legacy_sink(
    muda::DeviceBuffer<Solve>& diag,
    muda::DeviceBuffer<Solve>& offdiag,
    muda::DeviceBuffer<IndexT>& old_to_chain,
    muda::DeviceBuffer<IndexT>& fem_fixed,
    muda::DeviceBuffer<IndexT>& abd_vertex_to_body,
    muda::DeviceBuffer<ABDJacobi>& abd_jacobians,
    muda::DeviceBuffer<IndexT>& abd_body_fixed)
{
    StructuredContactAssemblySink<Store, Solve> sink;
    sink.sink = StructuredDeviceAssemblySink<Store, Solve>{
        diag.view(),
        offdiag.view(),
        old_to_chain.view(),
        2,
        16,
        {},
        {}};
    sink.fem_vertex_offset = 5;
    sink.fem_vertex_count = 1;
    sink.fem_old_dof_offset = 12;
    sink.fem_vertex_is_fixed = fem_fixed.view();
    sink.abd_vertex_offset = 0;
    sink.abd_vertex_count = 1;
    sink.abd_body_count = 1;
    sink.abd_old_dof_offset = 0;
    sink.abd_vertex_to_body = abd_vertex_to_body.view();
    sink.abd_vertex_to_J = abd_jacobians.view();
    sink.abd_body_is_fixed = abd_body_fixed.view();
    sink.offband_policy = StructuredContactOffbandPolicy::Drop;
    return sink;
}

template <typename Store, typename Solve>
StructuredContactAssemblySink<Store, Solve> make_abd_legacy_sink(
    muda::DeviceBuffer<Solve>& diag,
    muda::DeviceBuffer<Solve>& offdiag,
    muda::DeviceBuffer<IndexT>& old_to_chain,
    muda::DeviceBuffer<IndexT>& abd_vertex_to_body,
    muda::DeviceBuffer<ABDJacobi>& abd_jacobians,
    muda::DeviceBuffer<IndexT>& abd_body_fixed,
    SizeT horizon,
    StructuredContactOffbandPolicy policy = StructuredContactOffbandPolicy::Drop)
{
    StructuredContactAssemblySink<Store, Solve> sink;
    sink.sink = StructuredDeviceAssemblySink<Store, Solve>{
        diag.view(),
        offdiag.view(),
        old_to_chain.view(),
        horizon,
        16,
        {},
        {}};
    sink.abd_vertex_offset = 0;
    sink.abd_vertex_count = static_cast<IndexT>(abd_vertex_to_body.size());
    sink.abd_body_count = static_cast<IndexT>(abd_body_fixed.size());
    sink.abd_old_dof_offset = 0;
    sink.abd_vertex_to_body = abd_vertex_to_body.view();
    sink.abd_vertex_to_J = abd_jacobians.view();
    sink.abd_body_is_fixed = abd_body_fixed.view();
    sink.offband_policy = policy;
    return sink;
}

template <typename Store, typename Solve>
StructuredContactAssemblySink<Store, Solve> make_abd_fem_policy_legacy_sink(
    muda::DeviceBuffer<Solve>& diag,
    muda::DeviceBuffer<Solve>& offdiag,
    muda::DeviceBuffer<IndexT>& old_to_chain,
    muda::DeviceBuffer<IndexT>& fem_fixed,
    muda::DeviceBuffer<IndexT>& abd_vertex_to_body,
    muda::DeviceBuffer<ABDJacobi>& abd_jacobians,
    muda::DeviceBuffer<IndexT>& abd_body_fixed,
    StructuredContactOffbandPolicy policy)
{
    auto sink = make_abd_fem_legacy_sink<Store, Solve>(diag,
                                                       offdiag,
                                                       old_to_chain,
                                                       fem_fixed,
                                                       abd_vertex_to_body,
                                                       abd_jacobians,
                                                       abd_body_fixed);
    sink.sink.matrix.horizon = 3;
    sink.fem_old_dof_offset = 12;
    sink.offband_policy = policy;
    return sink;
}

SocuAssemblyDofLane make_lane(SizeT block,
                              SizeT lane,
                              std::uint8_t component,
                              Float weight)
{
    SocuAssemblyDofLane out;
    out.block = static_cast<std::uint32_t>(block);
    out.lane = static_cast<std::uint16_t>(lane);
    out.component = component;
    out.weight = weight;
    return out;
}

void append_abd_lanes(std::vector<SocuAssemblyDofLane>& lanes,
                      SizeT block,
                      SizeT first_lane,
                      const Vector3& x_bar)
{
    for(IndexT q = 0; q < 12; ++q)
    {
        const auto component =
            static_cast<std::uint8_t>(q < 3 ? q : (q - 3) / 3);
        const Float weight = q < 3 ? Float{1} : x_bar((q - 3) % 3);
        lanes.push_back(make_lane(block,
                                  static_cast<SizeT>(first_lane + q),
                                  component,
                                  weight));
    }
}

SocuContactAssemblyPlan make_manual_abd_fem_plan(const Vector3& x_bar)
{
    SocuContactAssemblyPlan plan;
    plan.side_plan.key.horizon = 2;
    plan.side_plan.key.block_size = 16;

    std::vector<SocuAssemblySideRecord> sides(2);
    sides[0].global_vertex = 0;
    sides[0].old_dof = 0;
    sides[0].dof_count = 12;
    sides[0].abd_body = 0;
    sides[0].abd_jacobian_index = 0;
    sides[0].kind = SocuAssemblySideKind::Abd;
    sides[0].writable = true;
    sides[0].block = 0;
    sides[0].lane = 0;
    sides[0].first_lane = 0;
    sides[0].lane_count = 12;

    sides[1].global_vertex = 5;
    sides[1].old_dof = 12;
    sides[1].dof_count = 3;
    sides[1].kind = SocuAssemblySideKind::Fem;
    sides[1].writable = true;
    sides[1].block = 1;
    sides[1].lane = 0;
    sides[1].first_lane = 12;
    sides[1].lane_count = 3;

    std::vector<SocuAssemblyDofLane> lanes;
    lanes.reserve(15);
    append_abd_lanes(lanes, 0, 0, x_bar);
    for(IndexT q = 0; q < 3; ++q)
        lanes.push_back(make_lane(1, static_cast<SizeT>(q), q, Float{1}));

    std::vector<SocuContactSourceHeader> sources(1);
    sources[0].source_id = 0;
    sources[0].reporter_id = 10;
    sources[0].model = SocuContactModelKind::SimplexNormal;
    sources[0].family = SocuContactFamily::PP;
    sources[0].stencil_size = 2;
    sources[0].contact_count = 1;
    sources[0].first_program = 0;
    sources[0].program_count = 1;
    sources[0].first_source_to_program = 0;

    std::vector<SocuContactProgramHeader> programs(1);
    programs[0].source_id = 0;
    programs[0].local_contact_id = 0;
    programs[0].model = SocuContactModelKind::SimplexNormal;
    programs[0].family = SocuContactFamily::PP;
    programs[0].program_kind = SocuContactProgramKind::Exact;
    programs[0].first_task = 0;
    programs[0].task_count = 1;
    programs[0].stencil_size = 2;
    programs[0].side_ids[0] = 0;
    programs[0].side_ids[1] = 1;

    std::vector<SocuContactMicroTask> tasks(1);
    tasks[0].row_side = 0;
    tasks[0].col_side = 1;
    tasks[0].local_row_vertex = 0;
    tasks[0].local_col_vertex = 1;
    tasks[0].band = SocuAssemblyBand::FirstOffdiag;
    tasks[0].write_kind = SocuAssemblyWriteKind::ExactAbdFem;
    tasks[0].block_or_left_block = 0;
    tasks[0].flags =
        static_cast<std::uint8_t>(SocuContactTaskFlag::TransposedFirstOffdiag);

    std::vector<SocuContactSourceToProgram> maps(1);
    maps[0].program_id = 0;
    maps[0].status = SocuContactProgramMapStatus::Valid;

    plan.side_plan.sorted_side_vertices = std::vector<IndexT>{0, 5};
    plan.side_plan.sides = sides;
    plan.side_plan.lanes = lanes;
    plan.side_plan.coverage.mode = SocuVertexSideCoverageMode::Global;
    plan.side_plan.coverage.covered_vertex_count = 2;
    plan.side_plan.coverage.complete_for_current_contacts = true;
    plan.program_plan.sources = sources;
    plan.program_plan.programs = programs;
    plan.program_plan.tasks = tasks;
    plan.program_plan.source_to_program = maps;
    return plan;
}

SocuContactAssemblyPlan make_manual_fem_abd_plan(const Vector3& x_bar)
{
    SocuContactAssemblyPlan plan;
    plan.side_plan.key.horizon = 2;
    plan.side_plan.key.block_size = 16;

    std::vector<SocuAssemblySideRecord> sides(2);
    sides[0].global_vertex = 5;
    sides[0].old_dof = 12;
    sides[0].dof_count = 3;
    sides[0].kind = SocuAssemblySideKind::Fem;
    sides[0].writable = true;
    sides[0].block = 1;
    sides[0].lane = 0;
    sides[0].first_lane = 0;
    sides[0].lane_count = 3;

    sides[1].global_vertex = 0;
    sides[1].old_dof = 0;
    sides[1].dof_count = 12;
    sides[1].abd_body = 0;
    sides[1].abd_jacobian_index = 0;
    sides[1].kind = SocuAssemblySideKind::Abd;
    sides[1].writable = true;
    sides[1].block = 0;
    sides[1].lane = 0;
    sides[1].first_lane = 3;
    sides[1].lane_count = 12;

    std::vector<SocuAssemblyDofLane> lanes;
    lanes.reserve(15);
    for(IndexT q = 0; q < 3; ++q)
        lanes.push_back(make_lane(1, static_cast<SizeT>(q), q, Float{1}));
    append_abd_lanes(lanes, 0, 0, x_bar);

    std::vector<SocuContactSourceHeader> sources(1);
    sources[0].source_id = 0;
    sources[0].reporter_id = 10;
    sources[0].model = SocuContactModelKind::SimplexNormal;
    sources[0].family = SocuContactFamily::PP;
    sources[0].stencil_size = 2;
    sources[0].contact_count = 1;
    sources[0].first_program = 0;
    sources[0].program_count = 1;
    sources[0].first_source_to_program = 0;

    std::vector<SocuContactProgramHeader> programs(1);
    programs[0].source_id = 0;
    programs[0].local_contact_id = 0;
    programs[0].model = SocuContactModelKind::SimplexNormal;
    programs[0].family = SocuContactFamily::PP;
    programs[0].program_kind = SocuContactProgramKind::Exact;
    programs[0].first_task = 0;
    programs[0].task_count = 1;
    programs[0].stencil_size = 2;
    programs[0].side_ids[0] = 0;
    programs[0].side_ids[1] = 1;

    std::vector<SocuContactMicroTask> tasks(1);
    tasks[0].row_side = 0;
    tasks[0].col_side = 1;
    tasks[0].local_row_vertex = 0;
    tasks[0].local_col_vertex = 1;
    tasks[0].band = SocuAssemblyBand::FirstOffdiag;
    tasks[0].write_kind = SocuAssemblyWriteKind::ExactFemAbd;
    tasks[0].block_or_left_block = 0;

    std::vector<SocuContactSourceToProgram> maps(1);
    maps[0].program_id = 0;
    maps[0].status = SocuContactProgramMapStatus::Valid;

    plan.side_plan.sorted_side_vertices = std::vector<IndexT>{0, 5};
    plan.side_plan.sides = sides;
    plan.side_plan.lanes = lanes;
    plan.side_plan.coverage.mode = SocuVertexSideCoverageMode::Global;
    plan.side_plan.coverage.covered_vertex_count = 2;
    plan.side_plan.coverage.complete_for_current_contacts = true;
    plan.program_plan.sources = sources;
    plan.program_plan.programs = programs;
    plan.program_plan.tasks = tasks;
    plan.program_plan.source_to_program = maps;
    return plan;
}

SocuContactAssemblyPlan make_manual_abd_abd_plan(const Vector3& x_bar0,
                                                 const Vector3& x_bar1,
                                                 bool same_body)
{
    SocuContactAssemblyPlan plan;
    plan.side_plan.key.horizon = same_body ? 1 : 2;
    plan.side_plan.key.block_size = 16;

    std::vector<SocuAssemblySideRecord> sides(2);
    sides[0].global_vertex = 0;
    sides[0].old_dof = 0;
    sides[0].dof_count = 12;
    sides[0].abd_body = 0;
    sides[0].abd_jacobian_index = 0;
    sides[0].kind = SocuAssemblySideKind::Abd;
    sides[0].writable = true;
    sides[0].block = 0;
    sides[0].lane = 0;
    sides[0].first_lane = 0;
    sides[0].lane_count = 12;

    sides[1].global_vertex = 1;
    sides[1].old_dof = same_body ? 0 : 12;
    sides[1].dof_count = 12;
    sides[1].abd_body = same_body ? 0 : 1;
    sides[1].abd_jacobian_index = 1;
    sides[1].kind = SocuAssemblySideKind::Abd;
    sides[1].writable = true;
    sides[1].block = same_body ? 0 : 1;
    sides[1].lane = 0;
    sides[1].first_lane = 12;
    sides[1].lane_count = 12;

    std::vector<SocuAssemblyDofLane> lanes;
    lanes.reserve(24);
    append_abd_lanes(lanes, sides[0].block, 0, x_bar0);
    append_abd_lanes(lanes, sides[1].block, 0, x_bar1);

    std::vector<SocuContactSourceHeader> sources(1);
    sources[0].source_id = 0;
    sources[0].reporter_id = 10;
    sources[0].model = SocuContactModelKind::SimplexNormal;
    sources[0].family = SocuContactFamily::PP;
    sources[0].stencil_size = 2;
    sources[0].contact_count = 1;
    sources[0].first_program = 0;
    sources[0].program_count = 1;
    sources[0].first_source_to_program = 0;

    std::vector<SocuContactProgramHeader> programs(1);
    programs[0].source_id = 0;
    programs[0].local_contact_id = 0;
    programs[0].model = SocuContactModelKind::SimplexNormal;
    programs[0].family = SocuContactFamily::PP;
    programs[0].program_kind = SocuContactProgramKind::Exact;
    programs[0].first_task = 0;
    programs[0].task_count = 1;
    programs[0].stencil_size = 2;
    programs[0].side_ids[0] = 0;
    programs[0].side_ids[1] = 1;

    std::vector<SocuContactMicroTask> tasks(1);
    tasks[0].row_side = 0;
    tasks[0].col_side = 1;
    tasks[0].local_row_vertex = 0;
    tasks[0].local_col_vertex = 1;
    tasks[0].band = same_body ? SocuAssemblyBand::Diag
                              : SocuAssemblyBand::FirstOffdiag;
    tasks[0].write_kind = same_body
                              ? SocuAssemblyWriteKind::ExactAbdAbdSameBody
                              : SocuAssemblyWriteKind::ExactAbdAbdCrossBody;
    tasks[0].block_or_left_block = 0;
    tasks[0].flags =
        same_body ? static_cast<std::uint8_t>(
                        SocuContactTaskFlag::MirrorDiagBlock
                        | SocuContactTaskFlag::SameAbdBody)
                  : static_cast<std::uint8_t>(
                        SocuContactTaskFlag::TransposedFirstOffdiag);

    std::vector<SocuContactSourceToProgram> maps(1);
    maps[0].program_id = 0;
    maps[0].status = SocuContactProgramMapStatus::Valid;

    plan.side_plan.sorted_side_vertices = std::vector<IndexT>{0, 1};
    plan.side_plan.sides = sides;
    plan.side_plan.lanes = lanes;
    plan.program_plan.sources = sources;
    plan.program_plan.programs = programs;
    plan.program_plan.tasks = tasks;
    plan.program_plan.source_to_program = maps;
    return plan;
}

SocuContactAssemblyPlan make_manual_abd_fem_fallback_plan(
    const Vector3& x_bar,
    SocuContactProgramKind program_kind,
    SocuAssemblyWriteKind abd_write_kind,
    SocuAssemblyWriteKind fem_write_kind)
{
    SocuContactAssemblyPlan plan;
    plan.side_plan.key.horizon = 3;
    plan.side_plan.key.block_size = 16;

    std::vector<SocuAssemblySideRecord> sides(2);
    sides[0].global_vertex = 0;
    sides[0].old_dof = 0;
    sides[0].dof_count = 12;
    sides[0].abd_body = 0;
    sides[0].abd_jacobian_index = 0;
    sides[0].kind = SocuAssemblySideKind::Abd;
    sides[0].writable = true;
    sides[0].block = 0;
    sides[0].lane = 0;
    sides[0].first_lane = 0;
    sides[0].lane_count = 12;

    sides[1].global_vertex = 5;
    sides[1].old_dof = 12;
    sides[1].dof_count = 3;
    sides[1].kind = SocuAssemblySideKind::Fem;
    sides[1].writable = true;
    sides[1].block = 2;
    sides[1].lane = 0;
    sides[1].first_lane = 12;
    sides[1].lane_count = 3;

    std::vector<SocuAssemblyDofLane> lanes;
    lanes.reserve(15);
    append_abd_lanes(lanes, 0, 0, x_bar);
    for(IndexT q = 0; q < 3; ++q)
        lanes.push_back(make_lane(2, static_cast<SizeT>(q), q, Float{1}));

    std::vector<SocuContactSourceHeader> sources(1);
    sources[0].source_id = 0;
    sources[0].reporter_id = 10;
    sources[0].model = SocuContactModelKind::SimplexNormal;
    sources[0].family = SocuContactFamily::PP;
    sources[0].stencil_size = 2;
    sources[0].contact_count = 1;
    sources[0].first_program = 0;
    sources[0].program_count = 1;
    sources[0].first_source_to_program = 0;

    std::vector<SocuContactProgramHeader> programs(1);
    programs[0].source_id = 0;
    programs[0].local_contact_id = 0;
    programs[0].model = SocuContactModelKind::SimplexNormal;
    programs[0].family = SocuContactFamily::PP;
    programs[0].program_kind = program_kind;
    programs[0].first_task = 0;
    programs[0].task_count = 2;
    programs[0].stencil_size = 2;
    programs[0].side_ids[0] = 0;
    programs[0].side_ids[1] = 1;

    std::vector<SocuContactMicroTask> tasks(2);
    tasks[0].row_side = 0;
    tasks[0].col_side = 0;
    tasks[0].local_row_vertex = 0;
    tasks[0].local_col_vertex = 0;
    tasks[0].band = SocuAssemblyBand::Diag;
    tasks[0].write_kind = abd_write_kind;
    tasks[0].block_or_left_block = 0;

    tasks[1].row_side = 1;
    tasks[1].col_side = 1;
    tasks[1].local_row_vertex = 1;
    tasks[1].local_col_vertex = 1;
    tasks[1].band = SocuAssemblyBand::Diag;
    tasks[1].write_kind = fem_write_kind;
    tasks[1].block_or_left_block = 2;

    std::vector<SocuContactSourceToProgram> maps(1);
    maps[0].program_id = 0;
    maps[0].status = SocuContactProgramMapStatus::Valid;

    plan.side_plan.sorted_side_vertices = std::vector<IndexT>{0, 5};
    plan.side_plan.sides = sides;
    plan.side_plan.lanes = lanes;
    plan.program_plan.sources = sources;
    plan.program_plan.programs = programs;
    plan.program_plan.tasks = tasks;
    plan.program_plan.source_to_program = maps;
    return plan;
}

SocuContactAssemblyPlan make_manual_diag_scalar_plan(bool abd,
                                                     const Vector3& x_bar)
{
    SocuContactAssemblyPlan plan;
    plan.side_plan.key.horizon = 1;
    plan.side_plan.key.block_size = 16;

    std::vector<SocuAssemblySideRecord> sides(1);
    sides[0].global_vertex = 0;
    sides[0].old_dof = 0;
    sides[0].dof_count = abd ? 12 : 3;
    sides[0].abd_body = abd ? 0 : -1;
    sides[0].abd_jacobian_index = abd ? 0 : -1;
    sides[0].kind = abd ? SocuAssemblySideKind::Abd : SocuAssemblySideKind::Fem;
    sides[0].writable = true;
    sides[0].block = 0;
    sides[0].lane = 0;
    sides[0].first_lane = 0;
    sides[0].lane_count = abd ? 12 : 3;

    std::vector<SocuAssemblyDofLane> lanes;
    if(abd)
        append_abd_lanes(lanes, 0, 0, x_bar);
    else
    {
        for(IndexT q = 0; q < 3; ++q)
            lanes.push_back(make_lane(0, static_cast<SizeT>(q), q, Float{1}));
    }

    std::vector<SocuContactSourceHeader> sources(1);
    sources[0].source_id = 0;
    sources[0].reporter_id = 10;
    sources[0].model = SocuContactModelKind::SimplexNormal;
    sources[0].family = SocuContactFamily::PH;
    sources[0].stencil_size = 1;
    sources[0].contact_count = 1;
    sources[0].first_program = 0;
    sources[0].program_count = 1;
    sources[0].first_source_to_program = 0;

    std::vector<SocuContactProgramHeader> programs(1);
    programs[0].source_id = 0;
    programs[0].local_contact_id = 0;
    programs[0].model = SocuContactModelKind::SimplexNormal;
    programs[0].family = SocuContactFamily::PH;
    programs[0].program_kind = SocuContactProgramKind::Diag;
    programs[0].first_task = 0;
    programs[0].task_count = 1;
    programs[0].stencil_size = 1;
    programs[0].side_ids[0] = 0;

    std::vector<SocuContactMicroTask> tasks(1);
    tasks[0].row_side = 0;
    tasks[0].col_side = 0;
    tasks[0].local_row_vertex = 0;
    tasks[0].local_col_vertex = 0;
    tasks[0].band = SocuAssemblyBand::Diag;
    tasks[0].write_kind = abd ? SocuAssemblyWriteKind::DiagScalarAbd
                              : SocuAssemblyWriteKind::DiagScalarFem;
    tasks[0].block_or_left_block = 0;

    std::vector<SocuContactSourceToProgram> maps(1);
    maps[0].program_id = 0;
    maps[0].status = SocuContactProgramMapStatus::Valid;

    plan.side_plan.sorted_side_vertices = std::vector<IndexT>{0};
    plan.side_plan.sides = sides;
    plan.side_plan.lanes = lanes;
    plan.program_plan.sources = sources;
    plan.program_plan.programs = programs;
    plan.program_plan.tasks = tasks;
    plan.program_plan.source_to_program = maps;
    return plan;
}

SocuContactAssemblyPlan make_manual_fem_fem_no_mirror_plan()
{
    SocuContactAssemblyPlan plan;
    plan.side_plan.key.horizon = 2;
    plan.side_plan.key.block_size = 16;

    std::vector<SocuAssemblySideRecord> sides(2);
    sides[0].global_vertex = 0;
    sides[0].old_dof = 0;
    sides[0].dof_count = 3;
    sides[0].kind = SocuAssemblySideKind::Fem;
    sides[0].writable = true;
    sides[0].block = 0;
    sides[0].lane = 0;
    sides[0].first_lane = 0;
    sides[0].lane_count = 3;

    sides[1].global_vertex = 1;
    sides[1].old_dof = 3;
    sides[1].dof_count = 3;
    sides[1].kind = SocuAssemblySideKind::Fem;
    sides[1].writable = true;
    sides[1].block = 0;
    sides[1].lane = 3;
    sides[1].first_lane = 3;
    sides[1].lane_count = 3;

    std::vector<SocuAssemblyDofLane> lanes;
    lanes.reserve(6);
    for(IndexT q = 0; q < 3; ++q)
        lanes.push_back(make_lane(0, static_cast<SizeT>(q), q, Float{1}));
    for(IndexT q = 0; q < 3; ++q)
        lanes.push_back(make_lane(0, static_cast<SizeT>(3 + q), q, Float{1}));

    std::vector<SocuContactSourceHeader> sources(1);
    sources[0].source_id = 0;
    sources[0].reporter_id = 10;
    sources[0].model = SocuContactModelKind::SimplexNormal;
    sources[0].family = SocuContactFamily::PP;
    sources[0].stencil_size = 2;
    sources[0].contact_count = 1;
    sources[0].first_program = 0;
    sources[0].program_count = 1;
    sources[0].first_source_to_program = 0;

    std::vector<SocuContactProgramHeader> programs(1);
    programs[0].source_id = 0;
    programs[0].local_contact_id = 0;
    programs[0].model = SocuContactModelKind::SimplexNormal;
    programs[0].family = SocuContactFamily::PP;
    programs[0].program_kind = SocuContactProgramKind::Exact;
    programs[0].first_task = 0;
    programs[0].task_count = 1;
    programs[0].stencil_size = 2;
    programs[0].side_ids[0] = 0;
    programs[0].side_ids[1] = 1;

    std::vector<SocuContactMicroTask> tasks(1);
    tasks[0].row_side = 0;
    tasks[0].col_side = 1;
    tasks[0].local_row_vertex = 0;
    tasks[0].local_col_vertex = 1;
    tasks[0].band = SocuAssemblyBand::Diag;
    tasks[0].write_kind = SocuAssemblyWriteKind::ExactFemFem;
    tasks[0].block_or_left_block = 0;

    std::vector<SocuContactSourceToProgram> maps(1);
    maps[0].program_id = 0;
    maps[0].status = SocuContactProgramMapStatus::Valid;

    plan.side_plan.sorted_side_vertices = std::vector<IndexT>{0, 1};
    plan.side_plan.sides = sides;
    plan.side_plan.lanes = lanes;
    plan.program_plan.sources = sources;
    plan.program_plan.programs = programs;
    plan.program_plan.tasks = tasks;
    plan.program_plan.source_to_program = maps;
    return plan;
}

SocuContactAssemblyPlan make_manual_status_plan()
{
    SocuContactAssemblyPlan plan;
    plan.side_plan.key.horizon = 2;
    plan.side_plan.key.block_size = 16;

    std::vector<SocuContactSourceHeader> sources(1);
    sources[0].source_id = 0;
    sources[0].reporter_id = 10;
    sources[0].model = SocuContactModelKind::SimplexNormal;
    sources[0].family = SocuContactFamily::PP;
    sources[0].stencil_size = 2;
    sources[0].contact_count = 5;
    sources[0].first_program = 0;
    sources[0].program_count = 1;
    sources[0].first_source_to_program = 0;

    std::vector<SocuContactProgramHeader> programs(1);
    programs[0].source_id = 0;
    programs[0].local_contact_id = 4;
    programs[0].program_kind = SocuContactProgramKind::Skipped;
    programs[0].stencil_size = 2;

    std::vector<SocuContactSourceToProgram> maps(5);
    maps[0].status = SocuContactProgramMapStatus::Skipped;
    maps[1].status = SocuContactProgramMapStatus::Dropped;
    maps[2].status = SocuContactProgramMapStatus::MixedRejected;
    maps[3].status = SocuContactProgramMapStatus::Missing;
    maps[4].program_id = 0;
    maps[4].status = SocuContactProgramMapStatus::Valid;

    plan.program_plan.sources = sources;
    plan.program_plan.programs = programs;
    plan.program_plan.source_to_program = maps;
    return plan;
}

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream ifs{path};
    REQUIRE(ifs.good());
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}
}  // namespace

TEST_CASE("cuda_mixed_socu_contact_program_writer_fem_fem_matches_legacy",
          "[cuda_mixed_socu][contract][socu_approx][m3]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact writer tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;
    constexpr SizeT Horizon = 2;
    constexpr SizeT BlockSize = 16;

    std::vector<SocuNativeVertexDescriptor> vertices(3);
    vertices[0] = make_vertex(SocuNativeDescriptorKind::Fem, 0, 3, 0, 0);
    vertices[1] = make_vertex(SocuNativeDescriptorKind::Fem, 3, 3, 0, 3);
    vertices[2] = make_vertex(SocuNativeDescriptorKind::Fem, 6, 3, 1, 0);
    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertex_buffer{vertices};
    muda::DeviceBuffer<Vector2i> pps{
        std::vector<Vector2i>{Vector2i{0, 1}, Vector2i{0, 2}}};

    SocuContactAssemblyPlan plan;
    SocuContactAssemblyPlanM2Workspace workspace;
    build_socu_contact_assembly_plan_m2(
        plan,
        workspace,
        make_fem_pp_input(vertex_buffer, pps));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    muda::DeviceBuffer<int> debug_results;
    debug_results.resize(3);
    zero_device_buffer(debug_results);
    debug_compare_matches_kernel<<<1, 1>>>(
        SocuContactProgramDebugCompare{socu_contact_assembly_plan_view(plan)},
        debug_results.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<int> debug_host;
    debug_results.copy_to(debug_host);
    REQUIRE(debug_host.size() == 3);
    CHECK(debug_host[0] == 1);
    CHECK(debug_host[1] == 1);
    CHECK(debug_host[2] == 0);

    SocuNativeMatrixBuilder<Solve> writer_matrix;
    writer_matrix.reserve(Horizon, BlockSize, 1);
    writer_matrix.clear();

    const auto layout = writer_matrix.layout();
    muda::DeviceBuffer<Solve> legacy_diag;
    muda::DeviceBuffer<Solve> legacy_offdiag;
    legacy_diag.resize(layout.diag_element_count);
    legacy_offdiag.resize(layout.offdiag_element_count);
    zero_device_buffer(legacy_diag);
    zero_device_buffer(legacy_offdiag);

    std::vector<IndexT> old_to_chain(9, -1);
    old_to_chain[0] = 0;
    old_to_chain[1] = 1;
    old_to_chain[2] = 2;
    old_to_chain[3] = 3;
    old_to_chain[4] = 4;
    old_to_chain[5] = 5;
    old_to_chain[6] = 16;
    old_to_chain[7] = 17;
    old_to_chain[8] = 18;
    muda::DeviceBuffer<IndexT> old_to_chain_buffer{old_to_chain};
    muda::DeviceBuffer<IndexT> fem_fixed{std::vector<IndexT>{0, 0, 0}};
    muda::DeviceBuffer<IndexT> writer_counters;
    writer_counters.resize(
        static_cast<SizeT>(SocuContactProgramWriterCounterSlot::Count));
    zero_device_buffer(writer_counters);

    const auto same_block_H = make_matrix<Store, 6>(Store{1});
    const auto first_offdiag_H = make_matrix<Store, 6>(Store{101});
    write_fem_pp_with_program_writer_kernel<Store, Solve>
        <<<1, 1>>>(make_writer<Store, Solve>(plan, writer_matrix, writer_counters),
                   same_block_H,
                   first_offdiag_H);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    write_fem_pp_with_legacy_sink_kernel<Store, Solve>
        <<<1, 1>>>(make_fem_legacy_sink<Store, Solve>(
                       legacy_diag,
                       legacy_offdiag,
                       old_to_chain_buffer,
                       fem_fixed),
                   same_block_H,
                   first_offdiag_H);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto snapshot = writer_matrix.snapshot();
    std::vector<Solve> legacy_D;
    std::vector<Solve> legacy_E;
    legacy_diag.copy_to(legacy_D);
    legacy_offdiag.copy_to(legacy_E);
    require_vectors_close(snapshot.D, legacy_D, 1e-8);
    require_vectors_close(snapshot.E, legacy_E, 1e-8);

    std::vector<IndexT> counters;
    writer_counters.copy_to(counters);
    REQUIRE(counters.size()
            == static_cast<SizeT>(SocuContactProgramWriterCounterSlot::Count));
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ContactWrite)]
          == 2);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ExactTaskWrite)]
          == 6);

    writer_matrix.clear();
    zero_device_buffer(writer_counters);
    write_invalid_contact_with_program_writer_kernel<Store, Solve>
        <<<1, 1>>>(make_writer<Store, Solve>(plan, writer_matrix, writer_counters),
                   same_block_H);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto empty_snapshot = writer_matrix.snapshot();
    for(const auto value : empty_snapshot.D)
        CHECK(static_cast<double>(value) == Catch::Approx(0.0));
    for(const auto value : empty_snapshot.E)
        CHECK(static_cast<double>(value) == Catch::Approx(0.0));
    writer_counters.copy_to(counters);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ProgramMissing)]
          == 2);
}

TEST_CASE("cuda_mixed_socu_contact_program_writer_abd_fem_projection_matches_legacy",
          "[cuda_mixed_socu][contract][socu_approx][m3]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact writer tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;
    constexpr SizeT Horizon = 2;
    constexpr SizeT BlockSize = 16;

    const Vector3 x_bar{Float{2}, Float{-1}, Float{0.5}};
    auto plan = make_manual_abd_fem_plan(x_bar);

    SocuNativeMatrixBuilder<Solve> writer_matrix;
    writer_matrix.reserve(Horizon, BlockSize, 1);
    writer_matrix.clear();

    const auto layout = writer_matrix.layout();
    muda::DeviceBuffer<Solve> legacy_diag;
    muda::DeviceBuffer<Solve> legacy_offdiag;
    legacy_diag.resize(layout.diag_element_count);
    legacy_offdiag.resize(layout.offdiag_element_count);
    zero_device_buffer(legacy_diag);
    zero_device_buffer(legacy_offdiag);

    std::vector<IndexT> old_to_chain(15, -1);
    for(IndexT i = 0; i < 12; ++i)
        old_to_chain[static_cast<SizeT>(i)] = i;
    old_to_chain[12] = 16;
    old_to_chain[13] = 17;
    old_to_chain[14] = 18;
    muda::DeviceBuffer<IndexT> old_to_chain_buffer{old_to_chain};
    muda::DeviceBuffer<IndexT> fem_fixed{std::vector<IndexT>{0}};
    muda::DeviceBuffer<IndexT> abd_vertex_to_body{std::vector<IndexT>{0}};
    muda::DeviceBuffer<ABDJacobi> abd_jacobians{
        std::vector<ABDJacobi>{ABDJacobi{x_bar}}};
    muda::DeviceBuffer<IndexT> abd_body_fixed{std::vector<IndexT>{0}};
    muda::DeviceBuffer<IndexT> writer_counters;
    writer_counters.resize(
        static_cast<SizeT>(SocuContactProgramWriterCounterSlot::Count));
    zero_device_buffer(writer_counters);

    auto H6 = make_matrix<Store, 6>(Store{5});
    const auto H3 = subblock3(H6, 0, 3);

    write_abd_fem_with_program_writer_kernel<Store, Solve>
        <<<1, 1>>>(make_writer<Store, Solve>(plan, writer_matrix, writer_counters),
                   H6);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    write_abd_fem_with_legacy_sink_kernel<Store, Solve>
        <<<1, 1>>>(make_abd_fem_legacy_sink<Store, Solve>(
                       legacy_diag,
                       legacy_offdiag,
                       old_to_chain_buffer,
                       fem_fixed,
                       abd_vertex_to_body,
                       abd_jacobians,
                       abd_body_fixed),
                   H3);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto snapshot = writer_matrix.snapshot();
    std::vector<Solve> legacy_D;
    std::vector<Solve> legacy_E;
    legacy_diag.copy_to(legacy_D);
    legacy_offdiag.copy_to(legacy_E);
    require_vectors_close(snapshot.D, legacy_D, 1e-8);
    require_vectors_close(snapshot.E, legacy_E, 1e-8);

    std::vector<IndexT> counters;
    writer_counters.copy_to(counters);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ContactWrite)]
          == 1);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ExactTaskWrite)]
          == 1);
}

TEST_CASE("cuda_mixed_socu_contact_program_writer_fem_abd_orientation_matches_legacy",
          "[cuda_mixed_socu][contract][socu_approx][m3]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact writer tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;
    constexpr SizeT Horizon = 2;
    constexpr SizeT BlockSize = 16;

    const Vector3 x_bar{Float{-0.25}, Float{1.5}, Float{3}};
    auto plan = make_manual_fem_abd_plan(x_bar);

    SocuNativeMatrixBuilder<Solve> writer_matrix;
    writer_matrix.reserve(Horizon, BlockSize, 1);
    writer_matrix.clear();

    const auto layout = writer_matrix.layout();
    muda::DeviceBuffer<Solve> legacy_diag;
    muda::DeviceBuffer<Solve> legacy_offdiag;
    legacy_diag.resize(layout.diag_element_count);
    legacy_offdiag.resize(layout.offdiag_element_count);
    zero_device_buffer(legacy_diag);
    zero_device_buffer(legacy_offdiag);

    std::vector<IndexT> old_to_chain(15, -1);
    for(IndexT i = 0; i < 12; ++i)
        old_to_chain[static_cast<SizeT>(i)] = i;
    old_to_chain[12] = 16;
    old_to_chain[13] = 17;
    old_to_chain[14] = 18;
    muda::DeviceBuffer<IndexT> old_to_chain_buffer{old_to_chain};
    muda::DeviceBuffer<IndexT> fem_fixed{std::vector<IndexT>{0}};
    muda::DeviceBuffer<IndexT> abd_vertex_to_body{std::vector<IndexT>{0}};
    muda::DeviceBuffer<ABDJacobi> abd_jacobians{
        std::vector<ABDJacobi>{ABDJacobi{x_bar}}};
    muda::DeviceBuffer<IndexT> abd_body_fixed{std::vector<IndexT>{0}};
    muda::DeviceBuffer<IndexT> writer_counters;
    writer_counters.resize(
        static_cast<SizeT>(SocuContactProgramWriterCounterSlot::Count));
    zero_device_buffer(writer_counters);

    const auto H6 = make_matrix<Store, 6>(Store{301});
    const auto H3 = subblock3(H6, 0, 3);

    write_abd_fem_with_program_writer_kernel<Store, Solve>
        <<<1, 1>>>(make_writer<Store, Solve>(plan, writer_matrix, writer_counters),
                   H6);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    write_fem_abd_with_legacy_sink_kernel<Store, Solve>
        <<<1, 1>>>(make_abd_fem_legacy_sink<Store, Solve>(
                       legacy_diag,
                       legacy_offdiag,
                       old_to_chain_buffer,
                       fem_fixed,
                       abd_vertex_to_body,
                       abd_jacobians,
                       abd_body_fixed),
                   H3);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto snapshot = writer_matrix.snapshot();
    std::vector<Solve> legacy_D;
    std::vector<Solve> legacy_E;
    legacy_diag.copy_to(legacy_D);
    legacy_offdiag.copy_to(legacy_E);
    require_vectors_close(snapshot.D, legacy_D, 1e-8);
    require_vectors_close(snapshot.E, legacy_E, 1e-8);

    std::vector<IndexT> counters;
    writer_counters.copy_to(counters);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ContactWrite)]
          == 1);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ExactTaskWrite)]
          == 1);
}

TEST_CASE("cuda_mixed_socu_contact_program_writer_diag_mirror_flag_contract",
          "[cuda_mixed_socu][contract][socu_approx][m3]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact writer tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;
    constexpr SizeT Horizon = 2;
    constexpr SizeT BlockSize = 16;

    auto plan = make_manual_fem_fem_no_mirror_plan();
    SocuNativeMatrixBuilder<Solve> writer_matrix;
    writer_matrix.reserve(Horizon, BlockSize, 1);
    writer_matrix.clear();

    muda::DeviceBuffer<IndexT> writer_counters;
    writer_counters.resize(
        static_cast<SizeT>(SocuContactProgramWriterCounterSlot::Count));
    zero_device_buffer(writer_counters);

    const auto H6 = make_matrix<Store, 6>(Store{41});
    write_abd_fem_with_program_writer_kernel<Store, Solve>
        <<<1, 1>>>(make_writer<Store, Solve>(plan, writer_matrix, writer_counters),
                   H6);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto snapshot = writer_matrix.snapshot();
    for(IndexT row = 0; row < 3; ++row)
    {
        for(IndexT col = 0; col < 3; ++col)
        {
            const SizeT forward_index =
                (static_cast<SizeT>(row) * BlockSize) + (3 + col);
            const SizeT mirrored_index =
                (static_cast<SizeT>(3 + col) * BlockSize) + row;
            CHECK(static_cast<double>(snapshot.D[forward_index])
                  == Catch::Approx(static_cast<double>(H6(row, 3 + col))));
            CHECK(static_cast<double>(snapshot.D[mirrored_index])
                  == Catch::Approx(0.0));
        }
    }
}

TEST_CASE("cuda_mixed_socu_contact_program_writer_status_counters",
          "[cuda_mixed_socu][contract][socu_approx][m3]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact writer tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;
    constexpr SizeT Horizon = 2;
    constexpr SizeT BlockSize = 16;

    auto plan = make_manual_status_plan();
    SocuNativeMatrixBuilder<Solve> writer_matrix;
    writer_matrix.reserve(Horizon, BlockSize, 1);
    writer_matrix.clear();

    muda::DeviceBuffer<IndexT> writer_counters;
    writer_counters.resize(
        static_cast<SizeT>(SocuContactProgramWriterCounterSlot::Count));
    zero_device_buffer(writer_counters);

    write_status_contacts_with_program_writer_kernel<Store, Solve>
        <<<1, 1>>>(make_writer<Store, Solve>(plan, writer_matrix, writer_counters),
                   make_matrix<Store, 6>(Store{7}));
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto snapshot = writer_matrix.snapshot();
    for(const auto value : snapshot.D)
        CHECK(static_cast<double>(value) == Catch::Approx(0.0));
    for(const auto value : snapshot.E)
        CHECK(static_cast<double>(value) == Catch::Approx(0.0));

    std::vector<IndexT> counters;
    writer_counters.copy_to(counters);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ContactWrite)]
          == 0);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ProgramSkipped)]
          == 2);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ProgramDropped)]
          == 1);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ProgramMixedRejected)]
          == 1);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ProgramMissing)]
          == 1);
    CHECK(counters[static_cast<SizeT>(
              SocuContactProgramWriterCounterSlot::ExactTaskWrite)]
          == 0);
}

TEST_CASE("cuda_mixed_socu_contact_program_writer_abd_abd_exact_matches_legacy",
          "[cuda_mixed_socu][contract][socu_approx][m4]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact writer tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;
    constexpr SizeT BlockSize = 16;

    const Vector3 x0{Float{2}, Float{-1}, Float{0.5}};
    const Vector3 x1{Float{-0.25}, Float{1.5}, Float{3}};

    for(const bool same_body : {true, false})
    {
        CAPTURE(same_body);
        auto plan = make_manual_abd_abd_plan(x0, x1, same_body);
        const SizeT horizon = same_body ? 1 : 2;

        SocuNativeMatrixBuilder<Solve> writer_matrix;
        writer_matrix.reserve(horizon, BlockSize, 1);
        writer_matrix.clear();

        const auto layout = writer_matrix.layout();
        muda::DeviceBuffer<Solve> legacy_diag;
        muda::DeviceBuffer<Solve> legacy_offdiag;
        legacy_diag.resize(layout.diag_element_count);
        legacy_offdiag.resize(layout.offdiag_element_count);
        zero_device_buffer(legacy_diag);
        zero_device_buffer(legacy_offdiag);

        std::vector<IndexT> old_to_chain(same_body ? 12 : 24, -1);
        for(IndexT i = 0; i < 12; ++i)
            old_to_chain[static_cast<SizeT>(i)] = i;
        if(!same_body)
        {
            for(IndexT i = 0; i < 12; ++i)
                old_to_chain[static_cast<SizeT>(12 + i)] = 16 + i;
        }
        muda::DeviceBuffer<IndexT> old_to_chain_buffer{old_to_chain};
        muda::DeviceBuffer<IndexT> abd_vertex_to_body{
            same_body ? std::vector<IndexT>{0, 0} : std::vector<IndexT>{0, 1}};
        muda::DeviceBuffer<ABDJacobi> abd_jacobians{
            std::vector<ABDJacobi>{ABDJacobi{x0}, ABDJacobi{x1}}};
        muda::DeviceBuffer<IndexT> abd_body_fixed{
            same_body ? std::vector<IndexT>{0} : std::vector<IndexT>{0, 0}};
        muda::DeviceBuffer<IndexT> writer_counters;
        writer_counters.resize(
            static_cast<SizeT>(SocuContactProgramWriterCounterSlot::Count));
        zero_device_buffer(writer_counters);

        const auto H6 = make_signed_matrix<Store, 6>(Store{11});
        const auto H3 = subblock3(H6, 0, 3);

        write_abd_fem_with_program_writer_kernel<Store, Solve>
            <<<1, 1>>>(make_writer<Store, Solve>(
                           plan,
                           writer_matrix,
                           writer_counters),
                       H6);
        REQUIRE(cudaGetLastError() == cudaSuccess);

        write_contact_block_with_legacy_sink_kernel<Store, Solve>
            <<<1, 1>>>(make_abd_legacy_sink<Store, Solve>(
                           legacy_diag,
                           legacy_offdiag,
                           old_to_chain_buffer,
                           abd_vertex_to_body,
                           abd_jacobians,
                           abd_body_fixed,
                           horizon),
                       0,
                       1,
                       H3,
                       same_body);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        const auto snapshot = writer_matrix.snapshot();
        std::vector<Solve> legacy_D;
        std::vector<Solve> legacy_E;
        legacy_diag.copy_to(legacy_D);
        legacy_offdiag.copy_to(legacy_E);
        require_vectors_close(snapshot.D, legacy_D, 1e-8);
        require_vectors_close(snapshot.E, legacy_E, 1e-8);

        std::vector<IndexT> counters;
        writer_counters.copy_to(counters);
        CHECK(counters[static_cast<SizeT>(
                  SocuContactProgramWriterCounterSlot::ContactWrite)]
              == 1);
        CHECK(counters[static_cast<SizeT>(
                  SocuContactProgramWriterCounterSlot::ExactTaskWrite)]
              == 1);
    }
}

TEST_CASE("cuda_mixed_socu_contact_program_writer_fem_policy_fallbacks_match_legacy",
          "[cuda_mixed_socu][contract][socu_approx][m4]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact writer tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;
    constexpr SizeT Horizon = 3;
    constexpr SizeT BlockSize = 16;

    for(const auto policy : {StructuredContactOffbandPolicy::Drop,
                            StructuredContactOffbandPolicy::Diag,
                            StructuredContactOffbandPolicy::DiagLump})
    {
        CAPTURE(static_cast<int>(policy));
        std::vector<SocuNativeVertexDescriptor> vertices(3);
        vertices[0] = make_vertex(SocuNativeDescriptorKind::Fem, 0, 3, 0, 0);
        vertices[1] = make_vertex(SocuNativeDescriptorKind::Fem, 3, 3, 1, 0);
        vertices[2] = make_vertex(SocuNativeDescriptorKind::Fem, 6, 3, 2, 0);
        muda::DeviceBuffer<SocuNativeVertexDescriptor> vertex_buffer{vertices};
        muda::DeviceBuffer<Vector2i> pps{std::vector<Vector2i>{Vector2i{0, 2}}};

        SocuContactAssemblyPlan plan;
        SocuContactAssemblyPlanM2Workspace workspace;
        build_socu_contact_assembly_plan_m2(
            plan,
            workspace,
            make_fem_pp_input(vertex_buffer, pps, Horizon, policy));
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        SocuNativeMatrixBuilder<Solve> writer_matrix;
        writer_matrix.reserve(Horizon, BlockSize, 1);
        writer_matrix.clear();

        const auto layout = writer_matrix.layout();
        muda::DeviceBuffer<Solve> legacy_diag;
        muda::DeviceBuffer<Solve> legacy_offdiag;
        legacy_diag.resize(layout.diag_element_count);
        legacy_offdiag.resize(layout.offdiag_element_count);
        zero_device_buffer(legacy_diag);
        zero_device_buffer(legacy_offdiag);

        std::vector<IndexT> old_to_chain(9, -1);
        old_to_chain[0] = 0;
        old_to_chain[1] = 1;
        old_to_chain[2] = 2;
        old_to_chain[3] = 16;
        old_to_chain[4] = 17;
        old_to_chain[5] = 18;
        old_to_chain[6] = 32;
        old_to_chain[7] = 33;
        old_to_chain[8] = 34;
        muda::DeviceBuffer<IndexT> old_to_chain_buffer{old_to_chain};
        muda::DeviceBuffer<IndexT> fem_fixed{std::vector<IndexT>{0, 0, 0}};
        muda::DeviceBuffer<IndexT> writer_counters;
        writer_counters.resize(
            static_cast<SizeT>(SocuContactProgramWriterCounterSlot::Count));
        zero_device_buffer(writer_counters);

        auto H6 = make_signed_matrix<Store, 6>(Store{23});
        symmetrize_block3(H6, 0);
        symmetrize_block3(H6, 3);
        write_abd_fem_with_program_writer_kernel<Store, Solve>
            <<<1, 1>>>(make_writer<Store, Solve>(
                           plan,
                           writer_matrix,
                           writer_counters),
                       H6);
        REQUIRE(cudaGetLastError() == cudaSuccess);

        if(policy != StructuredContactOffbandPolicy::Drop)
        {
            write_pair_half_with_legacy_sink_kernel<Store, Solve>
                <<<1, 1>>>(make_fem_legacy_sink<Store, Solve>(
                               legacy_diag,
                               legacy_offdiag,
                               old_to_chain_buffer,
                               fem_fixed,
                               Horizon,
                               policy),
                           Vector2i{0, 2},
                           H6);
            REQUIRE(cudaGetLastError() == cudaSuccess);
        }
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        const auto snapshot = writer_matrix.snapshot();
        std::vector<Solve> legacy_D;
        std::vector<Solve> legacy_E;
        legacy_diag.copy_to(legacy_D);
        legacy_offdiag.copy_to(legacy_E);
        require_vectors_close(snapshot.D, legacy_D, 1e-8);
        require_vectors_close(snapshot.E, legacy_E, 1e-8);

        std::vector<IndexT> counters;
        writer_counters.copy_to(counters);
        if(policy == StructuredContactOffbandPolicy::Drop)
        {
            CHECK(counters[static_cast<SizeT>(
                      SocuContactProgramWriterCounterSlot::ProgramDropped)]
                  == 1);
            CHECK(counters[static_cast<SizeT>(
                      SocuContactProgramWriterCounterSlot::ContactWrite)]
                  == 0);
        }
        else if(policy == StructuredContactOffbandPolicy::Diag)
        {
            CHECK(counters[static_cast<SizeT>(
                      SocuContactProgramWriterCounterSlot::DiagBlockTaskWrite)]
                  == 2);
        }
        else
        {
            CHECK(counters[static_cast<SizeT>(
                      SocuContactProgramWriterCounterSlot::LumpScalarTaskWrite)]
                  == 2);
        }
    }
}

TEST_CASE("cuda_mixed_socu_contact_program_writer_abd_policy_fallbacks_match_legacy",
          "[cuda_mixed_socu][contract][socu_approx][m4]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact writer tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;
    constexpr SizeT Horizon = 3;
    constexpr SizeT BlockSize = 16;

    const Vector3 x_bar{Float{2}, Float{-1}, Float{0.5}};
    for(const auto policy : {StructuredContactOffbandPolicy::Diag,
                            StructuredContactOffbandPolicy::DiagLump})
    {
        CAPTURE(static_cast<int>(policy));
        const bool lump = policy == StructuredContactOffbandPolicy::DiagLump;
        auto plan = make_manual_abd_fem_fallback_plan(
            x_bar,
            lump ? SocuContactProgramKind::DiagLump
                 : SocuContactProgramKind::Diag,
            lump ? SocuAssemblyWriteKind::LumpScalarAbd
                 : SocuAssemblyWriteKind::DiagBlockAbd,
            lump ? SocuAssemblyWriteKind::LumpScalarFem
                 : SocuAssemblyWriteKind::DiagBlockFem);

        SocuNativeMatrixBuilder<Solve> writer_matrix;
        writer_matrix.reserve(Horizon, BlockSize, 1);
        writer_matrix.clear();

        const auto layout = writer_matrix.layout();
        muda::DeviceBuffer<Solve> legacy_diag;
        muda::DeviceBuffer<Solve> legacy_offdiag;
        legacy_diag.resize(layout.diag_element_count);
        legacy_offdiag.resize(layout.offdiag_element_count);
        zero_device_buffer(legacy_diag);
        zero_device_buffer(legacy_offdiag);

        std::vector<IndexT> old_to_chain(15, -1);
        for(IndexT i = 0; i < 12; ++i)
            old_to_chain[static_cast<SizeT>(i)] = i;
        old_to_chain[12] = 32;
        old_to_chain[13] = 33;
        old_to_chain[14] = 34;
        muda::DeviceBuffer<IndexT> old_to_chain_buffer{old_to_chain};
        muda::DeviceBuffer<IndexT> fem_fixed{std::vector<IndexT>{0}};
        muda::DeviceBuffer<IndexT> abd_vertex_to_body{std::vector<IndexT>{0}};
        muda::DeviceBuffer<ABDJacobi> abd_jacobians{
            std::vector<ABDJacobi>{ABDJacobi{x_bar}}};
        muda::DeviceBuffer<IndexT> abd_body_fixed{std::vector<IndexT>{0}};
        muda::DeviceBuffer<IndexT> writer_counters;
        writer_counters.resize(
            static_cast<SizeT>(SocuContactProgramWriterCounterSlot::Count));
        zero_device_buffer(writer_counters);

        auto H6 = make_signed_matrix<Store, 6>(Store{53});
        symmetrize_block3(H6, 0);
        symmetrize_block3(H6, 3);
        write_abd_fem_with_program_writer_kernel<Store, Solve>
            <<<1, 1>>>(make_writer<Store, Solve>(
                           plan,
                           writer_matrix,
                           writer_counters),
                       H6);
        REQUIRE(cudaGetLastError() == cudaSuccess);

        write_pair_half_with_legacy_sink_kernel<Store, Solve>
            <<<1, 1>>>(make_abd_fem_policy_legacy_sink<Store, Solve>(
                           legacy_diag,
                           legacy_offdiag,
                           old_to_chain_buffer,
                           fem_fixed,
                           abd_vertex_to_body,
                           abd_jacobians,
                           abd_body_fixed,
                           policy),
                       Vector2i{0, 5},
                       H6);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        const auto snapshot = writer_matrix.snapshot();
        std::vector<Solve> legacy_D;
        std::vector<Solve> legacy_E;
        legacy_diag.copy_to(legacy_D);
        legacy_offdiag.copy_to(legacy_E);
        require_vectors_close(snapshot.D, legacy_D, 1e-8);
        require_vectors_close(snapshot.E, legacy_E, 1e-8);

        std::vector<IndexT> counters;
        writer_counters.copy_to(counters);
        CHECK(counters[static_cast<SizeT>(
                  lump ? SocuContactProgramWriterCounterSlot::LumpScalarTaskWrite
                       : SocuContactProgramWriterCounterSlot::DiagBlockTaskWrite)]
              == 2);
    }
}

TEST_CASE("cuda_mixed_socu_contact_program_writer_diag_scalar_compatibility_golden",
          "[cuda_mixed_socu][contract][socu_approx][m4]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact writer tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;
    constexpr SizeT Horizon = 1;
    constexpr SizeT BlockSize = 16;

    const Vector3 x_bar{Float{-0.25}, Float{1.5}, Float{3}};
    for(const bool abd : {false, true})
    {
        CAPTURE(abd);
        auto plan = make_manual_diag_scalar_plan(abd, x_bar);
        SocuNativeMatrixBuilder<Solve> writer_matrix;
        writer_matrix.reserve(Horizon, BlockSize, 1);
        writer_matrix.clear();

        muda::DeviceBuffer<IndexT> writer_counters;
        writer_counters.resize(
            static_cast<SizeT>(SocuContactProgramWriterCounterSlot::Count));
        zero_device_buffer(writer_counters);

        const auto H6 = make_signed_matrix<Store, 6>(Store{71});
        write_abd_fem_with_program_writer_kernel<Store, Solve>
            <<<1, 1>>>(make_writer<Store, Solve>(
                           plan,
                           writer_matrix,
                           writer_counters),
                       H6);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        const auto snapshot = writer_matrix.snapshot();
        const SizeT lane_count = abd ? 12 : 3;
        for(SizeT lane = 0; lane < lane_count; ++lane)
        {
            const IndexT component =
                static_cast<IndexT>(abd ? (lane < 3 ? lane : (lane - 3) / 3)
                                        : lane);
            const double weight =
                static_cast<double>(abd ? (lane < 3 ? Float{1}
                                                    : x_bar((lane - 3) % 3))
                                        : Float{1});
            const double expected =
                weight * static_cast<double>(H6(component, component)) * weight;
            const SizeT index = lane * BlockSize + lane;
            CHECK(static_cast<double>(snapshot.D[index])
                  == Catch::Approx(expected));
        }
        for(SizeT row = 0; row < BlockSize; ++row)
        {
            for(SizeT col = 0; col < BlockSize; ++col)
            {
                if(row == col && row < lane_count)
                    continue;
                CHECK(static_cast<double>(snapshot.D[row * BlockSize + col])
                      == Catch::Approx(0.0));
            }
        }

        std::vector<IndexT> counters;
        writer_counters.copy_to(counters);
        CHECK(counters[static_cast<SizeT>(
                  SocuContactProgramWriterCounterSlot::DiagScalarTaskWrite)]
              == 1);
    }
}

TEST_CASE("cuda_mixed_socu_contact_program_writer_source_isolation",
          "[cuda_mixed_socu][contract][socu_approx][m3][m4]")
{
    const auto root = std::filesystem::current_path();
    const auto writer =
        read_text_file(root / "src/backends/cuda_mixed_socu/linear_system/"
                              "socu_contact_program_writer.h");
    CHECK(writer.find("structured_contact_assembly_sink.h") == std::string::npos);
    CHECK(writer.find("socu_contact_program_debug_compare.h")
          == std::string::npos);
    CHECK(writer.find("old_to_chain") == std::string::npos);
    CHECK(writer.find("classify_dof_pair") == std::string::npos);
    CHECK(writer.find("SocuNativeContactStencilTarget") == std::string::npos);
    CHECK(writer.find("DiagBlockTaskWrite") != std::string::npos);
    CHECK(writer.find("DiagScalarTaskWrite") != std::string::npos);
    CHECK(writer.find("LumpScalarTaskWrite") != std::string::npos);
    CHECK(writer.find("ExactAbdAbdSameBody") != std::string::npos);
    CHECK(writer.find("ExactAbdAbdCrossBody") != std::string::npos);
    CHECK(writer.find("write_contact(") != std::string::npos);

    const auto debug_compare =
        read_text_file(root / "src/backends/cuda_mixed_socu/linear_system/"
                              "socu_contact_program_debug_compare.h");
    CHECK(debug_compare.find("SocuContactProgramDebugCompare")
          != std::string::npos);

    const char* build_dir_env = std::getenv("SOCU_NATIVE_CONTACT_BUILD_DIR");
    const auto compile_commands_path =
        build_dir_env != nullptr
            ? std::filesystem::path{build_dir_env} / "compile_commands.json"
            : root / "build/compile_commands.json";
    REQUIRE(std::filesystem::exists(compile_commands_path));
    const auto compile_commands = read_text_file(compile_commands_path);
    CHECK(compile_commands.find("socu_contact_program_writer.cu")
          != std::string::npos);
}
