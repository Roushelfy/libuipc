#include <app/app.h>
#include <linear_system/socu_contact_direct_evaluator.h>
#include <linear_system/socu_contact_executor.h>
#include <mixed_precision/policy.h>

#include <cuda_runtime.h>
#include <Eigen/Core>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_buffer_2d.h>
#include <muda/ext/linear_system/device_triplet_matrix.h>

#include <algorithm>
#include <array>
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
using uipc::Vector3i;
using uipc::Vector4i;
using uipc::span;

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

SocuNativeVertexDescriptor make_vertex(SocuNativeDescriptorKind kind,
                                       bool fixed,
                                       IndexT old_dof,
                                       IndexT dof_count,
                                       SizeT block,
                                       SizeT lane,
                                       IndexT abd_body = -1,
                                       IndexT abd_j_index = -1)
{
    SocuNativeVertexDescriptor out;
    out.kind = kind;
    out.fixed = fixed;
    out.old_dof = old_dof;
    out.dof_count = dof_count;
    out.block = block;
    out.lane = lane;
    out.abd_body = abd_body;
    out.abd_j_index = abd_j_index;
    out.epoch = 31;
    out.active = true;
    return out;
}

std::vector<SocuNativeVertexDescriptor> executor_fixture_vertices()
{
    std::vector<SocuNativeVertexDescriptor> vertices(7);
    vertices[0] =
        make_vertex(SocuNativeDescriptorKind::Fem, false, 0, 3, 0, 0);
    vertices[1] =
        make_vertex(SocuNativeDescriptorKind::Fem, false, 3, 3, 0, 3);
    vertices[2] =
        make_vertex(SocuNativeDescriptorKind::Fem, false, 16, 3, 1, 0);
    vertices[3] =
        make_vertex(SocuNativeDescriptorKind::Fem, true, 19, 3, 1, 3);
    vertices[4] =
        make_vertex(SocuNativeDescriptorKind::Abd, false, 32, 12, 2, 0, 0, 5);
    vertices[5] =
        make_vertex(SocuNativeDescriptorKind::Fem, false, 48, 3, 2, 12);
    vertices[6] =
        make_vertex(SocuNativeDescriptorKind::Abd, true, 64, 12, 3, 0, 1, 6);
    return vertices;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 3> make_block3(Scalar base)
{
    Eigen::Matrix<Scalar, 3, 3> block;
    for(IndexT row = 0; row < 3; ++row)
    {
        for(IndexT col = 0; col < 3; ++col)
            block(row, col) = static_cast<Scalar>(base + Scalar{10} * row + col);
    }
    return block;
}

SocuContactAssemblyPlanM2BuildInput make_executor_input(
    const muda::DeviceBuffer<SocuNativeVertexDescriptor>& vertices,
    const muda::DeviceBuffer<Vector2i>& pp_contacts,
    const muda::DeviceBuffer<Vector2i>& ph_contacts,
    const muda::DeviceBuffer<Vector2i>& friction_pp_contacts,
    StructuredContactOffbandPolicy policy,
    bool build_hot_block_plan = false,
    SocuContactExecutionStrategy hot_block_strategy =
        SocuContactExecutionStrategy::DirectScatter,
    SizeT hot_block_threshold = 0)
{
    SocuVertexSidePlanKey side_key;
    side_key.ordering_epoch = 2;
    side_key.native_descriptor_epoch = 31;
    side_key.fixed_mapping_epoch = 7;
    side_key.vertex_projection_epoch = 11;
    side_key.horizon = 4;
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
    input.pp_contacts = pp_contacts.view();
    input.ph_contacts = ph_contacts.view();
    input.friction_pp_contacts = friction_pp_contacts.view();
    input.pp_source =
        SocuContactM2SourceInput{0, 100, SocuContactModelKind::SimplexNormal};
    input.ph_source = SocuContactM2SourceInput{
        1,
        101,
        SocuContactModelKind::VertexHalfPlaneNormal};
    input.friction_pp_source = SocuContactM2SourceInput{
        2,
        200,
        SocuContactModelKind::SimplexFrictional};
    input.offband_policy = policy;
    input.side_coverage_mode = SocuVertexSideCoverageMode::ActiveSetTemporary;
    input.build_hot_block_plan = build_hot_block_plan;
    input.hot_block_strategy = hot_block_strategy;
    input.hot_block_threshold = hot_block_threshold;
    return input;
}

SocuContactAssemblyPlan build_executor_plan(
    const std::vector<Vector2i>& pp_host,
    const std::vector<Vector2i>& ph_host,
    const std::vector<Vector2i>& friction_pp_host,
    StructuredContactOffbandPolicy policy,
    SocuContactAssemblyPlanM2Workspace& workspace,
    bool build_hot_block_plan = false,
    SocuContactExecutionStrategy hot_block_strategy =
        SocuContactExecutionStrategy::DirectScatter,
    SizeT hot_block_threshold = 0)
{
    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices{
        executor_fixture_vertices()};
    muda::DeviceBuffer<Vector2i> pp_contacts{pp_host};
    muda::DeviceBuffer<Vector2i> ph_contacts{ph_host};
    muda::DeviceBuffer<Vector2i> friction_pp_contacts{friction_pp_host};

    SocuContactAssemblyPlan plan;
    build_socu_contact_assembly_plan_m2_active_set_temporary(
        plan,
        workspace,
        make_executor_input(vertices,
                            pp_contacts,
                            ph_contacts,
                            friction_pp_contacts,
                            policy,
                            build_hot_block_plan,
                            hot_block_strategy,
                            hot_block_threshold));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    return plan;
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

template <typename Scalar>
void require_all_zero(const std::vector<Scalar>& values)
{
    for(std::size_t i = 0; i < values.size(); ++i)
    {
        CAPTURE(i);
        CHECK(static_cast<double>(values[i]) == Catch::Approx(0.0).margin(0.0));
    }
}

template <typename Store, typename Solve, typename Evaluator>
__global__ void write_reference_with_program_writer_kernel(
    SocuContactAssemblyPlanView plan,
    SocuNativeMatrixView<Solve> matrix,
    Evaluator evaluator)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    const SocuContactProgramWriter<Store, Solve> writer{plan, matrix, {}};
    for(SizeT source_index = 0; source_index < plan.sources.size();
        ++source_index)
    {
        const auto source = plan.sources.data()[source_index];
        for(SizeT local_contact = 0; local_contact < source.contact_count;
            ++local_contact)
        {
            const auto map = plan.program_for(source.source_id, local_contact);
            if(map.status != SocuContactProgramMapStatus::Valid
               || map.program_id == SocuInvalidContactProgramId
               || static_cast<SizeT>(map.program_id) >= plan.programs.size())
            {
                SocuDeterministicContactHessian<Store> empty_hessian;
                writer.write_contact(source.source_id, local_contact, empty_hessian);
                continue;
            }

            const auto program =
                plan.programs.data()[static_cast<SizeT>(map.program_id)];
            writer.write_contact(source.source_id,
                                 local_contact,
                                 evaluator(program));
        }
    }
}

template <typename Store, typename Solve>
struct ExecutorRunResult
{
    SocuNativeMatrixSnapshot<Solve> executor_snapshot;
    SocuNativeMatrixSnapshot<Solve> writer_snapshot;
    std::vector<IndexT> executor_counters;
};

template <typename Store, typename Solve>
ExecutorRunResult<Store, Solve> run_executor_against_writer(
    SocuContactAssemblyPlan& plan,
    SizeT horizon = 4,
    SizeT block_size = 16)
{
    SocuNativeMatrixBuilder<Solve> executor_matrix;
    SocuNativeMatrixBuilder<Solve> writer_matrix;
    executor_matrix.reserve(horizon, block_size, 1);
    writer_matrix.reserve(horizon, block_size, 1);
    executor_matrix.clear();
    writer_matrix.clear();

    using Evaluator = SocuDeterministicContactEvaluator<Store>;
    muda::DeviceBuffer<IndexT> executor_counters;
    executor_counters.resize(
        static_cast<SizeT>(SocuContactExecutorCounterSlot::Count));
    zero_device_buffer(executor_counters);

    launch_socu_contact_executor<Store, Solve>(
        socu_contact_assembly_plan_view(plan),
        executor_matrix.view(),
        Evaluator{},
        executor_counters.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);

    write_reference_with_program_writer_kernel<Store, Solve, Evaluator>
        <<<1, 1>>>(socu_contact_assembly_plan_view(plan),
                   writer_matrix.view(),
                   Evaluator{});
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    ExecutorRunResult<Store, Solve> result;
    result.executor_snapshot = executor_matrix.snapshot();
    result.writer_snapshot = writer_matrix.snapshot();
    executor_counters.copy_to(result.executor_counters);
    return result;
}

IndexT counter_value(const std::vector<IndexT>& counters,
                     SocuContactExecutorCounterSlot slot)
{
    const auto index = static_cast<SizeT>(slot);
    REQUIRE(index < counters.size());
    return counters[index];
}

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream ifs{path};
    REQUIRE(ifs.good());
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

template <typename StoreT>
struct SocuContactDirectStencilOrderReferenceEvaluator
{
    using Alu = ActivePolicy::AluScalar;
    using Vec3A = Eigen::Matrix<Alu, 3, 1>;
    using Vec6A = Eigen::Matrix<Alu, 6, 1>;
    using Vec9A = Eigen::Matrix<Alu, 9, 1>;
    using Vec12A = Eigen::Matrix<Alu, 12, 1>;
    using Mat3A = Eigen::Matrix<Alu, 3, 3>;
    using Mat6A = Eigen::Matrix<Alu, 6, 6>;
    using Mat9A = Eigen::Matrix<Alu, 9, 9>;
    using Mat12A = Eigen::Matrix<Alu, 12, 12>;

    SocuContactDirectSceneView<StoreT> scene;
    SocuContactDirectSourceTable sources;

    MUDA_DEVICE bool supports_program(
        const SocuContactProgramHeader& program) const noexcept
    {
        const auto source = sources.source_for(program);
        return source.source_id != SocuInvalidContactSourceId
               && source.stencil_size == program.stencil_size
               && program.local_contact_id >= 0
               && static_cast<SizeT>(program.local_contact_id)
                      < static_cast<SizeT>(source.contact_count);
    }

    template <typename MatT>
    MUDA_DEVICE void copy_dense_contact_order(
        SocuDeterministicContactHessian<StoreT>& H,
        const MatT& dense,
        SizeT stencil_size) const noexcept
    {
        for(SizeT local_row = 0; local_row < stencil_size; ++local_row)
        {
            for(SizeT local_col = 0; local_col < stencil_size; ++local_col)
            {
                for(SizeT r = 0; r < 3; ++r)
                {
                    for(SizeT c = 0; c < 3; ++c)
                    {
                        const SizeT h_row = local_row * 3 + r;
                        const SizeT h_col = local_col * 3 + c;
                        const SizeT dense_row = local_row * 3 + r;
                        const SizeT dense_col = local_col * 3 + c;
                        H.values[h_row
                                 * SocuDeterministicContactHessian<
                                       StoreT>::MaxDofCount
                                 + h_col] =
                            safe_cast<StoreT>(dense(dense_row, dense_col));
                    }
                }
            }
        }
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT> operator()(
        SizeT,
        const SocuContactProgramHeader& program) const noexcept
    {
        SocuDeterministicContactHessian<StoreT> H;
        H.stencil_size = program.stencil_size;

        const auto source = sources.source_for(program);
        if(source.source_id == SocuInvalidContactSourceId
           || program.local_contact_id < 0
           || static_cast<SizeT>(program.local_contact_id)
                  >= static_cast<SizeT>(source.contact_count))
            return H;

        switch(program.model)
        {
            case SocuContactModelKind::SimplexNormal:
                return evaluate_simplex_normal(program, source, H);
            case SocuContactModelKind::SimplexFrictional:
                return evaluate_simplex_frictional(program, source, H);
            case SocuContactModelKind::VertexHalfPlaneNormal:
                return evaluate_vertex_half_plane_normal(program, source, H);
            case SocuContactModelKind::VertexHalfPlaneFrictional:
                return evaluate_vertex_half_plane_frictional(program, source, H);
        }
        return H;
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT> operator()(
        const SocuContactProgramHeader& program) const noexcept
    {
        return (*this)(SizeT{0}, program);
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT> evaluate_simplex_normal(
        const SocuContactProgramHeader& program,
        const SocuContactDirectSourceEntry& source,
        SocuDeterministicContactHessian<StoreT> H) const noexcept
    {
        using namespace sym::codim_ipc_simplex_contact;

        const SizeT contact_id = static_cast<SizeT>(program.local_contact_id);
        switch(program.family)
        {
            case SocuContactFamily::PT:
            {
                const auto PT = source.stencil4.data()[contact_id];
                Vector4i cids = {scene.contact_element_ids.data()[PT[0]],
                                 scene.contact_element_ids.data()[PT[1]],
                                 scene.contact_element_ids.data()[PT[2]],
                                 scene.contact_element_ids.data()[PT[3]]};
                const Alu kt2 =
                    safe_cast<Alu>(PT_kappa(scene.contact_tabular, cids)
                                   * scene.dt * scene.dt);
                const Alu thickness = safe_cast<Alu>(PT_thickness(
                    scene.thicknesses.data()[PT[0]],
                    scene.thicknesses.data()[PT[1]],
                    scene.thicknesses.data()[PT[2]],
                    scene.thicknesses.data()[PT[3]]));
                const Alu d_hat = safe_cast<Alu>(PT_d_hat(
                    scene.d_hats.data()[PT[0]],
                    scene.d_hats.data()[PT[1]],
                    scene.d_hats.data()[PT[2]],
                    scene.d_hats.data()[PT[3]]));
                const Vector4i flag = distance::point_triangle_distance_flag(
                    scene.positions.data()[PT[0]],
                    scene.positions.data()[PT[1]],
                    scene.positions.data()[PT[2]],
                    scene.positions.data()[PT[3]]);
                Vec12A G;
                Mat12A dense;
                PT_barrier_gradient_hessian(
                    G,
                    dense,
                    flag,
                    kt2,
                    d_hat,
                    thickness,
                    scene.positions.data()[PT[0]].template cast<Alu>(),
                    scene.positions.data()[PT[1]].template cast<Alu>(),
                    scene.positions.data()[PT[2]].template cast<Alu>(),
                    scene.positions.data()[PT[3]].template cast<Alu>());
                make_spd(dense);
                copy_dense_contact_order(H, dense, 4);
                return H;
            }
            case SocuContactFamily::EE:
            {
                const auto EE = source.stencil4.data()[contact_id];
                Vector4i cids = {scene.contact_element_ids.data()[EE[0]],
                                 scene.contact_element_ids.data()[EE[1]],
                                 scene.contact_element_ids.data()[EE[2]],
                                 scene.contact_element_ids.data()[EE[3]]};
                const Alu kt2 =
                    safe_cast<Alu>(EE_kappa(scene.contact_tabular, cids)
                                   * scene.dt * scene.dt);
                const Alu thickness = safe_cast<Alu>(EE_thickness(
                    scene.thicknesses.data()[EE[0]],
                    scene.thicknesses.data()[EE[1]],
                    scene.thicknesses.data()[EE[2]],
                    scene.thicknesses.data()[EE[3]]));
                const Alu d_hat = safe_cast<Alu>(EE_d_hat(
                    scene.d_hats.data()[EE[0]],
                    scene.d_hats.data()[EE[1]],
                    scene.d_hats.data()[EE[2]],
                    scene.d_hats.data()[EE[3]]));
                const Vector4i flag = distance::edge_edge_distance_flag(
                    scene.positions.data()[EE[0]],
                    scene.positions.data()[EE[1]],
                    scene.positions.data()[EE[2]],
                    scene.positions.data()[EE[3]]);
                Vec12A G;
                Mat12A dense;
                mollified_EE_barrier_gradient_hessian(
                    G,
                    dense,
                    flag,
                    kt2,
                    d_hat,
                    thickness,
                    scene.rest_positions.data()[EE[0]].template cast<Alu>(),
                    scene.rest_positions.data()[EE[1]].template cast<Alu>(),
                    scene.rest_positions.data()[EE[2]].template cast<Alu>(),
                    scene.rest_positions.data()[EE[3]].template cast<Alu>(),
                    scene.positions.data()[EE[0]].template cast<Alu>(),
                    scene.positions.data()[EE[1]].template cast<Alu>(),
                    scene.positions.data()[EE[2]].template cast<Alu>(),
                    scene.positions.data()[EE[3]].template cast<Alu>());
                make_spd(dense);
                copy_dense_contact_order(H, dense, 4);
                return H;
            }
            case SocuContactFamily::PE:
            {
                const auto PE = source.stencil3.data()[contact_id];
                Vector3i cids = {scene.contact_element_ids.data()[PE[0]],
                                 scene.contact_element_ids.data()[PE[1]],
                                 scene.contact_element_ids.data()[PE[2]]};
                const Alu kt2 =
                    safe_cast<Alu>(PE_kappa(scene.contact_tabular, cids)
                                   * scene.dt * scene.dt);
                const Alu thickness = safe_cast<Alu>(PE_thickness(
                    scene.thicknesses.data()[PE[0]],
                    scene.thicknesses.data()[PE[1]],
                    scene.thicknesses.data()[PE[2]]));
                const Alu d_hat = safe_cast<Alu>(PE_d_hat(
                    scene.d_hats.data()[PE[0]],
                    scene.d_hats.data()[PE[1]],
                    scene.d_hats.data()[PE[2]]));
                const Vector3i flag = distance::point_edge_distance_flag(
                    scene.positions.data()[PE[0]],
                    scene.positions.data()[PE[1]],
                    scene.positions.data()[PE[2]]);
                Vec9A G;
                Mat9A dense;
                PE_barrier_gradient_hessian(
                    G,
                    dense,
                    flag,
                    kt2,
                    d_hat,
                    thickness,
                    scene.positions.data()[PE[0]].template cast<Alu>(),
                    scene.positions.data()[PE[1]].template cast<Alu>(),
                    scene.positions.data()[PE[2]].template cast<Alu>());
                make_spd(dense);
                copy_dense_contact_order(H, dense, 3);
                return H;
            }
            case SocuContactFamily::PP:
            {
                const auto PP = source.stencil2.data()[contact_id];
                Vector2i cids = {scene.contact_element_ids.data()[PP[0]],
                                 scene.contact_element_ids.data()[PP[1]]};
                const Alu kt2 =
                    safe_cast<Alu>(PP_kappa(scene.contact_tabular, cids)
                                   * scene.dt * scene.dt);
                const Alu thickness = safe_cast<Alu>(
                    PP_thickness(scene.thicknesses.data()[PP[0]],
                                 scene.thicknesses.data()[PP[1]]));
                const Alu d_hat = safe_cast<Alu>(
                    PP_d_hat(scene.d_hats.data()[PP[0]],
                             scene.d_hats.data()[PP[1]]));
                const Vector2i flag = distance::point_point_distance_flag(
                    scene.positions.data()[PP[0]],
                    scene.positions.data()[PP[1]]);
                Vec6A G;
                Mat6A dense;
                PP_barrier_gradient_hessian(
                    G,
                    dense,
                    flag,
                    kt2,
                    d_hat,
                    thickness,
                    scene.positions.data()[PP[0]].template cast<Alu>(),
                    scene.positions.data()[PP[1]].template cast<Alu>());
                make_spd(dense);
                copy_dense_contact_order(H, dense, 2);
                return H;
            }
            case SocuContactFamily::PH:
                return H;
        }
        return H;
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT>
    evaluate_simplex_frictional(
        const SocuContactProgramHeader& program,
        const SocuContactDirectSourceEntry& source,
        SocuDeterministicContactHessian<StoreT> H) const noexcept
    {
        using namespace sym::codim_ipc_contact;

        const SizeT contact_id = static_cast<SizeT>(program.local_contact_id);
        const Alu epsvdt = safe_cast<Alu>(scene.eps_velocity * scene.dt);
        switch(program.family)
        {
            case SocuContactFamily::PT:
            {
                const auto PT = source.stencil4.data()[contact_id];
                Vector4i cids = {scene.contact_element_ids.data()[PT[0]],
                                 scene.contact_element_ids.data()[PT[1]],
                                 scene.contact_element_ids.data()[PT[2]],
                                 scene.contact_element_ids.data()[PT[3]]};
                const auto coeff = PT_contact_coeff(scene.contact_tabular, cids);
                const Alu kt2 = safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt);
                const Alu thickness = safe_cast<Alu>(PT_thickness(
                    scene.thicknesses.data()[PT[0]],
                    scene.thicknesses.data()[PT[1]],
                    scene.thicknesses.data()[PT[2]],
                    scene.thicknesses.data()[PT[3]]));
                const Alu d_hat = safe_cast<Alu>(PT_d_hat(
                    scene.d_hats.data()[PT[0]],
                    scene.d_hats.data()[PT[1]],
                    scene.d_hats.data()[PT[2]],
                    scene.d_hats.data()[PT[3]]));
                Vec12A G;
                Mat12A dense;
                PT_friction_gradient_hessian(
                    G,
                    dense,
                    kt2,
                    d_hat,
                    thickness,
                    safe_cast<Alu>(coeff.mu),
                    epsvdt,
                    scene.prev_positions.data()[PT[0]].template cast<Alu>(),
                    scene.prev_positions.data()[PT[1]].template cast<Alu>(),
                    scene.prev_positions.data()[PT[2]].template cast<Alu>(),
                    scene.prev_positions.data()[PT[3]].template cast<Alu>(),
                    scene.positions.data()[PT[0]].template cast<Alu>(),
                    scene.positions.data()[PT[1]].template cast<Alu>(),
                    scene.positions.data()[PT[2]].template cast<Alu>(),
                    scene.positions.data()[PT[3]].template cast<Alu>());
                make_spd(dense);
                copy_dense_contact_order(H, dense, 4);
                return H;
            }
            case SocuContactFamily::EE:
            {
                const auto EE = source.stencil4.data()[contact_id];
                Vector4i cids = {scene.contact_element_ids.data()[EE[0]],
                                 scene.contact_element_ids.data()[EE[1]],
                                 scene.contact_element_ids.data()[EE[2]],
                                 scene.contact_element_ids.data()[EE[3]]};
                const auto coeff = EE_contact_coeff(scene.contact_tabular, cids);
                const Alu kt2 = safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt);
                const Alu thickness = safe_cast<Alu>(EE_thickness(
                    scene.thicknesses.data()[EE[0]],
                    scene.thicknesses.data()[EE[1]],
                    scene.thicknesses.data()[EE[2]],
                    scene.thicknesses.data()[EE[3]]));
                const Alu d_hat = safe_cast<Alu>(EE_d_hat(
                    scene.d_hats.data()[EE[0]],
                    scene.d_hats.data()[EE[1]],
                    scene.d_hats.data()[EE[2]],
                    scene.d_hats.data()[EE[3]]));
                const Vec3A rest0 =
                    scene.rest_positions.data()[EE[0]].template cast<Alu>();
                const Vec3A rest1 =
                    scene.rest_positions.data()[EE[1]].template cast<Alu>();
                const Vec3A rest2 =
                    scene.rest_positions.data()[EE[2]].template cast<Alu>();
                const Vec3A rest3 =
                    scene.rest_positions.data()[EE[3]].template cast<Alu>();
                Alu eps_x;
                distance::edge_edge_mollifier_threshold(
                    rest0, rest1, rest2, rest3, static_cast<Alu>(1e-3), eps_x);
                const Vec3A prev0 =
                    scene.prev_positions.data()[EE[0]].template cast<Alu>();
                const Vec3A prev1 =
                    scene.prev_positions.data()[EE[1]].template cast<Alu>();
                const Vec3A prev2 =
                    scene.prev_positions.data()[EE[2]].template cast<Alu>();
                const Vec3A prev3 =
                    scene.prev_positions.data()[EE[3]].template cast<Alu>();
                Vec12A G;
                Mat12A dense;
                if(distance::need_mollify(prev0, prev1, prev2, prev3, eps_x))
                {
                    G.setZero();
                    dense.setZero();
                }
                else
                {
                    EE_friction_gradient_hessian(
                        G,
                        dense,
                        kt2,
                        d_hat,
                        thickness,
                        safe_cast<Alu>(coeff.mu),
                        epsvdt,
                        prev0,
                        prev1,
                        prev2,
                        prev3,
                        scene.positions.data()[EE[0]].template cast<Alu>(),
                        scene.positions.data()[EE[1]].template cast<Alu>(),
                        scene.positions.data()[EE[2]].template cast<Alu>(),
                        scene.positions.data()[EE[3]].template cast<Alu>());
                    make_spd(dense);
                }
                copy_dense_contact_order(H, dense, 4);
                return H;
            }
            case SocuContactFamily::PE:
            {
                const auto PE = source.stencil3.data()[contact_id];
                Vector3i cids = {scene.contact_element_ids.data()[PE[0]],
                                 scene.contact_element_ids.data()[PE[1]],
                                 scene.contact_element_ids.data()[PE[2]]};
                const auto coeff = PE_contact_coeff(scene.contact_tabular, cids);
                const Alu kt2 = safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt);
                const Alu thickness = safe_cast<Alu>(PE_thickness(
                    scene.thicknesses.data()[PE[0]],
                    scene.thicknesses.data()[PE[1]],
                    scene.thicknesses.data()[PE[2]]));
                const Alu d_hat = safe_cast<Alu>(PE_d_hat(
                    scene.d_hats.data()[PE[0]],
                    scene.d_hats.data()[PE[1]],
                    scene.d_hats.data()[PE[2]]));
                Vec9A G;
                Mat9A dense;
                PE_friction_gradient_hessian(
                    G,
                    dense,
                    kt2,
                    d_hat,
                    thickness,
                    safe_cast<Alu>(coeff.mu),
                    epsvdt,
                    scene.prev_positions.data()[PE[0]].template cast<Alu>(),
                    scene.prev_positions.data()[PE[1]].template cast<Alu>(),
                    scene.prev_positions.data()[PE[2]].template cast<Alu>(),
                    scene.positions.data()[PE[0]].template cast<Alu>(),
                    scene.positions.data()[PE[1]].template cast<Alu>(),
                    scene.positions.data()[PE[2]].template cast<Alu>());
                make_spd(dense);
                copy_dense_contact_order(H, dense, 3);
                return H;
            }
            case SocuContactFamily::PP:
            {
                const auto PP = source.stencil2.data()[contact_id];
                Vector2i cids = {scene.contact_element_ids.data()[PP[0]],
                                 scene.contact_element_ids.data()[PP[1]]};
                const auto coeff = PP_contact_coeff(scene.contact_tabular, cids);
                const Alu kt2 = safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt);
                const Alu thickness = safe_cast<Alu>(
                    PP_thickness(scene.thicknesses.data()[PP[0]],
                                 scene.thicknesses.data()[PP[1]]));
                const Alu d_hat = safe_cast<Alu>(
                    PP_d_hat(scene.d_hats.data()[PP[0]],
                             scene.d_hats.data()[PP[1]]));
                Vec6A G;
                Mat6A dense;
                PP_friction_gradient_hessian(
                    G,
                    dense,
                    kt2,
                    d_hat,
                    thickness,
                    safe_cast<Alu>(coeff.mu),
                    epsvdt,
                    scene.prev_positions.data()[PP[0]].template cast<Alu>(),
                    scene.prev_positions.data()[PP[1]].template cast<Alu>(),
                    scene.positions.data()[PP[0]].template cast<Alu>(),
                    scene.positions.data()[PP[1]].template cast<Alu>());
                make_spd(dense);
                copy_dense_contact_order(H, dense, 2);
                return H;
            }
            case SocuContactFamily::PH:
                return H;
        }
        return H;
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT>
    evaluate_vertex_half_plane_normal(
        const SocuContactProgramHeader& program,
        const SocuContactDirectSourceEntry& source,
        SocuDeterministicContactHessian<StoreT> H) const noexcept
    {
        if(program.family != SocuContactFamily::PH)
            return H;
        using namespace sym::ipc_vertex_half_contact;

        const auto PH =
            source.stencil2.data()[static_cast<SizeT>(program.local_contact_id)];
        const IndexT vI = PH(0);
        const IndexT HI = PH(1);
        const ContactCoeff coeff =
            scene.contact_tabular(scene.contact_element_ids.data()[vI],
                                  scene.contact_element_ids.data()[HI
                                      + scene.half_plane_vertex_offset]);
        Vec3A G;
        Mat3A dense;
        PH_barrier_gradient_hessian(
            G,
            dense,
            safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt),
            safe_cast<Alu>(scene.d_hats.data()[vI]),
            safe_cast<Alu>(scene.thicknesses.data()[vI]),
            scene.positions.data()[vI].template cast<Alu>(),
            scene.half_plane_positions.data()[HI].template cast<Alu>(),
            scene.half_plane_normals.data()[HI].template cast<Alu>());
        copy_dense_contact_order(H, dense, 1);
        return H;
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT>
    evaluate_vertex_half_plane_frictional(
        const SocuContactProgramHeader& program,
        const SocuContactDirectSourceEntry& source,
        SocuDeterministicContactHessian<StoreT> H) const noexcept
    {
        if(program.family != SocuContactFamily::PH)
            return H;
        using namespace sym::ipc_vertex_half_contact;

        const auto PH =
            source.stencil2.data()[static_cast<SizeT>(program.local_contact_id)];
        const IndexT vI = PH(0);
        const IndexT HI = PH(1);
        const ContactCoeff coeff =
            scene.contact_tabular(scene.contact_element_ids.data()[vI],
                                  scene.contact_element_ids.data()[HI
                                      + scene.half_plane_vertex_offset]);
        Vec3A G;
        Mat3A dense;
        PH_friction_gradient_hessian(
            G,
            dense,
            safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt),
            safe_cast<Alu>(scene.d_hats.data()[vI]),
            safe_cast<Alu>(scene.thicknesses.data()[vI]),
            safe_cast<Alu>(coeff.mu),
            safe_cast<Alu>(scene.eps_velocity * scene.dt),
            scene.prev_positions.data()[vI].template cast<Alu>(),
            scene.positions.data()[vI].template cast<Alu>(),
            scene.half_plane_positions.data()[HI].template cast<Alu>(),
            scene.half_plane_normals.data()[HI].template cast<Alu>());
        make_spd(dense);
        copy_dense_contact_order(H, dense, 1);
        return H;
    }
};

template <typename Store>
Eigen::Matrix<Store, 3, 3> hessian_block(
    const SocuDeterministicContactHessian<Store>& H,
    SizeT local_row,
    SizeT local_col)
{
    Eigen::Matrix<Store, 3, 3> block;
    for(SizeT row = 0; row < 3; ++row)
    {
        for(SizeT col = 0; col < 3; ++col)
        {
            block(row, col) =
                H(static_cast<IndexT>(local_row * 3 + row),
                  static_cast<IndexT>(local_col * 3 + col));
        }
    }
    return block;
}

template <typename Store, typename Stencil>
void populate_reference_triplets(
    muda::DeviceTripletMatrix<Store, 3>& matrix,
    SizeT vertex_count,
    const Stencil& stencil,
    SizeT stencil_size,
    const SocuDeterministicContactHessian<Store>& H)
{
    const SizeT half_size = stencil_size * (stencil_size + 1) / 2;
    matrix.resize(vertex_count, vertex_count, half_size);

    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<Eigen::Matrix<Store, 3, 3>> blocks;
    rows.reserve(half_size);
    cols.reserve(half_size);
    blocks.reserve(half_size);

    for(SizeT local_row = 0; local_row < stencil_size; ++local_row)
    {
        for(SizeT local_col = local_row; local_col < stencil_size; ++local_col)
        {
            rows.push_back(static_cast<int>(stencil(static_cast<Eigen::Index>(
                local_row))));
            cols.push_back(static_cast<int>(stencil(static_cast<Eigen::Index>(
                local_col))));
            blocks.push_back(hessian_block(H, local_row, local_col));
        }
    }

    matrix.row_indices().copy_from(rows.data());
    matrix.col_indices().copy_from(cols.data());
    matrix.values().copy_from(blocks.data());
}

template <typename Store>
double hessian_abs_sum(const SocuDeterministicContactHessian<Store>& H,
                       SizeT dof_count)
{
    double sum = 0.0;
    for(SizeT row = 0; row < dof_count; ++row)
    {
        for(SizeT col = 0; col < dof_count; ++col)
            sum += std::abs(static_cast<double>(
                H(static_cast<IndexT>(row), static_cast<IndexT>(col))));
    }
    return sum;
}
}  // namespace

TEST_CASE("cuda_mixed_socu_contact_executor_matches_program_writer",
          "[cuda_mixed_socu][contract][socu_approx][m5]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact executor tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_executor_plan(
        {Vector2i{1, 2}, Vector2i{2, 4}},
        {Vector2i{2, 99}},
        {Vector2i{1, 2}},
        StructuredContactOffbandPolicy::Drop,
        workspace);

    std::vector<SocuContactProgramBucket> buckets;
    plan.program_plan.buckets.copy_to(buckets);
    REQUIRE(buckets.size() >= 3);

    const auto result = run_executor_against_writer<Store, Solve>(plan);
    require_vectors_close(result.executor_snapshot.D,
                          result.writer_snapshot.D,
                          1e-8);
    require_vectors_close(result.executor_snapshot.E,
                          result.writer_snapshot.E,
                          1e-8);
    require_vectors_close(result.executor_snapshot.rhs,
                          result.writer_snapshot.rhs,
                          1e-8);

    CHECK(counter_value(result.executor_counters,
                        SocuContactExecutorCounterSlot::BucketVisit)
          == static_cast<IndexT>(buckets.size()));
    CHECK(counter_value(result.executor_counters,
                        SocuContactExecutorCounterSlot::ExactProgram)
          == 4);
    CHECK(counter_value(result.executor_counters,
                        SocuContactExecutorCounterSlot::TaskWrite)
          > 0);
}

TEST_CASE("cuda_mixed_socu_contact_executor_recompute_owner_reduce_matches_writer",
          "[cuda_mixed_socu][contract][socu_approx][m6][socu_contact_hot_blocks]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact executor tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_executor_plan({Vector2i{0, 2},
                                     Vector2i{0, 2},
                                     Vector2i{0, 2},
                                     Vector2i{0, 2}},
                                    {},
                                    {},
                                    StructuredContactOffbandPolicy::Drop,
                                    workspace,
                                    true,
                                    SocuContactExecutionStrategy::Recompute,
                                    2);
    REQUIRE(plan.program_plan.hot_blocks.ranges.size() == 3);
    REQUIRE(plan.program_plan.hot_blocks.strategy
            == SocuContactExecutionStrategy::Recompute);

    const auto result = run_executor_against_writer<Store, Solve>(plan);
    require_vectors_close(result.executor_snapshot.D,
                          result.writer_snapshot.D,
                          1e-8);
    require_vectors_close(result.executor_snapshot.E,
                          result.writer_snapshot.E,
                          1e-8);
    CHECK(counter_value(result.executor_counters,
                        SocuContactExecutorCounterSlot::DirectHotTaskSkipped)
          == static_cast<IndexT>(plan.program_plan.hot_blocks.eligible_task_count));
    CHECK(counter_value(result.executor_counters,
                        SocuContactExecutorCounterSlot::HotBlockRangeVisit)
          == static_cast<IndexT>(plan.program_plan.hot_blocks.ranges.size()));
}

TEST_CASE("cuda_mixed_socu_contact_executor_cached_microblock_matches_writer",
          "[cuda_mixed_socu][contract][socu_approx][m6][socu_contact_hot_blocks]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact executor tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_executor_plan({Vector2i{0, 2},
                                     Vector2i{0, 2},
                                     Vector2i{0, 2}},
                                    {},
                                    {},
                                    StructuredContactOffbandPolicy::Drop,
                                    workspace,
                                    true,
                                    SocuContactExecutionStrategy::CachedMicroblock,
                                    2);
    REQUIRE(plan.program_plan.hot_blocks.ranges.size() == 3);
    REQUIRE(plan.program_plan.hot_blocks.strategy
            == SocuContactExecutionStrategy::CachedMicroblock);

    const auto result = run_executor_against_writer<Store, Solve>(plan);
    require_vectors_close(result.executor_snapshot.D,
                          result.writer_snapshot.D,
                          1e-8);
    require_vectors_close(result.executor_snapshot.E,
                          result.writer_snapshot.E,
                          1e-8);
    CHECK(counter_value(result.executor_counters,
                        SocuContactExecutorCounterSlot::HotBlockRangeVisit)
          == static_cast<IndexT>(plan.program_plan.hot_blocks.ranges.size()));
    CHECK(counter_value(result.executor_counters,
                        SocuContactExecutorCounterSlot::HotMicroblockCacheWrite)
          > 0);
}

TEST_CASE("cuda_mixed_socu_contact_executor_detect_only_keeps_direct_scatter",
          "[cuda_mixed_socu][contract][socu_approx][m6][socu_contact_hot_blocks]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact executor tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_executor_plan({Vector2i{0, 2},
                                     Vector2i{0, 2},
                                     Vector2i{0, 2}},
                                    {},
                                    {},
                                    StructuredContactOffbandPolicy::Drop,
                                    workspace,
                                    true,
                                    SocuContactExecutionStrategy::DetectOnly,
                                    2);
    REQUIRE(plan.program_plan.hot_blocks.detect_only);
    REQUIRE(plan.program_plan.hot_blocks.ranges.size() == 3);

    const auto result = run_executor_against_writer<Store, Solve>(plan);
    require_vectors_close(result.executor_snapshot.D,
                          result.writer_snapshot.D,
                          1e-8);
    require_vectors_close(result.executor_snapshot.E,
                          result.writer_snapshot.E,
                          1e-8);
    CHECK(counter_value(result.executor_counters,
                        SocuContactExecutorCounterSlot::DirectHotTaskSkipped)
          == 0);
    CHECK(counter_value(result.executor_counters,
                        SocuContactExecutorCounterSlot::HotBlockRangeVisit)
          == 0);
}

TEST_CASE("cuda_mixed_socu_contact_executor_policy_buckets_match_writer",
          "[cuda_mixed_socu][contract][socu_approx][m5]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact executor tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    const std::vector<Vector2i> offband_pp{Vector2i{0, 4}};

    {
        SocuContactAssemblyPlanM2Workspace workspace;
        auto plan = build_executor_plan(offband_pp,
                                        {},
                                        {},
                                        StructuredContactOffbandPolicy::Drop,
                                        workspace);
        const auto result = run_executor_against_writer<Store, Solve>(plan);
        require_all_zero(result.executor_snapshot.D);
        require_all_zero(result.executor_snapshot.E);
        require_vectors_close(result.executor_snapshot.D,
                              result.writer_snapshot.D,
                              0.0);
        CHECK(counter_value(result.executor_counters,
                            SocuContactExecutorCounterSlot::DropProgram)
              == 1);
        CHECK(counter_value(result.executor_counters,
                            SocuContactExecutorCounterSlot::TaskWrite)
              == 0);
    }

    {
        SocuContactAssemblyPlanM2Workspace workspace;
        auto plan = build_executor_plan(offband_pp,
                                        {},
                                        {},
                                        StructuredContactOffbandPolicy::Diag,
                                        workspace);
        const auto result = run_executor_against_writer<Store, Solve>(plan);
        require_vectors_close(result.executor_snapshot.D,
                              result.writer_snapshot.D,
                              1e-8);
        require_vectors_close(result.executor_snapshot.E,
                              result.writer_snapshot.E,
                              1e-8);
        CHECK(counter_value(result.executor_counters,
                            SocuContactExecutorCounterSlot::DiagProgram)
              == 1);
        CHECK(counter_value(result.executor_counters,
                            SocuContactExecutorCounterSlot::TaskWrite)
              > 0);
    }

    {
        SocuContactAssemblyPlanM2Workspace workspace;
        auto plan = build_executor_plan(offband_pp,
                                        {},
                                        {},
                                        StructuredContactOffbandPolicy::DiagLump,
                                        workspace);
        const auto result = run_executor_against_writer<Store, Solve>(plan);
        require_vectors_close(result.executor_snapshot.D,
                              result.writer_snapshot.D,
                              1e-8);
        require_vectors_close(result.executor_snapshot.E,
                              result.writer_snapshot.E,
                              1e-8);
        CHECK(counter_value(result.executor_counters,
                            SocuContactExecutorCounterSlot::DiagLumpProgram)
              == 1);
        CHECK(counter_value(result.executor_counters,
                            SocuContactExecutorCounterSlot::TaskWrite)
              > 0);
    }
}

TEST_CASE("cuda_mixed_socu_contact_executor_bucket_order_is_stable",
          "[cuda_mixed_socu][contract][socu_approx][m5]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact executor tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    SocuContactAssemblyPlanM2Workspace workspace_a;
    auto plan_a = build_executor_plan(
        {Vector2i{1, 2}, Vector2i{0, 4}},
        {Vector2i{2, 99}},
        {Vector2i{1, 2}},
        StructuredContactOffbandPolicy::Diag,
        workspace_a);

    std::vector<SocuContactProgramBucket> buckets;
    plan_a.program_plan.buckets.copy_to(buckets);
    REQUIRE(buckets.size() > 1);

    SocuContactAssemblyPlanM2Workspace workspace_b;
    auto plan_b = build_executor_plan(
        {Vector2i{1, 2}, Vector2i{0, 4}},
        {Vector2i{2, 99}},
        {Vector2i{1, 2}},
        StructuredContactOffbandPolicy::Diag,
        workspace_b);
    std::reverse(buckets.begin(), buckets.end());
    plan_b.program_plan.buckets = buckets;

    const auto result_a = run_executor_against_writer<Store, Solve>(plan_a);
    const auto result_b = run_executor_against_writer<Store, Solve>(plan_b);
    require_vectors_close(result_a.executor_snapshot.D,
                          result_b.executor_snapshot.D,
                          1e-8);
    require_vectors_close(result_a.executor_snapshot.E,
                          result_b.executor_snapshot.E,
                          1e-8);
}

TEST_CASE("cuda_mixed_socu_contact_executor_triplet_source_views_smoke",
          "[cuda_mixed_socu][contract][socu_approx][m5]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact executor tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_executor_plan({Vector2i{1, 2}},
                                    {Vector2i{2, 99}},
                                    {},
                                    StructuredContactOffbandPolicy::Drop,
                                    workspace);

    std::vector<SocuContactProgramHeader> programs;
    plan.program_plan.programs.copy_to(programs);
    REQUIRE(programs.size() == 2);
    CHECK(programs[0].model == SocuContactModelKind::SimplexNormal);
    CHECK(programs[0].family == SocuContactFamily::PP);
    CHECK(programs[1].model == SocuContactModelKind::VertexHalfPlaneNormal);
    CHECK(programs[1].family == SocuContactFamily::PH);

    muda::DeviceTripletMatrix<Store, 3> pp_hessians;
    pp_hessians.resize(7, 7, 3);
    const std::vector<int> pp_rows{1, 1, 2};
    const std::vector<int> pp_cols{1, 2, 2};
    const std::vector<Eigen::Matrix<Store, 3, 3>> pp_blocks{
        make_block3<Store>(Store{10}),
        make_block3<Store>(Store{20}),
        make_block3<Store>(Store{30})};
    pp_hessians.row_indices().copy_from(pp_rows.data());
    pp_hessians.col_indices().copy_from(pp_cols.data());
    pp_hessians.values().copy_from(pp_blocks.data());

    muda::DeviceTripletMatrix<Store, 3> ph_hessians;
    ph_hessians.resize(7, 7, 1);
    const std::vector<int> ph_rows{2};
    const std::vector<int> ph_cols{2};
    const std::vector<Eigen::Matrix<Store, 3, 3>> ph_blocks{
        make_block3<Store>(Store{70})};
    ph_hessians.row_indices().copy_from(ph_rows.data());
    ph_hessians.col_indices().copy_from(ph_cols.data());
    ph_hessians.values().copy_from(ph_blocks.data());

    SocuContactEvaluatorSourceTable<Store> sources;
    sources.simplex_normal.pp_hessians = pp_hessians.view();
    sources.vertex_half_plane_normal.ph_hessians = ph_hessians.view();

    SocuNativeMatrixBuilder<Solve> matrix;
    matrix.reserve(4, 16, 1);
    matrix.clear();
    launch_socu_contact_executor<Store, Solve>(
        socu_contact_assembly_plan_view(plan),
        matrix.view(),
        SocuContactTripletEvaluator<Store>{socu_contact_assembly_plan_view(plan),
                                           sources});
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto snapshot = matrix.snapshot();
    const auto& layout = snapshot.layout;
    const auto diag_index = [&](SizeT block, SizeT row, SizeT col)
    {
        return (block * layout.block_size + row) * layout.block_size + col;
    };
    const auto first_offdiag_index = [&](SizeT left_block, SizeT row, SizeT col)
    {
        return (left_block * layout.block_size + row) * layout.block_size + col;
    };

    CHECK(static_cast<double>(snapshot.D[diag_index(0, 3, 3)])
          == Catch::Approx(10.0).margin(1e-8));
    CHECK(static_cast<double>(snapshot.E[first_offdiag_index(0, 0, 3)])
          == Catch::Approx(20.0).margin(1e-8));
    CHECK(static_cast<double>(snapshot.D[diag_index(1, 0, 0)])
          == Catch::Approx(100.0).margin(1e-8));
}

TEST_CASE("cuda_mixed_socu_contact_executor_source_id_indexed_triplets",
          "[cuda_mixed_socu][contract][socu_approx][m65]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact executor tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices{
        executor_fixture_vertices()};
    muda::DeviceBuffer<Vector2i> pp_contacts_a{
        std::vector<Vector2i>{Vector2i{1, 2}}};
    muda::DeviceBuffer<Vector2i> pp_contacts_b{
        std::vector<Vector2i>{Vector2i{1, 2}}};

    SocuVertexSidePlanKey side_key;
    side_key.ordering_epoch = 2;
    side_key.native_descriptor_epoch = 31;
    side_key.fixed_mapping_epoch = 7;
    side_key.vertex_projection_epoch = 11;
    side_key.horizon = 4;
    side_key.block_size = 16;

    SocuContactProgramPlanKey program_key;
    program_key.side_key = side_key;
    program_key.contact_topology_epoch = 17;
    program_key.contact_layout_hash = 19;
    program_key.contact_content_hash = 23;
    program_key.offband_policy = StructuredContactOffbandPolicy::Drop;

    std::vector<SocuContactM2SourceInput> source_inputs(2);
    source_inputs[0].source_id = 1;
    source_inputs[0].reporter_id = 101;
    source_inputs[0].model = SocuContactModelKind::SimplexNormal;
    source_inputs[0].family = SocuContactFamily::PP;
    source_inputs[0].stencil_size = 2;
    source_inputs[0].stencil2 = pp_contacts_b.view();
    source_inputs[1].source_id = 0;
    source_inputs[1].reporter_id = 100;
    source_inputs[1].model = SocuContactModelKind::SimplexNormal;
    source_inputs[1].family = SocuContactFamily::PP;
    source_inputs[1].stencil_size = 2;
    source_inputs[1].stencil2 = pp_contacts_a.view();

    SocuContactAssemblyPlanM2BuildInput input;
    input.side_key = side_key;
    input.program_key = program_key;
    input.vertex_descriptors = vertices.view();
    input.sources = span<const SocuContactM2SourceInput>{source_inputs};
    input.offband_policy = StructuredContactOffbandPolicy::Drop;
    input.side_coverage_mode = SocuVertexSideCoverageMode::ActiveSetTemporary;

    SocuContactAssemblyPlan plan;
    SocuContactAssemblyPlanM2Workspace workspace;
    build_socu_contact_assembly_plan_m2(plan, workspace, input);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(plan.program_plan.last_stats.source_id_validation_status
            == SocuContactSourceIdValidationStatus::ValidDense);
    REQUIRE(plan.program_plan.programs.size() == 2);
    REQUIRE(plan.program_plan.last_stats.valid_program_map_count == 2);

    std::vector<SocuContactSourceHeader> plan_sources;
    std::vector<SocuContactProgramHeader> plan_programs;
    plan.program_plan.sources.copy_to(plan_sources);
    plan.program_plan.programs.copy_to(plan_programs);
    REQUIRE(plan_sources.size() == 2);
    REQUIRE(plan_programs.size() == 2);
    CHECK(plan_sources[0].source_id == 0);
    CHECK(plan_sources[0].reporter_id == 100);
    CHECK(plan_sources[1].source_id == 1);
    CHECK(plan_sources[1].reporter_id == 101);
    CHECK(plan_programs[0].source_id == 0);
    CHECK(plan_programs[1].source_id == 1);

    muda::DeviceTripletMatrix<Store, 3> pp_hessians_a;
    pp_hessians_a.resize(7, 7, 3);
    const std::vector<int> rows{1, 1, 2};
    const std::vector<int> cols{1, 2, 2};
    const std::vector<Eigen::Matrix<Store, 3, 3>> blocks_a{
        make_block3<Store>(Store{10}),
        make_block3<Store>(Store{20}),
        make_block3<Store>(Store{30})};
    pp_hessians_a.row_indices().copy_from(rows.data());
    pp_hessians_a.col_indices().copy_from(cols.data());
    pp_hessians_a.values().copy_from(blocks_a.data());

    muda::DeviceTripletMatrix<Store, 3> pp_hessians_b;
    pp_hessians_b.resize(7, 7, 3);
    const std::vector<Eigen::Matrix<Store, 3, 3>> blocks_b{
        make_block3<Store>(Store{40}),
        make_block3<Store>(Store{50}),
        make_block3<Store>(Store{60})};
    pp_hessians_b.row_indices().copy_from(rows.data());
    pp_hessians_b.col_indices().copy_from(cols.data());
    pp_hessians_b.values().copy_from(blocks_b.data());

    std::vector<SocuContactEvaluatorSourceEntry<Store>> source_entries(2);
    source_entries[0].source_id = 0;
    source_entries[0].model = SocuContactModelKind::SimplexNormal;
    source_entries[0].family = SocuContactFamily::PP;
    source_entries[0].hessians = pp_hessians_a.view();
    source_entries[1].source_id = 1;
    source_entries[1].model = SocuContactModelKind::SimplexNormal;
    source_entries[1].family = SocuContactFamily::PP;
    source_entries[1].hessians = pp_hessians_b.view();
    muda::DeviceBuffer<SocuContactEvaluatorSourceEntry<Store>>
        source_entry_buffer{source_entries};

    SocuContactEvaluatorSourceTable<Store> sources;
    sources.source_entries = source_entry_buffer.view().as_const();

    SocuNativeMatrixBuilder<Solve> matrix;
    matrix.reserve(4, 16, 1);
    matrix.clear();
    launch_socu_contact_executor<Store, Solve>(
        socu_contact_assembly_plan_view(plan),
        matrix.view(),
        SocuContactTripletEvaluator<Store>{socu_contact_assembly_plan_view(plan),
                                           sources});
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto snapshot = matrix.snapshot();
    const auto& layout = snapshot.layout;
    const auto diag_index = [&](SizeT block, SizeT row, SizeT col)
    {
        return (block * layout.block_size + row) * layout.block_size + col;
    };
    const auto first_offdiag_index = [&](SizeT left_block, SizeT row, SizeT col)
    {
        return (left_block * layout.block_size + row) * layout.block_size + col;
    };

    CHECK(static_cast<double>(snapshot.D[diag_index(0, 3, 3)])
          == Catch::Approx(50.0).margin(1e-8));
    CHECK(static_cast<double>(snapshot.E[first_offdiag_index(0, 0, 3)])
          == Catch::Approx(70.0).margin(1e-8));
    CHECK(static_cast<double>(snapshot.D[diag_index(1, 0, 0)])
          == Catch::Approx(90.0).margin(1e-8));
}

TEST_CASE("cuda_mixed_socu_contact_executor_empty_buckets_are_noop",
          "[cuda_mixed_socu][contract][socu_approx][m5]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact executor tests");

    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_executor_plan({},
                                    {},
                                    {},
                                    StructuredContactOffbandPolicy::Drop,
                                    workspace);
    REQUIRE(plan.program_plan.buckets.size() == 0);

    const auto result = run_executor_against_writer<Store, Solve>(plan);
    require_all_zero(result.executor_snapshot.D);
    require_all_zero(result.executor_snapshot.E);
    require_all_zero(result.executor_snapshot.rhs);
    CHECK(counter_value(result.executor_counters,
                        SocuContactExecutorCounterSlot::BucketVisit)
          == 0);
    CHECK(counter_value(result.executor_counters,
                        SocuContactExecutorCounterSlot::ProgramVisit)
          == 0);
}

TEST_CASE("cuda_mixed_socu_contact_direct_evaluator_flags_unsupported_sources",
          "[cuda_mixed_socu][contract][socu_approx][m67]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact direct evaluator tests");

    using Store = ActivePolicy::StoreScalar;

    SocuContactAssemblyPlanM2Workspace workspace;
    auto plan = build_executor_plan({Vector2i{1, 2}},
                                    {},
                                    {},
                                    StructuredContactOffbandPolicy::Drop,
                                    workspace);
    const auto plan_view = socu_contact_assembly_plan_view(plan);
    REQUIRE(plan_view.programs.size() == 1);

    SocuContactDirectSourceTable direct_sources;
    SocuContactDirectEvaluator<Store> direct_evaluator{
        plan_view,
        SocuContactDirectSceneView<Store>{},
        direct_sources};

    muda::DeviceBuffer<SocuDeterministicContactHessian<Store>> hessians;
    muda::DeviceBuffer<IndexT> unsupported_flags;
    hessians.resize(plan_view.programs.size());
    unsupported_flags.resize(plan_view.programs.size());

    launch_socu_contact_direct_evaluate_programs<Store>(
        plan_view,
        direct_evaluator,
        hessians.view(),
        unsupported_flags.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<IndexT> flags;
    std::vector<SocuDeterministicContactHessian<Store>> direct_host;
    unsupported_flags.copy_to(flags);
    hessians.copy_to(direct_host);
    REQUIRE(flags.size() == 1);
    REQUIRE(direct_host.size() == 1);
    CHECK(flags[0] == IndexT{1});
    CHECK(static_cast<double>(direct_host[0](0, 0))
          == Catch::Approx(0.0).margin(0.0));

    launch_socu_contact_replace_direct_unsupported_programs<Store>(
        plan_view,
        hessians.view(),
        unsupported_flags.view().as_const(),
        SocuDeterministicContactEvaluator<Store>{});
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<SocuContactProgramHeader> programs;
    std::vector<SocuDeterministicContactHessian<Store>> fallback_host;
    plan.program_plan.programs.copy_to(programs);
    hessians.copy_to(fallback_host);
    REQUIRE(programs.size() == 1);
    REQUIRE(fallback_host.size() == 1);
    CHECK(fallback_host[0].stencil_size == programs[0].stencil_size);
    CHECK(static_cast<double>(fallback_host[0](0, 0)) != 0.0);
}

TEST_CASE("cuda_mixed_socu_contact_direct_evaluator_per_family_triplet_parity",
          "[cuda_mixed_socu][contract][socu_approx][m67]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact direct evaluator tests");

    using Store = ActivePolicy::StoreScalar;

    constexpr SizeT SourceCount = 10;
    enum SourceIndex : SizeT
    {
        NormalPT = 0,
        NormalEE,
        NormalPE,
        NormalPP,
        NormalPH,
        FrictionPT,
        FrictionEE,
        FrictionPE,
        FrictionPP,
        FrictionPH,
    };

    const std::array<const char*, SourceCount> source_names{
        "normal/PT",
        "normal/EE",
        "normal/PE",
        "normal/PP",
        "normal/PH",
        "friction/PT",
        "friction/EE",
        "friction/PE",
        "friction/PP",
        "friction/PH",
    };
    const std::array<SocuContactModelKind, SourceCount> models{
        SocuContactModelKind::SimplexNormal,
        SocuContactModelKind::SimplexNormal,
        SocuContactModelKind::SimplexNormal,
        SocuContactModelKind::SimplexNormal,
        SocuContactModelKind::VertexHalfPlaneNormal,
        SocuContactModelKind::SimplexFrictional,
        SocuContactModelKind::SimplexFrictional,
        SocuContactModelKind::SimplexFrictional,
        SocuContactModelKind::SimplexFrictional,
        SocuContactModelKind::VertexHalfPlaneFrictional,
    };
    const std::array<SocuContactFamily, SourceCount> families{
        SocuContactFamily::PT,
        SocuContactFamily::EE,
        SocuContactFamily::PE,
        SocuContactFamily::PP,
        SocuContactFamily::PH,
        SocuContactFamily::PT,
        SocuContactFamily::EE,
        SocuContactFamily::PE,
        SocuContactFamily::PP,
        SocuContactFamily::PH,
    };
    const std::array<std::uint16_t, SourceCount> stencil_sizes{
        4, 4, 3, 2, 2, 4, 4, 3, 2, 2,
    };

    constexpr SizeT VertexCount = 8;
    std::vector<SocuNativeVertexDescriptor> vertices(VertexCount);
    for(SizeT i = 0; i < VertexCount; ++i)
    {
        vertices[i] = make_vertex(SocuNativeDescriptorKind::Fem,
                                  false,
                                  static_cast<IndexT>(i * 3),
                                  3,
                                  0,
                                  i * 3);
    }
    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertex_buffer{vertices};

    const std::vector<Vector4i> pt_host{Vector4i{0, 1, 2, 3}};
    const std::vector<Vector4i> ee_host{Vector4i{0, 1, 2, 3}};
    const std::vector<Vector3i> pe_host{Vector3i{4, 5, 6}};
    const std::vector<Vector2i> pp_host{Vector2i{6, 7}};
    const std::vector<Vector2i> ph_host{Vector2i{0, 0}};
    muda::DeviceBuffer<Vector4i> pt_contacts{pt_host};
    muda::DeviceBuffer<Vector4i> ee_contacts{ee_host};
    muda::DeviceBuffer<Vector3i> pe_contacts{pe_host};
    muda::DeviceBuffer<Vector2i> pp_contacts{pp_host};
    muda::DeviceBuffer<Vector2i> ph_contacts{ph_host};

    std::vector<SocuContactM2SourceInput> source_inputs(SourceCount);
    std::vector<SocuContactDirectSourceEntry> direct_entries(SourceCount);
    const auto configure_source =
        [&](SourceIndex index,
            muda::CBufferView<Vector4i> stencil4,
            muda::CBufferView<Vector3i> stencil3,
            muda::CBufferView<Vector2i> stencil2)
    {
        auto& source = source_inputs[static_cast<SizeT>(index)];
        source.source_id = static_cast<SocuContactSourceId>(index);
        source.reporter_id = static_cast<std::uint32_t>(500 + index);
        source.model = models[static_cast<SizeT>(index)];
        source.family = families[static_cast<SizeT>(index)];
        source.stencil_size = stencil_sizes[static_cast<SizeT>(index)];
        source.stencil4 = stencil4;
        source.stencil3 = stencil3;
        source.stencil2 = stencil2;

        auto& direct = direct_entries[static_cast<SizeT>(index)];
        direct.source_id = source.source_id;
        direct.model = source.model;
        direct.family = source.family;
        direct.stencil_size = source.stencil_size;
        direct.contact_count = 1;
        direct.stencil4 = stencil4;
        direct.stencil3 = stencil3;
        direct.stencil2 = stencil2;
    };
    configure_source(NormalPT, pt_contacts.view(), {}, {});
    configure_source(NormalEE, ee_contacts.view(), {}, {});
    configure_source(NormalPE, {}, pe_contacts.view(), {});
    configure_source(NormalPP, {}, {}, pp_contacts.view());
    configure_source(NormalPH, {}, {}, ph_contacts.view());
    configure_source(FrictionPT, pt_contacts.view(), {}, {});
    configure_source(FrictionEE, ee_contacts.view(), {}, {});
    configure_source(FrictionPE, {}, pe_contacts.view(), {});
    configure_source(FrictionPP, {}, {}, pp_contacts.view());
    configure_source(FrictionPH, {}, {}, ph_contacts.view());
    std::rotate(source_inputs.begin(),
                source_inputs.begin() + 5,
                source_inputs.end());

    SocuVertexSidePlanKey side_key;
    side_key.ordering_epoch = 3;
    side_key.native_descriptor_epoch = 31;
    side_key.fixed_mapping_epoch = 7;
    side_key.vertex_projection_epoch = 11;
    side_key.horizon = 1;
    side_key.block_size = 32;

    SocuContactProgramPlanKey program_key;
    program_key.side_key = side_key;
    program_key.contact_topology_epoch = 19;
    program_key.contact_layout_hash = 23;
    program_key.contact_content_hash = 29;
    program_key.offband_policy = StructuredContactOffbandPolicy::Drop;

    SocuContactAssemblyPlanM2BuildInput input;
    input.side_key = side_key;
    input.program_key = program_key;
    input.vertex_descriptors = vertex_buffer.view();
    input.sources = span<const SocuContactM2SourceInput>{source_inputs};
    input.offband_policy = StructuredContactOffbandPolicy::Drop;
    input.side_coverage_mode = SocuVertexSideCoverageMode::ActiveSetTemporary;

    SocuContactAssemblyPlan plan;
    SocuContactAssemblyPlanM2Workspace workspace;
    build_socu_contact_assembly_plan_m2(plan, workspace, input);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(plan.program_plan.last_stats.source_id_validation_status
            == SocuContactSourceIdValidationStatus::ValidDense);
    REQUIRE(plan.program_plan.programs.size() == SourceCount);

    std::vector<SocuContactProgramHeader> programs;
    plan.program_plan.programs.copy_to(programs);
    REQUIRE(programs.size() == SourceCount);
    for(SizeT i = 0; i < SourceCount; ++i)
    {
        CAPTURE(source_names[i]);
        CHECK(programs[i].source_id == static_cast<SocuContactSourceId>(i));
        CHECK(programs[i].local_contact_id == 0);
        CHECK(programs[i].model == models[i]);
        CHECK(programs[i].family == families[i]);
        CHECK(programs[i].stencil_size == stencil_sizes[i]);
        CHECK(programs[i].program_kind == SocuContactProgramKind::Exact);
    }

    const std::vector<Vector3> positions_host{
        Vector3{0.08, 0.12, 0.28},
        Vector3{1.20, 0.05, 0.15},
        Vector3{0.15, 1.10, 0.25},
        Vector3{0.05, 0.20, 1.25},
        Vector3{-0.30, 0.40, 0.10},
        Vector3{1.10, 0.40, 0.00},
        Vector3{-0.20, 1.20, 0.30},
        Vector3{0.70, -0.80, 0.45},
    };
    std::vector<Vector3> prev_positions_host = positions_host;
    std::vector<Vector3> rest_positions_host = positions_host;
    for(SizeT i = 0; i < prev_positions_host.size(); ++i)
    {
        prev_positions_host[i](0) -= Float{0.01} * static_cast<Float>(i + 1);
        prev_positions_host[i](1) += Float{0.006} * static_cast<Float>(i + 1);
        rest_positions_host[i](2) += Float{0.002} * static_cast<Float>(i);
    }
    muda::DeviceBuffer<Vector3> positions{positions_host};
    muda::DeviceBuffer<Vector3> prev_positions{prev_positions_host};
    muda::DeviceBuffer<Vector3> rest_positions{rest_positions_host};
    muda::DeviceBuffer<Float> thicknesses{std::vector<Float>(VertexCount, 0.01)};
    muda::DeviceBuffer<Float> d_hats{std::vector<Float>(VertexCount, 10.0)};
    muda::DeviceBuffer<IndexT> contact_element_ids{
        std::vector<IndexT>(VertexCount + 1, 0)};

    muda::DeviceBuffer2D<ContactCoeff> contact_tabular{muda::Extent2D{1, 1}};
    contact_tabular.copy_from(std::vector<ContactCoeff>{ContactCoeff{3.0, 0.45}});

    muda::DeviceBuffer<Vector3> half_plane_positions{
        std::vector<Vector3>{Vector3{0.0, 0.0, -0.20}}};
    muda::DeviceBuffer<Vector3> half_plane_normals{
        std::vector<Vector3>{Vector3{0.0, 0.0, 1.0}}};

    SocuContactDirectSceneView<Store> scene;
    scene.contact_tabular = contact_tabular.cviewer();
    scene.positions = positions.view().as_const();
    scene.prev_positions = prev_positions.view().as_const();
    scene.rest_positions = rest_positions.view().as_const();
    scene.thicknesses = thicknesses.view().as_const();
    scene.contact_element_ids = contact_element_ids.view().as_const();
    scene.d_hats = d_hats.view().as_const();
    scene.dt = 0.1;
    scene.eps_velocity = 0.02;
    scene.half_plane_positions = half_plane_positions.view().as_const();
    scene.half_plane_normals = half_plane_normals.view().as_const();
    scene.half_plane_vertex_offset = static_cast<IndexT>(VertexCount);

    muda::DeviceBuffer<SocuContactDirectSourceEntry> direct_source_buffer{
        direct_entries};
    SocuContactDirectSourceTable direct_source_table;
    direct_source_table.source_entries = direct_source_buffer.view().as_const();

    const auto plan_view = socu_contact_assembly_plan_view(plan);
    SocuContactDirectEvaluator<Store> direct_evaluator{
        plan_view,
        scene,
        direct_source_table};
    SocuContactDirectStencilOrderReferenceEvaluator<Store> reference_evaluator{
        scene,
        direct_source_table};

    muda::DeviceBuffer<SocuDeterministicContactHessian<Store>> direct_hessians;
    muda::DeviceBuffer<SocuDeterministicContactHessian<Store>> reference_hessians;
    muda::DeviceBuffer<IndexT> direct_unsupported_flags;
    muda::DeviceBuffer<IndexT> reference_unsupported_flags;
    direct_hessians.resize(SourceCount);
    reference_hessians.resize(SourceCount);
    direct_unsupported_flags.resize(SourceCount);
    reference_unsupported_flags.resize(SourceCount);

    launch_socu_contact_direct_evaluate_programs<Store>(
        plan_view,
        direct_evaluator,
        direct_hessians.view(),
        direct_unsupported_flags.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    launch_socu_contact_direct_evaluate_programs<Store>(
        plan_view,
        reference_evaluator,
        reference_hessians.view(),
        reference_unsupported_flags.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<IndexT> direct_flags;
    std::vector<IndexT> reference_flags;
    direct_unsupported_flags.copy_to(direct_flags);
    reference_unsupported_flags.copy_to(reference_flags);
    REQUIRE(direct_flags.size() == SourceCount);
    REQUIRE(reference_flags.size() == SourceCount);
    for(SizeT i = 0; i < SourceCount; ++i)
    {
        CAPTURE(source_names[i]);
        CHECK(direct_flags[i] == 0);
        CHECK(reference_flags[i] == 0);
    }

    std::vector<SocuDeterministicContactHessian<Store>> reference_host;
    reference_hessians.copy_to(reference_host);
    REQUIRE(reference_host.size() == SourceCount);
    for(SizeT i = 0; i < SourceCount; ++i)
    {
        CAPTURE(source_names[i]);
        const SizeT meaningful_stencil =
            families[i] == SocuContactFamily::PH ? SizeT{1}
                                                 : static_cast<SizeT>(
                                                       stencil_sizes[i]);
        CHECK(hessian_abs_sum(reference_host[i], meaningful_stencil * 3)
              > 0.0);
    }

    std::array<muda::DeviceTripletMatrix<Store, 3>, SourceCount> triplet_matrices;
    populate_reference_triplets(
        triplet_matrices[NormalPT],
        VertexCount,
        pt_host[0],
        4,
        reference_host[NormalPT]);
    populate_reference_triplets(
        triplet_matrices[NormalEE],
        VertexCount,
        ee_host[0],
        4,
        reference_host[NormalEE]);
    populate_reference_triplets(
        triplet_matrices[NormalPE],
        VertexCount,
        pe_host[0],
        3,
        reference_host[NormalPE]);
    populate_reference_triplets(
        triplet_matrices[NormalPP],
        VertexCount,
        pp_host[0],
        2,
        reference_host[NormalPP]);
    populate_reference_triplets(
        triplet_matrices[NormalPH],
        VertexCount,
        ph_host[0],
        1,
        reference_host[NormalPH]);
    populate_reference_triplets(
        triplet_matrices[FrictionPT],
        VertexCount,
        pt_host[0],
        4,
        reference_host[FrictionPT]);
    populate_reference_triplets(
        triplet_matrices[FrictionEE],
        VertexCount,
        ee_host[0],
        4,
        reference_host[FrictionEE]);
    populate_reference_triplets(
        triplet_matrices[FrictionPE],
        VertexCount,
        pe_host[0],
        3,
        reference_host[FrictionPE]);
    populate_reference_triplets(
        triplet_matrices[FrictionPP],
        VertexCount,
        pp_host[0],
        2,
        reference_host[FrictionPP]);
    populate_reference_triplets(
        triplet_matrices[FrictionPH],
        VertexCount,
        ph_host[0],
        1,
        reference_host[FrictionPH]);

    std::vector<SocuContactEvaluatorSourceEntry<Store>> triplet_entries(SourceCount);
    for(SizeT i = 0; i < SourceCount; ++i)
    {
        triplet_entries[i].source_id = static_cast<SocuContactSourceId>(i);
        triplet_entries[i].model = models[i];
        triplet_entries[i].family = families[i];
        triplet_entries[i].hessians = triplet_matrices[i].view();
    }
    muda::DeviceBuffer<SocuContactEvaluatorSourceEntry<Store>>
        triplet_source_buffer{triplet_entries};
    SocuContactEvaluatorSourceTable<Store> triplet_source_table;
    triplet_source_table.source_entries = triplet_source_buffer.view().as_const();

    muda::DeviceBuffer<SocuContactDirectCompareProgramStats> compare_stats;
    compare_stats.resize(SourceCount);
    launch_socu_contact_compare_direct_triplet_programs<Store>(
        plan_view,
        direct_hessians.view().as_const(),
        SocuContactTripletEvaluator<Store>{plan_view, triplet_source_table},
        compare_stats.view(),
        Float{1e-6});
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<SocuContactDirectCompareProgramStats> stats_host;
    compare_stats.copy_to(stats_host);
    REQUIRE(stats_host.size() == SourceCount);
    for(SizeT i = 0; i < SourceCount; ++i)
    {
        CAPTURE(source_names[i]);
        CHECK(stats_host[i].compared_entry_count
              == static_cast<IndexT>(stencil_sizes[i] * 3 * stencil_sizes[i] * 3));
        CHECK(stats_host[i].mismatch_count == 0);
        CHECK(static_cast<double>(stats_host[i].max_abs_error)
              == Catch::Approx(0.0).margin(1e-6));
    }
}

TEST_CASE("cuda_mixed_socu_contact_executor_source_isolation",
          "[cuda_mixed_socu][contract][socu_approx][m5]")
{
    const auto root = std::filesystem::current_path();
    const auto header =
        read_text_file(root / "src/backends/cuda_mixed_socu/linear_system/"
                              "socu_contact_executor.h");
    const auto tu =
        read_text_file(root / "src/backends/cuda_mixed_socu/linear_system/"
                              "socu_contact_executor.cu");

    for(const auto* text : {&header, &tu})
    {
        CHECK(text->find("structured_contact_assembly_sink.h")
              == std::string::npos);
        CHECK(text->find("socu_contact_program_debug_compare.h")
              == std::string::npos);
        CHECK(text->find("old_to_chain") == std::string::npos);
        CHECK(text->find("classify_dof_pair") == std::string::npos);
        CHECK(text->find("SocuNativeContactStencilTarget") == std::string::npos);
    }

    CHECK(header.find("socu_contact_program_writer.h") != std::string::npos);
    CHECK(header.find("socu_contact_execute_buckets_kernel")
          != std::string::npos);

    const char* build_dir_env = std::getenv("SOCU_NATIVE_CONTACT_BUILD_DIR");
    const auto compile_commands_path =
        build_dir_env != nullptr
            ? std::filesystem::path{build_dir_env} / "compile_commands.json"
            : root / "build/compile_commands.json";
    REQUIRE(std::filesystem::exists(compile_commands_path));
    const auto compile_commands = read_text_file(compile_commands_path);
    CHECK(compile_commands.find("socu_contact_executor.cu")
          != std::string::npos);
}
