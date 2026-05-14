#include <app/app.h>
#include <linear_system/socu_contact_executor.h>
#include <mixed_precision/policy.h>

#include <cuda_runtime.h>
#include <Eigen/Core>
#include <muda/buffer/device_buffer.h>
#include <muda/ext/linear_system/device_triplet_matrix.h>

#include <algorithm>
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
using uipc::IndexT;
using uipc::SizeT;
using uipc::Vector2i;
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
    source_inputs[0].source_id = 0;
    source_inputs[0].reporter_id = 100;
    source_inputs[0].model = SocuContactModelKind::SimplexNormal;
    source_inputs[0].family = SocuContactFamily::PP;
    source_inputs[0].stencil_size = 2;
    source_inputs[0].stencil2 = pp_contacts_a.view();
    source_inputs[1].source_id = 1;
    source_inputs[1].reporter_id = 101;
    source_inputs[1].model = SocuContactModelKind::SimplexNormal;
    source_inputs[1].family = SocuContactFamily::PP;
    source_inputs[1].stencil_size = 2;
    source_inputs[1].stencil2 = pp_contacts_b.view();

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
