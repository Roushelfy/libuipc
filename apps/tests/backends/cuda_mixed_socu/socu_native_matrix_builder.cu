#include <app/app.h>
#include <linear_system/socu_native_matrix_builder.h>
#include <mixed_precision/policy.h>

#include <cuda_runtime.h>

#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

#ifndef UIPC_WITH_SOCU_NATIVE
#define UIPC_WITH_SOCU_NATIVE 0
#endif

#if UIPC_WITH_SOCU_NATIVE
#include <socu_native/solver.h>
#endif

namespace
{
using namespace uipc::backend::cuda_mixed;

struct SolverPlanDeleter
{
#if UIPC_WITH_SOCU_NATIVE
    void operator()(socu_native::SolverPlan* plan) const
    {
        socu_native::destroy_solver_plan(plan);
    }
#endif
};

struct StreamGuard
{
    cudaStream_t stream = nullptr;

    StreamGuard()
    {
        REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
                == cudaSuccess);
    }

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;

    ~StreamGuard()
    {
        if(stream != nullptr)
            cudaStreamDestroy(stream);
    }
};

template <typename Scalar>
__global__ void write_storage_fixture(SocuNativeMatrixView<Scalar> view)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    const Scalar h[9] = {Scalar{1},
                         Scalar{2},
                         Scalar{3},
                         Scalar{4},
                         Scalar{5},
                         Scalar{6},
                         Scalar{7},
                         Scalar{8},
                         Scalar{9}};

    view.add_diag_scalar(0, 1, 2, Scalar{3.5});
    view.add_diag_block3_row_major(1, 0, 1, h);
    view.add_first_offdiag_scalar(0, 2, 1, Scalar{7});
    view.add_first_offdiag_block3_row_major(1, 1, 0, h);
    view.add_offdiag_scalar(6, 0, 0, Scalar{11});
    view.add_offdiag_block3_row_major(7, 0, 1, h);
    view.add_rhs_scalar(2, 3, 0, Scalar{4.25});
    view.add_rhs_scalar(2, 3, 0, Scalar{0.75});
}

template <typename Scalar>
__global__ void write_out_of_bounds_fixture(SocuNativeMatrixView<Scalar> view)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    view.add_diag_scalar(0, view.block_size, 0, Scalar{1});
    view.add_diag_scalar(view.block_count, 0, 0, Scalar{2});
    view.add_first_offdiag_scalar(0, view.block_size, 0, Scalar{3});
    view.add_first_offdiag_scalar(view.first_offdiag_block_count, 0, 0, Scalar{4});
    view.add_offdiag_scalar(0, view.block_size, 0, Scalar{5});
    view.add_offdiag_scalar(view.offdiag_block_count, 0, 0, Scalar{6});
    view.add_rhs_scalar(0, view.block_size, 0, Scalar{7});
    view.add_rhs_scalar(0, 0, view.nrhs, Scalar{8});
}

template <typename Scalar>
__global__ void fill_spd_diagonal_fixture(SocuNativeMatrixView<Scalar> view)
{
    const uipc::SizeT block = static_cast<uipc::SizeT>(blockIdx.x);
    const uipc::SizeT lane  = static_cast<uipc::SizeT>(threadIdx.x);
    if(block >= view.block_count || lane >= view.block_size)
        return;

    view.add_diag_scalar(block, lane, lane, Scalar{2});
    view.add_rhs_scalar(block, lane, 0, Scalar{1});
}

template <typename Scalar>
void require_all_zero(const std::vector<Scalar>& values)
{
    for(const Scalar value : values)
    {
        REQUIRE(static_cast<double>(value) == Catch::Approx(0.0));
    }
}

template <typename Scalar>
void require_all_close(const std::vector<Scalar>& values,
                       Scalar                     expected,
                       double                     tolerance)
{
    for(const Scalar value : values)
    {
        REQUIRE(std::isfinite(static_cast<double>(value)));
        REQUIRE(static_cast<double>(value)
                == Catch::Approx(static_cast<double>(expected)).margin(tolerance));
    }
}
}  // namespace

TEST_CASE("cuda_mixed_socu_native_matrix_builder_layout",
          "[cuda_mixed_socu][contract][socu_native_builder]")
{
    const auto layout = make_socu_native_storage_layout(7, 12, 2);

    REQUIRE(layout.block_count == 7);
    REQUIRE(layout.block_size == 12);
    REQUIRE(layout.nrhs == 2);
    REQUIRE(layout.diag_block_count == 7);
    REQUIRE(layout.first_offdiag_block_count == 6);
    REQUIRE(layout.offdiag_block_count == 8);
    REQUIRE(layout.rhs_block_count == 7);
    REQUIRE(layout.diag_element_count == 7 * 12 * 12);
    REQUIRE(layout.offdiag_element_count == 8 * 12 * 12);
    REQUIRE(layout.rhs_element_count == 7 * 12 * 2);
    REQUIRE(layout.recursive_iterations == 3);
    REQUIRE(layout.offdiag_levels.size() == 3);
    CHECK(layout.offdiag_levels[0].stride == 1);
    CHECK(layout.offdiag_levels[0].block_offset == 0);
    CHECK(layout.offdiag_levels[0].block_count == 6);
    CHECK(layout.offdiag_levels[1].stride == 2);
    CHECK(layout.offdiag_levels[1].block_offset == 6);
    CHECK(layout.offdiag_levels[1].block_count == 2);
    CHECK(layout.offdiag_levels[2].stride == 4);
    CHECK(layout.offdiag_levels[2].block_offset == 8);
    CHECK(layout.offdiag_levels[2].block_count == 0);
    CHECK(socu_native_offdiag_storage_block(layout, 1, 0) == 6);
    CHECK(socu_native_offdiag_storage_block(layout, 1, 1) == 7);

    const auto single = make_socu_native_storage_layout(1, 4, 1);
    REQUIRE(single.offdiag_block_count == 0);
    REQUIRE(single.first_offdiag_block_count == 0);
    REQUIRE(single.offdiag_element_count == 0);
}

#if UIPC_WITH_SOCU_NATIVE
TEST_CASE("cuda_mixed_socu_native_matrix_builder_layout_matches_socu_native",
          "[cuda_mixed_socu][contract][socu_native_builder][socu_native]")
{
    for(const int horizon : {1, 2, 7, 8, 16})
    {
        for(const int block_size : {12, 32, 64})
        {
            for(const int nrhs : {1, 2})
            {
                const auto local = make_socu_native_storage_layout(
                    static_cast<uipc::SizeT>(horizon),
                    static_cast<uipc::SizeT>(block_size),
                    static_cast<uipc::SizeT>(nrhs));
                const auto native = socu_native::describe_problem_layout(
                    socu_native::ProblemShape{horizon, block_size, nrhs});

                CAPTURE(horizon, block_size, nrhs);
                CHECK(local.diag_block_count == native.diag_block_count);
                CHECK(local.offdiag_block_count == native.off_diag_block_count);
                CHECK(local.rhs_block_count == native.rhs_block_count);
                CHECK(local.diag_element_count == native.diag_element_count);
                CHECK(local.offdiag_element_count == native.off_diag_element_count);
                CHECK(local.rhs_element_count == native.rhs_element_count);
                REQUIRE(local.offdiag_levels.size() == native.off_diag_levels.size());
                for(std::size_t i = 0; i < local.offdiag_levels.size(); ++i)
                {
                    CHECK(local.offdiag_levels[i].stride
                          == native.off_diag_levels[i].stride);
                    CHECK(local.offdiag_levels[i].block_offset
                          == native.off_diag_levels[i].block_offset);
                    CHECK(local.offdiag_levels[i].block_count
                          == native.off_diag_levels[i].block_count);
                }
            }
        }
    }
}
#endif

TEST_CASE("cuda_mixed_socu_native_matrix_builder_device_writes",
          "[cuda_mixed_socu][contract][socu_native_builder]")
{
    using Scalar = ActivePolicy::SolveScalar;

    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        SKIP("no CUDA device is available for SOCU native storage tests");
    }

    StreamGuard stream;
    SocuNativeMatrixBuilder<Scalar> builder;
    builder.reserve(7, 4, 1);
    builder.clear(stream.stream);

    std::vector<SocuNativeBlockMeta> metadata(7);
    metadata[0] = SocuNativeBlockMeta{0, 4, 4, 0};
    metadata[1] = SocuNativeBlockMeta{4, 3, 3, 1, 42};
    metadata[2] = SocuNativeBlockMeta{8, 2, 2, 2};
    builder.set_block_metadata(metadata, stream.stream);

    write_storage_fixture<<<1, 1, 0, stream.stream>>>(builder.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    const auto snapshot = builder.snapshot(stream.stream);
    const auto& layout = snapshot.layout;

    CHECK(snapshot.D[layout.block_size + 2]
          == Catch::Approx(static_cast<double>(Scalar{3.5})));

    for(uipc::SizeT row = 0; row < 3; ++row)
    {
        for(uipc::SizeT col = 0; col < 3; ++col)
        {
            const uipc::SizeT index =
                (1 * layout.block_size + row) * layout.block_size + (1 + col);
            CHECK(snapshot.D[index]
                  == Catch::Approx(static_cast<double>(1 + row * 3 + col)));
        }
    }

    CHECK(snapshot.E[(0 * layout.block_size + 2) * layout.block_size + 1]
          == Catch::Approx(7.0));
    for(uipc::SizeT row = 0; row < 3; ++row)
    {
        for(uipc::SizeT col = 0; col < 3; ++col)
        {
            const uipc::SizeT index =
                (1 * layout.block_size + (1 + row)) * layout.block_size + col;
            CHECK(snapshot.E[index]
                  == Catch::Approx(static_cast<double>(1 + row * 3 + col)));
        }
    }

    CHECK(snapshot.E[(6 * layout.block_size + 0) * layout.block_size + 0]
          == Catch::Approx(11.0));
    for(uipc::SizeT row = 0; row < 3; ++row)
    {
        for(uipc::SizeT col = 0; col < 3; ++col)
        {
            const uipc::SizeT index =
                (7 * layout.block_size + row) * layout.block_size + (1 + col);
            CHECK(snapshot.E[index]
                  == Catch::Approx(static_cast<double>(1 + row * 3 + col)));
        }
    }

    CHECK(snapshot.rhs[(2 * layout.block_size + 3) * layout.nrhs]
          == Catch::Approx(5.0));
    REQUIRE(snapshot.blocks.size() == metadata.size());
    CHECK(snapshot.blocks[1].old_dof_begin == 4);
    CHECK(snapshot.blocks[1].active_lane_count == 3);
    CHECK(snapshot.blocks[1].padding_lane_begin == 3);
    CHECK(snapshot.blocks[1].padding_lane_count == 1);
    CHECK(snapshot.blocks[1].ordering_epoch == 42);
}

TEST_CASE("cuda_mixed_socu_native_matrix_builder_bounds_and_clear_contract",
          "[cuda_mixed_socu][contract][socu_native_builder]")
{
    using Scalar = ActivePolicy::SolveScalar;

    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        SKIP("no CUDA device is available for SOCU native storage tests");
    }

    StreamGuard stream;
    SocuNativeMatrixBuilder<Scalar> builder;
    builder.reserve(2, 4, 1);
    builder.clear(stream.stream);

    std::vector<SocuNativeBlockMeta> metadata(2);
    metadata[0] = SocuNativeBlockMeta{0, 4, 4, 0, 7};
    metadata[1] = SocuNativeBlockMeta{4, 2, 2, 2, 7};
    builder.set_block_metadata(metadata, stream.stream);

    write_out_of_bounds_fixture<<<1, 1, 0, stream.stream>>>(builder.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    auto snapshot = builder.snapshot(stream.stream);
    require_all_zero(snapshot.D);
    require_all_zero(snapshot.E);
    require_all_zero(snapshot.rhs);
    REQUIRE(snapshot.blocks.size() == metadata.size());
    CHECK(snapshot.blocks[0].old_dof_begin == 0);
    CHECK(snapshot.blocks[1].active_lane_count == 2);

    builder.clear(stream.stream);
    snapshot = builder.snapshot(stream.stream);
    require_all_zero(snapshot.D);
    require_all_zero(snapshot.E);
    require_all_zero(snapshot.rhs);
    REQUIRE(snapshot.blocks.size() == metadata.size());
    CHECK(snapshot.blocks[0].ordering_epoch == 7);
    CHECK(snapshot.blocks[1].padding_lane_count == 2);
}

TEST_CASE("cuda_mixed_socu_native_matrix_builder_solver_contract",
          "[cuda_mixed_socu][contract][socu_native_builder][socu_native]")
{
#if !UIPC_WITH_SOCU_NATIVE
    SKIP("socu_native is not enabled in this build");
#else
    using Scalar = ActivePolicy::SolveScalar;

    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        SKIP("no CUDA device is available for SOCU native storage tests");
    }

    StreamGuard stream;
    SocuNativeMatrixBuilder<Scalar> builder;
    builder.reserve(7, 12, 1);
    builder.clear(stream.stream);

    fill_spd_diagonal_fixture<<<7, 12, 0, stream.stream>>>(builder.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    socu_native::SolverPlanOptions options;
    options.backend    = socu_native::SolverBackend::NativeProof;
    options.graph_mode = socu_native::GraphMode::Off;

    const socu_native::ProblemShape shape{static_cast<int>(builder.block_count()),
                                          static_cast<int>(builder.block_size()),
                                          static_cast<int>(builder.nrhs())};
    const auto capability =
        socu_native::query_solver_capability<Scalar>(
            shape,
            socu_native::SolverOperation::FactorAndSolve,
            options);
    INFO("capability.reason=" << capability.reason);
    REQUIRE(capability.supported);

    std::unique_ptr<socu_native::SolverPlan, SolverPlanDeleter> plan(
        socu_native::create_solver_plan<Scalar>(shape, options));
    socu_native::factor_and_solve_inplace_async(
        plan.get(),
        builder.diag_data(),
        builder.offdiag_data(),
        builder.rhs_data(),
        socu_native::LaunchOptions{stream.stream});
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    const auto snapshot = builder.snapshot(stream.stream);
    require_all_close(snapshot.rhs, Scalar{0.5}, 1e-10);
#endif
}
