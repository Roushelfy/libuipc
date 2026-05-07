#include <app/app.h>
#include <linear_system/socu_native_matrix_builder.h>
#include <linear_system/socu_approx_kernels.h>
#include <mixed_precision/policy.h>
#include <utils/assembly_sink.h>

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

template <typename Scalar, int Rows, int Cols>
struct DeviceFixedMatrix
{
    Scalar values[Rows * Cols];

    MUDA_DEVICE Scalar operator()(uipc::IndexT row, uipc::IndexT col) const noexcept
    {
        return values[row * Cols + col];
    }
};

template <typename StoreT, typename SolveT>
__global__ void write_native_chain_base_sink_fixture(
    StructuredDeviceAssemblySink<StoreT, SolveT> sink)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    DeviceFixedMatrix<StoreT, 4, 4> diag_block{};
    DeviceFixedMatrix<StoreT, 4, 4> first_block{};
    for(uipc::IndexT row = 0; row < 4; ++row)
    {
        for(uipc::IndexT col = 0; col < 4; ++col)
        {
            diag_block.values[row * 4 + col] =
                static_cast<StoreT>(1 + row * 10 + col);
            first_block.values[row * 4 + col] =
                static_cast<StoreT>(101 + row * 10 + col);
        }
    }

    sink.template add_dense_block_fixed<4, 4>(0, diag_block);
    sink.template add_dense_block_between_fixed<4, 4>(0, 4, first_block);
    sink.add_hessian_scalar_status(5, 2, StoreT{7.25});

    const auto offband = sink.add_hessian_scalar_status(0, 8, StoreT{999});
    const auto skipped = sink.add_hessian_scalar_status(12, 12, StoreT{999});
    if(offband != StructuredSinkWriteClass::OffBand
       || skipped != StructuredSinkWriteClass::Skipped)
    {
        sink.add_hessian_scalar_status(0, 0, StoreT{123456});
    }
}

template <typename StoreT, typename SolveT>
__global__ void write_native_abd_dense_block_fixture(
    StructuredDeviceAssemblySink<StoreT, SolveT> fast_sink,
    StructuredDeviceAssemblySink<StoreT, SolveT> legacy_sink,
    muda::BufferView<uipc::IndexT>               status)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    DeviceFixedMatrix<StoreT, 12, 12> H{};
    for(uipc::IndexT row = 0; row < 12; ++row)
    {
        for(uipc::IndexT col = 0; col < 12; ++col)
        {
            H.values[row * 12 + col] =
                static_cast<StoreT>(1 + row * 17 + col);
        }
    }

    const bool used_fast_path =
        fast_sink.template try_add_native_dense_block_upper_subblocks_fixed<3, 4>(
            0,
            H);
    legacy_sink.template add_dense_block_upper_subblocks_fixed<3, 4>(0, H);
    *status.data(0) = used_fast_path ? 1 : 0;
}

template <typename StoreT, typename SolveT>
__global__ void write_native_diag_rhs_provider_fixture(
    SocuNativeMatrixView<SolveT>              native,
    StructuredDeviceMatrixSink<StoreT, SolveT> structured,
    muda::CBufferView<SocuNativeDofDescriptor> dofs,
    muda::CBufferView<SocuNativeVertexDescriptor> vertices)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    const SolveT fem_H[9] = {SolveT{1},
                             SolveT{2},
                             SolveT{3},
                             SolveT{4},
                             SolveT{5},
                             SolveT{6},
                             SolveT{7},
                             SolveT{8},
                             SolveT{9}};
    native.add_vertex_diag_block_row_major(vertices[0], fem_H);
    for(uipc::IndexT row = 0; row < 3; ++row)
    {
        for(uipc::IndexT col = 0; col < 3; ++col)
        {
            structured.add_hessian_scalar(row,
                                          col,
                                          static_cast<StoreT>(
                                              fem_H[row * 3 + col]));
        }
    }

    const SolveT fem_rhs[3] = {SolveT{10}, SolveT{20}, SolveT{30}};
    native.add_vertex_rhs_vector(vertices[0], 0, fem_rhs);

    const SolveT fixed_H[9] = {SolveT{100},
                               SolveT{101},
                               SolveT{102},
                               SolveT{103},
                               SolveT{104},
                               SolveT{105},
                               SolveT{106},
                               SolveT{107},
                               SolveT{108}};
    const SolveT fixed_rhs[3] = {SolveT{1000}, SolveT{1001}, SolveT{1002}};
    native.add_vertex_diag_block_row_major(vertices[1], fixed_H);
    native.add_vertex_rhs_vector(vertices[1], 0, fixed_rhs);

    native.add_diag_scalar(dofs[7], SolveT{4.5});
    native.add_rhs_scalar(dofs[7], 1, SolveT{-2.5});
    structured.add_hessian_scalar(7, 7, StoreT{4.5});

    native.add_diag_scalar(dofs[6], SolveT{999});
    native.add_rhs_scalar(dofs[6], 0, SolveT{999});
}

template <typename Solve>
SocuNativeMatrixView<Solve> make_test_native_view(
    muda::BufferView<Solve> diag,
    muda::BufferView<Solve> offdiag,
    uipc::SizeT             horizon,
    uipc::SizeT             block_size,
    uipc::SizeT             nrhs)
{
    const auto layout = make_socu_native_storage_layout(horizon, block_size, nrhs);
    return SocuNativeMatrixView<Solve>{diag,
                                       offdiag,
                                       {},
                                       {},
                                       horizon,
                                       block_size,
                                       nrhs,
                                       layout.first_offdiag_block_count,
                                       layout.offdiag_block_count};
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

TEST_CASE("cuda_mixed_socu_native_diag_rhs_provider_matches_structured_sink",
          "[cuda_mixed_socu][contract][socu_native_builder][socu_native_provider][m5]")
{
    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        SKIP("no CUDA device is available for SOCU native storage tests");
    }

    constexpr uipc::SizeT  Horizon   = 2;
    constexpr uipc::SizeT  BlockSize = 4;
    constexpr uipc::SizeT  Nrhs      = 2;
    constexpr uipc::IndexT Epoch     = 23;

    const std::vector<uipc::IndexT> old_to_chain{0, 1, 2, 4, 5, 6, -1, 7};
    const std::vector<uipc::IndexT> old_dof_to_atom(old_to_chain.size(), -1);
    auto dofs = build_socu_native_dof_descriptors(
        uipc::span<const uipc::IndexT>{old_to_chain.data(), old_to_chain.size()},
        uipc::span<const uipc::IndexT>{old_dof_to_atom.data(),
                                       old_dof_to_atom.size()},
        Horizon,
        BlockSize,
        Epoch);
    const auto dof_span =
        uipc::span<const SocuNativeDofDescriptor>{dofs.data(), dofs.size()};

    std::vector<SocuNativeVertexDescriptor> vertices;
    vertices.push_back(make_socu_native_vertex_descriptor(
        SocuNativeDescriptorKind::Fem, false, 0, 3, -1, -1, Epoch, dof_span));
    vertices.push_back(make_socu_native_vertex_descriptor(
        SocuNativeDescriptorKind::Fem, true, 3, 3, -1, -1, Epoch, dof_span));
    REQUIRE(vertices[0].writable());
    REQUIRE(!vertices[1].writable());
    REQUIRE(!dofs[6].active);
    REQUIRE(dofs[7].active);

    StreamGuard stream;
    SocuNativeMatrixBuilder<Solve> builder;
    builder.reserve(Horizon, BlockSize, Nrhs);
    builder.clear(stream.stream);

    const auto& layout = builder.layout();
    muda::DeviceBuffer<uipc::IndexT> old_to_chain_device{old_to_chain};
    muda::DeviceBuffer<SocuNativeDofDescriptor> dofs_device{dofs};
    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertices_device{vertices};
    muda::DeviceBuffer<Solve> structured_diag;
    muda::DeviceBuffer<Solve> structured_first_offdiag;
    structured_diag.resize(layout.diag_element_count);
    structured_first_offdiag.resize(layout.first_offdiag_block_count * BlockSize
                                    * BlockSize);
    REQUIRE(cudaMemsetAsync(structured_diag.data(),
                            0,
                            structured_diag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(structured_first_offdiag.data(),
                            0,
                            structured_first_offdiag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);

    StructuredDeviceMatrixSink<Store, Solve> structured{
        structured_diag.view(),
        structured_first_offdiag.view(),
        old_to_chain_device.view(),
        Horizon,
        BlockSize,
        {}};

    write_native_diag_rhs_provider_fixture<Store, Solve>
        <<<1, 1, 0, stream.stream>>>(builder.view(),
                                     structured,
                                     dofs_device.view(),
                                     vertices_device.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    const auto snapshot = builder.snapshot(stream.stream);
    std::vector<Solve> structured_diag_host;
    std::vector<Solve> structured_first_offdiag_host;
    structured_diag.copy_to(structured_diag_host);
    structured_first_offdiag.copy_to(structured_first_offdiag_host);

    REQUIRE(snapshot.D.size() == structured_diag_host.size());
    REQUIRE(snapshot.E.size() == structured_first_offdiag_host.size());
    for(std::size_t i = 0; i < snapshot.D.size(); ++i)
    {
        CHECK(static_cast<double>(snapshot.D[i])
              == Catch::Approx(static_cast<double>(structured_diag_host[i]))
                     .margin(1e-8));
    }
    for(std::size_t i = 0; i < snapshot.E.size(); ++i)
    {
        CHECK(static_cast<double>(snapshot.E[i])
              == Catch::Approx(static_cast<double>(
                                    structured_first_offdiag_host[i]))
                     .margin(1e-8));
    }

    std::vector<Solve> expected_rhs(snapshot.rhs.size(), Solve{0});
    const auto rhs_index = [&](uipc::SizeT block,
                               uipc::SizeT lane,
                               uipc::SizeT rhs_col)
    {
        return (block * layout.block_size + lane) * layout.nrhs + rhs_col;
    };
    expected_rhs[rhs_index(0, 0, 0)] = Solve{10};
    expected_rhs[rhs_index(0, 1, 0)] = Solve{20};
    expected_rhs[rhs_index(0, 2, 0)] = Solve{30};
    expected_rhs[rhs_index(1, 3, 1)] = Solve{-2.5};
    REQUIRE(snapshot.rhs.size() == expected_rhs.size());
    for(std::size_t i = 0; i < snapshot.rhs.size(); ++i)
    {
        CHECK(static_cast<double>(snapshot.rhs[i])
              == Catch::Approx(static_cast<double>(expected_rhs[i]))
                     .margin(1e-8));
    }
}

TEST_CASE("cuda_mixed_socu_native_chain_base_sink_matches_structured_sink",
          "[cuda_mixed_socu][contract][socu_native_builder][socu_native_provider][m6]")
{
    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        SKIP("no CUDA device is available for SOCU native storage tests");
    }

    constexpr uipc::SizeT  Horizon   = 3;
    constexpr uipc::SizeT  BlockSize = 4;
    constexpr uipc::SizeT  Nrhs      = 1;
    constexpr uipc::IndexT Epoch     = 31;

    const auto layout = make_socu_native_storage_layout(Horizon, BlockSize, Nrhs);
    const std::vector<uipc::IndexT> old_to_chain{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1};
    const std::vector<uipc::IndexT> old_dof_to_atom(old_to_chain.size(), -1);
    auto dofs = build_socu_native_dof_descriptors(
        uipc::span<const uipc::IndexT>{old_to_chain.data(), old_to_chain.size()},
        uipc::span<const uipc::IndexT>{old_dof_to_atom.data(),
                                       old_dof_to_atom.size()},
        Horizon,
        BlockSize,
        Epoch);

    StreamGuard stream;
    muda::DeviceBuffer<uipc::IndexT> old_to_chain_device{old_to_chain};
    muda::DeviceBuffer<SocuNativeDofDescriptor> dofs_device{dofs};
    muda::DeviceBuffer<Solve> primary_diag;
    muda::DeviceBuffer<Solve> primary_offdiag;
    muda::DeviceBuffer<Solve> compare_diag;
    muda::DeviceBuffer<Solve> compare_offdiag;
    primary_diag.resize(layout.diag_element_count);
    primary_offdiag.resize(layout.offdiag_element_count);
    compare_diag.resize(layout.diag_element_count);
    compare_offdiag.resize(layout.offdiag_element_count);
    REQUIRE(cudaMemsetAsync(primary_diag.data(),
                            0,
                            primary_diag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(primary_offdiag.data(),
                            0,
                            primary_offdiag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(compare_diag.data(),
                            0,
                            compare_diag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(compare_offdiag.data(),
                            0,
                            compare_offdiag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);

    StructuredDeviceAssemblySink<Store, Solve> sink{
        primary_diag.view(),
        primary_offdiag.view(),
        old_to_chain_device.view(),
        Horizon,
        BlockSize,
        {},
        {}};
    sink.matrix.use_native_matrix = true;
    sink.matrix.native_matrix =
        make_test_native_view(primary_diag.view(),
                              primary_offdiag.view(),
                              Horizon,
                              BlockSize,
                              Nrhs);
    sink.matrix.native_dof_descriptors = dofs_device.view();
    sink.matrix.compare_enabled = true;
    sink.matrix.compare_uses_native_matrix = false;
    sink.matrix.compare_diag = compare_diag.view();
    sink.matrix.compare_first_offdiag = compare_offdiag.view();

    write_native_chain_base_sink_fixture<Store, Solve>
        <<<1, 1, 0, stream.stream>>>(sink);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    std::vector<Solve> primary_diag_host;
    std::vector<Solve> primary_offdiag_host;
    std::vector<Solve> compare_diag_host;
    std::vector<Solve> compare_offdiag_host;
    primary_diag.copy_to(primary_diag_host);
    primary_offdiag.copy_to(primary_offdiag_host);
    compare_diag.copy_to(compare_diag_host);
    compare_offdiag.copy_to(compare_offdiag_host);

    REQUIRE(primary_diag_host.size() == compare_diag_host.size());
    REQUIRE(primary_offdiag_host.size() == compare_offdiag_host.size());
    for(std::size_t i = 0; i < primary_diag_host.size(); ++i)
    {
        CHECK(static_cast<double>(primary_diag_host[i])
              == Catch::Approx(static_cast<double>(compare_diag_host[i]))
                     .margin(1e-8));
    }
    for(std::size_t i = 0; i < primary_offdiag_host.size(); ++i)
    {
        CHECK(static_cast<double>(primary_offdiag_host[i])
              == Catch::Approx(static_cast<double>(compare_offdiag_host[i]))
                     .margin(1e-8));
    }
}

TEST_CASE("cuda_mixed_socu_native_abd_block_fast_path_matches_structured_sink",
          "[cuda_mixed_socu][contract][socu_native_builder][socu_native_provider][m6][m6b]")
{
    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        SKIP("no CUDA device is available for SOCU native storage tests");
    }

    constexpr uipc::SizeT  Horizon   = 1;
    constexpr uipc::SizeT  BlockSize = 12;
    constexpr uipc::SizeT  Nrhs      = 1;
    constexpr uipc::IndexT Epoch     = 37;

    const auto layout = make_socu_native_storage_layout(Horizon, BlockSize, Nrhs);
    std::vector<uipc::IndexT> old_to_chain(BlockSize);
    for(uipc::IndexT i = 0; i < static_cast<uipc::IndexT>(BlockSize); ++i)
        old_to_chain[static_cast<std::size_t>(i)] =
            static_cast<uipc::IndexT>(BlockSize - 1) - i;
    const std::vector<uipc::IndexT> old_dof_to_atom(old_to_chain.size(), -1);
    auto dofs = build_socu_native_dof_descriptors(
        uipc::span<const uipc::IndexT>{old_to_chain.data(), old_to_chain.size()},
        uipc::span<const uipc::IndexT>{old_dof_to_atom.data(),
                                       old_dof_to_atom.size()},
        Horizon,
        BlockSize,
        Epoch);

    StreamGuard stream;
    muda::DeviceBuffer<uipc::IndexT> old_to_chain_device{old_to_chain};
    muda::DeviceBuffer<SocuNativeDofDescriptor> dofs_device{dofs};
    muda::DeviceBuffer<Solve> native_diag;
    muda::DeviceBuffer<Solve> compare_diag;
    muda::DeviceBuffer<Solve> legacy_diag;
    muda::DeviceBuffer<uipc::IndexT> status;
    muda::DeviceBuffer<uipc::IndexT> counters;
    native_diag.resize(layout.diag_element_count);
    compare_diag.resize(layout.diag_element_count);
    legacy_diag.resize(layout.diag_element_count);
    status.resize(1);
    counters.resize(kStructuredAssemblyCounterCount);

    REQUIRE(cudaMemsetAsync(native_diag.data(),
                            0,
                            native_diag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(compare_diag.data(),
                            0,
                            compare_diag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(legacy_diag.data(),
                            0,
                            legacy_diag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(status.data(),
                            0,
                            status.size() * sizeof(uipc::IndexT),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(counters.data(),
                            0,
                            counters.size() * sizeof(uipc::IndexT),
                            stream.stream)
            == cudaSuccess);

    muda::DeviceBuffer<Solve> empty_offdiag;
    StructuredDeviceAssemblySink<Store, Solve> fast_sink{
        native_diag.view(),
        empty_offdiag.view(),
        old_to_chain_device.view(),
        Horizon,
        BlockSize,
        counters.view(),
        {}};
    fast_sink.matrix.use_native_matrix = true;
    fast_sink.matrix.native_matrix =
        make_test_native_view(native_diag.view(),
                              empty_offdiag.view(),
                              Horizon,
                              BlockSize,
                              Nrhs);
    fast_sink.matrix.native_dof_descriptors = dofs_device.view();
    fast_sink.matrix.compare_enabled = true;
    fast_sink.matrix.compare_uses_native_matrix = false;
    fast_sink.matrix.compare_diag = compare_diag.view();
    fast_sink.matrix.compare_first_offdiag = empty_offdiag.view();

    StructuredDeviceAssemblySink<Store, Solve> legacy_sink{
        legacy_diag.view(),
        empty_offdiag.view(),
        old_to_chain_device.view(),
        Horizon,
        BlockSize,
        {},
        {}};

    write_native_abd_dense_block_fixture<Store, Solve>
        <<<1, 1, 0, stream.stream>>>(fast_sink, legacy_sink, status.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    std::vector<uipc::IndexT> status_host;
    std::vector<Solve> native_diag_host;
    std::vector<Solve> compare_diag_host;
    std::vector<Solve> legacy_diag_host;
    std::vector<uipc::IndexT> counters_host;
    status.copy_to(status_host);
    native_diag.copy_to(native_diag_host);
    compare_diag.copy_to(compare_diag_host);
    legacy_diag.copy_to(legacy_diag_host);
    counters.copy_to(counters_host);

    REQUIRE(status_host.size() == 1);
    REQUIRE(status_host[0] == 1);
    REQUIRE(counters_host.size() == kStructuredAssemblyCounterCount);
    const auto counter = [&](StructuredAssemblyCounterSlot slot) -> uipc::IndexT
    {
        return counters_host[static_cast<std::size_t>(slot)];
    };
    CHECK(counter(StructuredAssemblyCounterSlot::NativeChainBaseSameBlockDenseHit)
          == 1);
    CHECK(counter(StructuredAssemblyCounterSlot::NativeChainBaseAdjacentDenseHit)
          == 0);
    CHECK(counter(StructuredAssemblyCounterSlot::NativeChainBaseDenseMiss) == 0);
    CHECK(counter(StructuredAssemblyCounterSlot::NativeChainBaseScalarFallback) == 0);
    REQUIRE(native_diag_host.size() == legacy_diag_host.size());
    REQUIRE(compare_diag_host.size() == legacy_diag_host.size());
    for(std::size_t i = 0; i < legacy_diag_host.size(); ++i)
    {
        CAPTURE(i);
        CHECK(static_cast<double>(native_diag_host[i])
              == Catch::Approx(static_cast<double>(legacy_diag_host[i]))
                     .margin(1e-8));
        CHECK(static_cast<double>(compare_diag_host[i])
              == Catch::Approx(static_cast<double>(legacy_diag_host[i]))
                     .margin(1e-8));
    }
}

TEST_CASE("cuda_mixed_socu_native_abd_adjacent_block_fast_path_matches_structured_sink",
          "[cuda_mixed_socu][contract][socu_native_builder][socu_native_provider][m6][m6b]")
{
    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        SKIP("no CUDA device is available for SOCU native storage tests");
    }

    constexpr uipc::SizeT  Horizon   = 2;
    constexpr uipc::SizeT  BlockSize = 8;
    constexpr uipc::SizeT  Nrhs      = 1;
    constexpr uipc::IndexT Epoch     = 38;

    const auto layout = make_socu_native_storage_layout(Horizon, BlockSize, Nrhs);
    const std::vector<uipc::IndexT> old_to_chain{
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12};
    const std::vector<uipc::IndexT> old_dof_to_atom(old_to_chain.size(), -1);
    auto dofs = build_socu_native_dof_descriptors(
        uipc::span<const uipc::IndexT>{old_to_chain.data(), old_to_chain.size()},
        uipc::span<const uipc::IndexT>{old_dof_to_atom.data(),
                                       old_dof_to_atom.size()},
        Horizon,
        BlockSize,
        Epoch);

    StreamGuard stream;
    muda::DeviceBuffer<uipc::IndexT> old_to_chain_device{old_to_chain};
    muda::DeviceBuffer<SocuNativeDofDescriptor> dofs_device{dofs};
    muda::DeviceBuffer<Solve> native_diag;
    muda::DeviceBuffer<Solve> native_offdiag;
    muda::DeviceBuffer<Solve> compare_diag;
    muda::DeviceBuffer<Solve> compare_offdiag;
    muda::DeviceBuffer<Solve> legacy_diag;
    muda::DeviceBuffer<Solve> legacy_offdiag;
    muda::DeviceBuffer<uipc::IndexT> status;
    muda::DeviceBuffer<uipc::IndexT> counters;
    native_diag.resize(layout.diag_element_count);
    native_offdiag.resize(layout.offdiag_element_count);
    compare_diag.resize(layout.diag_element_count);
    compare_offdiag.resize(layout.first_offdiag_block_count * BlockSize * BlockSize);
    legacy_diag.resize(layout.diag_element_count);
    legacy_offdiag.resize(layout.first_offdiag_block_count * BlockSize * BlockSize);
    status.resize(1);
    counters.resize(kStructuredAssemblyCounterCount);

    REQUIRE(cudaMemsetAsync(native_diag.data(),
                            0,
                            native_diag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(native_offdiag.data(),
                            0,
                            native_offdiag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(compare_diag.data(),
                            0,
                            compare_diag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(compare_offdiag.data(),
                            0,
                            compare_offdiag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(legacy_diag.data(),
                            0,
                            legacy_diag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(legacy_offdiag.data(),
                            0,
                            legacy_offdiag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(status.data(),
                            0,
                            status.size() * sizeof(uipc::IndexT),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(counters.data(),
                            0,
                            counters.size() * sizeof(uipc::IndexT),
                            stream.stream)
            == cudaSuccess);

    StructuredDeviceAssemblySink<Store, Solve> fast_sink{
        native_diag.view(),
        native_offdiag.view(),
        old_to_chain_device.view(),
        Horizon,
        BlockSize,
        counters.view(),
        {}};
    fast_sink.matrix.use_native_matrix = true;
    fast_sink.matrix.native_matrix =
        make_test_native_view(native_diag.view(),
                              native_offdiag.view(),
                              Horizon,
                              BlockSize,
                              Nrhs);
    fast_sink.matrix.native_dof_descriptors = dofs_device.view();
    fast_sink.matrix.compare_enabled = true;
    fast_sink.matrix.compare_uses_native_matrix = false;
    fast_sink.matrix.compare_diag = compare_diag.view();
    fast_sink.matrix.compare_first_offdiag = compare_offdiag.view();

    StructuredDeviceAssemblySink<Store, Solve> legacy_sink{
        legacy_diag.view(),
        legacy_offdiag.view(),
        old_to_chain_device.view(),
        Horizon,
        BlockSize,
        {},
        {}};

    write_native_abd_dense_block_fixture<Store, Solve>
        <<<1, 1, 0, stream.stream>>>(fast_sink, legacy_sink, status.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    std::vector<uipc::IndexT> status_host;
    std::vector<uipc::IndexT> counters_host;
    std::vector<Solve> native_diag_host;
    std::vector<Solve> native_offdiag_host;
    std::vector<Solve> compare_diag_host;
    std::vector<Solve> compare_offdiag_host;
    std::vector<Solve> legacy_diag_host;
    std::vector<Solve> legacy_offdiag_host;
    status.copy_to(status_host);
    counters.copy_to(counters_host);
    native_diag.copy_to(native_diag_host);
    native_offdiag.copy_to(native_offdiag_host);
    compare_diag.copy_to(compare_diag_host);
    compare_offdiag.copy_to(compare_offdiag_host);
    legacy_diag.copy_to(legacy_diag_host);
    legacy_offdiag.copy_to(legacy_offdiag_host);

    REQUIRE(status_host.size() == 1);
    REQUIRE(status_host[0] == 1);
    REQUIRE(counters_host.size() == kStructuredAssemblyCounterCount);
    const auto counter = [&](StructuredAssemblyCounterSlot slot) -> uipc::IndexT
    {
        return counters_host[static_cast<std::size_t>(slot)];
    };
    CHECK(counter(StructuredAssemblyCounterSlot::NativeChainBaseSameBlockDenseHit)
          == 0);
    CHECK(counter(StructuredAssemblyCounterSlot::NativeChainBaseAdjacentDenseHit)
          == 1);
    CHECK(counter(StructuredAssemblyCounterSlot::NativeChainBaseDenseMiss) == 0);
    CHECK(counter(StructuredAssemblyCounterSlot::NativeChainBaseScalarFallback) == 0);

    REQUIRE(native_diag_host.size() == legacy_diag_host.size());
    REQUIRE(compare_diag_host.size() == legacy_diag_host.size());
    for(std::size_t i = 0; i < legacy_diag_host.size(); ++i)
    {
        CAPTURE(i);
        CHECK(static_cast<double>(native_diag_host[i])
              == Catch::Approx(static_cast<double>(legacy_diag_host[i]))
                     .margin(1e-8));
        CHECK(static_cast<double>(compare_diag_host[i])
              == Catch::Approx(static_cast<double>(legacy_diag_host[i]))
                     .margin(1e-8));
    }

    REQUIRE(native_offdiag_host.size() == legacy_offdiag_host.size());
    REQUIRE(compare_offdiag_host.size() == legacy_offdiag_host.size());
    for(std::size_t i = 0; i < legacy_offdiag_host.size(); ++i)
    {
        CAPTURE(i);
        CHECK(static_cast<double>(native_offdiag_host[i])
              == Catch::Approx(static_cast<double>(legacy_offdiag_host[i]))
                     .margin(1e-8));
        CHECK(static_cast<double>(compare_offdiag_host[i])
              == Catch::Approx(static_cast<double>(legacy_offdiag_host[i]))
                     .margin(1e-8));
    }
}

TEST_CASE("cuda_mixed_socu_native_diag_rhs_workspace_matches_legacy_init",
          "[cuda_mixed_socu][contract][socu_native_builder][socu_native_provider][m5]")
{
#if !UIPC_WITH_SOCU_NATIVE
    SKIP("socu_native is not enabled in this build");
#else
    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;

    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        SKIP("no CUDA device is available for SOCU native storage tests");
    }

    constexpr uipc::SizeT  Horizon   = 3;
    constexpr uipc::SizeT  BlockSize = 4;
    constexpr uipc::SizeT  Nrhs      = 1;
    constexpr uipc::IndexT Epoch     = 24;
    const StructuredChainShape shape{Horizon, BlockSize, Nrhs, true};
    const auto layout = make_socu_native_storage_layout(Horizon, BlockSize, Nrhs);

    const std::vector<uipc::IndexT> old_to_chain{0, 2, 4, 5, 8, 11};
    std::vector<uipc::IndexT> chain_to_old(Horizon * BlockSize, -1);
    for(std::size_t old = 0; old < old_to_chain.size(); ++old)
        chain_to_old[static_cast<std::size_t>(old_to_chain[old])] =
            static_cast<uipc::IndexT>(old);
    const std::vector<uipc::IndexT> old_dof_to_atom(old_to_chain.size(), -1);
    auto dofs = build_socu_native_dof_descriptors(
        uipc::span<const uipc::IndexT>{old_to_chain.data(), old_to_chain.size()},
        uipc::span<const uipc::IndexT>{old_dof_to_atom.data(),
                                       old_dof_to_atom.size()},
        Horizon,
        BlockSize,
        Epoch);

    const std::vector<Store> b_host{Store{1.25},
                                    Store{-2.5},
                                    Store{3.75},
                                    Store{-4.0},
                                    Store{5.5},
                                    Store{-6.25}};
    StreamGuard stream;
    muda::DeviceDenseVector<Store> b;
    b.resize(b_host.size());
    b.buffer_view().copy_from(b_host.data());

    muda::DeviceBuffer<uipc::IndexT> chain_to_old_device{chain_to_old};
    muda::DeviceBuffer<SocuNativeDofDescriptor> dof_device{dofs};

    muda::DeviceBuffer<Solve> legacy_diag;
    muda::DeviceBuffer<Solve> legacy_offdiag;
    muda::DeviceBuffer<Solve> legacy_rhs;
    muda::DeviceBuffer<Solve> legacy_rhs_original;
    muda::DeviceBuffer<Solve> native_diag;
    muda::DeviceBuffer<Solve> native_offdiag;
    muda::DeviceBuffer<Solve> native_rhs;
    muda::DeviceBuffer<Solve> native_rhs_original;
    legacy_diag.resize(layout.diag_element_count);
    legacy_offdiag.resize(layout.offdiag_element_count);
    legacy_rhs.resize(layout.rhs_element_count);
    legacy_rhs_original.resize(layout.rhs_element_count);
    native_diag.resize(layout.diag_element_count);
    native_offdiag.resize(layout.offdiag_element_count);
    native_rhs.resize(layout.rhs_element_count);
    native_rhs_original.resize(layout.rhs_element_count);

    constexpr double DampingShift = 0.125;
    socu_approx::initialize_structured_workspace<Store, Solve>(
        stream.stream,
        shape,
        b.view(),
        legacy_diag.view(),
        legacy_offdiag.view(),
        legacy_rhs.view(),
        legacy_rhs_original.view(),
        chain_to_old_device.view(),
        DampingShift);
    socu_approx::initialize_socu_native_diag_rhs_workspace<Store, Solve>(
        stream.stream,
        shape,
        b.view(),
        native_diag.view(),
        native_offdiag.view(),
        native_rhs.view(),
        native_rhs_original.view(),
        chain_to_old_device.view(),
        dof_device.view(),
        DampingShift);

    muda::DeviceBuffer<double> diff_sums;
    muda::DeviceBuffer<uipc::IndexT> mismatch_count;
    diff_sums.resize(3);
    mismatch_count.resize(1);
    socu_approx::compare_socu_native_diag_rhs_workspace<Solve>(
        stream.stream,
        legacy_diag.view(),
        legacy_offdiag.view(),
        legacy_rhs.view(),
        native_diag.view(),
        native_offdiag.view(),
        native_rhs.view(),
        diff_sums.view(),
        mismatch_count.view(),
        1e-9,
        1e-10);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    std::vector<uipc::IndexT> mismatch_host;
    std::vector<double>       diff_sums_host;
    mismatch_count.copy_to(mismatch_host);
    diff_sums.copy_to(diff_sums_host);
    REQUIRE(mismatch_host.size() == 1);
    REQUIRE(diff_sums_host.size() == 3);
    CHECK(mismatch_host[0] == 0);
    CHECK(diff_sums_host[0] == Catch::Approx(0.0).margin(1e-12));
    CHECK(diff_sums_host[1] == Catch::Approx(0.0).margin(1e-12));
    CHECK(diff_sums_host[2] == Catch::Approx(0.0).margin(1e-12));

    std::vector<Solve> legacy_rhs_original_host;
    std::vector<Solve> native_rhs_original_host;
    legacy_rhs_original.copy_to(legacy_rhs_original_host);
    native_rhs_original.copy_to(native_rhs_original_host);
    REQUIRE(legacy_rhs_original_host.size() == native_rhs_original_host.size());
    for(std::size_t i = 0; i < legacy_rhs_original_host.size(); ++i)
    {
        CHECK(static_cast<double>(legacy_rhs_original_host[i])
              == Catch::Approx(static_cast<double>(
                                    native_rhs_original_host[i]))
                     .margin(1e-12));
    }
#endif
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
