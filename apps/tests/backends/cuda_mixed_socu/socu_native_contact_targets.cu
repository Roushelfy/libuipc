#include <app/app.h>
#include <linear_system/socu_native_contact_targets.h>
#include <linear_system/socu_native_contact_writer.h>

#include <cuda_runtime.h>
#include <muda/buffer/device_buffer.h>

#include <cmath>
#include <initializer_list>
#include <vector>

namespace
{
using namespace uipc::backend::cuda_mixed;
using uipc::IndexT;
using uipc::SizeT;
using uipc::Vector3;
using uipc::Vector2i;
using uipc::Vector3i;
using uipc::Vector4i;

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

struct ContactTargetFixture
{
    static constexpr uipc::SizeT Horizon   = 5;
    static constexpr uipc::SizeT BlockSize = 16;

    std::vector<uipc::IndexT> old_to_chain;
    std::vector<uipc::IndexT> old_dof_to_atom;
    std::vector<SocuNativeDofDescriptor> dofs;

    ContactTargetFixture()
        : old_to_chain(24, uipc::IndexT{-1})
        , old_dof_to_atom(24, uipc::IndexT{-1})
    {
        map_triplet(0, {5, 2, 9});       // FEM vertex 0, block 0, arbitrary lanes
        map_triplet(3, {18, 17, 19});    // FEM vertex 1, block 1
        map_triplet(6, {50, 49, 51});    // FEM vertex 2, block 3

        const uipc::IndexT abd_lanes[12] = {11, 10, 9, 8, 7, 6,
                                            5,  4,  3, 2, 1, 0};
        for(uipc::IndexT i = 0; i < 12; ++i)
            old_to_chain[static_cast<uipc::SizeT>(9 + i)] = 32 + abd_lanes[i];

        for(uipc::SizeT old = 0; old < old_dof_to_atom.size(); ++old)
            old_dof_to_atom[old] = static_cast<uipc::IndexT>(old / 3);

        dofs = build_socu_native_dof_descriptors(
            uipc::span<const uipc::IndexT>{old_to_chain.data(), old_to_chain.size()},
            uipc::span<const uipc::IndexT>{old_dof_to_atom.data(),
                                           old_dof_to_atom.size()},
            Horizon,
            BlockSize,
            17);
    }

    void map_triplet(uipc::IndexT old_begin,
                     std::initializer_list<uipc::IndexT> chain_dofs)
    {
        uipc::IndexT local = 0;
        for(const auto chain : chain_dofs)
            old_to_chain[static_cast<uipc::SizeT>(old_begin + local++)] = chain;
    }

    SocuNativeVertexDescriptor vertex(SocuNativeDescriptorKind kind,
                                      uipc::IndexT             old_dof,
                                      uipc::IndexT             dof_count,
                                      bool                     fixed = false,
                                      uipc::IndexT             body = -1,
                                      uipc::IndexT             j_index = -1) const
    {
        return make_socu_native_vertex_descriptor(
            kind,
            fixed,
            old_dof,
            dof_count,
            body,
            j_index,
            17,
            uipc::span<const SocuNativeDofDescriptor>{dofs.data(), dofs.size()});
    }

    uipc::span<const SocuNativeDofDescriptor> dof_span() const noexcept
    {
        return {dofs.data(), dofs.size()};
    }
};

struct StreamGuard
{
    cudaStream_t stream = nullptr;

    StreamGuard()
    {
        REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
                == cudaSuccess);
    }

    StreamGuard(const StreamGuard&)            = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;

    ~StreamGuard()
    {
        if(stream != nullptr)
            cudaStreamDestroy(stream);
    }
};

template <typename Solve>
SocuNativeMatrixView<Solve> make_contact_test_native_view(
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

template <typename StoreT, typename SolveT>
__global__ void write_native_contact_exact_diff_fixture(
    SocuNativeContactExactWriter<StoreT, SolveT>  native_writer,
    muda::CBufferView<SocuNativeContactStencilTarget> pt_targets,
    muda::BufferView<IndexT> status)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    Eigen::Matrix<StoreT, 12, 12> H;
    for(IndexT row = 0; row < 12; ++row)
    {
        for(IndexT col = 0; col < 12; ++col)
        {
            H(row, col) =
                static_cast<StoreT>(0.125 + 0.5 * row + 0.03125 * col);
        }
    }

    IndexT native_consumed = 0;
    for(IndexT target_index = 0; target_index < 10; ++target_index)
    {
        const auto target = pt_targets.data()[static_cast<SizeT>(target_index)];
        const auto H3 = H.template block<3, 3>(target.local_row_vertex * 3,
                                               target.local_col_vertex * 3);
        if(native_writer.write_half_block(target, H3))
            ++native_consumed;
    }
    status.data()[0] = native_consumed;
    status.data()[1] = 0;
}

template <typename StoreT, typename SolveT>
__global__ void write_native_contact_direct_lane_fixture(
    SocuNativeContactExactWriter<StoreT, SolveT> native_writer,
    SocuNativeContactStencilTarget               diag_target,
    SocuNativeContactStencilTarget               offdiag_target,
    muda::BufferView<IndexT>                     status)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    Eigen::Matrix<StoreT, 3, 3> H_diag;
    Eigen::Matrix<StoreT, 3, 3> H_offdiag;
    for(IndexT row = 0; row < 3; ++row)
    {
        for(IndexT col = 0; col < 3; ++col)
        {
            H_diag(row, col) =
                static_cast<StoreT>(1.0 + 10.0 * row + col);
            H_offdiag(row, col) =
                static_cast<StoreT>(100.0 + 10.0 * row + col);
        }
    }

    status.data()[0] = native_writer.write_half_block(diag_target, H_diag) ? 1 : 0;
    status.data()[1] =
        native_writer.write_half_block(offdiag_target, H_offdiag) ? 1 : 0;
}
}  // namespace

TEST_CASE("cuda_mixed_socu_native_contact_target_arbitrary_lane_exact",
          "[cuda_mixed_socu][contract][socu_native_contact][m8]")
{
    ContactTargetFixture fixture;
    const auto fem0 = fixture.vertex(SocuNativeDescriptorKind::Fem, 0, 3);

    CHECK(fem0.mapped());
    CHECK(!fem0.active);

    const auto classification = socu_native_contact_classify_half_block(
        fem0,
        fem0,
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize);
    CHECK(classification.cls == SocuNativeBandClass::Diag);
    CHECK(classification.diag_scalar_count == 9);
    CHECK(classification.offband_scalar_count == 0);
    CHECK(classification.skipped_scalar_count == 0);
    CHECK(classification.fully_writable_in_band());

    const std::vector<SocuNativeVertexDescriptor> stencil{fem0};
    const auto stencil_class = socu_native_contact_classify_stencil_half(
        uipc::span<const SocuNativeVertexDescriptor>{stencil.data(), stencil.size()},
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize);
    const auto policy = socu_native_contact_make_stencil_policy(
        stencil_class,
        StructuredContactOffbandPolicy::Drop);
    CHECK(policy.write_mode == SocuNativeContactWriteMode::ExactInBand);
    CHECK(!policy.whole_stencil_fallback());

    const auto target = socu_native_contact_make_half_block_target(
        3,
        0,
        0,
        fem0,
        fem0,
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        StructuredContactOffbandPolicy::Drop,
        policy.write_mode);
    CHECK(target.exact_in_band());
    CHECK(target.half_block_class == SocuNativeBandClass::Diag);
    CHECK(target.block_or_left_block == 0);
    CHECK(target.row_lane == 5);
    CHECK(target.col_lane == 5);
    CHECK(target.direct_lanes_valid);
    CHECK(target.row_direct_lanes[0] == 5);
    CHECK(target.row_direct_lanes[1] == 2);
    CHECK(target.row_direct_lanes[2] == 9);
    CHECK(target.col_direct_lanes[0] == 5);
    CHECK(target.col_direct_lanes[1] == 2);
    CHECK(target.col_direct_lanes[2] == 9);
    CHECK(target.row_old_dof == 0);
    CHECK(target.row_dof_count == 3);
}

TEST_CASE("cuda_mixed_socu_native_contact_target_adjacent_orientation",
          "[cuda_mixed_socu][contract][socu_native_contact][m8]")
{
    ContactTargetFixture fixture;
    const auto fem0 = fixture.vertex(SocuNativeDescriptorKind::Fem, 0, 3);
    const auto fem1 = fixture.vertex(SocuNativeDescriptorKind::Fem, 3, 3);

    const auto reverse = socu_native_contact_make_half_block_target(
        42,
        1,
        0,
        fem1,
        fem0,
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        StructuredContactOffbandPolicy::Drop);
    CHECK(reverse.exact_in_band());
    CHECK(reverse.half_block_class == SocuNativeBandClass::FirstOffdiag);
    CHECK(reverse.block_or_left_block == 0);
    CHECK(reverse.row_lane == 2);
    CHECK(reverse.col_lane == 5);
    CHECK(!reverse.transposed_first_offdiag);
    CHECK(reverse.contact_id == 42);
    CHECK(reverse.local_row_vertex == 1);
    CHECK(reverse.local_col_vertex == 0);

    const auto forward = socu_native_contact_make_half_block_target(
        42,
        0,
        1,
        fem0,
        fem1,
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        StructuredContactOffbandPolicy::Drop);
    CHECK(forward.exact_in_band());
    CHECK(forward.block_or_left_block == 0);
    CHECK(forward.row_lane == 2);
    CHECK(forward.col_lane == 5);
    CHECK(forward.transposed_first_offdiag);
    CHECK(forward.direct_lanes_valid);
    CHECK(forward.row_direct_lanes[0] == 5);
    CHECK(forward.row_direct_lanes[1] == 2);
    CHECK(forward.row_direct_lanes[2] == 9);
    CHECK(forward.col_direct_lanes[0] == 2);
    CHECK(forward.col_direct_lanes[1] == 1);
    CHECK(forward.col_direct_lanes[2] == 3);
}

TEST_CASE("cuda_mixed_socu_native_contact_direct_lane_writer_uses_targets",
          "[cuda_mixed_socu][contract][socu_native_contact][m8][v2]")
{
    if(!has_cuda_device())
        SKIP("CUDA device is required");

    using Store = double;
    using Solve = double;

    ContactTargetFixture fixture;
    const auto fem0 = fixture.vertex(SocuNativeDescriptorKind::Fem, 0, 3);
    const auto fem1 = fixture.vertex(SocuNativeDescriptorKind::Fem, 3, 3);

    const auto diag_target = socu_native_contact_make_half_block_target(
        0,
        0,
        0,
        fem0,
        fem0,
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        StructuredContactOffbandPolicy::Drop);
    const auto offdiag_target = socu_native_contact_make_half_block_target(
        0,
        0,
        1,
        fem0,
        fem1,
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        StructuredContactOffbandPolicy::Drop);

    REQUIRE(diag_target.direct_lanes_valid);
    REQUIRE(offdiag_target.direct_lanes_valid);
    REQUIRE(offdiag_target.transposed_first_offdiag);

    StreamGuard stream;
    const auto layout = make_socu_native_storage_layout(
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        1);

    muda::DeviceBuffer<Solve> diag;
    muda::DeviceBuffer<Solve> offdiag;
    muda::DeviceBuffer<IndexT> status;
    diag.resize(layout.diag_element_count);
    offdiag.resize(layout.offdiag_element_count);
    status.resize(2);

    REQUIRE(cudaMemsetAsync(diag.data(),
                            0,
                            diag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(offdiag.data(),
                            0,
                            offdiag.size() * sizeof(Solve),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(status.data(),
                            0,
                            status.size() * sizeof(IndexT),
                            stream.stream)
            == cudaSuccess);

    StructuredDeviceMatrixSink<Store, Solve> matrix;
    matrix.use_native_matrix = true;
    matrix.native_matrix =
        make_contact_test_native_view(diag.view(),
                                      offdiag.view(),
                                      ContactTargetFixture::Horizon,
                                      ContactTargetFixture::BlockSize,
                                      1);
    SocuNativeContactExactWriter<Store, Solve> writer{matrix, {}, {}};

    write_native_contact_direct_lane_fixture<Store, Solve>
        <<<1, 1, 0, stream.stream>>>(writer,
                                     diag_target,
                                     offdiag_target,
                                     status.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    std::vector<IndexT> status_host;
    std::vector<Solve> diag_host;
    std::vector<Solve> offdiag_host;
    status.copy_to(status_host);
    diag.copy_to(diag_host);
    offdiag.copy_to(offdiag_host);

    REQUIRE(status_host.size() == 2);
    CHECK(status_host[0] == 1);
    CHECK(status_host[1] == 1);

    const auto diag_index = [&](SizeT row, SizeT col)
    {
        return (SizeT{0} * ContactTargetFixture::BlockSize + row)
                   * ContactTargetFixture::BlockSize
               + col;
    };
    const auto offdiag_index = [&](SizeT row, SizeT col)
    {
        return (SizeT{0} * ContactTargetFixture::BlockSize + row)
                   * ContactTargetFixture::BlockSize
               + col;
    };

    const SizeT fem0_lanes[3] = {5, 2, 9};
    const SizeT fem1_lanes[3] = {2, 1, 3};
    for(IndexT row = 0; row < 3; ++row)
    {
        for(IndexT col = 0; col < 3; ++col)
        {
            const auto diag_expected = 1.0 + 10.0 * row + col;
            CHECK(diag_host[diag_index(fem0_lanes[row], fem0_lanes[col])]
                  == Catch::Approx(diag_expected).margin(1e-12));

            const auto offdiag_expected = 100.0 + 10.0 * row + col;
            CHECK(offdiag_host[offdiag_index(fem1_lanes[col], fem0_lanes[row])]
                  == Catch::Approx(offdiag_expected).margin(1e-12));
        }
    }
}

TEST_CASE("cuda_mixed_socu_native_contact_target_stencil_policy",
          "[cuda_mixed_socu][contract][socu_native_contact][m8]")
{
    ContactTargetFixture fixture;
    const auto fem0 = fixture.vertex(SocuNativeDescriptorKind::Fem, 0, 3);
    const auto fem1 = fixture.vertex(SocuNativeDescriptorKind::Fem, 3, 3);
    const auto fem2 = fixture.vertex(SocuNativeDescriptorKind::Fem, 6, 3);

    const std::vector<SocuNativeVertexDescriptor> stencil{fem0, fem1, fem2};
    const auto stencil_class = socu_native_contact_classify_stencil_half(
        uipc::span<const SocuNativeVertexDescriptor>{stencil.data(), stencil.size()},
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize);
    CHECK(stencil_class.diag_half_block_count == 3);
    CHECK(stencil_class.first_offdiag_half_block_count == 1);
    CHECK(stencil_class.offband_half_block_count == 2);
    CHECK(!stencil_class.fully_in_band());

    const auto lump_policy = socu_native_contact_make_stencil_policy(
        stencil_class,
        StructuredContactOffbandPolicy::DiagLump);
    CHECK(lump_policy.write_mode == SocuNativeContactWriteMode::DiagLumpFallback);
    CHECK(lump_policy.whole_stencil_fallback());

    const auto inband_under_lump = socu_native_contact_make_half_block_target(
        8,
        0,
        1,
        fem0,
        fem1,
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        StructuredContactOffbandPolicy::DiagLump,
        lump_policy.write_mode);
    CHECK(inband_under_lump.write_mode
          == SocuNativeContactWriteMode::DiagLumpFallback);
    CHECK(inband_under_lump.whole_stencil_fallback());

    const auto drop_policy = socu_native_contact_make_stencil_policy(
        stencil_class,
        StructuredContactOffbandPolicy::Drop);
    CHECK(drop_policy.write_mode == SocuNativeContactWriteMode::DropOffBand);
    CHECK(!drop_policy.whole_stencil_fallback());

    const auto dropped = socu_native_contact_make_half_block_target(
        8,
        0,
        2,
        fem0,
        fem2,
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        StructuredContactOffbandPolicy::Drop,
        drop_policy.write_mode);
    CHECK(dropped.write_mode == SocuNativeContactWriteMode::DropOffBand);
    CHECK(dropped.half_block_class == SocuNativeBandClass::OffBand);
}

TEST_CASE("cuda_mixed_socu_native_contact_target_skip_and_abd_metadata",
          "[cuda_mixed_socu][contract][socu_native_contact][m8]")
{
    ContactTargetFixture fixture;
    const auto fem0_fixed =
        fixture.vertex(SocuNativeDescriptorKind::Fem, 0, 3, true);
    const auto fem1 = fixture.vertex(SocuNativeDescriptorKind::Fem, 3, 3);
    const auto abd0 =
        fixture.vertex(SocuNativeDescriptorKind::Abd, 9, 12, false, 5, 2);

    const auto skipped = socu_native_contact_make_half_block_target(
        9,
        0,
        1,
        fem0_fixed,
        fem1,
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        StructuredContactOffbandPolicy::Drop);
    CHECK(skipped.write_mode == SocuNativeContactWriteMode::Skipped);
    CHECK(skipped.half_block_class == SocuNativeBandClass::Skipped);

    CHECK(abd0.mapped());
    CHECK(!abd0.active);
    const auto abd_fem = socu_native_contact_make_half_block_target(
        11,
        2,
        1,
        abd0,
        fem1,
        fixture.dof_span(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        StructuredContactOffbandPolicy::Drop);
    CHECK(abd_fem.exact_in_band());
    CHECK(abd_fem.half_block_class == SocuNativeBandClass::FirstOffdiag);
    CHECK(abd_fem.row_kind == SocuNativeDescriptorKind::Abd);
    CHECK(abd_fem.col_kind == SocuNativeDescriptorKind::Fem);
    CHECK(abd_fem.row_abd_body == 5);
    CHECK(abd_fem.row_jacobian_index == 2);
    CHECK(abd_fem.col_jacobian_index == -1);
}

TEST_CASE("cuda_mixed_socu_native_simplex_contact_target_table_device_rebuild",
          "[cuda_mixed_socu][contract][socu_native_contact][m8]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU native contact target tests");

    ContactTargetFixture fixture;
    std::vector<SocuNativeVertexDescriptor> vertices(16);
    vertices[0] =
        fixture.vertex(SocuNativeDescriptorKind::Fem, 0, 3, false);
    vertices[1] =
        fixture.vertex(SocuNativeDescriptorKind::Fem, 3, 3, false);
    vertices[2] =
        fixture.vertex(SocuNativeDescriptorKind::Fem, 6, 3, false);
    vertices[10] =
        fixture.vertex(SocuNativeDescriptorKind::Abd, 9, 12, false, 5, 2);

    std::vector<Vector4i> pts{Vector4i{0, 1, 2, 10}};
    std::vector<Vector2i> pps{Vector2i{0, 1}};

    muda::DeviceBuffer<IndexT> old_to_chain{fixture.old_to_chain};
    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertex_device{vertices};
    muda::DeviceBuffer<Vector4i> pt_device{pts};
    muda::DeviceBuffer<Vector4i> ee_device;
    muda::DeviceBuffer<Vector3i> pe_device;
    muda::DeviceBuffer<Vector2i> pp_device{pps};
    muda::DeviceBuffer<SocuNativeContactStencilTarget> pt_targets;
    muda::DeviceBuffer<SocuNativeContactStencilTarget> ee_targets;
    muda::DeviceBuffer<SocuNativeContactStencilTarget> pe_targets;
    muda::DeviceBuffer<SocuNativeContactStencilTarget> pp_targets;
    pt_targets.resize(10);
    pp_targets.resize(3);

    cudaGetLastError();
    rebuild_socu_native_simplex_contact_targets(
        cudaStreamLegacy,
        pt_targets.view(),
        ee_targets.view(),
        pe_targets.view(),
        pp_targets.view(),
        pt_device.view().as_const(),
        ee_device.view().as_const(),
        pe_device.view().as_const(),
        pp_device.view().as_const(),
        vertex_device.view().as_const(),
        old_to_chain.view().as_const(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        StructuredContactOffbandPolicy::DiagLump);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    std::vector<SocuNativeContactStencilTarget> pt_host;
    std::vector<SocuNativeContactStencilTarget> pp_host;
    pt_targets.copy_to(pt_host);
    pp_targets.copy_to(pp_host);

    REQUIRE(pt_host.size() == 10);
    CHECK(pt_host[0].write_mode
          == SocuNativeContactWriteMode::DiagLumpFallback);
    CHECK(pt_host[0].local_row_vertex == 0);
    CHECK(pt_host[0].local_col_vertex == 0);
    CHECK(pt_host[0].half_block_class == SocuNativeBandClass::Diag);
    CHECK(pt_host[2].write_mode
          == SocuNativeContactWriteMode::DiagLumpFallback);
    CHECK(pt_host[2].half_block_class == SocuNativeBandClass::OffBand);
    CHECK(pt_host[6].col_kind == SocuNativeDescriptorKind::Abd);
    CHECK(pt_host[6].col_abd_body == 5);
    CHECK(pt_host[6].col_jacobian_index == 2);

    REQUIRE(pp_host.size() == 3);
    CHECK(pp_host[0].exact_in_band());
    CHECK(pp_host[0].half_block_class == SocuNativeBandClass::Diag);
    CHECK(pp_host[0].block_or_left_block == 0);
    CHECK(pp_host[0].row_lane == 5);
    CHECK(pp_host[0].col_lane == 5);

    CHECK(pp_host[1].exact_in_band());
    CHECK(pp_host[1].half_block_class == SocuNativeBandClass::FirstOffdiag);
    CHECK(pp_host[1].block_or_left_block == 0);
    CHECK(pp_host[1].row_lane == 2);
    CHECK(pp_host[1].col_lane == 5);
    CHECK(pp_host[1].transposed_first_offdiag);
    CHECK(pp_host[1].local_row_vertex == 0);
    CHECK(pp_host[1].local_col_vertex == 1);

    CHECK(pp_host[2].exact_in_band());
    CHECK(pp_host[2].half_block_class == SocuNativeBandClass::Diag);
    CHECK(pp_host[2].block_or_left_block == 1);
    CHECK(pp_host[2].row_lane == 2);
    CHECK(pp_host[2].col_lane == 2);

    std::vector<Vector2i> phs{Vector2i{0, 7}, Vector2i{10, 3}};
    muda::DeviceBuffer<Vector2i> ph_device{phs};
    muda::DeviceBuffer<SocuNativeContactStencilTarget> ph_targets;
    ph_targets.resize(phs.size());

    cudaGetLastError();
    rebuild_socu_native_vertex_half_plane_contact_targets(
        cudaStreamLegacy,
        ph_targets.view(),
        ph_device.view().as_const(),
        vertex_device.view().as_const(),
        old_to_chain.view().as_const(),
        ContactTargetFixture::Horizon,
        ContactTargetFixture::BlockSize,
        StructuredContactOffbandPolicy::DiagLump);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    std::vector<SocuNativeContactStencilTarget> ph_host;
    ph_targets.copy_to(ph_host);

    REQUIRE(ph_host.size() == 2);
    CHECK(ph_host[0].exact_in_band());
    CHECK(ph_host[0].row_global_vertex == 0);
    CHECK(ph_host[0].col_global_vertex == 0);
    CHECK(ph_host[0].local_row_vertex == 0);
    CHECK(ph_host[0].local_col_vertex == 0);
    CHECK(ph_host[0].half_block_class == SocuNativeBandClass::Diag);
    CHECK(ph_host[0].block_or_left_block == 0);
    CHECK(ph_host[0].row_lane == 5);
    CHECK(ph_host[0].col_lane == 5);

    CHECK(ph_host[1].exact_in_band());
    CHECK(ph_host[1].row_global_vertex == 10);
    CHECK(ph_host[1].col_global_vertex == 10);
    CHECK(ph_host[1].row_kind == SocuNativeDescriptorKind::Abd);
    CHECK(ph_host[1].row_abd_body == 5);
    CHECK(ph_host[1].row_jacobian_index == 2);
    CHECK(ph_host[1].half_block_class == SocuNativeBandClass::Diag);
    CHECK(ph_host[1].block_or_left_block == 2);
    CHECK(ph_host[1].row_lane == 11);
    CHECK(ph_host[1].col_lane == 11);
}

TEST_CASE("cuda_mixed_socu_native_simplex_contact_exact_writer_matrix_diff",
          "[cuda_mixed_socu][contract][socu_native_contact][m8]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU native contact writer tests");

    using Store = double;
    using Solve = double;
    constexpr SizeT  Horizon   = 2;
    constexpr SizeT  BlockSize = 32;
    constexpr SizeT  Nrhs      = 1;
    constexpr IndexT Epoch     = 23;

    const auto layout = make_socu_native_storage_layout(Horizon, BlockSize, Nrhs);

    std::vector<IndexT> old_to_chain(18, IndexT{-1});
    old_to_chain[0] = 5;
    old_to_chain[1] = 2;
    old_to_chain[2] = 9;
    old_to_chain[3] = static_cast<IndexT>(BlockSize + 4);
    old_to_chain[4] = static_cast<IndexT>(BlockSize + 7);
    old_to_chain[5] = static_cast<IndexT>(BlockSize + 1);
    for(IndexT local = 0; local < 12; ++local)
        old_to_chain[static_cast<SizeT>(6 + local)] =
            static_cast<IndexT>(BlockSize + 8 + local);

    std::vector<IndexT> old_dof_to_atom(old_to_chain.size(), IndexT{-1});
    for(SizeT old = 0; old < old_dof_to_atom.size(); ++old)
        old_dof_to_atom[old] = static_cast<IndexT>(old / 3);

    const auto dofs = build_socu_native_dof_descriptors(
        uipc::span<const IndexT>{old_to_chain.data(), old_to_chain.size()},
        uipc::span<const IndexT>{old_dof_to_atom.data(), old_dof_to_atom.size()},
        Horizon,
        BlockSize,
        Epoch);

    const auto make_vertex = [&](SocuNativeDescriptorKind kind,
                                 IndexT old_dof,
                                 IndexT dof_count,
                                 IndexT body,
                                 IndexT jacobian_index)
    {
        return make_socu_native_vertex_descriptor(
            kind,
            false,
            old_dof,
            dof_count,
            body,
            jacobian_index,
            Epoch,
            uipc::span<const SocuNativeDofDescriptor>{dofs.data(), dofs.size()});
    };

    std::vector<SocuNativeVertexDescriptor> vertices(12);
    vertices[0] = make_vertex(SocuNativeDescriptorKind::Fem, 0, 3, -1, -1);
    vertices[1] = make_vertex(SocuNativeDescriptorKind::Fem, 3, 3, -1, -1);
    vertices[10] = make_vertex(SocuNativeDescriptorKind::Abd, 6, 12, 0, 0);
    vertices[11] = make_vertex(SocuNativeDescriptorKind::Abd, 6, 12, 0, 1);

    std::vector<Vector4i> pts{Vector4i{0, 1, 10, 11}};
    std::vector<ABDJacobi> Js{ABDJacobi{Vector3{0.25, -0.5, 0.75}},
                              ABDJacobi{Vector3{-0.125, 0.375, 0.625}}};

    StreamGuard stream;
    muda::DeviceBuffer<IndexT> old_to_chain_device{old_to_chain};
    muda::DeviceBuffer<SocuNativeDofDescriptor> dofs_device{dofs};
    muda::DeviceBuffer<SocuNativeVertexDescriptor> vertex_device{vertices};
    muda::DeviceBuffer<Vector4i> pt_device{pts};
    muda::DeviceBuffer<Vector4i> empty_ee;
    muda::DeviceBuffer<Vector3i> empty_pe;
    muda::DeviceBuffer<Vector2i> empty_pp;
    muda::DeviceBuffer<SocuNativeContactStencilTarget> pt_targets;
    muda::DeviceBuffer<SocuNativeContactStencilTarget> empty_ee_targets;
    muda::DeviceBuffer<SocuNativeContactStencilTarget> empty_pe_targets;
    muda::DeviceBuffer<SocuNativeContactStencilTarget> empty_pp_targets;
    pt_targets.resize(10);

    rebuild_socu_native_simplex_contact_targets(
        stream.stream,
        pt_targets.view(),
        empty_ee_targets.view(),
        empty_pe_targets.view(),
        empty_pp_targets.view(),
        pt_device.view().as_const(),
        empty_ee.view().as_const(),
        empty_pe.view().as_const(),
        empty_pp.view().as_const(),
        vertex_device.view().as_const(),
        old_to_chain_device.view().as_const(),
        Horizon,
        BlockSize,
        StructuredContactOffbandPolicy::Drop);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    std::vector<SocuNativeContactStencilTarget> target_host;
    pt_targets.copy_to(target_host);
    REQUIRE(target_host.size() == 10);
    for(const auto& target : target_host)
    {
        CHECK(target.exact_in_band());
        CHECK(target.row_global_vertex >= 0);
        CHECK(target.col_global_vertex >= 0);
    }
    CHECK(target_host[0].row_global_vertex == 0);
    CHECK(target_host[0].col_global_vertex == 0);
    CHECK(!target_host[0].mirror_diag_block);
    CHECK(target_host[1].row_global_vertex == 0);
    CHECK(target_host[1].col_global_vertex == 1);
    CHECK(target_host[1].mirror_diag_block);
    CHECK(target_host[9].row_global_vertex == 11);
    CHECK(target_host[9].col_global_vertex == 11);
    CHECK(!target_host[9].mirror_diag_block);

    muda::DeviceBuffer<Solve> native_diag;
    muda::DeviceBuffer<Solve> native_offdiag;
    muda::DeviceBuffer<Solve> compare_diag;
    muda::DeviceBuffer<Solve> compare_offdiag;
    muda::DeviceBuffer<Solve> legacy_diag;
    muda::DeviceBuffer<Solve> legacy_offdiag;
    muda::DeviceBuffer<IndexT> counters;
    muda::DeviceBuffer<IndexT> status;
    muda::DeviceBuffer<ABDJacobi> abd_J_device{Js};

    const SizeT first_offdiag_elements =
        layout.first_offdiag_block_count * BlockSize * BlockSize;
    native_diag.resize(layout.diag_element_count);
    native_offdiag.resize(layout.offdiag_element_count);
    compare_diag.resize(layout.diag_element_count);
    compare_offdiag.resize(first_offdiag_elements);
    legacy_diag.resize(layout.diag_element_count);
    legacy_offdiag.resize(first_offdiag_elements);
    counters.resize(kStructuredAssemblyCounterCount);
    status.resize(2);

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
    REQUIRE(cudaMemsetAsync(counters.data(),
                            0,
                            counters.size() * sizeof(IndexT),
                            stream.stream)
            == cudaSuccess);
    REQUIRE(cudaMemsetAsync(status.data(),
                            0,
                            status.size() * sizeof(IndexT),
                            stream.stream)
            == cudaSuccess);

    StructuredDeviceAssemblySink<Store, Solve> native_sink{
        native_diag.view(),
        native_offdiag.view(),
        old_to_chain_device.view(),
        Horizon,
        BlockSize,
        counters.view(),
        {}};
    native_sink.matrix.use_native_matrix = true;
    native_sink.matrix.native_matrix =
        make_contact_test_native_view(native_diag.view(),
                                      native_offdiag.view(),
                                      Horizon,
                                      BlockSize,
                                      Nrhs);
    native_sink.matrix.native_dof_descriptors = dofs_device.view();
    native_sink.matrix.compare_enabled = true;
    native_sink.matrix.compare_uses_native_matrix = false;
    native_sink.matrix.compare_diag = compare_diag.view();
    native_sink.matrix.compare_first_offdiag = compare_offdiag.view();

    SocuNativeContactExactWriter<Store, Solve> native_writer{
        native_sink.matrix,
        abd_J_device.view().as_const()};

    write_native_contact_exact_diff_fixture<Store, Solve>
        <<<1, 1, 0, stream.stream>>>(native_writer,
                                     pt_targets.view().as_const(),
                                     status.view());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream.stream) == cudaSuccess);

    std::vector<IndexT> status_host;
    std::vector<Solve> native_diag_host;
    std::vector<Solve> native_offdiag_host;
    std::vector<Solve> compare_diag_host;
    std::vector<Solve> compare_offdiag_host;
    status.copy_to(status_host);
    native_diag.copy_to(native_diag_host);
    native_offdiag.copy_to(native_offdiag_host);
    compare_diag.copy_to(compare_diag_host);
    compare_offdiag.copy_to(compare_offdiag_host);

    REQUIRE(status_host.size() == 2);
    CHECK(status_host[0] == 10);
    CHECK(status_host[1] == 0);

    REQUIRE(native_diag_host.size() == compare_diag_host.size());
    for(std::size_t i = 0; i < compare_diag_host.size(); ++i)
    {
        CAPTURE(i);
        CHECK(native_diag_host[i]
              == Catch::Approx(compare_diag_host[i]).margin(1e-9));
    }

    REQUIRE(native_offdiag_host.size() >= compare_offdiag_host.size());
    for(std::size_t i = 0; i < compare_offdiag_host.size(); ++i)
    {
        CAPTURE(i);
        CHECK(native_offdiag_host[i]
              == Catch::Approx(compare_offdiag_host[i]).margin(1e-9));
    }

    for(std::size_t i = compare_offdiag_host.size();
        i < native_offdiag_host.size();
        ++i)
    {
        CAPTURE(i);
        CHECK(std::abs(native_offdiag_host[i]) == Catch::Approx(0.0).margin(1e-12));
    }
}
