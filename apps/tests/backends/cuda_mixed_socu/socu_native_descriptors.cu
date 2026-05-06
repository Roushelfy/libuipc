#include <app/app.h>
#include <linear_system/socu_native_descriptors.h>

#include <vector>

namespace
{
using namespace uipc::backend::cuda_mixed;

struct DescriptorFixture
{
    static constexpr uipc::SizeT Horizon   = 5;
    static constexpr uipc::SizeT BlockSize = 16;

    std::vector<uipc::IndexT> old_to_chain;
    std::vector<uipc::IndexT> old_dof_to_atom;
    std::vector<SocuNativeDofDescriptor> dofs;

    explicit DescriptorFixture(uipc::IndexT epoch = 7)
        : old_to_chain(33, uipc::IndexT{-1})
        , old_dof_to_atom(33, uipc::IndexT{-1})
    {
        map_range(0, 0, 3);    // FEM vertex 0 -> block 0, lanes 0..2
        map_range(3, 16, 3);   // FEM vertex 1 -> block 1, lanes 0..2
        map_range(6, 48, 3);   // FEM vertex 2 -> block 3, lanes 0..2
        map_range(9, 32, 12);  // ABD body 0   -> block 2, lanes 0..11
        map_range(21, 64, 12); // ABD body 1   -> block 4, lanes 0..11

        for(uipc::SizeT old = 0; old < old_dof_to_atom.size(); ++old)
            old_dof_to_atom[old] = static_cast<uipc::IndexT>(old / 3);

        dofs = build_socu_native_dof_descriptors(
            uipc::span<const uipc::IndexT>{old_to_chain.data(), old_to_chain.size()},
            uipc::span<const uipc::IndexT>{old_dof_to_atom.data(),
                                           old_dof_to_atom.size()},
            Horizon,
            BlockSize,
            epoch);
    }

    void map_range(uipc::IndexT old_begin,
                   uipc::IndexT chain_begin,
                   uipc::IndexT count)
    {
        for(uipc::IndexT i = 0; i < count; ++i)
            old_to_chain[static_cast<uipc::SizeT>(old_begin + i)] =
                chain_begin + i;
    }

    SocuNativeVertexDescriptor vertex(SocuNativeDescriptorKind kind,
                                      uipc::IndexT             old_dof,
                                      uipc::IndexT             dof_count,
                                      uipc::IndexT             epoch = 7,
                                      bool                     fixed = false) const
    {
        return make_socu_native_vertex_descriptor(
            kind,
            fixed,
            old_dof,
            dof_count,
            kind == SocuNativeDescriptorKind::Abd ? (old_dof - 9) / 12 : -1,
            kind == SocuNativeDescriptorKind::Abd ? (old_dof - 9) / 12 : -1,
            epoch,
            uipc::span<const SocuNativeDofDescriptor>{dofs.data(), dofs.size()});
    }
};
}  // namespace

TEST_CASE("cuda_mixed_socu_native_descriptor_dof_table",
          "[cuda_mixed_socu][contract][socu_native_descriptor]")
{
    DescriptorFixture fixture;

    REQUIRE(fixture.dofs.size() == fixture.old_to_chain.size());
    CHECK(fixture.dofs[0].active);
    CHECK(fixture.dofs[0].old_dof == 0);
    CHECK(fixture.dofs[0].chain_dof == 0);
    CHECK(fixture.dofs[0].atom == 0);
    CHECK(fixture.dofs[0].block == 0);
    CHECK(fixture.dofs[0].lane == 0);
    CHECK(fixture.dofs[0].epoch == 7);

    CHECK(fixture.dofs[4].active);
    CHECK(fixture.dofs[4].block == 1);
    CHECK(fixture.dofs[4].lane == 1);
    CHECK(fixture.dofs[10].active);
    CHECK(fixture.dofs[10].block == 2);
    CHECK(fixture.dofs[10].lane == 1);

    CHECK(fixture.dofs[32].active);

    std::vector<uipc::IndexT> invalid_old_to_chain(3, uipc::IndexT{-1});
    auto invalid = build_socu_native_dof_descriptors(
        uipc::span<const uipc::IndexT>{invalid_old_to_chain.data(),
                                       invalid_old_to_chain.size()},
        {},
        1,
        DescriptorFixture::BlockSize,
        3);
    CHECK(!invalid[0].active);
    CHECK(invalid[0].chain_dof == -1);
}

TEST_CASE("cuda_mixed_socu_native_descriptor_vertex_table",
          "[cuda_mixed_socu][contract][socu_native_descriptor]")
{
    DescriptorFixture fixture;
    const std::vector<uipc::IndexT> fem_fixed{0, 1, 0};
    const std::vector<uipc::IndexT> abd_v2b{0, 1};
    const std::vector<uipc::IndexT> abd_fixed{0, 1};

    SocuNativeVertexDescriptorBuildInput input;
    input.global_vertex_count = 12;
    input.horizon = DescriptorFixture::Horizon;
    input.block_size = DescriptorFixture::BlockSize;
    input.epoch = 7;
    input.dofs = uipc::span<const SocuNativeDofDescriptor>{fixture.dofs.data(),
                                                           fixture.dofs.size()};
    input.fem_vertex_offset = 0;
    input.fem_vertex_count = 3;
    input.fem_old_dof_offset = 0;
    input.fem_vertex_is_fixed =
        uipc::span<const uipc::IndexT>{fem_fixed.data(), fem_fixed.size()};
    input.abd_vertex_offset = 10;
    input.abd_vertex_count = 2;
    input.abd_old_dof_offset = 9;
    input.abd_body_count = 2;
    input.abd_vertex_to_body =
        uipc::span<const uipc::IndexT>{abd_v2b.data(), abd_v2b.size()};
    input.abd_body_is_fixed =
        uipc::span<const uipc::IndexT>{abd_fixed.data(), abd_fixed.size()};

    const auto vertices = build_socu_native_vertex_descriptors(input);
    REQUIRE(vertices.size() == 12);

    CHECK(vertices[0].kind == SocuNativeDescriptorKind::Fem);
    CHECK(vertices[0].writable());
    CHECK(vertices[0].old_dof == 0);
    CHECK(vertices[0].dof_count == 3);
    CHECK(vertices[0].block == 0);
    CHECK(vertices[0].lane == 0);
    CHECK(vertices[0].epoch == 7);

    CHECK(vertices[1].kind == SocuNativeDescriptorKind::Fem);
    CHECK(vertices[1].fixed);
    CHECK(!vertices[1].writable());

    CHECK(!vertices[5].mapped());

    CHECK(vertices[10].kind == SocuNativeDescriptorKind::Abd);
    CHECK(vertices[10].writable());
    CHECK(vertices[10].old_dof == 9);
    CHECK(vertices[10].dof_count == 12);
    CHECK(vertices[10].block == 2);
    CHECK(vertices[10].lane == 0);
    CHECK(vertices[10].abd_body == 0);
    CHECK(vertices[10].abd_j_index == 0);

    CHECK(vertices[11].kind == SocuNativeDescriptorKind::Abd);
    CHECK(vertices[11].fixed);
    CHECK(vertices[11].abd_body == 1);
    CHECK(!vertices[11].writable());
}

TEST_CASE("cuda_mixed_socu_native_descriptor_band_classification",
          "[cuda_mixed_socu][contract][socu_native_descriptor]")
{
    DescriptorFixture fixture;
    const auto old_to_chain =
        uipc::span<const uipc::IndexT>{fixture.old_to_chain.data(),
                                       fixture.old_to_chain.size()};

    const auto fem0 = fixture.vertex(SocuNativeDescriptorKind::Fem, 0, 3);
    const auto fem1 = fixture.vertex(SocuNativeDescriptorKind::Fem, 3, 3);
    const auto fem2 = fixture.vertex(SocuNativeDescriptorKind::Fem, 6, 3);
    const auto abd0 = fixture.vertex(SocuNativeDescriptorKind::Abd, 9, 12);
    const auto abd1 = fixture.vertex(SocuNativeDescriptorKind::Abd, 21, 12);
    const auto fixed_fem =
        fixture.vertex(SocuNativeDescriptorKind::Fem, 3, 3, 7, true);

    const auto diag = socu_native_classify_half_block(
        fem0,
        fem0,
        old_to_chain,
        DescriptorFixture::Horizon,
        DescriptorFixture::BlockSize);
    CHECK(diag.cls == SocuNativeBandClass::Diag);
    CHECK(diag.diag_scalar_count == 9);
    CHECK(diag.fully_in_band());

    const auto first = socu_native_classify_half_block(
        fem0,
        fem1,
        old_to_chain,
        DescriptorFixture::Horizon,
        DescriptorFixture::BlockSize);
    CHECK(first.cls == SocuNativeBandClass::FirstOffdiag);
    CHECK(first.first_offdiag_scalar_count == 9);
    CHECK(first.fully_in_band());

    const auto off = socu_native_classify_half_block(
        fem0,
        fem2,
        old_to_chain,
        DescriptorFixture::Horizon,
        DescriptorFixture::BlockSize);
    CHECK(off.cls == SocuNativeBandClass::OffBand);
    CHECK(off.offband_scalar_count == 9);
    CHECK(!off.fully_in_band());

    CHECK(socu_native_classify_half_block(abd0,
                                          fem2,
                                          old_to_chain,
                                          DescriptorFixture::Horizon,
                                          DescriptorFixture::BlockSize)
              .cls == SocuNativeBandClass::FirstOffdiag);
    CHECK(socu_native_classify_half_block(fem1,
                                          abd0,
                                          old_to_chain,
                                          DescriptorFixture::Horizon,
                                          DescriptorFixture::BlockSize)
              .cls == SocuNativeBandClass::FirstOffdiag);
    CHECK(socu_native_classify_half_block(abd0,
                                          abd1,
                                          old_to_chain,
                                          DescriptorFixture::Horizon,
                                          DescriptorFixture::BlockSize)
              .cls == SocuNativeBandClass::OffBand);
    CHECK(socu_native_classify_half_block(fixed_fem,
                                          fem0,
                                          old_to_chain,
                                          DescriptorFixture::Horizon,
                                          DescriptorFixture::BlockSize)
              .cls == SocuNativeBandClass::Skipped);

    const auto target = socu_native_classify_old_dof_pair(
        old_to_chain,
        DescriptorFixture::Horizon,
        DescriptorFixture::BlockSize,
        2,
        3);
    CHECK(target.cls == SocuNativeBandClass::FirstOffdiag);
    CHECK(target.left_block == 0);
    CHECK(target.row_lane == 0);
    CHECK(target.col_lane == 2);
    CHECK(target.transposed_first_offdiag);

    const std::vector<SocuNativeVertexDescriptor> stencil{fem0, fem1, fem2};
    const auto stencil_class = socu_native_classify_stencil_half(
        uipc::span<const SocuNativeVertexDescriptor>{stencil.data(), stencil.size()},
        old_to_chain,
        DescriptorFixture::Horizon,
        DescriptorFixture::BlockSize);
    CHECK(stencil_class.diag_half_block_count == 3);
    CHECK(stencil_class.first_offdiag_half_block_count == 1);
    CHECK(stencil_class.offband_half_block_count == 2);
    CHECK(!stencil_class.fully_in_band());
}

TEST_CASE("cuda_mixed_socu_native_descriptor_reorder_epoch",
          "[cuda_mixed_socu][contract][socu_native_descriptor]")
{
    DescriptorFixture epoch7{7};

    std::vector<uipc::IndexT> reordered = epoch7.old_to_chain;
    reordered[0] = 16;
    reordered[1] = 17;
    reordered[2] = 18;
    const auto epoch8_dofs = build_socu_native_dof_descriptors(
        uipc::span<const uipc::IndexT>{reordered.data(), reordered.size()},
        uipc::span<const uipc::IndexT>{epoch7.old_dof_to_atom.data(),
                                       epoch7.old_dof_to_atom.size()},
        DescriptorFixture::Horizon,
        DescriptorFixture::BlockSize,
        8);

    SocuNativeDescriptorTable table7;
    table7.epoch = 7;
    table7.horizon = DescriptorFixture::Horizon;
    table7.block_size = DescriptorFixture::BlockSize;
    table7.dofs = epoch7.dofs;
    CHECK(table7.valid_for(7));
    CHECK(!table7.valid_for(8));

    SocuNativeDescriptorTable table8;
    table8.epoch = 8;
    table8.horizon = DescriptorFixture::Horizon;
    table8.block_size = DescriptorFixture::BlockSize;
    table8.dofs = epoch8_dofs;
    CHECK(table8.valid_for(8));
    REQUIRE(table8.dofs[0].active);
    CHECK(table8.dofs[0].block == 1);
    CHECK(table8.dofs[0].lane == 0);
    CHECK(table8.dofs[0].epoch == 8);
}
