#include <app/app.h>
#include <linear_system/socu_native_contact_targets.h>

#include <initializer_list>
#include <vector>

namespace
{
using namespace uipc::backend::cuda_mixed;

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
