#include <muda/ext/eigen/eigen_core_cxx20.h>

#include <app/app.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <contact_system/rcc_bonded_pt_beta_carry.h>
#include <contact_system/rcc_bonded_pt_lookup.h>
#include <contact_system/rcc_bonded_pt_system.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_buffer_2d.h>
#include <uipc/core/rcc_bonded_pt_oracle.h>

namespace
{
bool same_topo(const uipc::Vector4i& lhs, const uipc::Vector4i& rhs)
{
    return (lhs.array() == rhs.array()).all();
}

uipc::core::RCCBondedPTEntry entry_by_key(const uipc::core::RCCBondedPTState& state,
                                          uipc::U64 key)
{
    const auto index = state.find_key(key);
    REQUIRE(index != uipc::core::RCCBondedPTState::npos);
    return state.entry(index);
}

uipc::core::RCCBondedPTRestShape rest_shape_from_positions(
    const uipc::Vector4i& topo,
    const std::vector<uipc::Vector3>& positions)
{
    uipc::core::RCCBondedPTRestShapeInput input;
    input.topo = topo;
    input.point = positions[topo[0]];
    input.tri0 = positions[topo[1]];
    input.tri1 = positions[topo[2]];
    input.tri2 = positions[topo[3]];
    input.min_separate_distance = 0.05;
    input.triangle_degeneracy_tol = 1e-12;
    return uipc::core::build_rcc_bonded_pt_rest_shape_svts(input);
}

template <typename MutateCurrentPositions>
void check_two_lock_release_reason(MutateCurrentPositions mutate_current_positions,
                                   uipc::Float strain_threshold,
                                   uipc::Float gap_threshold,
                                   uipc::Float slip_threshold,
                                   uipc::U32 expected_flag,
                                   uipc::backend::cuda::RCCBondedPTReleaseContext
                                       release_context = {})
{
    using namespace muda;
    using namespace uipc;
    using namespace uipc::backend::cuda;
    using namespace uipc::core;

    const Vector4i stay{0, 1, 2, 3};
    const Vector4i release{4, 5, 6, 7};

    std::vector<Vector3> h_rest_positions(8, Vector3::Zero());
    h_rest_positions[0] = Vector3{0.25, 0.25, 0.10};
    h_rest_positions[1] = Vector3{0.0, 0.0, 0.0};
    h_rest_positions[2] = Vector3{1.0, 0.0, 0.0};
    h_rest_positions[3] = Vector3{0.0, 1.0, 0.0};

    h_rest_positions[4] = Vector3{2.25, 0.25, 0.10};
    h_rest_positions[5] = Vector3{2.0, 0.0, 0.0};
    h_rest_positions[6] = Vector3{3.0, 0.0, 0.0};
    h_rest_positions[7] = Vector3{2.0, 1.0, 0.0};

    const auto stay_rest = rest_shape_from_positions(stay, h_rest_positions);
    const auto release_rest = rest_shape_from_positions(release, h_rest_positions);
    REQUIRE(stay_rest.valid);
    REQUIRE(release_rest.valid);

    RCCBondedPTState host;
    host.push_locked(RCCBondedPTEntry{rcc_bonded_pt_key(stay),
                                      stay_rest.oriented_topo,
                                      0.72,
                                      5,
                                      RCCBondedPTReleaseNone,
                                      stay_rest.Dm_inv,
                                      stay_rest.rest_volume});
    host.push_locked(RCCBondedPTEntry{rcc_bonded_pt_key(release),
                                      release_rest.oriented_topo,
                                      0.88,
                                      7,
                                      RCCBondedPTReleaseNone,
                                      release_rest.Dm_inv,
                                      release_rest.rest_volume});
    host.sort_by_key();

    RCCBondedPTSystem::Impl owner;
    owner.set_enabled(true);
    owner.set_rest_shape_config(0.05, 1e-12);
    owner.set_release_config(strain_threshold, gap_threshold, slip_threshold);
    owner.upload(host);

    auto h_current_positions = h_rest_positions;
    mutate_current_positions(h_current_positions);

    std::vector<Vector4i> h_pairs = {release};
    std::vector<Float>    h_beta  = {0.99};
    DeviceBuffer<Vector4i> d_pairs;
    DeviceBuffer<Float>    d_beta;
    DeviceBuffer<Vector3>  d_positions;
    d_pairs.copy_from(h_pairs);
    d_beta.copy_from(h_beta);
    d_positions.copy_from(h_current_positions);

    owner.lock_from_rcc_pt_snapshot(
        d_pairs.view(), d_beta.view(), d_positions.view(), 0.90, release_context);

    auto locked = owner.download();
    REQUIRE(locked.size() == 1);
    CHECK(locked.counters().candidate_count == 1);
    CHECK(locked.counters().locked_count == 1);
    CHECK(locked.counters().released_count == 1);

    const auto stay_entry = entry_by_key(locked, rcc_bonded_pt_key(stay));
    CHECK(same_topo(stay_entry.topo, stay_rest.oriented_topo));
    CHECK(stay_entry.beta == Catch::Approx(0.72));
    CHECK(stay_entry.age == 6);
    CHECK(locked.find_key(rcc_bonded_pt_key(release)) == RCCBondedPTState::npos);

    REQUIRE(owner.released_keys().size() == 1);
    std::vector<U64> released_keys(1);
    std::vector<Vector4i> released_topos(1);
    std::vector<Float> released_beta(1);
    std::vector<IndexT> released_age(1);
    std::vector<U32> released_flags(1);
    owner.released_keys().copy_to(released_keys.data());
    owner.released_topos().copy_to(released_topos.data());
    owner.released_beta().copy_to(released_beta.data());
    owner.released_age().copy_to(released_age.data());
    owner.released_flags().copy_to(released_flags.data());

    CHECK(released_keys[0] == rcc_bonded_pt_key(release));
    CHECK(same_topo(released_topos[0], release_rest.oriented_topo));
    CHECK(released_beta[0] == Catch::Approx(0.88));
    CHECK(released_age[0] == 8);
    CHECK((released_flags[0] & expected_flag) != 0);
}
}  // namespace

TEST_CASE("rcc_bonded_pt_system_feeds_locked_keys_and_syncs_filter_counters",
          "[rcc_bonded_pt][owner][cuda]")
{
    using namespace muda;
    using namespace uipc;
    using namespace uipc::backend::cuda;
    using namespace uipc::core;

    const Vector4i locked{9, 3, 4, 5};
    const Vector4i unlocked{9, 3, 4, 6};

    RCCBondedPTState host;
    host.record_candidates(2);
    host.push_locked(RCCBondedPTEntry{rcc_bonded_pt_key(locked), locked, 0.95, 9});
    host.sort_by_key();

    RCCBondedPTSystem::Impl owner;
    owner.set_enabled(true);
    owner.upload(host);
    REQUIRE(owner.size() == 1);
    CHECK(owner.counters().candidate_count == 2);
    CHECK(owner.counters().locked_count == 1);

    std::vector<Vector4i> h_active = {locked, unlocked};
    DeviceBuffer<Vector4i> d_active;
    d_active.copy_from(h_active);

    SimplexTrajectoryFilter::Impl filter;
    filter.PTs = d_active.view();
    owner.feed_filter_keys(filter);
    filter.filter_rcc_bonded_pt_locked_active_pairs();
    owner.sync_filter_skipped_count(filter);

    CHECK(owner.counters().filter_skipped_count == 1);
    owner.sync_filter_skipped_count(filter);
    CHECK(owner.counters().filter_skipped_count == 1);
    CHECK(owner.counters().locked_count == 1);
    REQUIRE(filter.PTs.size() == 1);

    std::vector<Vector4i> h_filtered(filter.PTs.size());
    filter.PTs.copy_to(h_filtered.data());
    CHECK(same_topo(h_filtered[0], unlocked));

    auto roundtrip = owner.download();
    CHECK(roundtrip.counters().candidate_count == 2);
    CHECK(roundtrip.counters().filter_skipped_count == 1);
    CHECK(roundtrip.counters().locked_count == 1);

    owner.clear();
    CHECK(owner.empty());
    CHECK(owner.counters().locked_count == 0);
}

TEST_CASE("rcc_bonded_pt_system_locks_from_rcc_beta_snapshot_on_device",
          "[rcc_bonded_pt][owner][producer][cuda]")
{
    using namespace muda;
    using namespace uipc;
    using namespace uipc::backend::cuda;
    using namespace uipc::core;

    const Vector4i refreshed{9, 3, 4, 5};
    const Vector4i carried{10, 6, 7, 8};
    const Vector4i fresh{12, 1, 2, 3};
    const Vector4i rejected{13, 1, 2, 3};
    const Vector4i degenerate{14, 15, 16, 17};

    std::vector<Vector4i> h_pairs = {refreshed, fresh, fresh, rejected, degenerate};
    std::vector<Float>    h_beta  = {0.98, 0.95, 0.95, 0.50, 0.96};
    std::vector<Vector3>  h_positions(18, Vector3::Zero());
    h_positions[1] = Vector3{0.0, 0.0, 0.0};
    h_positions[2] = Vector3{1.0, 0.0, 0.0};
    h_positions[3] = Vector3{0.0, 1.0, 0.0};
    h_positions[12] = Vector3{0.25, 0.25, 0.0};

    h_positions[4] = Vector3{1.0, 1.0, 0.0};
    h_positions[5] = Vector3{0.0, 2.0, 0.0};
    h_positions[9] = Vector3{0.25, 1.25, 0.02};

    h_positions[6] = Vector3{4.0, 0.0, 0.0};
    h_positions[7] = Vector3{5.0, 0.0, 0.0};
    h_positions[8] = Vector3{4.0, 1.0, 0.0};
    h_positions[10] = Vector3{4.25, 0.25, 0.05};

    h_positions[15] = Vector3{0.0, 0.0, 0.0};
    h_positions[16] = Vector3{1.0, 0.0, 0.0};
    h_positions[17] = Vector3{2.0, 0.0, 0.0};
    h_positions[14] = Vector3{0.5, 0.0, 0.0};

    const auto refreshed_rest = rest_shape_from_positions(refreshed, h_positions);
    const auto carried_rest = rest_shape_from_positions(carried, h_positions);
    REQUIRE(refreshed_rest.valid);
    REQUIRE(carried_rest.valid);

    RCCBondedPTState host;
    host.push_locked(RCCBondedPTEntry{rcc_bonded_pt_key(refreshed),
                                      refreshed_rest.oriented_topo,
                                      0.91,
                                      4,
                                      RCCBondedPTReleaseNone,
                                      refreshed_rest.Dm_inv,
                                      refreshed_rest.rest_volume});
    host.push_locked(RCCBondedPTEntry{rcc_bonded_pt_key(carried),
                                      carried_rest.oriented_topo,
                                      0.93,
                                      2,
                                      RCCBondedPTReleaseNone,
                                      carried_rest.Dm_inv,
                                      carried_rest.rest_volume});
    host.sort_by_key();

    RCCBondedPTSystem::Impl owner;
    owner.set_enabled(true);
    owner.set_rest_shape_config(0.05, 1e-12);
    owner.upload(host);

    DeviceBuffer<Vector4i> d_pairs;
    DeviceBuffer<Float>    d_beta;
    DeviceBuffer<Vector3>  d_positions;
    d_pairs.copy_from(h_pairs);
    d_beta.copy_from(h_beta);
    d_positions.copy_from(h_positions);

    owner.lock_from_rcc_pt_snapshot(
        d_pairs.view(), d_beta.view(), d_positions.view(), 0.90);

    RCCBondedPTRestShapeInput fresh_input;
    fresh_input.topo = fresh;
    fresh_input.point = h_positions[fresh[0]];
    fresh_input.tri0 = h_positions[fresh[1]];
    fresh_input.tri1 = h_positions[fresh[2]];
    fresh_input.tri2 = h_positions[fresh[3]];
    fresh_input.min_separate_distance = 0.05;
    fresh_input.triangle_degeneracy_tol = 1e-12;
    const auto fresh_rest = build_rcc_bonded_pt_rest_shape_svts(fresh_input);
    REQUIRE(fresh_rest.valid);

    auto locked = owner.download();
    REQUIRE(locked.size() == 3);
    CHECK(locked.counters().candidate_count == 5);
    CHECK(locked.counters().locked_count == 3);
    CHECK(locked.counters().duplicate_suppressed_count == 1);
    CHECK(locked.counters().degenerate_rejected_count == 1);

    const auto refreshed_entry = entry_by_key(locked, rcc_bonded_pt_key(refreshed));
    CHECK(same_topo(refreshed_entry.topo, refreshed_rest.oriented_topo));
    CHECK(refreshed_entry.beta == Catch::Approx(0.98));
    CHECK(refreshed_entry.age == 5);
    CHECK(refreshed_entry.Dm_inv.isApprox(refreshed_rest.Dm_inv));
    CHECK(refreshed_entry.rest_volume == Catch::Approx(refreshed_rest.rest_volume));

    const auto carried_entry = entry_by_key(locked, rcc_bonded_pt_key(carried));
    CHECK(same_topo(carried_entry.topo, carried_rest.oriented_topo));
    CHECK(carried_entry.beta == Catch::Approx(0.93));
    CHECK(carried_entry.age == 3);
    CHECK(carried_entry.Dm_inv.isApprox(carried_rest.Dm_inv));
    CHECK(carried_entry.rest_volume == Catch::Approx(carried_rest.rest_volume));

    const auto fresh_entry = entry_by_key(locked, rcc_bonded_pt_key(fresh));
    CHECK(same_topo(fresh_entry.topo, fresh_rest.oriented_topo));
    CHECK(fresh_entry.beta == Catch::Approx(0.95));
    CHECK(fresh_entry.age == 1);
    CHECK(fresh_entry.Dm_inv.isApprox(fresh_rest.Dm_inv, 1e-12));
    CHECK(fresh_entry.rest_volume == Catch::Approx(fresh_rest.rest_volume));

    CHECK(locked.find_key(rcc_bonded_pt_key(rejected)) == RCCBondedPTState::npos);
    CHECK(locked.find_key(rcc_bonded_pt_key(degenerate)) == RCCBondedPTState::npos);

    std::vector<Vector4i> h_active = {refreshed, carried, fresh, rejected, degenerate};
    DeviceBuffer<Vector4i> d_active;
    d_active.copy_from(h_active);

    SimplexTrajectoryFilter::Impl filter;
    filter.PTs = d_active.view();
    owner.feed_filter_keys(filter);
    filter.filter_rcc_bonded_pt_locked_active_pairs();
    owner.sync_filter_skipped_count(filter);

    CHECK(owner.counters().filter_skipped_count == 3);
    REQUIRE(filter.PTs.size() == 2);

    std::vector<Vector4i> h_filtered(filter.PTs.size());
    filter.PTs.copy_to(h_filtered.data());
    CHECK(same_topo(h_filtered[0], rejected));
    CHECK(same_topo(h_filtered[1], degenerate));
}

TEST_CASE("rcc_bonded_pt_system_releases_strained_locks_without_relocking",
          "[rcc_bonded_pt][release][cuda]")
{
    using namespace muda;
    using namespace uipc;
    using namespace uipc::backend::cuda;
    using namespace uipc::core;

    const Vector4i stay{0, 1, 2, 3};
    const Vector4i release{4, 5, 6, 7};

    std::vector<Vector3> h_rest_positions(8, Vector3::Zero());
    h_rest_positions[0] = Vector3{0.25, 0.25, 0.10};
    h_rest_positions[1] = Vector3{0.0, 0.0, 0.0};
    h_rest_positions[2] = Vector3{1.0, 0.0, 0.0};
    h_rest_positions[3] = Vector3{0.0, 1.0, 0.0};

    h_rest_positions[4] = Vector3{2.25, 0.25, 0.10};
    h_rest_positions[5] = Vector3{2.0, 0.0, 0.0};
    h_rest_positions[6] = Vector3{3.0, 0.0, 0.0};
    h_rest_positions[7] = Vector3{2.0, 1.0, 0.0};

    const auto stay_rest = rest_shape_from_positions(stay, h_rest_positions);
    const auto release_rest = rest_shape_from_positions(release, h_rest_positions);
    REQUIRE(stay_rest.valid);
    REQUIRE(release_rest.valid);

    RCCBondedPTState host;
    host.push_locked(RCCBondedPTEntry{rcc_bonded_pt_key(stay),
                                      stay_rest.oriented_topo,
                                      0.72,
                                      5,
                                      RCCBondedPTReleaseNone,
                                      stay_rest.Dm_inv,
                                      stay_rest.rest_volume});
    host.push_locked(RCCBondedPTEntry{rcc_bonded_pt_key(release),
                                      release_rest.oriented_topo,
                                      0.88,
                                      7,
                                      RCCBondedPTReleaseNone,
                                      release_rest.Dm_inv,
                                      release_rest.rest_volume});
    host.sort_by_key();

    RCCBondedPTSystem::Impl owner;
    owner.set_enabled(true);
    owner.set_rest_shape_config(0.05, 1e-12);
    owner.set_release_config(0.25, 1e30, 1e30);
    owner.upload(host);

    auto h_current_positions = h_rest_positions;
    h_current_positions[4] = Vector3{2.25, 0.25, 0.55};

    std::vector<Vector4i> h_pairs = {release};
    std::vector<Float>    h_beta  = {0.99};
    DeviceBuffer<Vector4i> d_pairs;
    DeviceBuffer<Float>    d_beta;
    DeviceBuffer<Vector3>  d_positions;
    d_pairs.copy_from(h_pairs);
    d_beta.copy_from(h_beta);
    d_positions.copy_from(h_current_positions);

    owner.lock_from_rcc_pt_snapshot(
        d_pairs.view(), d_beta.view(), d_positions.view(), 0.90);

    auto locked = owner.download();
    REQUIRE(locked.size() == 1);
    CHECK(locked.counters().candidate_count == 1);
    CHECK(locked.counters().locked_count == 1);
    CHECK(locked.counters().released_count == 1);

    const auto stay_entry = entry_by_key(locked, rcc_bonded_pt_key(stay));
    CHECK(same_topo(stay_entry.topo, stay_rest.oriented_topo));
    CHECK(stay_entry.beta == Catch::Approx(0.72));
    CHECK(stay_entry.age == 6);
    CHECK(locked.find_key(rcc_bonded_pt_key(release)) == RCCBondedPTState::npos);

    REQUIRE(owner.released_keys().size() == 1);
    std::vector<U64> released_keys(1);
    std::vector<Vector4i> released_topos(1);
    std::vector<Float> released_beta(1);
    std::vector<IndexT> released_age(1);
    std::vector<U32> released_flags(1);
    owner.released_keys().copy_to(released_keys.data());
    owner.released_topos().copy_to(released_topos.data());
    owner.released_beta().copy_to(released_beta.data());
    owner.released_age().copy_to(released_age.data());
    owner.released_flags().copy_to(released_flags.data());

    CHECK(released_keys[0] == rcc_bonded_pt_key(release));
    CHECK(same_topo(released_topos[0], release_rest.oriented_topo));
    CHECK(released_beta[0] == Catch::Approx(0.88));
    CHECK(released_age[0] == 8);
    CHECK((released_flags[0] & RCCBondedPTReleaseStrain) != 0);
}

TEST_CASE("rcc_bonded_pt_system_releases_large_normal_gap",
          "[rcc_bonded_pt][release][gap][cuda]")
{
    using namespace uipc;
    using namespace uipc::core;

    check_two_lock_release_reason(
        [](std::vector<Vector3>& positions)
        {
            positions[4] = Vector3{2.25, 0.25, 0.40};
        },
        1e30,
        0.05,
        1e30,
        RCCBondedPTReleaseGap);
}

TEST_CASE("rcc_bonded_pt_system_releases_large_tangential_slip",
          "[rcc_bonded_pt][release][slip][cuda]")
{
    using namespace uipc;
    using namespace uipc::core;

    check_two_lock_release_reason(
        [](std::vector<Vector3>& positions)
        {
            positions[4] = Vector3{2.65, 0.25, 0.10};
        },
        1e30,
        1e30,
        0.05,
        RCCBondedPTReleaseSlip);
}

TEST_CASE("rcc_bonded_pt_system_releases_sticky_side_failure",
          "[rcc_bonded_pt][release][sticky][cuda]")
{
    using namespace muda;
    using namespace uipc;
    using namespace uipc::backend::cuda;
    using namespace uipc::core;

    std::vector<IndexT> h_sticky_sign(8, 0);
    h_sticky_sign[1] = 1;
    h_sticky_sign[2] = 1;
    h_sticky_sign[3] = 1;
    h_sticky_sign[5] = -1;
    h_sticky_sign[6] = -1;
    h_sticky_sign[7] = -1;

    std::vector<Vector3> h_normals(8, Vector3::UnitZ());
    DeviceBuffer<IndexT> d_sticky_sign;
    DeviceBuffer<Vector3> d_normals;
    d_sticky_sign.copy_from(h_sticky_sign);
    d_normals.copy_from(h_normals);

    RCCBondedPTReleaseContext context;
    context.sticky_side_enabled = true;
    context.sticky_sign = d_sticky_sign.view();
    context.vertex_normal = d_normals.view();

    check_two_lock_release_reason(
        [](std::vector<Vector3>&) {},
        1e30,
        1e30,
        1e30,
        RCCBondedPTReleaseStickySide,
        context);
}

TEST_CASE("rcc_bonded_pt_system_releases_disabled_contact_policy",
          "[rcc_bonded_pt][release][policy][cuda]")
{
    using namespace muda;
    using namespace uipc;
    using namespace uipc::backend::cuda;
    using namespace uipc::core;

    std::vector<IndexT> h_contact_ids = {0, 1, 1, 1, 2, 3, 3, 3};
    std::vector<IndexT> h_subscene_ids(8, 0);

    constexpr IndexT cid_count = 4;
    std::vector<IndexT> h_contact_mask(cid_count * cid_count, 1);
    h_contact_mask[2 * cid_count + 3] = 0;
    h_contact_mask[3 * cid_count + 2] = 0;

    std::vector<IndexT> h_subscene_mask = {1};
    std::vector<RCCAdhesiveCoeff> h_adhesive(cid_count * cid_count);
    for(auto& coeff : h_adhesive)
        coeff.enabled = 1;

    DeviceBuffer<IndexT> d_contact_ids;
    DeviceBuffer<IndexT> d_subscene_ids;
    DeviceBuffer2D<IndexT> d_contact_mask(Extent2D{cid_count, cid_count});
    DeviceBuffer2D<IndexT> d_subscene_mask(Extent2D{1, 1});
    DeviceBuffer2D<RCCAdhesiveCoeff> d_adhesive(Extent2D{cid_count, cid_count});
    d_contact_ids.copy_from(h_contact_ids);
    d_subscene_ids.copy_from(h_subscene_ids);
    d_contact_mask.view().copy_from(h_contact_mask.data());
    d_subscene_mask.view().copy_from(h_subscene_mask.data());
    d_adhesive.view().copy_from(h_adhesive.data());

    RCCBondedPTReleaseContext context;
    context.policy_enabled = true;
    context.contact_element_ids = d_contact_ids.view();
    context.subscene_element_ids = d_subscene_ids.view();
    context.contact_mask_tabular = d_contact_mask.view();
    context.subscene_mask_tabular = d_subscene_mask.view();
    context.adhesive_tabular = d_adhesive.view();

    check_two_lock_release_reason(
        [](std::vector<Vector3>&) {},
        1e30,
        1e30,
        1e30,
        RCCBondedPTReleasePolicy,
        context);
}

TEST_CASE("rcc_bonded_pt_beta_carry_merges_released_beta_without_overwrite",
          "[rcc_bonded_pt][release][beta_carry][cuda]")
{
    using namespace muda;
    using namespace uipc;
    using namespace uipc::backend::cuda;

    std::vector<U64>   h_prev_keys = {5, 9};
    std::vector<Float> h_prev_beta = {0.4, 0.7};
    std::vector<U64>   h_released_keys = {3, 5};
    std::vector<Float> h_released_beta = {0.8, 0.99};

    DeviceBuffer<U64>   d_prev_keys;
    DeviceBuffer<Float> d_prev_beta;
    DeviceBuffer<U64>   d_released_keys;
    DeviceBuffer<Float> d_released_beta;
    d_prev_keys.copy_from(h_prev_keys);
    d_prev_beta.copy_from(h_prev_beta);
    d_released_keys.copy_from(h_released_keys);
    d_released_beta.copy_from(h_released_beta);

    RCCBondedPTBetaCarryScratch scratch;
    scratch.merge_released_beta(
        d_prev_keys, d_prev_beta, d_released_keys.view(), d_released_beta.view());

    REQUIRE(d_prev_keys.size() == 3);
    std::vector<U64> merged_keys(3);
    std::vector<Float> merged_beta(3);
    d_prev_keys.view().copy_to(merged_keys.data());
    d_prev_beta.view().copy_to(merged_beta.data());

    CHECK(merged_keys[0] == 3);
    CHECK(merged_beta[0] == Catch::Approx(0.8));
    CHECK(merged_keys[1] == 5);
    CHECK(merged_beta[1] == Catch::Approx(0.4));
    CHECK(merged_keys[2] == 9);
    CHECK(merged_beta[2] == Catch::Approx(0.7));
}
