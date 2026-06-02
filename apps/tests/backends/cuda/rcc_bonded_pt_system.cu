#include <muda/ext/eigen/eigen_core_cxx20.h>

#include <app/app.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <contact_system/rcc_bonded_pt_lookup.h>
#include <contact_system/rcc_bonded_pt_system.h>
#include <muda/buffer/device_buffer.h>

namespace
{
bool same_topo(const uipc::Vector4i& lhs, const uipc::Vector4i& rhs)
{
    return (lhs.array() == rhs.array()).all();
}

uipc::Matrix3x3 diag3(uipc::Float x, uipc::Float y, uipc::Float z)
{
    uipc::Matrix3x3 m = uipc::Matrix3x3::Zero();
    m(0, 0) = x;
    m(1, 1) = y;
    m(2, 2) = z;
    return m;
}

uipc::core::RCCBondedPTEntry entry_by_key(const uipc::core::RCCBondedPTState& state,
                                          uipc::U64 key)
{
    const auto index = state.find_key(key);
    REQUIRE(index != uipc::core::RCCBondedPTState::npos);
    return state.entry(index);
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
    const Matrix3x3 refreshed_dm_inv = diag3(1.0, 2.0, 3.0);
    const Matrix3x3 carried_dm_inv = diag3(4.0, 5.0, 6.0);

    RCCBondedPTState host;
    host.push_locked(RCCBondedPTEntry{rcc_bonded_pt_key(refreshed),
                                      refreshed,
                                      0.91,
                                      4,
                                      RCCBondedPTReleaseNone,
                                      refreshed_dm_inv,
                                      0.2});
    host.push_locked(RCCBondedPTEntry{rcc_bonded_pt_key(carried),
                                      carried,
                                      0.93,
                                      2,
                                      RCCBondedPTReleaseNone,
                                      carried_dm_inv,
                                      0.3});
    host.sort_by_key();

    RCCBondedPTSystem::Impl owner;
    owner.set_enabled(true);
    owner.upload(host);

    std::vector<Vector4i> h_pairs = {refreshed, fresh, fresh, rejected};
    std::vector<Float>    h_beta  = {0.98, 0.95, 0.95, 0.50};

    DeviceBuffer<Vector4i> d_pairs;
    DeviceBuffer<Float>    d_beta;
    d_pairs.copy_from(h_pairs);
    d_beta.copy_from(h_beta);

    owner.lock_from_rcc_pt_snapshot(d_pairs.view(), d_beta.view(), 0.90);

    auto locked = owner.download();
    REQUIRE(locked.size() == 3);
    CHECK(locked.counters().candidate_count == 4);
    CHECK(locked.counters().locked_count == 3);
    CHECK(locked.counters().duplicate_suppressed_count == 1);

    const auto refreshed_entry = entry_by_key(locked, rcc_bonded_pt_key(refreshed));
    CHECK(same_topo(refreshed_entry.topo, refreshed));
    CHECK(refreshed_entry.beta == Catch::Approx(0.98));
    CHECK(refreshed_entry.age == 5);
    CHECK(refreshed_entry.Dm_inv.isApprox(refreshed_dm_inv));
    CHECK(refreshed_entry.rest_volume == Catch::Approx(0.2));

    const auto carried_entry = entry_by_key(locked, rcc_bonded_pt_key(carried));
    CHECK(same_topo(carried_entry.topo, carried));
    CHECK(carried_entry.beta == Catch::Approx(0.93));
    CHECK(carried_entry.age == 3);
    CHECK(carried_entry.Dm_inv.isApprox(carried_dm_inv));
    CHECK(carried_entry.rest_volume == Catch::Approx(0.3));

    const auto fresh_entry = entry_by_key(locked, rcc_bonded_pt_key(fresh));
    CHECK(same_topo(fresh_entry.topo, fresh));
    CHECK(fresh_entry.beta == Catch::Approx(0.95));
    CHECK(fresh_entry.age == 1);
    CHECK(fresh_entry.Dm_inv.isApprox(Matrix3x3::Identity()));
    CHECK(fresh_entry.rest_volume == Catch::Approx(0.0));

    CHECK(locked.find_key(rcc_bonded_pt_key(rejected)) == RCCBondedPTState::npos);

    std::vector<Vector4i> h_active = {refreshed, carried, fresh, rejected};
    DeviceBuffer<Vector4i> d_active;
    d_active.copy_from(h_active);

    SimplexTrajectoryFilter::Impl filter;
    filter.PTs = d_active.view();
    owner.feed_filter_keys(filter);
    filter.filter_rcc_bonded_pt_locked_active_pairs();
    owner.sync_filter_skipped_count(filter);

    CHECK(owner.counters().filter_skipped_count == 3);
    REQUIRE(filter.PTs.size() == 1);

    std::vector<Vector4i> h_filtered(filter.PTs.size());
    filter.PTs.copy_to(h_filtered.data());
    CHECK(same_topo(h_filtered[0], rejected));
}
