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
