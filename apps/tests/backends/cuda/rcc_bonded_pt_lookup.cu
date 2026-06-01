#include <muda/ext/eigen/eigen_core_cxx20.h>

#include <app/app.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <contact_system/rcc_bonded_pt_lookup.h>
#include <contact_system/rcc_bonded_pt_state_bridge.h>
#include <muda/buffer/device_buffer.h>

#include <array>

namespace
{
bool same_topo(const uipc::Vector4i& lhs, const uipc::Vector4i& rhs)
{
    return (lhs.array() == rhs.array()).all();
}
}  // namespace

TEST_CASE("rcc_bonded_pt_lookup_preserves_existing_pt_key_semantics",
          "[rcc_bonded_pt][lookup][cuda]")
{
    using namespace muda;
    using namespace uipc;
    using namespace uipc::backend::cuda;
    using namespace uipc::core;

    const Vector4i locked{9, 3, 4, 5};
    const Vector4i locked_permuted{9, 5, 3, 4};
    const Vector4i second_locked{2, 8, 1, 7};
    const Vector4i unlocked{9, 3, 4, 6};
    const Vector4i different_point{8, 3, 4, 5};

    const U64 locked_key = rcc_bonded_pt_key(locked);
    CHECK(locked_key
          == sym::codim_ipc_rcc_adhesive::PT_pair_key(
              locked[0], locked[1], locked[2], locked[3]));
    CHECK(rcc_bonded_pt_key(locked_permuted) == locked_key);
    CHECK(rcc_bonded_pt_key(different_point) != locked_key);

    RCCBondedPTState host;
    host.push_locked(RCCBondedPTEntry{
        rcc_bonded_pt_key(second_locked), second_locked, 0.75, 3});
    host.push_locked(RCCBondedPTEntry{locked_key, locked, 0.95, 9});
    host.sort_by_key();

    RCCBondedPTStateBridge bridge;
    bridge.upload(host);
    REQUIRE(bridge.size() == 2);

    const std::array<Vector4i, 5> h_queries = {
        locked, locked_permuted, unlocked, different_point, second_locked};
    DeviceBuffer<Vector4i> d_queries(h_queries.size());
    DeviceBuffer<IndexT>   d_hits(h_queries.size());
    DeviceBuffer<IndexT>   d_lower_bounds(h_queries.size());
    d_queries.view().copy_from(h_queries.data());
    d_hits.fill(0);
    d_lower_bounds.fill(-1);

    ParallelFor()
        .kernel_name("rcc_bonded_pt_lookup_preserves_existing_pt_key_semantics")
        .apply(d_queries.size(),
               [queries      = d_queries.cviewer().name("queries"),
                locked_keys  = bridge.locked_keys(),
                hits         = d_hits.viewer().name("hits"),
                lower_bounds = d_lower_bounds.viewer().name("lower_bounds")] __device__(
                   int i) mutable
               {
                   const auto& pt  = queries(i);
                   const U64   key = rcc_bonded_pt_key(pt);
                   lower_bounds(i) = rcc_bonded_pt_lower_bound(locked_keys, key);
                   hits(i) = rcc_bonded_pt_is_locked(locked_keys, key) ? 1 : 0;
               });

    std::array<IndexT, 5> h_hits{};
    std::array<IndexT, 5> h_lower_bounds{};
    d_hits.view().copy_to(h_hits.data());
    d_lower_bounds.view().copy_to(h_lower_bounds.data());

    const IndexT locked_index =
        static_cast<IndexT>(host.find_key(rcc_bonded_pt_key(locked)));
    const IndexT second_index =
        static_cast<IndexT>(host.find_key(rcc_bonded_pt_key(second_locked)));

    CHECK(h_hits[0] == 1);
    CHECK(h_hits[1] == 1);
    CHECK(h_hits[2] == 0);
    CHECK(h_hits[3] == 0);
    CHECK(h_hits[4] == 1);
    CHECK(h_lower_bounds[0] == locked_index);
    CHECK(h_lower_bounds[1] == locked_index);
    CHECK(h_lower_bounds[4] == second_index);
    CHECK(same_topo(host.locked_topos()[locked_index], locked));
    CHECK(same_topo(host.locked_topos()[second_index], second_locked));

    bridge.clear();
    REQUIRE(bridge.empty());
    DeviceBuffer<IndexT> d_empty_hit(1);
    DeviceBuffer<IndexT> d_empty_lower_bound(1);
    d_empty_hit.fill(1);
    d_empty_lower_bound.fill(-1);

    ParallelFor()
        .kernel_name("rcc_bonded_pt_lookup_empty_set_misses")
        .apply(1,
               [locked_keys       = bridge.locked_keys(),
                empty_hit         = d_empty_hit.viewer().name("empty_hit"),
                empty_lower_bound = d_empty_lower_bound.viewer().name(
                    "empty_lower_bound")] __device__(int i) mutable
               {
                   const U64 key = rcc_bonded_pt_key(9, 3, 4, 5);
                   empty_lower_bound(i) = rcc_bonded_pt_lower_bound(locked_keys, key);
                   empty_hit(i) = rcc_bonded_pt_is_locked(locked_keys, key) ? 1 : 0;
               });

    IndexT h_empty_hit         = 1;
    IndexT h_empty_lower_bound = -1;
    d_empty_hit.view().copy_to(&h_empty_hit);
    d_empty_lower_bound.view().copy_to(&h_empty_lower_bound);
    CHECK(h_empty_hit == 0);
    CHECK(h_empty_lower_bound == 0);
}

TEST_CASE("rcc_bonded_pt_filter_compacts_active_pts_before_friction_copy",
          "[rcc_bonded_pt][filter][cuda]")
{
    using namespace muda;
    using namespace uipc;
    using namespace uipc::backend::cuda;
    using namespace uipc::core;

    const Vector4i locked{9, 3, 4, 5};
    const Vector4i locked_permuted{2, 7, 8, 1};
    const Vector4i unlocked{9, 3, 4, 6};
    const Vector4i second_unlocked{4, 1, 2, 3};

    RCCBondedPTState host;
    host.push_locked(RCCBondedPTEntry{rcc_bonded_pt_key(locked), locked, 0.95, 9});
    host.push_locked(RCCBondedPTEntry{
        rcc_bonded_pt_key(locked_permuted), locked_permuted, 0.75, 3});
    host.sort_by_key();

    RCCBondedPTStateBridge bridge;
    bridge.upload(host);

    std::vector<Vector4i> h_active = {
        locked, unlocked, locked_permuted, second_unlocked};
    DeviceBuffer<Vector4i> d_active;
    d_active.copy_from(h_active);

    SimplexTrajectoryFilter::Impl impl;
    impl.PTs = d_active.view();
    impl.set_rcc_bonded_pt_locked_keys(bridge.locked_keys());
    impl.filter_rcc_bonded_pt_locked_active_pairs();

    CHECK(impl.rcc_bonded_pt_filter_skipped_count() == 2);
    REQUIRE(impl.PTs.size() == 2);

    std::vector<Vector4i> h_filtered(impl.PTs.size());
    impl.PTs.copy_to(h_filtered.data());
    CHECK(same_topo(h_filtered[0], unlocked));
    CHECK(same_topo(h_filtered[1], second_unlocked));
    for(const Vector4i& pt : h_filtered)
        CHECK(host.find_key(rcc_bonded_pt_key(pt)) == RCCBondedPTState::npos);

    GlobalTrajectoryFilter::RecordFrictionCandidatesInfo info;
    impl.record_friction_candidates(info);
    REQUIRE(impl.friction_PT.size() == 2);

    std::vector<Vector4i> h_friction;
    impl.friction_PT.copy_to(h_friction);
    CHECK(same_topo(h_friction[0], unlocked));
    CHECK(same_topo(h_friction[1], second_unlocked));

    impl.clear_rcc_bonded_pt_locked_keys();
    impl.PTs = d_active.view();
    impl.filter_rcc_bonded_pt_locked_active_pairs();
    CHECK(impl.rcc_bonded_pt_filter_skipped_count() == 0);
    CHECK(impl.PTs.size() == h_active.size());
}
