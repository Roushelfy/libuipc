#include <muda/ext/eigen/eigen_core_cxx20.h>

#include <app/app.h>
#include <contact_system/rcc_bonded_pt_state_bridge.h>

namespace
{
bool same_topo(const uipc::Vector4i& lhs, const uipc::Vector4i& rhs)
{
    return (lhs.array() == rhs.array()).all();
}
}  // namespace

TEST_CASE("rcc_bonded_pt_state_bridge_roundtrips_payloads_and_counters",
          "[rcc_bonded_pt][backend_state][cuda]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::backend::cuda;

    constexpr U64 stay_key    = 17;
    constexpr U64 release_key = 31;
    const Vector4i stay_topo{1, 2, 3, 4};
    const Vector4i release_topo{8, 5, 6, 7};

    RCCBondedPTState host;
    host.record_candidates(4);
    host.record_degenerate_rejected(1);
    host.record_filter_skipped(2);
    host.record_duplicate_suppressed(1);
    host.push_locked(RCCBondedPTEntry{release_key, release_topo, 0.625, 3});
    host.push_locked(RCCBondedPTEntry{stay_key, stay_topo, 0.875, 6});
    host.sort_by_key();
    REQUIRE(host.mark_released(release_key,
                               RCCBondedPTReleaseGap | RCCBondedPTReleaseSlip));

    RCCBondedPTStateBridge bridge;
    bridge.upload(host);

    CHECK(bridge.size() == 2);
    CHECK_FALSE(bridge.empty());
    CHECK(bridge.locked_keys().size() == 2);
    CHECK(bridge.locked_topos().size() == 2);
    CHECK(bridge.locked_beta().size() == 2);
    CHECK(bridge.locked_age().size() == 2);
    CHECK(bridge.release_flags().size() == 2);
    CHECK(bridge.counters().candidate_count == 4);
    CHECK(bridge.counters().locked_count == 2);

    auto roundtrip = bridge.download();
    REQUIRE(roundtrip.validate());
    REQUIRE(roundtrip.size() == 2);

    CHECK(roundtrip.locked_keys()[0] == stay_key);
    CHECK(same_topo(roundtrip.locked_topos()[0], stay_topo));
    CHECK(roundtrip.locked_beta()[0] == Catch::Approx(0.875));
    CHECK(roundtrip.locked_age()[0] == 6);
    CHECK(roundtrip.release_flags()[0] == RCCBondedPTReleaseNone);

    CHECK(roundtrip.locked_keys()[1] == release_key);
    CHECK(same_topo(roundtrip.locked_topos()[1], release_topo));
    CHECK(roundtrip.locked_beta()[1] == Catch::Approx(0.625));
    CHECK(roundtrip.locked_age()[1] == 3);
    CHECK((roundtrip.release_flags()[1] & RCCBondedPTReleaseGap) != 0);
    CHECK((roundtrip.release_flags()[1] & RCCBondedPTReleaseSlip) != 0);

    CHECK(roundtrip.counters().candidate_count == 4);
    CHECK(roundtrip.counters().locked_count == 2);
    CHECK(roundtrip.counters().released_count == 0);
    CHECK(roundtrip.counters().degenerate_rejected_count == 1);
    CHECK(roundtrip.counters().filter_skipped_count == 2);
    CHECK(roundtrip.counters().duplicate_suppressed_count == 1);

    auto released = roundtrip.extract_released();
    REQUIRE(released.size() == 1);
    REQUIRE(released[0].key == release_key);

    bridge.upload(roundtrip);
    auto after_extract = bridge.download();
    REQUIRE(after_extract.validate());
    REQUIRE(after_extract.size() == 1);
    CHECK(after_extract.locked_keys()[0] == stay_key);
    CHECK(after_extract.release_flags()[0] == RCCBondedPTReleaseNone);
    CHECK(after_extract.counters().candidate_count == 4);
    CHECK(after_extract.counters().locked_count == 1);
    CHECK(after_extract.counters().released_count == 1);
    CHECK(after_extract.counters().degenerate_rejected_count == 1);
    CHECK(after_extract.counters().filter_skipped_count == 2);
    CHECK(after_extract.counters().duplicate_suppressed_count == 1);

    bridge.clear();
    CHECK(bridge.empty());
    auto empty = bridge.download();
    CHECK(empty.empty());
    CHECK(empty.counters().locked_count == 0);
}
