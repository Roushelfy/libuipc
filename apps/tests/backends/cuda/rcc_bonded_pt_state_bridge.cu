#include <muda/ext/eigen/eigen_core_cxx20.h>

#include <app/app.h>
#include <contact_system/rcc_bonded_pt_state_bridge.h>

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
    const Matrix3x3 stay_dm_inv = diag3(1.0, 1.5, 2.0);
    const Matrix3x3 release_dm_inv = diag3(2.5, 3.0, 3.5);

    RCCBondedPTState host;
    host.record_candidates(4);
    host.record_degenerate_rejected(1);
    host.record_filter_skipped(2);
    host.record_duplicate_suppressed(1);
    host.push_locked(RCCBondedPTEntry{release_key,
                                      release_topo,
                                      0.625,
                                      3,
                                      RCCBondedPTReleaseNone,
                                      release_dm_inv,
                                      0.75});
    host.push_locked(RCCBondedPTEntry{
        stay_key, stay_topo, 0.875, 6, RCCBondedPTReleaseNone, stay_dm_inv, 0.5});
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
    CHECK(bridge.locked_dm_inv().size() == 2);
    CHECK(bridge.locked_rest_volume().size() == 2);
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
    CHECK(roundtrip.locked_dm_inv()[0].isApprox(stay_dm_inv));
    CHECK(roundtrip.locked_rest_volume()[0] == Catch::Approx(0.5));

    CHECK(roundtrip.locked_keys()[1] == release_key);
    CHECK(same_topo(roundtrip.locked_topos()[1], release_topo));
    CHECK(roundtrip.locked_beta()[1] == Catch::Approx(0.625));
    CHECK(roundtrip.locked_age()[1] == 3);
    CHECK((roundtrip.release_flags()[1] & RCCBondedPTReleaseGap) != 0);
    CHECK((roundtrip.release_flags()[1] & RCCBondedPTReleaseSlip) != 0);
    CHECK(roundtrip.locked_dm_inv()[1].isApprox(release_dm_inv));
    CHECK(roundtrip.locked_rest_volume()[1] == Catch::Approx(0.75));

    CHECK(roundtrip.counters().candidate_count == 4);
    CHECK(roundtrip.counters().locked_count == 2);
    CHECK(roundtrip.counters().released_count == 0);
    CHECK(roundtrip.counters().degenerate_rejected_count == 1);
    CHECK(roundtrip.counters().filter_skipped_count == 2);
    CHECK(roundtrip.counters().duplicate_suppressed_count == 1);

    auto released = roundtrip.extract_released();
    REQUIRE(released.size() == 1);
    REQUIRE(released[0].key == release_key);
    CHECK(released[0].Dm_inv.isApprox(release_dm_inv));
    CHECK(released[0].rest_volume == Catch::Approx(0.75));

    bridge.upload(roundtrip);
    auto after_extract = bridge.download();
    REQUIRE(after_extract.validate());
    REQUIRE(after_extract.size() == 1);
    CHECK(after_extract.locked_keys()[0] == stay_key);
    CHECK(after_extract.release_flags()[0] == RCCBondedPTReleaseNone);
    CHECK(after_extract.locked_dm_inv()[0].isApprox(stay_dm_inv));
    CHECK(after_extract.locked_rest_volume()[0] == Catch::Approx(0.5));
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
