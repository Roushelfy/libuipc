#include <catch2/catch_all.hpp>
#include <uipc/core/rcc_bonded_pt_state.h>

namespace
{
bool same_topo(const uipc::Vector4i& lhs, const uipc::Vector4i& rhs)
{
    return (lhs.array() == rhs.array()).all();
}
}  // namespace

TEST_CASE("rcc_bonded_pt_state_keeps_payloads_zipped",
          "[rcc_bonded_pt][state]")
{
    using namespace uipc;
    using namespace uipc::core;

    constexpr U64 stay_key    = 11;
    constexpr U64 release_key = 7;
    const Vector4i stay_topo{10, 20, 21, 22};
    const Vector4i release_topo{30, 40, 41, 42};

    RCCBondedPTState state;
    state.push_locked(RCCBondedPTEntry{stay_key, stay_topo, 0.875, 5});
    state.push_locked(RCCBondedPTEntry{release_key, release_topo, 0.625, 2});

    REQUIRE(state.validate());
    state.sort_by_key();

    REQUIRE(state.size() == 2);
    REQUIRE(state.validate());
    CHECK(state.counters().locked_count == 2);
    CHECK(state.locked_keys()[0] == release_key);
    CHECK(same_topo(state.locked_topos()[0], release_topo));
    CHECK(state.locked_beta()[0] == Catch::Approx(0.625));
    CHECK(state.locked_age()[0] == 2);
    CHECK(state.release_flags()[0] == RCCBondedPTReleaseNone);

    CHECK(state.locked_keys()[1] == stay_key);
    CHECK(same_topo(state.locked_topos()[1], stay_topo));
    CHECK(state.locked_beta()[1] == Catch::Approx(0.875));
    CHECK(state.locked_age()[1] == 5);
    CHECK(state.release_flags()[1] == RCCBondedPTReleaseNone);

    CHECK(state.mark_released(release_key,
                              RCCBondedPTReleaseGap | RCCBondedPTReleaseSlip));
    CHECK_FALSE(state.mark_released(404, RCCBondedPTReleasePolicy));

    auto released = state.extract_released();

    REQUIRE(released.size() == 1);
    CHECK(released[0].key == release_key);
    CHECK(same_topo(released[0].topo, release_topo));
    CHECK(released[0].beta == Catch::Approx(0.625));
    CHECK(released[0].age == 2);
    CHECK((released[0].release_flags & RCCBondedPTReleaseGap) != 0);
    CHECK((released[0].release_flags & RCCBondedPTReleaseSlip) != 0);

    REQUIRE(state.size() == 1);
    REQUIRE(state.validate());
    CHECK(state.counters().locked_count == 1);
    CHECK(state.counters().released_count == 1);
    CHECK(state.locked_keys()[0] == stay_key);
    CHECK(same_topo(state.locked_topos()[0], stay_topo));
    CHECK(state.locked_beta()[0] == Catch::Approx(0.875));
    CHECK(state.locked_age()[0] == 5);
    CHECK(state.release_flags()[0] == RCCBondedPTReleaseNone);
}

TEST_CASE("rcc_bonded_pt_state_counters_are_reportable",
          "[rcc_bonded_pt][state][counters]")
{
    using namespace uipc;
    using namespace uipc::core;

    RCCBondedPTState state;
    state.record_candidates(5);
    state.record_degenerate_rejected(2);
    state.record_filter_skipped(3);
    state.record_duplicate_suppressed(1);
    state.push_locked(RCCBondedPTEntry{17, Vector4i{1, 2, 3, 4}, 0.9, 6});
    state.push_locked(RCCBondedPTEntry{23, Vector4i{5, 6, 7, 8}, 0.8, 4});

    CHECK(state.counters().candidate_count == 5);
    CHECK(state.counters().locked_count == 2);
    CHECK(state.counters().released_count == 0);
    CHECK(state.counters().degenerate_rejected_count == 2);
    CHECK(state.counters().filter_skipped_count == 3);
    CHECK(state.counters().duplicate_suppressed_count == 1);

    REQUIRE(state.mark_released(17, RCCBondedPTReleasePolicy));
    auto released = state.extract_released();
    REQUIRE(released.size() == 1);

    CHECK(state.counters().locked_count == 1);
    CHECK(state.counters().released_count == 1);

    state.clear_counters();
    CHECK(state.counters().candidate_count == 0);
    CHECK(state.counters().locked_count == 1);
    CHECK(state.counters().released_count == 0);
    CHECK(state.counters().degenerate_rejected_count == 0);
    CHECK(state.counters().filter_skipped_count == 0);
    CHECK(state.counters().duplicate_suppressed_count == 0);

    RCCBondedPTCounters restored;
    restored.candidate_count             = 8;
    restored.locked_count                = 999;
    restored.released_count              = 5;
    restored.degenerate_rejected_count   = 4;
    restored.filter_skipped_count        = 3;
    restored.duplicate_suppressed_count = 2;
    state.set_counters(restored);

    CHECK(state.counters().candidate_count == 8);
    CHECK(state.counters().locked_count == 1);
    CHECK(state.counters().released_count == 5);
    CHECK(state.counters().degenerate_rejected_count == 4);
    CHECK(state.counters().filter_skipped_count == 3);
    CHECK(state.counters().duplicate_suppressed_count == 2);
}
