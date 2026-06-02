#include <catch2/catch_all.hpp>
#include <uipc/core/rcc_bonded_pt_state_accessor_feature.h>

namespace
{
class FakeRCCBondedPTStateAccessor final
    : public uipc::core::RCCBondedPTStateAccessorFeatureOverrider
{
  public:
    explicit FakeRCCBondedPTStateAccessor(uipc::core::RCCBondedPTState state)
        : m_state{std::move(state)}
    {
    }

    uipc::SizeT get_locked_pair_count() override
    {
        ++query_count;
        return m_state.size();
    }

    uipc::core::RCCBondedPTCounters get_counters() override
    {
        ++query_count;
        return m_state.counters();
    }

    uipc::core::RCCBondedPTState do_dump_state() override
    {
        ++query_count;
        return m_state;
    }

    uipc::SizeT query_count = 0;

  private:
    uipc::core::RCCBondedPTState m_state;
};
}  // namespace

TEST_CASE("rcc_bonded_pt_state_accessor_feature_reports_live_contract",
          "[rcc_bonded_pt][accessor][state]")
{
    using namespace uipc;
    using namespace uipc::core;

    RCCBondedPTState state;
    state.record_candidates(3);
    state.record_filter_skipped(2);
    state.push_locked(
        RCCBondedPTEntry{11, Vector4i{1, 2, 3, 4}, 0.9, 7});
    state.sort_by_key();

    auto overrider = std::make_shared<FakeRCCBondedPTStateAccessor>(state);
    RCCBondedPTStateAccessorFeature feature{overrider};

    CHECK(feature.name() == RCCBondedPTStateAccessorFeature::FeatureName);
    CHECK(feature.locked_pair_count() == 1);
    const auto counters = feature.counters();
    CHECK(counters.candidate_count == 3);
    CHECK(counters.filter_skipped_count == 2);

    auto snapshot = feature.dump_state();
    REQUIRE(snapshot.size() == 1);
    CHECK(snapshot.locked_keys()[0] == 11);
    CHECK(overrider->query_count == 3);
}
