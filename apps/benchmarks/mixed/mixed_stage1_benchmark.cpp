#include "mixed_bench_utils.h"
#include <benchmark/benchmark.h>
#include <uipc/common/logger.h>
#include <fmt/format.h>
#include <array>
#include <string>

namespace uipc::bench::mixed
{
static void BM_Stage1(benchmark::State& state, const Stage1RunSpec& spec)
{
    static const bool kLoggerReady = []()
    {
        logger::set_level(Logger::Level::warn);
        return true;
    }();
    benchmark::DoNotOptimize(kLoggerReady);

    for(auto _ : state)
    {
        auto result = run_stage1_case(spec);
        if(!result.ok)
        {
            state.SkipWithError(result.error.c_str());
            break;
        }
        benchmark::DoNotOptimize(result.timer_report_non_empty);
    }

    state.counters["frames"]    = spec.init_only ? 0 : spec.frames;
    state.counters["telemetry"] = spec.telemetry_enabled ? 1 : 0;
    state.counters["init_only"] = spec.init_only ? 1 : 0;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations())
                            * static_cast<int64_t>(spec.init_only ? 1 : spec.frames));
}

static void register_stage1_for_backend(std::string backend)
{
    constexpr std::array scenarios = {Stage1Scenario::AbdGravity,
                                      Stage1Scenario::FemGravity,
                                      Stage1Scenario::FemGroundContact};

    for(auto scenario : scenarios)
    {
        const auto scenario_str = std::string{scenario_name(scenario)};

        {
            Stage1RunSpec init_spec{
                .backend           = backend,
                .scenario          = scenario,
                .telemetry_enabled = false,
                .frames            = 0,
                .init_only         = true,
            };
            auto init_name = fmt::format("Mixed.Stage1.Init.{}.{}", scenario_str, backend);
            benchmark::RegisterBenchmark(init_name.c_str(),
                                         [init_spec](benchmark::State& state)
                                         { BM_Stage1(state, init_spec); });
        }

        {
            Stage1RunSpec run_spec{
                .backend           = backend,
                .scenario          = scenario,
                .telemetry_enabled = false,
                .frames            = 20,
                .init_only         = false,
            };
            auto run_name = fmt::format("Mixed.Stage1.Advance20F.TelemetryOff.{}.{}",
                                        scenario_str,
                                        backend);
            benchmark::RegisterBenchmark(run_name.c_str(),
                                         [run_spec](benchmark::State& state)
                                         { BM_Stage1(state, run_spec); });
        }

        {
            Stage1RunSpec run_spec{
                .backend           = backend,
                .scenario          = scenario,
                .telemetry_enabled = true,
                .frames            = 20,
                .init_only         = false,
            };
            auto run_name = fmt::format("Mixed.Stage1.Advance20F.TelemetryOn.{}.{}",
                                        scenario_str,
                                        backend);
            benchmark::RegisterBenchmark(run_name.c_str(),
                                         [run_spec](benchmark::State& state)
                                         { BM_Stage1(state, run_spec); });
        }
    }
}

const int kRegisterStage1 = []()
{
    register_stage1_for_backend("cuda_mixed");
    if(enable_cuda_baseline())
        register_stage1_for_backend("cuda");
    return 0;
}();
}  // namespace uipc::bench::mixed
