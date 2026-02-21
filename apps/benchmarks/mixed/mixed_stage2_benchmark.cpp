#include "mixed_bench_utils.h"
#include <benchmark/benchmark.h>
#include <uipc/common/logger.h>
#include <fmt/format.h>
#include <array>
#include <string>

namespace uipc::bench::mixed
{
static void BM_Stage2(benchmark::State& state, const MixedRunSpec& spec)
{
    static const bool kLoggerReady = []()
    {
        logger::set_level(Logger::Level::warn);
        return true;
    }();
    benchmark::DoNotOptimize(kLoggerReady);

    for(auto _ : state)
    {
        auto result = run_mixed_case(spec);
        if(!result.ok)
        {
            state.SkipWithError(result.error.c_str());
            break;
        }
        benchmark::DoNotOptimize(result.timer_report_non_empty);
        benchmark::DoNotOptimize(result.error_jsonl_non_empty);
    }

    state.counters["frames"]    = spec.init_only ? 0 : spec.frames;
    state.counters["telemetry"] = spec.telemetry_enabled ? 1 : 0;
    state.counters["run_mode"]  = static_cast<int>(spec.run_mode);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations())
                            * static_cast<int64_t>(spec.init_only ? 1 : spec.frames));
}

static void register_stage2_perf_for_backend(const std::string& backend)
{
    const auto workspace_tag = env_or_default("UIPC_BENCH_WORKSPACE_TAG", "default");

    {
        MixedRunSpec spec{
            .backend           = backend,
            .scenario          = MixedScenario::WreckingBall,
            .telemetry_enabled = false,
            .frames            = 100,
            .init_only         = false,
            .run_mode          = MixedRunMode::Perf,
            .suite_name        = "stage2",
            .workspace_tag     = workspace_tag,
        };
        auto name = fmt::format("Mixed.Stage2.Perf.Advance100F.TelemetryOff.{}.{}",
                                scenario_name(spec.scenario),
                                backend);
        benchmark::RegisterBenchmark(name.c_str(),
                                     [spec](benchmark::State& state)
                                     { BM_Stage2(state, spec); });
    }

    {
        MixedRunSpec spec{
            .backend           = backend,
            .scenario          = MixedScenario::WreckingBall,
            .telemetry_enabled = true,
            .frames            = 100,
            .init_only         = false,
            .run_mode          = MixedRunMode::Perf,
            .suite_name        = "stage2",
            .workspace_tag     = workspace_tag,
        };
        auto name = fmt::format("Mixed.Stage2.Perf.Advance100F.TelemetryOn.{}.{}",
                                scenario_name(spec.scenario),
                                backend);
        benchmark::RegisterBenchmark(name.c_str(),
                                     [spec](benchmark::State& state)
                                     { BM_Stage2(state, spec); });
    }
}

static void register_stage2_quality_for_mixed()
{
    const auto workspace_tag = env_or_default("UIPC_BENCH_WORKSPACE_TAG", "default");
    const auto reference_root =
        env_or_default("UIPC_BENCH_ERROR_REFERENCE_ROOT", "");

    constexpr std::array scenarios = {MixedScenario::WreckingBall,
                                      MixedScenario::FemGroundContact};

    for(auto scenario : scenarios)
    {
        {
            MixedRunSpec spec{
                .backend           = "cuda_mixed",
                .scenario          = scenario,
                .telemetry_enabled = false,
                .frames            = 30,
                .init_only         = false,
                .run_mode          = MixedRunMode::QualityReference,
                .suite_name        = "stage2",
                .workspace_tag     = workspace_tag,
                .dump_solution_x   = true,
            };
            auto name = fmt::format("Mixed.Stage2.Quality.Reference30F.{}.cuda_mixed",
                                    scenario_name(scenario));
            benchmark::RegisterBenchmark(name.c_str(),
                                         [spec](benchmark::State& state)
                                         { BM_Stage2(state, spec); });
        }

        {
            MixedRunSpec spec{
                .backend              = "cuda_mixed",
                .scenario             = scenario,
                .telemetry_enabled    = true,
                .frames               = 30,
                .init_only            = false,
                .run_mode             = MixedRunMode::QualityCompare,
                .suite_name           = "stage2",
                .workspace_tag        = workspace_tag,
                .error_tracker_enable = true,
                .error_reference_root = reference_root,
            };
            auto name = fmt::format("Mixed.Stage2.Quality.Compare30F.{}.cuda_mixed",
                                    scenario_name(scenario));
            benchmark::RegisterBenchmark(name.c_str(),
                                         [spec](benchmark::State& state)
                                         { BM_Stage2(state, spec); });
        }
    }
}

const int kRegisterStage2 = []()
{
    register_stage2_perf_for_backend("cuda_mixed");
    register_stage2_quality_for_mixed();

    if(enable_cuda_baseline())
        register_stage2_perf_for_backend("cuda");

    return 0;
}();
}  // namespace uipc::bench::mixed
