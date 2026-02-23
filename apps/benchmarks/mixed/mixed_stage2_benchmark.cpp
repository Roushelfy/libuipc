#include "mixed_bench_utils.h"
#include <benchmark/benchmark.h>
#include <uipc/common/logger.h>
#include <fmt/format.h>
#include <array>
#include <string>
#include <utility>
#include <cstdlib>

namespace uipc::bench::mixed
{
static int env_int_or_default(const char* key, int fallback)
{
    const char* v = std::getenv(key);
    if(!v || *v == '\0')
        return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if(end == v || *end != '\0' || parsed <= 0)
        return fallback;
    return static_cast<int>(parsed);
}

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
    const int  frame_scale   = env_int_or_default("UIPC_BENCH_STAGE2_FRAME_SCALE", 1);

    const auto register_perf_case = [&](MixedScenario scenario, int base_frames)
    {
        const int frames = base_frames * frame_scale;
        for(bool telemetry_on : {false, true})
        {
            MixedRunSpec spec{
                .backend           = backend,
                .scenario          = scenario,
                .telemetry_enabled = telemetry_on,
                .frames            = frames,
                .init_only         = false,
                .run_mode          = MixedRunMode::Perf,
                .suite_name        = "stage2",
                .workspace_tag     = workspace_tag,
            };
            auto name = fmt::format("Mixed.Stage2.Perf.Advance{}F.{}.{}.{}",
                                    frames,
                                    telemetry_on ? "TelemetryOn" : "TelemetryOff",
                                    scenario_name(spec.scenario),
                                    backend);
            benchmark::RegisterBenchmark(name.c_str(),
                                         [spec](benchmark::State& state)
                                         { BM_Stage2(state, spec); });
        }
    };

    register_perf_case(MixedScenario::WreckingBall, 100);
    register_perf_case(MixedScenario::FemHeavyNoContact, 80);
    register_perf_case(MixedScenario::FemHeavyGroundContact, 80);
}

static void register_stage2_quality_for_mixed()
{
    const auto workspace_tag = env_or_default("UIPC_BENCH_WORKSPACE_TAG", "default");
    const auto reference_root =
        env_or_default("UIPC_BENCH_ERROR_REFERENCE_ROOT", "");
    const int frame_scale = env_int_or_default("UIPC_BENCH_STAGE2_FRAME_SCALE", 1);

    constexpr std::array base_scenario_frames = {
        std::pair{MixedScenario::WreckingBall, 30},
        std::pair{MixedScenario::FemGroundContact, 30},
        std::pair{MixedScenario::FemHeavyNoContact, 20},
        std::pair{MixedScenario::FemHeavyGroundContact, 20}};

    for(const auto& [scenario, base_frames] : base_scenario_frames)
    {
        const int frames = base_frames * frame_scale;
        {
            MixedRunSpec spec{
                .backend           = "cuda_mixed",
                .scenario          = scenario,
                .telemetry_enabled = false,
                .frames            = frames,
                .init_only         = false,
                .run_mode          = MixedRunMode::QualityReference,
                .suite_name        = "stage2",
                .workspace_tag     = workspace_tag,
                .dump_solution_x   = true,
            };
            auto name = fmt::format("Mixed.Stage2.Quality.Reference{}F.{}.cuda_mixed",
                                    frames,
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
                .frames               = frames,
                .init_only            = false,
                .run_mode             = MixedRunMode::QualityCompare,
                .suite_name           = "stage2",
                .workspace_tag        = workspace_tag,
                .error_tracker_enable = true,
                .error_reference_root = reference_root,
            };
            auto name = fmt::format("Mixed.Stage2.Quality.Compare{}F.{}.cuda_mixed",
                                    frames,
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
