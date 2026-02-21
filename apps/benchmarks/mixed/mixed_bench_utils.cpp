#include "mixed_bench_utils.h"
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/common/timer.h>
#include <fmt/format.h>
#include <filesystem>
#include <exception>
#include <cstdlib>
#include <cstring>
#include <cctype>

namespace uipc::bench::mixed
{
namespace
{
std::string to_lower_copy(std::string value)
{
    for(auto& c : value)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

bool env_flag(const char* key)
{
    const char* v = std::getenv(key);
    if(!v)
        return false;
    auto value = to_lower_copy(v);
    return value == "1" || value == "on" || value == "true" || value == "yes";
}

std::filesystem::path find_recursive_first(const std::filesystem::path& root,
                                           const std::string& pattern_prefix,
                                           const std::string& pattern_suffix)
{
    namespace fs = std::filesystem;
    if(!fs::exists(root))
        return {};

    for(const auto& entry : fs::recursive_directory_iterator(root))
    {
        if(!entry.is_regular_file())
            continue;
        auto name = entry.path().filename().string();
        if(name.rfind(pattern_prefix, 0) == 0 && name.size() >= pattern_suffix.size()
           && name.substr(name.size() - pattern_suffix.size()) == pattern_suffix)
            return entry.path();
    }

    return {};
}

std::filesystem::path find_recursive_exact(const std::filesystem::path& root,
                                           const std::string& filename)
{
    namespace fs = std::filesystem;
    if(!fs::exists(root))
        return {};

    for(const auto& entry : fs::recursive_directory_iterator(root))
    {
        if(entry.is_regular_file() && entry.path().filename().string() == filename)
            return entry.path();
    }

    return {};
}

std::string make_workspace(const MixedRunSpec& spec)
{
    namespace fs = std::filesystem;

    const char* workspace_root_env = std::getenv("UIPC_BENCH_WORKSPACE_ROOT");
    fs::path    workspace_root = (workspace_root_env && std::strlen(workspace_root_env) > 0) ?
                                     fs::path{workspace_root_env} :
                                     fs::path{AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE)};

    fs::path workspace = workspace_root / spec.suite_name / spec.backend
                         / std::string{scenario_name(spec.scenario)}
                         / std::string{run_mode_name(spec.run_mode)}
                         / telemetry_mode_name(spec.telemetry_enabled)
                         / (spec.workspace_tag.empty() ? "default" : spec.workspace_tag);

    fs::create_directories(workspace);
    return (workspace / "").string();
}

std::string resolve_reference_dir(const std::string& reference_root, MixedScenario scenario)
{
    namespace fs = std::filesystem;
    if(reference_root.empty())
        return {};

    fs::path root = fs::path{reference_root};
    if(!fs::exists(root))
        return {};

    const auto scenario_token = std::string{scenario_name(scenario)};
    for(const auto& entry : fs::recursive_directory_iterator(root))
    {
        if(!entry.is_regular_file())
            continue;
        const auto filename = entry.path().filename().string();
        if(filename.rfind("x.", 0) != 0
           || filename.substr(filename.size() >= 4 ? filename.size() - 4 : 0) != ".mtx")
            continue;

        const auto parent_str = entry.path().parent_path().string();
        if(parent_str.find(scenario_token) != std::string::npos)
            return (entry.path().parent_path() / "").string();
    }

    auto direct_probe = find_recursive_first(root, "x.", ".mtx");
    if(!direct_probe.empty())
        return (direct_probe.parent_path() / "").string();

    return {};
}
}  // namespace

std::string env_or_default(const char* key, std::string_view fallback)
{
    const char* value = std::getenv(key);
    if(value && std::strlen(value) > 0)
        return std::string{value};
    return std::string{fallback};
}

bool enable_cuda_baseline()
{
    return env_flag("UIPC_BENCH_ENABLE_CUDA_BASELINE");
}

std::string telemetry_mode_name(bool telemetry_enabled)
{
    return telemetry_enabled ? "TelemetryOn" : "TelemetryOff";
}

std::string_view run_mode_name(MixedRunMode mode)
{
    switch(mode)
    {
        case MixedRunMode::Perf:
            return "Perf";
        case MixedRunMode::QualityReference:
            return "QualityReference";
        case MixedRunMode::QualityCompare:
            return "QualityCompare";
        default:
            return "Unknown";
    }
}

MixedRunResult run_mixed_case(const MixedRunSpec& spec)
{
    using namespace uipc;
    using namespace uipc::core;
    namespace fs = std::filesystem;

    MixedRunResult result;

    struct TimerGuard
    {
        ~TimerGuard() { Timer::disable_all(); }
    } guard;

    try
    {
        if(spec.telemetry_enabled)
            Timer::enable_all();
        else
            Timer::disable_all();

        result.workspace = make_workspace(spec);

        MixedConfigOptions options;
        options.telemetry_enabled     = spec.telemetry_enabled;
        options.error_tracker_enabled = spec.error_tracker_enable;
        options.dump_linear_system    = spec.dump_linear_system;
        options.dump_solution_x       = spec.dump_solution_x;
        options.dump_surface          = spec.dump_surface || env_flag("UIPC_BENCH_DUMP_SURFACE");

        if(spec.run_mode == MixedRunMode::QualityReference)
        {
            options.dump_solution_x = true;
        }
        else if(spec.run_mode == MixedRunMode::QualityCompare)
        {
            options.telemetry_enabled     = true;  // backend currently gates error tracker with telemetry.enable
            options.error_tracker_enabled = true;
            options.error_tracker_mode    = "offline";
            auto reference_dir            = resolve_reference_dir(spec.error_reference_root,
                                                       spec.scenario);
            if(reference_dir.empty())
            {
                result.ok = false;
                result.error =
                    fmt::format("missing reference dump dir for compare mode: {}",
                                spec.error_reference_root);
                return result;
            }
            options.error_tracker_reference_dir = reference_dir;
        }

        auto config = make_mixed_config(spec.scenario, options);

        Engine engine{spec.backend, result.workspace};
        World  world{engine};
        Scene  scene{config};

        populate_mixed_scene(spec.scenario, scene);
        world.init(scene);

        if(!world.is_valid())
        {
            result.ok = false;
            result.error =
                fmt::format("world.init failed: {} / {}", spec.backend, scenario_name(spec.scenario));
            return result;
        }

        if(!spec.init_only)
        {
            for(int i = 0; i < spec.frames; ++i)
            {
                world.advance();
                if(!world.is_valid())
                {
                    result.ok = false;
                    result.error = fmt::format("world.advance failed at frame {}: {} / {}",
                                               i,
                                               spec.backend,
                                               scenario_name(spec.scenario));
                    return result;
                }
                world.retrieve();
            }
        }

        if(spec.telemetry_enabled)
        {
            auto timer_json = Timer::report_as_json();
            result.timer_report_non_empty = !timer_json.empty();
            if(!result.timer_report_non_empty)
            {
                result.ok = false;
                result.error = fmt::format("empty timer report: {} / {}",
                                           spec.backend,
                                           scenario_name(spec.scenario));
                return result;
            }
        }

        if(spec.run_mode == MixedRunMode::QualityReference)
        {
            auto x_dump = find_recursive_first(result.workspace, "x.", ".mtx");
            if(x_dump.empty())
            {
                result.ok = false;
                result.error = fmt::format("reference dump x.*.mtx not found in workspace: {}",
                                           result.workspace);
                return result;
            }
            result.reference_dump_dir = (x_dump.parent_path() / "").string();
        }
        else if(spec.run_mode == MixedRunMode::QualityCompare)
        {
            auto error_jsonl = find_recursive_exact(result.workspace, "error.jsonl");
            if(error_jsonl.empty())
            {
                result.ok = false;
                result.error = fmt::format("error.jsonl not found in workspace: {}",
                                           result.workspace);
                return result;
            }
            result.error_jsonl = error_jsonl.string();
            result.error_jsonl_non_empty =
                fs::exists(error_jsonl) && fs::is_regular_file(error_jsonl)
                && fs::file_size(error_jsonl) > 0;
            if(!result.error_jsonl_non_empty)
            {
                result.ok = false;
                result.error = fmt::format("error.jsonl is empty: {}", error_jsonl.string());
                return result;
            }
        }
    }
    catch(const std::exception& e)
    {
        result.ok    = false;
        result.error = fmt::format("exception: {}", e.what());
    }
    catch(...)
    {
        result.ok    = false;
        result.error = "unknown exception";
    }

    return result;
}

Stage1RunResult run_stage1_case(const Stage1RunSpec& spec)
{
    MixedRunSpec run_spec;
    run_spec.backend           = spec.backend;
    run_spec.scenario          = spec.scenario;
    run_spec.telemetry_enabled = spec.telemetry_enabled;
    run_spec.frames            = spec.frames;
    run_spec.init_only         = spec.init_only;
    run_spec.run_mode          = MixedRunMode::Perf;
    run_spec.suite_name        = "stage1";
    run_spec.workspace_tag     = env_or_default("UIPC_BENCH_WORKSPACE_TAG", "default");

    auto mixed_result = run_mixed_case(run_spec);

    Stage1RunResult result;
    result.ok                     = mixed_result.ok;
    result.timer_report_non_empty = mixed_result.timer_report_non_empty;
    result.error                  = mixed_result.error;
    result.workspace              = mixed_result.workspace;
    result.error_jsonl            = mixed_result.error_jsonl;
    return result;
}
}  // namespace uipc::bench::mixed
