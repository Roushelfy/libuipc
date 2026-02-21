#pragma once
#include "mixed_scene_builders.h"
#include <string>
#include <string_view>

namespace uipc::bench::mixed
{
enum class MixedRunMode
{
    Perf,
    QualityReference,
    QualityCompare
};

struct MixedRunSpec
{
    std::string   backend;
    MixedScenario scenario             = MixedScenario::AbdGravity;
    bool          telemetry_enabled    = false;
    int           frames               = 20;
    bool          init_only            = false;
    MixedRunMode  run_mode             = MixedRunMode::Perf;
    std::string   suite_name           = "stage1";
    std::string   workspace_tag        = "default";
    bool          error_tracker_enable = false;
    std::string   error_reference_root;
    bool          dump_linear_system   = false;
    bool          dump_solution_x      = false;
};

struct MixedRunResult
{
    bool        ok                     = true;
    bool        timer_report_non_empty = false;
    bool        error_jsonl_non_empty  = false;
    std::string error;
    std::string workspace;
    std::string error_jsonl;
    std::string reference_dump_dir;
};

std::string      telemetry_mode_name(bool telemetry_enabled);
std::string_view run_mode_name(MixedRunMode mode);
MixedRunResult   run_mixed_case(const MixedRunSpec& spec);

struct Stage1RunSpec
{
    std::string    backend;
    Stage1Scenario scenario          = Stage1Scenario::AbdGravity;
    bool           telemetry_enabled = false;
    int            frames            = 20;
    bool           init_only         = false;
};

struct Stage1RunResult
{
    bool        ok                     = true;
    bool        timer_report_non_empty = false;
    std::string error;
    std::string workspace;
    std::string error_jsonl;
};

std::string     env_or_default(const char* key, std::string_view fallback);
bool            enable_cuda_baseline();
Stage1RunResult run_stage1_case(const Stage1RunSpec& spec);
}  // namespace uipc::bench::mixed
