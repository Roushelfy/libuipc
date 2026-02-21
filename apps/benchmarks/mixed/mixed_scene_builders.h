#pragma once
#include <uipc/common/json.h>
#include <uipc/core/scene.h>
#include <string>
#include <string_view>

namespace uipc::bench::mixed
{
enum class MixedScenario
{
    AbdGravity,
    FemGravity,
    FemGroundContact,
    WreckingBall
};

using Stage1Scenario = MixedScenario;

struct MixedConfigOptions
{
    bool        telemetry_enabled            = false;
    bool        error_tracker_enabled        = false;
    std::string error_tracker_mode           = "offline";
    std::string error_tracker_reference_dir;
    bool        dump_linear_system           = false;
    bool        dump_solution_x              = false;
};

std::string_view scenario_name(MixedScenario scenario);
Json             make_mixed_config(MixedScenario scenario,
                                   const MixedConfigOptions& options);
void             populate_mixed_scene(MixedScenario scenario, core::Scene& scene);

inline Json make_stage1_config(Stage1Scenario scenario, bool telemetry_enabled)
{
    MixedConfigOptions options;
    options.telemetry_enabled = telemetry_enabled;
    return make_mixed_config(scenario, options);
}

inline void populate_stage1_scene(Stage1Scenario scenario, core::Scene& scene)
{
    populate_mixed_scene(scenario, scene);
}
}  // namespace uipc::bench::mixed
