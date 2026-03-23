#pragma once

namespace uipc::backend::cuda_mixed
{
enum class SimEngineState
{
    None = 0,
    BuildSystems,
    InitScene,
    RebuildScene,

    PredictMotion,            // Common
    ComputeDyTopoEffect,      // Common
    SolveGlobalLinearSystem,  // Common
    LineSearch,               // Common
    UpdateVelocity,           // Common

    AdvanceNonPenetrate,  // AL-IPC
    RecoverNonPenetrate,  // AL-IPC
};
}
