#pragma once

namespace uipc::backend::cuda_mixed
{
enum class SimEngineState
{
    None = 0,
    BuildSystems,
    InitScene,
    RebuildScene,
    PredictMotion,
    ComputeDyTopoEffect,
    // ComputeGradientHessian,
    SolveGlobalLinearSystem,
    LineSearch,
    UpdateVelocity,
};
}

