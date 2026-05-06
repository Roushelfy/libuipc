#pragma once

namespace uipc::backend::cuda_mixed
{
enum class NewtonAssemblyMode
{
    FullSparse,
    GradientOnly,
    GradientStructuredHessian,
    FullSparseAndStructuredHessian,
};

constexpr const char* newton_assembly_mode_name(NewtonAssemblyMode mode) noexcept
{
    switch(mode)
    {
        case NewtonAssemblyMode::FullSparse:
            return "FullSparse";
        case NewtonAssemblyMode::GradientOnly:
            return "GradientOnly";
        case NewtonAssemblyMode::GradientStructuredHessian:
            return "GradientStructuredHessian";
        case NewtonAssemblyMode::FullSparseAndStructuredHessian:
            return "FullSparseAndStructuredHessian";
        default:
            return "Unknown";
    }
}
}  // namespace uipc::backend::cuda_mixed
