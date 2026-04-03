#pragma once
#include <string_view>

namespace uipc::tensor_core_lab
{
enum class Mode
{
    Fp64RefNoTc,
    Fp32NoTc,
    Tc32Tf32
};

inline constexpr std::string_view to_string(Mode mode) noexcept
{
    switch(mode)
    {
        case Mode::Fp64RefNoTc:
            return "fp64_ref_no_tc";
        case Mode::Fp32NoTc:
            return "fp32_no_tc";
        case Mode::Tc32Tf32:
            return "tc32_tf32";
    }
    return "unknown";
}
}  // namespace uipc::tensor_core_lab
