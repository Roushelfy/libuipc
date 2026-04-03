#pragma once
#include <string_view>

namespace uipc::tensor_core_lab
{
enum class OpKind
{
    Mas48Factorize,
    Mas48Inverse,
    Mas48Solve,
    Abd12Factorize,
    Abd12Inverse,
    Abd12Solve,
    Fem12Assemble,
    Joint24Assemble
};

inline constexpr std::string_view to_string(OpKind kind) noexcept
{
    switch(kind)
    {
        case OpKind::Mas48Factorize:
            return "mas48_factorize";
        case OpKind::Mas48Inverse:
            return "mas48_inverse";
        case OpKind::Mas48Solve:
            return "mas48_solve";
        case OpKind::Abd12Factorize:
            return "abd12_factorize";
        case OpKind::Abd12Inverse:
            return "abd12_inverse";
        case OpKind::Abd12Solve:
            return "abd12_solve";
        case OpKind::Fem12Assemble:
            return "fem12_assemble";
        case OpKind::Joint24Assemble:
            return "joint24_assemble";
    }
    return "unknown";
}
}  // namespace uipc::tensor_core_lab
