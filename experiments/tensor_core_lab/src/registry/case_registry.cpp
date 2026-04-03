#include <tcl/registry.h>

namespace uipc::tensor_core_lab
{
std::vector<int> smoke_batches(OpKind kind)
{
    switch(kind)
    {
        case OpKind::Mas48Factorize:
        case OpKind::Mas48Inverse:
        case OpKind::Mas48Solve:
            return {32};
        case OpKind::Abd12Factorize:
        case OpKind::Abd12Inverse:
        case OpKind::Abd12Solve:
            return {128};
        case OpKind::Fem12Assemble:
            return {256};
        case OpKind::Joint24Assemble:
            return {128};
    }
    return {};
}

std::vector<int> full_batches(OpKind kind)
{
    switch(kind)
    {
        case OpKind::Mas48Factorize:
        case OpKind::Mas48Inverse:
        case OpKind::Mas48Solve:
            return {256, 2048, 8192};
        case OpKind::Abd12Factorize:
        case OpKind::Abd12Inverse:
        case OpKind::Abd12Solve:
            return {4096, 32768, 131072};
        case OpKind::Fem12Assemble:
            return {8192, 65536, 262144};
        case OpKind::Joint24Assemble:
            return {4096, 32768, 131072};
    }
    return {};
}

std::vector<double> condition_scales()
{
    return {1.0e2, 1.0e4, 1.0e6};
}
}  // namespace uipc::tensor_core_lab
