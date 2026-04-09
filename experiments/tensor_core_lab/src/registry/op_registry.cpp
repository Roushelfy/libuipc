#include <tcl/registry.h>

namespace uipc::tensor_core_lab
{
std::vector<OpKind> supported_ops()
{
    return {
        OpKind::Mas48Factorize,
        OpKind::Mas48Inverse,
        OpKind::Mas48Solve,
        OpKind::Abd12Assemble,
        OpKind::Abd12Factorize,
        OpKind::Abd12Inverse,
        OpKind::Abd12Solve,
        OpKind::Fem12Assemble,
        OpKind::Joint24Assemble,
    };
}
}  // namespace uipc::tensor_core_lab
