#pragma once
#include <vector>

#include <tcl/op_kind.h>

namespace uipc::tensor_core_lab
{
std::vector<OpKind> supported_ops();
std::vector<int>    smoke_batches(OpKind kind);
std::vector<int>    full_batches(OpKind kind);
std::vector<double> condition_scales();
}  // namespace uipc::tensor_core_lab
