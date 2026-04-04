#pragma once

#include <string>
#include <vector>

#include <tcg/gemm_case.h>

namespace uipc::tensor_core_gemm
{
struct RegisteredGemmShape
{
    std::string shape_tag;
    std::string shape_group;
    GemmShape   logical_shape;
};

std::vector<RegisteredGemmShape> all_registered_shapes();
std::vector<RegisteredGemmShape> uipc_common_shapes();
std::vector<RegisteredGemmShape> representative_extra_shapes();
std::vector<int>                 smoke_batches(const RegisteredGemmShape& shape);
std::vector<int>                 full_batches(const RegisteredGemmShape& shape);
}  // namespace uipc::tensor_core_gemm
