#pragma once

#include <string>
#include <vector>

namespace uipc::tensor_core_gemm
{
enum class GemmLayoutVariant
{
    Raw,
    Padded
};

struct GemmShape
{
    int m = 0;
    int n = 0;
    int k = 0;
};

struct GemmCaseSpec
{
    std::string shape_tag;
    std::string shape_group;
    GemmLayoutVariant layout_variant = GemmLayoutVariant::Raw;
    int batch_count = 0;
    int seed = 0;
    int m = 0;
    int n = 0;
    int k = 0;
    int physical_m = 0;
    int physical_n = 0;
    int physical_k = 0;
};

struct GemmCaseData
{
    GemmCaseSpec        spec;
    std::vector<double> a_fp64;
    std::vector<double> b_fp64;
    std::vector<double> reference_fp64;
};

constexpr const char* to_string(GemmLayoutVariant variant) noexcept
{
    switch(variant)
    {
        case GemmLayoutVariant::Raw:
            return "raw";
        case GemmLayoutVariant::Padded:
            return "padded";
    }
    return "raw";
}

int round_up_to_multiple_of_16(int value) noexcept;

GemmCaseData make_gemm_case(const GemmShape&      shape,
                            const std::string&    shape_group,
                            GemmLayoutVariant     layout_variant,
                            int                   batch_count,
                            int                   seed);
}  // namespace uipc::tensor_core_gemm
