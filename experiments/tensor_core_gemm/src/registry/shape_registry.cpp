#include <tcg/registry.h>

#include <cstdint>

namespace uipc::tensor_core_gemm
{
namespace
{
std::vector<RegisteredGemmShape> make_shapes()
{
    return {
        {"3x3x3", "uipc_common_square", {3, 3, 3}},
        {"6x6x6", "uipc_common_square", {6, 6, 6}},
        {"9x9x9", "uipc_common_square", {9, 9, 9}},
        {"12x12x12", "uipc_common_square", {12, 12, 12}},
        {"24x24x24", "uipc_common_square", {24, 24, 24}},
        {"48x48x48", "uipc_common_square", {48, 48, 48}},
        {"3x3x12", "uipc_common_rect", {3, 3, 12}},
        {"12x12x3", "uipc_common_rect", {12, 12, 3}},
        {"9x12x9", "uipc_common_rect", {9, 12, 9}},
        {"12x12x9", "uipc_common_rect", {12, 12, 9}},
        {"3x24x3", "uipc_common_rect", {3, 24, 3}},
        {"24x24x3", "uipc_common_rect", {24, 24, 3}},
        {"9x24x9", "uipc_common_rect", {9, 24, 9}},
        {"24x24x9", "uipc_common_rect", {24, 24, 9}},
        {"16x16x16", "friendly_square", {16, 16, 16}},
        {"32x32x32", "friendly_square", {32, 32, 32}},
        {"64x64x64", "friendly_square", {64, 64, 64}},
        {"96x96x96", "friendly_square", {96, 96, 96}},
        {"128x128x128", "friendly_square", {128, 128, 128}},
        {"5x5x5", "awkward_square", {5, 5, 5}},
        {"7x7x7", "awkward_square", {7, 7, 7}},
        {"15x15x15", "awkward_square", {15, 15, 15}},
        {"20x20x20", "awkward_square", {20, 20, 20}},
        {"40x40x40", "awkward_square", {40, 40, 40}},
        {"16x32x16", "friendly_rect", {16, 32, 16}},
        {"32x32x16", "friendly_rect", {32, 32, 16}},
        {"32x64x32", "friendly_rect", {32, 64, 32}},
        {"64x128x64", "friendly_rect", {64, 128, 64}},
        {"5x3x7", "awkward_rect", {5, 3, 7}},
        {"7x5x15", "awkward_rect", {7, 5, 15}},
        {"15x7x20", "awkward_rect", {15, 7, 20}},
        {"20x15x40", "awkward_rect", {20, 15, 40}},
    };
}

std::uint64_t logical_flops(const RegisteredGemmShape& shape)
{
    return 2ull * static_cast<std::uint64_t>(shape.logical_shape.m)
           * static_cast<std::uint64_t>(shape.logical_shape.n)
           * static_cast<std::uint64_t>(shape.logical_shape.k);
}
}  // namespace

std::vector<RegisteredGemmShape> all_registered_shapes()
{
    return make_shapes();
}

std::vector<RegisteredGemmShape> uipc_common_shapes()
{
    std::vector<RegisteredGemmShape> out;
    for(const auto& shape : make_shapes())
    {
        if(shape.shape_group == "uipc_common_square" || shape.shape_group == "uipc_common_rect")
            out.push_back(shape);
    }
    return out;
}

std::vector<RegisteredGemmShape> representative_extra_shapes()
{
    return {
        {"16x16x16", "friendly_square", {16, 16, 16}},
        {"5x5x5", "awkward_square", {5, 5, 5}},
        {"16x32x16", "friendly_rect", {16, 32, 16}},
        {"15x7x20", "awkward_rect", {15, 7, 20}},
    };
}

std::vector<int> smoke_batches(const RegisteredGemmShape& shape)
{
    const auto flops = logical_flops(shape);
    if(flops <= 4096ull)
        return {16384};
    if(flops <= 65536ull)
        return {4096};
    return {1024};
}

std::vector<int> full_batches(const RegisteredGemmShape& shape)
{
    const auto flops = logical_flops(shape);
    if(flops <= 4096ull)
        return {16384, 65536, 262144};
    if(flops <= 65536ull)
        return {4096, 16384, 65536};
    return {1024, 4096, 16384};
}
}  // namespace uipc::tensor_core_gemm
