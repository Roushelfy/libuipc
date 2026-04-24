#pragma once

#include <sol/types.h>

#include <string_view>

namespace sol
{
AtomGraph make_rod(std::size_t count = 96);
AtomGraph make_cloth_grid(std::size_t width = 16, std::size_t height = 12);
AtomGraph make_tet_block(std::size_t nx = 5, std::size_t ny = 4, std::size_t nz = 3);
AtomGraph make_shuffled_cloth_grid();
AtomGraph make_shuffled_tet_block();
AtomGraph make_preset(std::string_view name);
} // namespace sol
