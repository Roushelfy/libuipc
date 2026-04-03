#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <utility>

namespace uipc::tensor_core_lab
{
void write_simple_json(
    const std::filesystem::path&                          path,
    std::span<const std::pair<std::string, std::string>> entries);
}  // namespace uipc::tensor_core_lab
