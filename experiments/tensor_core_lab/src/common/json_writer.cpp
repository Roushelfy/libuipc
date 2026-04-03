#include "json_writer.h"

#include <fstream>

namespace uipc::tensor_core_lab
{
void write_simple_json(
    const std::filesystem::path&                          path,
    std::span<const std::pair<std::string, std::string>> entries)
{
    std::ofstream out(path);
    out << "{\n";
    for(size_t i = 0; i < entries.size(); ++i)
    {
        out << "  \"" << entries[i].first << "\": \"" << entries[i].second << "\"";
        out << (i + 1 == entries.size() ? "\n" : ",\n");
    }
    out << "}\n";
}
}  // namespace uipc::tensor_core_lab
