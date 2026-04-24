#pragma once

#include <sol/types.h>

#include <nlohmann/json.hpp>

#include <filesystem>

namespace sol
{
nlohmann::json to_json(const AtomGraph& graph);
nlohmann::json to_json(const OrderingResult& ordering);
nlohmann::json to_json(const OrderingMetrics& metrics);
nlohmann::json to_json(const OrderingCandidate& candidate);
nlohmann::json to_json(const OrderingRun& run);

OrderingResult ordering_from_json(const nlohmann::json& json);

void write_json_report(const std::filesystem::path& path, const nlohmann::json& json);
void write_summary_csv(const std::filesystem::path& path, const OrderingRun& run);
void write_mapping_csv(const std::filesystem::path& path, const OrderingCandidate& candidate);
} // namespace sol
