#pragma once

#include <linear_system/socu_approx_report.h>
#include <linear_system/structured_chain_provider.h>

#include <uipc/common/json.h>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace uipc::backend
{
class WorldVisitor;
}

namespace uipc::backend::cuda_mixed::socu_approx
{
const Json* selected_candidate_json(const Json& report);
const Json* ordering_json(const Json& candidate);
const Json* metrics_json(const Json& report, const Json& candidate);
bool has_basic_ordering_schema(const Json& ordering);

std::filesystem::path absolute_workspace_path(std::string_view workspace,
                                               const std::string& path);
std::filesystem::path default_generated_ordering_report_path(
    std::string_view workspace);

Json generate_abd_init_time_ordering_report(SizeT            body_count,
                                            std::string_view orderer,
                                            std::string_view block_size);
Json generate_fem_init_time_ordering_report(backend::WorldVisitor& world,
                                            std::string_view       orderer,
                                            std::string_view       block_size);
Json generate_mixed_abd_fem_init_time_ordering_report(
    backend::WorldVisitor& world,
    SizeT                  body_count,
    std::string_view       orderer,
    std::string_view       block_size);
void write_json_report(const std::filesystem::path& path, const Json& report);

SizeT count_abd_bodies_from_scene(backend::WorldVisitor& world);
SizeT fem_vertex_count_from_scene(backend::WorldVisitor& world);

bool parse_atom_dof_count(const Json& ordering,
                          SizeT&      dof_count,
                          std::string& detail);
bool parse_block_layouts(const Json&                         ordering,
                         std::vector<SocuApproxBlockLayout>& blocks,
                         std::string&                       detail);
bool validate_atom_inverse_mapping(const Json& ordering, std::string& detail);
bool build_ordering_provider(const Json&                         ordering,
                             SizeT                               block_size,
                             const std::vector<SocuApproxBlockLayout>& blocks,
                             std::unique_ptr<StructuredChainProvider>& provider,
                             SizeT&                              padding_slot_count,
                             std::string&                        detail);
bool validate_dof_coverage(span<const StructuredDofSlot> slots,
                           SizeT                         dof_count,
                           SizeT                         chain_scalar_count,
                           std::vector<IndexT>&          old_to_chain,
                           std::vector<IndexT>&          chain_to_old,
                           StructuredQualityReport&      quality,
                           std::string&                  detail);
}  // namespace uipc::backend::cuda_mixed::socu_approx
