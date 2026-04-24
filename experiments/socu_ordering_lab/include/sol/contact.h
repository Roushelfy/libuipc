#pragma once

#include <sol/types.h>

#include <nlohmann/json_fwd.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace sol
{
struct ContactPrimitive
{
    std::string              kind{"PP"};
    std::vector<std::size_t> atoms;
    double                   stiffness{1.0};
    std::string              source{"synthetic"};
};

struct ContactContribution
{
    std::size_t primitive{};
    std::string kind;
    std::size_t atom_a{};
    std::size_t atom_b{};
    std::size_t chain_a{};
    std::size_t chain_b{};
    std::size_t block_a{};
    std::size_t block_b{};
    double      stiffness{};
    double      weight{};
    bool        near_band{};
};

struct ContactClassificationReport
{
    std::string                      scenario;
    std::vector<ContactPrimitive>    primitives;
    std::vector<ContactContribution> contributions;
    std::size_t                      active_contact_count{};
    std::size_t                      near_band_contact_count{};
    std::size_t                      mixed_contact_count{};
    std::size_t                      off_band_contact_count{};
    std::size_t                      near_band_contribution_count{};
    std::size_t                      off_band_contribution_count{};
    double                           near_band_ratio{};
    double                           off_band_ratio{};
    double                           contribution_near_band_ratio{};
    double                           contribution_off_band_ratio{};
    double                           weighted_near_band_contribution_norm{};
    double                           weighted_off_band_dropped_norm{};
    double                           weighted_near_band_ratio{};
    double                           weighted_off_band_ratio{};
    double                           contact_classify_time_ms{};
    double                           frame_boundary_reorder_time_ms{};
    std::size_t                      estimated_absorbed_hessian_contribution_count{};
    std::size_t                      estimated_dropped_contribution_count{};
    std::size_t                      newton_iteration_reorder_count{};
    bool                             permutation_fixed_within_frame{true};
};

std::vector<ContactPrimitive> make_contact_scenario(const OrderingResult& ordering,
                                                    std::string_view scenario,
                                                    std::size_t contact_count);

std::vector<ContactPrimitive> read_contact_primitives_csv(const std::filesystem::path& path);

ContactClassificationReport classify_contacts(const OrderingResult& ordering,
                                               const std::vector<ContactPrimitive>& primitives,
                                               std::string_view scenario,
                                               double frame_boundary_reorder_time_ms = 0.0);

nlohmann::json to_json(const ContactPrimitive& primitive);
nlohmann::json to_json(const ContactContribution& contribution);
nlohmann::json to_json(const ContactClassificationReport& report);

void write_contact_summary_csv(const std::filesystem::path& path,
                               const ContactClassificationReport& report);
} // namespace sol
