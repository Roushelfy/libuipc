#include <sol/contact.h>

#include <sol/ordering.h>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace sol
{
namespace
{
using Clock = std::chrono::steady_clock;

std::size_t abs_diff(std::size_t a, std::size_t b)
{
    return a > b ? a - b : b - a;
}

std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> fields;
    std::string              current;
    std::istringstream       in(line);
    while(std::getline(in, current, ','))
    {
        const auto first = current.find_first_not_of(" \t\r\n");
        const auto last  = current.find_last_not_of(" \t\r\n");
        fields.push_back(first == std::string::npos ? std::string{} : current.substr(first, last - first + 1));
    }
    return fields;
}

bool is_supported_kind(std::string_view kind)
{
    return kind == "PP" || kind == "PE" || kind == "PT" || kind == "EE";
}

std::size_t expected_atom_count(std::string_view kind)
{
    if(kind == "PP")
        return 2;
    if(kind == "PE")
        return 3;
    if(kind == "PT" || kind == "EE")
        return 4;
    throw std::invalid_argument("unsupported contact kind; expected PP, PE, PT, or EE");
}

void validate_primitive(const ContactPrimitive& primitive, std::size_t atom_count)
{
    if(!is_supported_kind(primitive.kind))
        throw std::invalid_argument("unsupported contact kind; expected PP, PE, PT, or EE");
    if(primitive.atoms.size() != expected_atom_count(primitive.kind))
        throw std::invalid_argument("contact primitive atom count does not match its kind");
    if(primitive.stiffness <= 0.0 || !std::isfinite(primitive.stiffness))
        throw std::invalid_argument("contact primitive stiffness must be positive and finite");

    std::unordered_set<std::size_t> unique;
    unique.reserve(primitive.atoms.size());
    for(const std::size_t atom : primitive.atoms)
    {
        if(atom >= atom_count)
            throw std::invalid_argument("contact primitive references an out-of-range atom");
        unique.insert(atom);
    }
    if(unique.size() != primitive.atoms.size())
        throw std::invalid_argument("contact primitive must not repeat an atom");
}

std::vector<std::vector<std::size_t>> atoms_by_block(const OrderingResult& ordering)
{
    std::vector<std::vector<std::size_t>> blocks(ordering.block_to_atom_range.size());
    for(std::size_t chain = 0; chain < ordering.chain_to_old.size(); ++chain)
    {
        const std::size_t old   = ordering.chain_to_old[chain];
        const std::size_t block = ordering.atom_to_block.at(old);
        if(block >= blocks.size())
            throw std::invalid_argument("ordering has an out-of-range atom_to_block entry");
        blocks[block].push_back(old);
    }
    return blocks;
}

std::size_t pick_atom(const std::vector<std::vector<std::size_t>>& blocks,
                      std::size_t block,
                      std::size_t salt)
{
    if(block >= blocks.size() || blocks[block].empty())
        throw std::invalid_argument("cannot pick an atom from an empty block");
    const auto& atoms = blocks[block];
    return atoms[salt % atoms.size()];
}

std::size_t pick_unique_atom(const std::vector<std::vector<std::size_t>>& blocks,
                             const std::vector<std::size_t>& candidate_blocks,
                             std::size_t salt,
                             const std::vector<std::size_t>& used)
{
    for(const std::size_t block : candidate_blocks)
    {
        if(block >= blocks.size())
            continue;
        const auto& atoms = blocks[block];
        if(atoms.empty())
            continue;
        for(std::size_t probe = 0; probe < atoms.size(); ++probe)
        {
            const std::size_t atom = atoms[(salt + probe) % atoms.size()];
            if(std::find(used.begin(), used.end(), atom) == used.end())
                return atom;
        }
    }
    throw std::invalid_argument("cannot build a unique synthetic contact primitive from the selected blocks");
}

std::size_t far_block(std::size_t block_count, std::size_t block)
{
    if(block_count < 3)
        return block_count == 0 ? 0 : block_count - 1;
    return block < block_count / 2 ? block_count - 1 : 0;
}

ContactPrimitive make_near_primitive(const std::vector<std::vector<std::size_t>>& blocks,
                                     std::size_t i,
                                     std::string_view kind = "PE",
                                     double stiffness = 1.0)
{
    const std::size_t block_count = blocks.size();
    const std::size_t b0 = block_count <= 1 ? 0 : i % (block_count - 1);
    const std::size_t b1 = block_count <= 1 ? 0 : b0 + 1;

    ContactPrimitive primitive;
    primitive.kind      = std::string(kind);
    primitive.stiffness = stiffness;
    primitive.source    = "synthetic_near_band";
    const std::vector<std::size_t> near_blocks = {b0, b1};
    primitive.atoms.push_back(pick_unique_atom(blocks, near_blocks, i, primitive.atoms));
    primitive.atoms.push_back(pick_unique_atom(blocks, near_blocks, i + 1, primitive.atoms));
    if(kind == "PE" || kind == "PT" || kind == "EE")
        primitive.atoms.push_back(pick_unique_atom(blocks, near_blocks, i + 2, primitive.atoms));
    if(kind == "PT" || kind == "EE")
        primitive.atoms.push_back(pick_unique_atom(blocks, near_blocks, i + 3, primitive.atoms));
    return primitive;
}

ContactPrimitive make_off_primitive(const std::vector<std::vector<std::size_t>>& blocks,
                                    std::size_t i,
                                    std::string_view kind = "PP",
                                    double stiffness = 1.0)
{
    const std::size_t block_count = blocks.size();
    const std::size_t b0 = block_count <= 1 ? 0 : i % block_count;
    const std::size_t b1 = far_block(block_count, b0);

    ContactPrimitive primitive;
    primitive.kind      = std::string(kind);
    primitive.stiffness = stiffness;
    primitive.source    = "synthetic_off_band";
    primitive.atoms.push_back(pick_unique_atom(blocks, {b0}, i, primitive.atoms));
    primitive.atoms.push_back(pick_unique_atom(blocks, {b1}, i + 1, primitive.atoms));
    if(kind == "PE" || kind == "PT" || kind == "EE")
        primitive.atoms.push_back(pick_unique_atom(blocks, {b1, b0}, i + 2, primitive.atoms));
    if(kind == "PT" || kind == "EE")
        primitive.atoms.push_back(pick_unique_atom(blocks, {b0, b1}, i + 3, primitive.atoms));
    return primitive;
}

ContactPrimitive make_mixed_primitive(const std::vector<std::vector<std::size_t>>& blocks,
                                      std::size_t i,
                                      double stiffness = 1.0)
{
    const std::size_t block_count = blocks.size();
    const std::size_t b0 = block_count <= 1 ? 0 : i % block_count;
    const std::size_t b1 = block_count <= 1 ? 0 : std::min(b0 + 1, block_count - 1);
    const std::size_t b2 = far_block(block_count, b0);

    ContactPrimitive primitive;
    primitive.kind      = "PT";
    primitive.stiffness = stiffness;
    primitive.source    = "synthetic_mixed";
    primitive.atoms.push_back(pick_unique_atom(blocks, {b0}, i, primitive.atoms));
    primitive.atoms.push_back(pick_unique_atom(blocks, {b1, b0}, i + 1, primitive.atoms));
    primitive.atoms.push_back(pick_unique_atom(blocks, {b2}, i + 2, primitive.atoms));
    primitive.atoms.push_back(pick_unique_atom(blocks, {b1, b0, b2}, i + 3, primitive.atoms));
    return primitive;
}

double contribution_weight(const OrderingResult& ordering,
                           std::size_t atom_a,
                           std::size_t atom_b,
                           double stiffness)
{
    const double dofs_a = static_cast<double>(ordering.atom_dof_count.at(atom_a));
    const double dofs_b = static_cast<double>(ordering.atom_dof_count.at(atom_b));
    return stiffness * std::sqrt(dofs_a * dofs_b);
}

} // namespace

std::vector<ContactPrimitive> make_contact_scenario(const OrderingResult& ordering,
                                                    std::string_view scenario,
                                                    std::size_t contact_count)
{
    validate_permutation(ordering, ordering.chain_to_old.size());
    if(ordering.block_to_atom_range.empty())
        return {};

    const auto blocks = atoms_by_block(ordering);
    std::vector<ContactPrimitive> primitives;
    primitives.reserve(contact_count);

    if(scenario == "near_band")
    {
        for(std::size_t i = 0; i < contact_count; ++i)
            primitives.push_back(make_near_primitive(blocks, i, i % 3 == 0 ? "PT" : "PE"));
        return primitives;
    }

    if(scenario == "mixed")
    {
        for(std::size_t i = 0; i < contact_count; ++i)
        {
            if(i % 4 == 0)
                primitives.push_back(make_mixed_primitive(blocks, i, 2.0));
            else if(i % 4 == 1)
                primitives.push_back(make_off_primitive(blocks, i, "EE", 1.5));
            else
                primitives.push_back(make_near_primitive(blocks, i, "PE"));
        }
        return primitives;
    }

    if(scenario == "adversarial")
    {
        for(std::size_t i = 0; i < contact_count; ++i)
            primitives.push_back(make_off_primitive(blocks, i, i % 2 == 0 ? "EE" : "PT", i == 0 ? 1000.0 : 5.0));
        return primitives;
    }

    throw std::invalid_argument("unsupported contact scenario; expected near_band, mixed, adversarial, or from_file");
}

std::vector<ContactPrimitive> read_contact_primitives_csv(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if(!in)
        throw std::runtime_error(fmt::format("failed to open contact CSV '{}'", path.string()));

    std::vector<ContactPrimitive> primitives;
    std::string                   line;
    std::size_t                   line_number = 0;
    while(std::getline(in, line))
    {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n");
        if(first == std::string::npos || line[first] == '#')
            continue;

        auto fields = split_csv_line(line);
        if(fields.empty())
            continue;
        if(fields[0] == "kind")
            continue;
        if(fields.size() < 4)
            throw std::invalid_argument(fmt::format("contact CSV line {} has too few fields", line_number));

        ContactPrimitive primitive;
        primitive.kind      = fields[0];
        primitive.stiffness = std::stod(fields[1]);
        primitive.source    = "from_file";
        for(std::size_t i = 2; i < fields.size(); ++i)
        {
            if(!fields[i].empty())
                primitive.atoms.push_back(static_cast<std::size_t>(std::stoull(fields[i])));
        }
        primitives.push_back(std::move(primitive));
    }
    return primitives;
}

ContactClassificationReport classify_contacts(const OrderingResult& ordering,
                                               const std::vector<ContactPrimitive>& primitives,
                                               std::string_view scenario,
                                               double frame_boundary_reorder_time_ms)
{
    validate_permutation(ordering, ordering.chain_to_old.size());

    const auto begin = Clock::now();
    ContactClassificationReport report;
    report.scenario = std::string(scenario);
    report.primitives = primitives;
    report.active_contact_count = primitives.size();
    report.frame_boundary_reorder_time_ms = frame_boundary_reorder_time_ms;
    report.newton_iteration_reorder_count = 0;
    report.permutation_fixed_within_frame = true;

    for(std::size_t primitive_index = 0; primitive_index < primitives.size(); ++primitive_index)
    {
        const auto& primitive = primitives[primitive_index];
        validate_primitive(primitive, ordering.chain_to_old.size());

        std::size_t primitive_near = 0;
        std::size_t primitive_off  = 0;
        for(std::size_t i = 0; i < primitive.atoms.size(); ++i)
        {
            for(std::size_t j = i + 1; j < primitive.atoms.size(); ++j)
            {
                const std::size_t atom_a = primitive.atoms[i];
                const std::size_t atom_b = primitive.atoms[j];
                ContactContribution contribution;
                contribution.primitive = primitive_index;
                contribution.kind      = primitive.kind;
                contribution.atom_a    = std::min(atom_a, atom_b);
                contribution.atom_b    = std::max(atom_a, atom_b);
                contribution.chain_a   = ordering.old_to_chain.at(contribution.atom_a);
                contribution.chain_b   = ordering.old_to_chain.at(contribution.atom_b);
                contribution.block_a   = ordering.atom_to_block.at(contribution.atom_a);
                contribution.block_b   = ordering.atom_to_block.at(contribution.atom_b);
                contribution.stiffness = primitive.stiffness;
                contribution.weight =
                    contribution_weight(ordering, contribution.atom_a, contribution.atom_b, primitive.stiffness);
                contribution.near_band = abs_diff(contribution.block_a, contribution.block_b) <= 1;

                if(contribution.near_band)
                {
                    ++primitive_near;
                    ++report.near_band_contribution_count;
                    report.weighted_near_band_contribution_norm += contribution.weight;
                }
                else
                {
                    ++primitive_off;
                    ++report.off_band_contribution_count;
                    report.weighted_off_band_dropped_norm += contribution.weight;
                }
                report.contributions.push_back(std::move(contribution));
            }
        }

        if(primitive_off == 0)
            ++report.near_band_contact_count;
        else if(primitive_near == 0)
            ++report.off_band_contact_count;
        else
        {
            ++report.mixed_contact_count;
            ++report.off_band_contact_count;
        }
    }

    if(report.active_contact_count != 0)
    {
        report.near_band_ratio =
            static_cast<double>(report.near_band_contact_count) / report.active_contact_count;
        report.off_band_ratio =
            static_cast<double>(report.off_band_contact_count) / report.active_contact_count;
    }

    const std::size_t contribution_count =
        report.near_band_contribution_count + report.off_band_contribution_count;
    if(contribution_count != 0)
    {
        report.contribution_near_band_ratio =
            static_cast<double>(report.near_band_contribution_count) / contribution_count;
        report.contribution_off_band_ratio =
            static_cast<double>(report.off_band_contribution_count) / contribution_count;
    }

    const double total_weight =
        report.weighted_near_band_contribution_norm + report.weighted_off_band_dropped_norm;
    if(total_weight > 0.0)
    {
        report.weighted_near_band_ratio = report.weighted_near_band_contribution_norm / total_weight;
        report.weighted_off_band_ratio  = report.weighted_off_band_dropped_norm / total_weight;
    }

    report.estimated_absorbed_hessian_contribution_count = report.near_band_contribution_count;
    report.estimated_dropped_contribution_count = report.off_band_contribution_count;
    report.contact_classify_time_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    return report;
}

nlohmann::json to_json(const ContactPrimitive& primitive)
{
    return {{"kind", primitive.kind},
            {"atoms", primitive.atoms},
            {"stiffness", primitive.stiffness},
            {"source", primitive.source}};
}

nlohmann::json to_json(const ContactContribution& contribution)
{
    return {{"primitive", contribution.primitive},
            {"kind", contribution.kind},
            {"atom_a", contribution.atom_a},
            {"atom_b", contribution.atom_b},
            {"chain_a", contribution.chain_a},
            {"chain_b", contribution.chain_b},
            {"block_a", contribution.block_a},
            {"block_b", contribution.block_b},
            {"stiffness", contribution.stiffness},
            {"weight", contribution.weight},
            {"near_band", contribution.near_band}};
}

nlohmann::json to_json(const ContactClassificationReport& report)
{
    nlohmann::json primitives = nlohmann::json::array();
    for(const auto& primitive : report.primitives)
        primitives.push_back(to_json(primitive));

    nlohmann::json contributions = nlohmann::json::array();
    for(const auto& contribution : report.contributions)
        contributions.push_back(to_json(contribution));

    return {{"scenario", report.scenario},
            {"active_contact_count", report.active_contact_count},
            {"near_band_contact_count", report.near_band_contact_count},
            {"mixed_contact_count", report.mixed_contact_count},
            {"off_band_contact_count", report.off_band_contact_count},
            {"near_band_contribution_count", report.near_band_contribution_count},
            {"off_band_contribution_count", report.off_band_contribution_count},
            {"near_band_ratio", report.near_band_ratio},
            {"off_band_ratio", report.off_band_ratio},
            {"contribution_near_band_ratio", report.contribution_near_band_ratio},
            {"contribution_off_band_ratio", report.contribution_off_band_ratio},
            {"weighted_near_band_contribution_norm", report.weighted_near_band_contribution_norm},
            {"weighted_off_band_dropped_norm", report.weighted_off_band_dropped_norm},
            {"weighted_near_band_ratio", report.weighted_near_band_ratio},
            {"weighted_off_band_ratio", report.weighted_off_band_ratio},
            {"contact_classify_time_ms", report.contact_classify_time_ms},
            {"frame_boundary_reorder_time_ms", report.frame_boundary_reorder_time_ms},
            {"estimated_absorbed_hessian_contribution_count",
             report.estimated_absorbed_hessian_contribution_count},
            {"estimated_dropped_contribution_count", report.estimated_dropped_contribution_count},
            {"newton_iteration_reorder_count", report.newton_iteration_reorder_count},
            {"permutation_fixed_within_frame", report.permutation_fixed_within_frame},
            {"primitives", primitives},
            {"contributions", contributions}};
}

void write_contact_summary_csv(const std::filesystem::path& path,
                               const ContactClassificationReport& report)
{
    if(path.empty())
        return;

    std::ofstream out(path);
    if(!out)
        throw std::runtime_error(fmt::format("failed to open contact summary CSV '{}'", path.string()));

    out << "scenario,active_contact_count,near_band_contact_count,mixed_contact_count,"
           "off_band_contact_count,near_band_contribution_count,off_band_contribution_count,"
           "near_band_ratio,off_band_ratio,contribution_near_band_ratio,"
           "contribution_off_band_ratio,weighted_near_band_contribution_norm,"
           "weighted_off_band_dropped_norm,weighted_near_band_ratio,weighted_off_band_ratio,"
           "contact_classify_time_ms,frame_boundary_reorder_time_ms,"
           "estimated_absorbed_hessian_contribution_count,estimated_dropped_contribution_count,"
           "newton_iteration_reorder_count,permutation_fixed_within_frame\n";
    out << report.scenario << ','
        << report.active_contact_count << ','
        << report.near_band_contact_count << ','
        << report.mixed_contact_count << ','
        << report.off_band_contact_count << ','
        << report.near_band_contribution_count << ','
        << report.off_band_contribution_count << ','
        << report.near_band_ratio << ','
        << report.off_band_ratio << ','
        << report.contribution_near_band_ratio << ','
        << report.contribution_off_band_ratio << ','
        << report.weighted_near_band_contribution_norm << ','
        << report.weighted_off_band_dropped_norm << ','
        << report.weighted_near_band_ratio << ','
        << report.weighted_off_band_ratio << ','
        << report.contact_classify_time_ms << ','
        << report.frame_boundary_reorder_time_ms << ','
        << report.estimated_absorbed_hessian_contribution_count << ','
        << report.estimated_dropped_contribution_count << ','
        << report.newton_iteration_reorder_count << ','
        << (report.permutation_fixed_within_frame ? 1 : 0) << '\n';
}
} // namespace sol
