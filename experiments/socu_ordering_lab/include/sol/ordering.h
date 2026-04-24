#pragma once

#include <sol/types.h>

#include <string_view>
#include <vector>

namespace sol
{
std::vector<std::size_t> resolve_block_sizes(std::string_view value);
std::vector<std::string> resolve_orderers(std::string_view value);

void validate_permutation(const OrderingResult& ordering, std::size_t atom_count);

OrderingCandidate make_ordering_candidate(const AtomGraph& graph,
                                           std::string_view orderer,
                                           std::size_t block_size);

OrderingRun run_ordering(const AtomGraph& graph,
                         std::string_view orderer,
                         std::string_view block_size);
} // namespace sol
