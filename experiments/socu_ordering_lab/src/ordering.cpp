#include <sol/ordering.h>

#include <sol/graph.h>

#include <metis.h>
#if SOL_HAS_CUSOLVER_RCM
#include <cusolverSp.h>
#include <cusparse.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace sol
{
namespace
{
using Clock = std::chrono::steady_clock;

struct CandidateBuild
{
    std::vector<std::size_t> chain_to_old;
    std::string             fallback_reason;
};

std::size_t abs_diff(std::size_t a, std::size_t b)
{
    return a > b ? a - b : b - a;
}

std::vector<std::size_t> original_order(std::size_t n)
{
    std::vector<std::size_t> out(n);
    std::iota(out.begin(), out.end(), 0);
    return out;
}

std::vector<std::size_t> rcm_order_from_adjacency(const std::vector<std::vector<std::size_t>>& adj)
{
    const std::size_t n = adj.size();
    std::vector<std::size_t> order;
    std::vector<char>        seen(n, 0);
    order.reserve(n);

    auto degree_less = [&adj](std::size_t a, std::size_t b)
    {
        const auto da = adj[a].size();
        const auto db = adj[b].size();
        return da == db ? a < b : da < db;
    };

    while(order.size() < n)
    {
        std::size_t start = 0;
        while(start < n && seen[start])
            ++start;
        if(start == n)
            break;
        for(std::size_t i = start + 1; i < n; ++i)
            if(!seen[i] && degree_less(i, start))
                start = i;

        std::queue<std::size_t> queue;
        queue.push(start);
        seen[start] = 1;
        while(!queue.empty())
        {
            const std::size_t u = queue.front();
            queue.pop();
            order.push_back(u);

            auto neighbors = adj[u];
            std::sort(neighbors.begin(), neighbors.end(), degree_less);
            for(const std::size_t v : neighbors)
            {
                if(v < n && !seen[v])
                {
                    seen[v] = 1;
                    queue.push(v);
                }
            }
        }
    }
    std::reverse(order.begin(), order.end());
    return order;
}

std::vector<std::size_t> rcm_subset_order(const AtomGraph& graph,
                                          const std::vector<std::size_t>& subset)
{
    if(subset.empty())
        return {};

    std::unordered_map<std::size_t, std::size_t> old_to_local;
    old_to_local.reserve(subset.size());
    for(std::size_t i = 0; i < subset.size(); ++i)
        old_to_local[subset[i]] = i;

    std::vector<std::set<std::size_t>> local_adj_set(subset.size());
    for(const auto& edge : graph.edges)
    {
        auto a = old_to_local.find(edge.a);
        auto b = old_to_local.find(edge.b);
        if(a == old_to_local.end() || b == old_to_local.end())
            continue;
        local_adj_set[a->second].insert(b->second);
        local_adj_set[b->second].insert(a->second);
    }

    std::vector<std::vector<std::size_t>> local_adj(subset.size());
    for(std::size_t i = 0; i < local_adj_set.size(); ++i)
        local_adj[i] = std::vector<std::size_t>(local_adj_set[i].begin(), local_adj_set[i].end());

    auto local_order = rcm_order_from_adjacency(local_adj);
    std::vector<std::size_t> out;
    out.reserve(subset.size());
    for(const std::size_t local : local_order)
        out.push_back(subset[local]);
    return out;
}

void build_metis_csr(const AtomGraph& graph,
                     std::vector<idx_t>& xadj,
                     std::vector<idx_t>& adjncy,
                     std::vector<idx_t>* vwgt)
{
    const auto adj = adjacency(graph);
    xadj.resize(adj.size() + 1);
    adjncy.clear();
    if(vwgt)
    {
        vwgt->resize(graph.atoms.size());
        for(std::size_t i = 0; i < graph.atoms.size(); ++i)
            (*vwgt)[i] = static_cast<idx_t>(graph.atoms[i].dof_count);
    }

    xadj[0] = 0;
    for(std::size_t i = 0; i < adj.size(); ++i)
    {
        for(const std::size_t neighbor : adj[i])
            adjncy.push_back(static_cast<idx_t>(neighbor));
        xadj[i + 1] = static_cast<idx_t>(adjncy.size());
    }
}

std::vector<std::vector<std::size_t>> checked_adjacency_for_csr(const AtomGraph& graph)
{
    if(graph.atoms.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("CSR orderer only supports graphs with <= INT_MAX atoms");
    auto adj = adjacency(graph);
    std::size_t nnz = 0;
    for(const auto& row : adj)
        nnz += row.size();
    if(nnz > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("CSR orderer only supports graphs with <= INT_MAX adjacency entries");
    return adj;
}

void build_int_csr(const AtomGraph& graph,
                   std::vector<int>& row_ptr,
                   std::vector<int>& col_ind)
{
    const auto adj = checked_adjacency_for_csr(graph);
    row_ptr.resize(adj.size() + 1);
    col_ind.clear();
    row_ptr[0] = 0;
    for(std::size_t row = 0; row < adj.size(); ++row)
    {
        for(const std::size_t col : adj[row])
            col_ind.push_back(static_cast<int>(col));
        row_ptr[row + 1] = static_cast<int>(col_ind.size());
    }
}

#if SOL_HAS_CUSOLVER_RCM
const char* cusolver_status_name(cusolverStatus_t status)
{
    switch(status)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:
        return "CUSOLVER_STATUS_UNKNOWN";
    }
}

const char* cusparse_status_name(cusparseStatus_t status)
{
    switch(status)
    {
    case CUSPARSE_STATUS_SUCCESS:
        return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
        return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:
        return "CUSPARSE_STATUS_UNKNOWN";
    }
}

struct CusolverSpHandle
{
    cusolverSpHandle_t handle{};

    CusolverSpHandle()
    {
        const auto status = cusolverSpCreate(&handle);
        if(status != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error(std::string("cusolverSpCreate failed: ")
                                     + cusolver_status_name(status));
    }

    ~CusolverSpHandle()
    {
        if(handle)
            cusolverSpDestroy(handle);
    }

    CusolverSpHandle(const CusolverSpHandle&)            = delete;
    CusolverSpHandle& operator=(const CusolverSpHandle&) = delete;
};

struct CusparseMatDescr
{
    cusparseMatDescr_t descr{};

    CusparseMatDescr()
    {
        auto status = cusparseCreateMatDescr(&descr);
        if(status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(std::string("cusparseCreateMatDescr failed: ")
                                     + cusparse_status_name(status));
        status = cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        if(status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(std::string("cusparseSetMatType failed: ")
                                     + cusparse_status_name(status));
        status = cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        if(status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(std::string("cusparseSetMatIndexBase failed: ")
                                     + cusparse_status_name(status));
    }

    ~CusparseMatDescr()
    {
        if(descr)
            cusparseDestroyMatDescr(descr);
    }

    CusparseMatDescr(const CusparseMatDescr&)            = delete;
    CusparseMatDescr& operator=(const CusparseMatDescr&) = delete;
};

struct NvidiaSymRcmContext
{
    CusolverSpHandle solver;
    CusparseMatDescr descr;
};

NvidiaSymRcmContext& nvidia_symrcm_context()
{
    static NvidiaSymRcmContext context;
    return context;
}

CandidateBuild nvidia_symrcm_order(const AtomGraph& graph)
{
    if(graph.atoms.empty())
        return {};
    if(graph.edges.empty())
        throw std::runtime_error("nvidia_symrcm requires at least one graph edge");

    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    build_int_csr(graph, row_ptr, col_ind);

    auto& context = nvidia_symrcm_context();
    std::vector<int> p(graph.atoms.size(), 0);

    const auto status = cusolverSpXcsrsymrcmHost(context.solver.handle,
                                                 static_cast<int>(graph.atoms.size()),
                                                 static_cast<int>(col_ind.size()),
                                                 context.descr.descr,
                                                 row_ptr.data(),
                                                 col_ind.data(),
                                                 p.data());
    if(status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cusolverSpXcsrsymrcmHost failed: ")
                                 + cusolver_status_name(status));

    CandidateBuild build;
    build.chain_to_old.reserve(p.size());
    for(const int old : p)
    {
        if(old < 0)
            throw std::runtime_error("nvidia_symrcm returned a negative permutation entry");
        build.chain_to_old.push_back(static_cast<std::size_t>(old));
    }
    return build;
}
#else
CandidateBuild nvidia_symrcm_order(const AtomGraph&)
{
    throw std::runtime_error("nvidia_symrcm unavailable: CUDAToolkit cusolver/cusparse were not found");
}
#endif

CandidateBuild metis_nd_order(const AtomGraph& graph)
{
    if(graph.atoms.empty())
        return {};
    if(graph.edges.empty())
        throw std::runtime_error("METIS_NodeND requires at least one graph edge");

    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
    std::vector<idx_t> vwgt;
    build_metis_csr(graph, xadj, adjncy, &vwgt);

    idx_t n = static_cast<idx_t>(graph.atoms.size());
    std::vector<idx_t> perm(n, 0);
    std::vector<idx_t> iperm(n, 0);

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_SEED]      = 0;

    const int ret = METIS_NodeND(&n, xadj.data(), adjncy.data(), vwgt.data(), options, perm.data(), iperm.data());
    if(ret != METIS_OK)
        throw std::runtime_error("METIS_NodeND failed with code " + std::to_string(ret));

    CandidateBuild build;
    build.chain_to_old.reserve(graph.atoms.size());
    for(const idx_t value : perm)
    {
        if(value < 0)
            throw std::runtime_error("METIS_NodeND returned a negative permutation entry");
        build.chain_to_old.push_back(static_cast<std::size_t>(value));
    }

    std::vector<std::size_t> old_to_chain(build.chain_to_old.size());
    for(std::size_t chain = 0; chain < build.chain_to_old.size(); ++chain)
        old_to_chain[build.chain_to_old[chain]] = chain;
    for(std::size_t old = 0; old < old_to_chain.size(); ++old)
    {
        if(iperm[old] < 0 || static_cast<std::size_t>(iperm[old]) != old_to_chain[old])
            throw std::runtime_error("METIS_NodeND perm/iperm contract did not match expected chain_to_old semantics");
    }
    return build;
}

std::vector<std::size_t> quotient_rcm_order(std::size_t part_count,
                                            const std::vector<std::set<std::size_t>>& quotient_edges)
{
    std::vector<std::vector<std::size_t>> adj(part_count);
    for(std::size_t i = 0; i < part_count; ++i)
        adj[i] = std::vector<std::size_t>(quotient_edges[i].begin(), quotient_edges[i].end());
    return rcm_order_from_adjacency(adj);
}

CandidateBuild metis_kway_rcm_order(const AtomGraph& graph, std::size_t block_size)
{
    if(graph.atoms.empty())
        return {};
    if(graph.edges.empty())
        throw std::runtime_error("METIS_PartGraphKway requires at least one graph edge");

    const std::size_t total_dofs = std::accumulate(graph.atoms.begin(),
                                                   graph.atoms.end(),
                                                   std::size_t{0},
                                                   [](std::size_t sum, const Atom& atom)
                                                   {
                                                       return sum + atom.dof_count;
                                                   });
    idx_t part_count = static_cast<idx_t>((total_dofs + block_size - 1) / block_size);
    if(part_count <= 1)
    {
        return CandidateBuild{rcm_order_from_adjacency(adjacency(graph)),
                              "single-part graph; used rcm within the only partition"};
    }

    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
    std::vector<idx_t> vwgt;
    build_metis_csr(graph, xadj, adjncy, &vwgt);

    idx_t n_vertices = static_cast<idx_t>(graph.atoms.size());
    idx_t n_weights  = 1;
    idx_t edge_cut   = 0;
    std::vector<idx_t> part(graph.atoms.size(), 0);

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_SEED]      = 0;

    const int ret = METIS_PartGraphKway(&n_vertices,
                                        &n_weights,
                                        xadj.data(),
                                        adjncy.data(),
                                        vwgt.data(),
                                        nullptr,
                                        nullptr,
                                        &part_count,
                                        nullptr,
                                        nullptr,
                                        options,
                                        &edge_cut,
                                        part.data());
    if(ret != METIS_OK)
        throw std::runtime_error("METIS_PartGraphKway failed with code " + std::to_string(ret));

    std::vector<std::vector<std::size_t>> parts(static_cast<std::size_t>(part_count));
    for(std::size_t atom = 0; atom < part.size(); ++atom)
    {
        if(part[atom] < 0 || part[atom] >= part_count)
            throw std::runtime_error("METIS_PartGraphKway returned an out-of-range partition id");
        parts[static_cast<std::size_t>(part[atom])].push_back(atom);
    }

    std::vector<std::set<std::size_t>> quotient_edges(static_cast<std::size_t>(part_count));
    for(const auto& edge : graph.edges)
    {
        const std::size_t pa = static_cast<std::size_t>(part[edge.a]);
        const std::size_t pb = static_cast<std::size_t>(part[edge.b]);
        if(pa == pb)
            continue;
        quotient_edges[pa].insert(pb);
        quotient_edges[pb].insert(pa);
    }

    CandidateBuild build;
    for(const std::size_t part_id : quotient_rcm_order(parts.size(), quotient_edges))
    {
        auto local = rcm_subset_order(graph, parts[part_id]);
        build.chain_to_old.insert(build.chain_to_old.end(), local.begin(), local.end());
    }
    return build;
}

OrderingResult finalize_ordering(const AtomGraph& graph,
                                 std::string_view orderer,
                                 std::size_t block_size,
                                 std::vector<std::size_t> chain_to_old)
{
    OrderingResult result;
    result.orderer      = std::string(orderer);
    result.block_size   = block_size;
    result.chain_to_old = std::move(chain_to_old);
    result.old_to_chain.assign(result.chain_to_old.size(), 0);

    std::vector<char> seen(result.chain_to_old.size(), 0);
    for(std::size_t chain = 0; chain < result.chain_to_old.size(); ++chain)
    {
        const std::size_t old = result.chain_to_old[chain];
        if(old >= result.chain_to_old.size())
            throw std::runtime_error("ordering contains an out-of-range atom id");
        if(seen[old])
            throw std::runtime_error("ordering contains a duplicate atom id");
        seen[old] = 1;
        result.old_to_chain[old] = chain;
    }

    result.atom_to_block.assign(graph.atoms.size(), 0);
    result.atom_block_offset.assign(graph.atoms.size(), 0);
    result.atom_dof_count.assign(graph.atoms.size(), 0);

    std::size_t chain_begin = 0;
    std::size_t block_dofs  = 0;
    std::size_t block       = 0;
    for(std::size_t chain = 0; chain < result.chain_to_old.size(); ++chain)
    {
        const std::size_t old  = result.chain_to_old[chain];
        const std::size_t dofs = graph.atoms[old].dof_count;
        if(dofs > block_size)
            throw std::runtime_error("atom dof_count exceeds block_size");
        if(block_dofs != 0 && block_dofs + dofs > block_size)
        {
            result.block_to_atom_range.push_back(BlockRange{block, chain_begin, chain, block_dofs});
            ++block;
            chain_begin = chain;
            block_dofs  = 0;
        }

        result.atom_to_block[old]     = block;
        result.atom_block_offset[old] = block_dofs;
        result.atom_dof_count[old]    = dofs;
        block_dofs += dofs;
    }

    if(!result.chain_to_old.empty())
        result.block_to_atom_range.push_back(
            BlockRange{block, chain_begin, result.chain_to_old.size(), block_dofs});

    validate_permutation(result, graph.atoms.size());
    return result;
}

OrderingMetrics score_ordering(const AtomGraph& graph,
                               const OrderingResult& ordering,
                               double ordering_time_ms)
{
    OrderingMetrics metrics;
    metrics.total_atoms       = graph.atoms.size();
    metrics.block_size        = ordering.block_size;
    metrics.block_count       = ordering.block_to_atom_range.size();
    metrics.ordering_time_ms  = ordering_time_ms;
    metrics.valid_permutation = true;

    for(const auto& atom : graph.atoms)
        metrics.total_dofs += atom.dof_count;

    if(!ordering.block_to_atom_range.empty())
    {
        double utilization_sum = 0.0;
        metrics.min_block_utilization = std::numeric_limits<double>::max();
        for(const auto& block : ordering.block_to_atom_range)
        {
            const double utilization = static_cast<double>(block.dof_count) / ordering.block_size;
            utilization_sum += utilization;
            metrics.min_block_utilization = std::min(metrics.min_block_utilization, utilization);
        }
        metrics.avg_block_utilization =
            utilization_sum / static_cast<double>(ordering.block_to_atom_range.size());
    }

    double block_distance_weight_sum = 0.0;
    for(const auto& edge : graph.edges)
    {
        const std::size_t block_a = ordering.atom_to_block[edge.a];
        const std::size_t block_b = ordering.atom_to_block[edge.b];
        const std::size_t dist    = abs_diff(block_a, block_b);
        metrics.max_block_distance = std::max(metrics.max_block_distance, dist);
        block_distance_weight_sum += static_cast<double>(dist) * edge.weight;
        metrics.edge_chain_span_sum += abs_diff(ordering.old_to_chain[edge.a], ordering.old_to_chain[edge.b]);

        if(dist <= 1)
        {
            ++metrics.near_band_edge_count;
            metrics.near_band_edge_weight += edge.weight;
        }
        else
        {
            ++metrics.off_band_edge_count;
            metrics.off_band_edge_weight += edge.weight;
        }
    }

    const std::size_t edge_count = metrics.near_band_edge_count + metrics.off_band_edge_count;
    const double      edge_weight = metrics.near_band_edge_weight + metrics.off_band_edge_weight;
    if(edge_count != 0)
    {
        metrics.near_band_ratio = static_cast<double>(metrics.near_band_edge_count) / edge_count;
        metrics.off_band_ratio  = static_cast<double>(metrics.off_band_edge_count) / edge_count;
    }
    if(edge_weight > 0.0)
    {
        metrics.weighted_near_band_ratio = metrics.near_band_edge_weight / edge_weight;
        metrics.weighted_off_band_ratio  = metrics.off_band_edge_weight / edge_weight;
        metrics.avg_block_distance       = block_distance_weight_sum / edge_weight;
    }
    return metrics;
}

double candidate_score(const OrderingMetrics& metrics)
{
    const double normalized_max_distance =
        metrics.block_count > 1
            ? static_cast<double>(metrics.max_block_distance) / static_cast<double>(metrics.block_count - 1)
            : 0.0;
    const double normalized_time = metrics.ordering_time_ms / (metrics.ordering_time_ms + 1.0);
    return 4.0 * metrics.weighted_near_band_ratio
           - 4.0 * metrics.weighted_off_band_ratio
           - 0.5 * normalized_max_distance
           + 1.0 * metrics.avg_block_utilization
           - 0.1 * normalized_time;
}

CandidateBuild build_candidate_order(const AtomGraph& graph,
                                     std::string_view orderer,
                                     std::size_t block_size)
{
    if(orderer == "original")
        return CandidateBuild{original_order(graph.atoms.size()), ""};
    if(orderer == "rcm")
        return CandidateBuild{rcm_order_from_adjacency(adjacency(graph)), ""};
    if(orderer == "metis_nd")
        return metis_nd_order(graph);
    if(orderer == "metis_kway_rcm")
        return metis_kway_rcm_order(graph, block_size);
    if(orderer == "nvidia_symrcm")
        return nvidia_symrcm_order(graph);
    throw std::invalid_argument("unknown orderer: " + std::string(orderer));
}
} // namespace

std::vector<std::size_t> resolve_block_sizes(std::string_view value)
{
    if(value == "32")
        return {32};
    if(value == "64")
        return {64};
    if(value == "auto")
        return {32, 64};
    throw std::invalid_argument("block-size must be 32, 64, or auto");
}

std::vector<std::string> resolve_orderers(std::string_view value)
{
    if(value == "auto_stable")
        return {"original", "rcm", "metis_kway_rcm"};
    if(value == "auto" || value == "all" || value == "auto_exhaustive")
        return {"original", "rcm", "nvidia_symrcm", "metis_nd", "metis_kway_rcm"};
    if(value == "original" || value == "rcm" || value == "nvidia_symrcm"
       || value == "metis_nd" || value == "metis_kway_rcm")
        return {std::string(value)};
    throw std::invalid_argument(
        "orderer must be original, rcm, nvidia_symrcm, metis_nd, metis_kway_rcm, auto_stable, auto_exhaustive, auto, or all");
}

void validate_permutation(const OrderingResult& ordering, std::size_t atom_count)
{
    if(ordering.block_size != 32 && ordering.block_size != 64)
        throw std::invalid_argument("ordering block_size must be 32 or 64");
    if(ordering.chain_to_old.size() != atom_count || ordering.old_to_chain.size() != atom_count
       || ordering.atom_to_block.size() != atom_count || ordering.atom_block_offset.size() != atom_count
       || ordering.atom_dof_count.size() != atom_count)
        throw std::invalid_argument("ordering mapping sizes must match atom count");

    std::vector<char> seen(atom_count, 0);
    for(std::size_t chain = 0; chain < atom_count; ++chain)
    {
        const std::size_t old = ordering.chain_to_old[chain];
        if(old >= atom_count)
            throw std::invalid_argument("chain_to_old contains an out-of-range atom");
        if(seen[old])
            throw std::invalid_argument("chain_to_old contains a duplicate atom");
        seen[old] = 1;
        if(ordering.old_to_chain[old] != chain)
            throw std::invalid_argument("old_to_chain is not the inverse of chain_to_old");
    }
    for(std::size_t old = 0; old < atom_count; ++old)
    {
        const std::size_t chain = ordering.old_to_chain[old];
        if(chain >= atom_count || ordering.chain_to_old[chain] != old)
            throw std::invalid_argument("chain_to_old is not the inverse of old_to_chain");
    }
}

OrderingCandidate make_ordering_candidate(const AtomGraph& graph,
                                           std::string_view orderer,
                                           std::size_t block_size)
{
    validate_graph(graph);
    OrderingCandidate candidate;
    candidate.orderer    = std::string(orderer);
    candidate.block_size = block_size;

    try
    {
        const auto start = Clock::now();
        auto       build = build_candidate_order(graph, orderer, block_size);
        const auto end   = Clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        candidate.ordering =
            finalize_ordering(graph, orderer, block_size, std::move(build.chain_to_old));
        candidate.metrics = score_ordering(graph, candidate.ordering, elapsed_ms);
        candidate.score   = candidate_score(candidate.metrics);
        candidate.ok      = true;
        candidate.fallback_reason = std::move(build.fallback_reason);
    }
    catch(const std::exception& e)
    {
        candidate.ok = false;
        candidate.fallback_reason = e.what();
    }
    return candidate;
}

OrderingRun run_ordering(const AtomGraph& graph,
                         std::string_view orderer,
                         std::string_view block_size)
{
    validate_graph(graph);
    OrderingRun run;
    run.graph_name = graph.name;

    for(const std::size_t block : resolve_block_sizes(block_size))
        for(const auto& candidate_orderer : resolve_orderers(orderer))
            run.candidates.push_back(make_ordering_candidate(graph, candidate_orderer, block));

    auto best = run.candidates.end();
    for(auto it = run.candidates.begin(); it != run.candidates.end(); ++it)
    {
        if(!it->ok)
            continue;
        if(best == run.candidates.end() || it->score > best->score)
            best = it;
    }
    if(best == run.candidates.end())
        throw std::runtime_error("all ordering candidates failed");
    run.selected_candidate = static_cast<std::size_t>(std::distance(run.candidates.begin(), best));
    return run;
}
} // namespace sol
