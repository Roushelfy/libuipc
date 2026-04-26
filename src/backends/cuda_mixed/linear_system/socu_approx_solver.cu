#include <linear_system/socu_approx_solver.h>

#include <linear_system/global_linear_system.h>
#include <linear_system/structured_chain_provider.h>
#include <mixed_precision/policy.h>
#include <sim_engine.h>
#include <uipc/common/exception.h>
#include <uipc/common/json.h>
#include <uipc/common/timer.h>

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <muda/buffer/device_buffer.h>
#include <muda/launch/parallel_for.h>
#include <muda/launch/kernel.h>
#include <muda/atomic.h>
#include <cub/warp/warp_reduce.cuh>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef UIPC_WITH_SOCU_NATIVE
#define UIPC_WITH_SOCU_NATIVE 0
#endif

#if UIPC_WITH_SOCU_NATIVE
#include <socu_native/problem_generator.h>
#include <socu_native/solver.h>
#endif

namespace uipc::backend::cuda_mixed
{
REGISTER_SIM_SYSTEM(SocuApproxSolver);

struct SocuApproxSolver::Runtime
{
#if UIPC_WITH_SOCU_NATIVE
    using Scalar = ActivePolicy::SolveScalar;

    struct SolverPlanDeleter
    {
        void operator()(socu_native::SolverPlan* plan) const
        {
            socu_native::destroy_solver_plan(plan);
        }
    };

    socu_native::ProblemShape shape{};
    socu_native::ProblemBufferLayout layout{};
    socu_native::SolverPlanOptions options{};
    std::unique_ptr<socu_native::SolverPlan, SolverPlanDeleter> plan;
    muda::DeviceBuffer<Scalar> device_diag;
    muda::DeviceBuffer<Scalar> device_off_diag;
    muda::DeviceBuffer<Scalar> device_diag_original;
    muda::DeviceBuffer<Scalar> device_off_diag_original;
    muda::DeviceBuffer<Scalar> device_rhs;
    muda::DeviceBuffer<Scalar> device_rhs_original;
    muda::DeviceBuffer<IndexT> device_old_to_chain;
    muda::DeviceBuffer<IndexT> device_chain_to_old;
    muda::DeviceBuffer<double> validation_sums;
    bool                       mappings_uploaded = false;

    Runtime(const socu_native::ProblemShape& shape_in,
            const socu_native::SolverPlanOptions& options_in)
        : shape(shape_in)
        , layout(socu_native::describe_problem_layout(shape))
        , options(options_in)
    {
    }

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    ~Runtime() = default;

    void reserve(bool debug_validation)
    {
        if(device_diag.capacity() < layout.diag_element_count)
            device_diag.reserve(layout.diag_element_count);
        if(device_off_diag.capacity() < layout.off_diag_element_count)
            device_off_diag.reserve(layout.off_diag_element_count);
        if(device_rhs.capacity() < layout.rhs_element_count)
            device_rhs.reserve(layout.rhs_element_count);
        device_diag.resize(layout.diag_element_count);
        device_off_diag.resize(layout.off_diag_element_count);
        device_rhs.resize(layout.rhs_element_count);

        if(debug_validation)
        {
            if(device_diag_original.capacity() < layout.diag_element_count)
                device_diag_original.reserve(layout.diag_element_count);
            if(device_off_diag_original.capacity() < layout.off_diag_element_count)
                device_off_diag_original.reserve(layout.off_diag_element_count);
            if(device_rhs_original.capacity() < layout.rhs_element_count)
                device_rhs_original.reserve(layout.rhs_element_count);
            if(validation_sums.capacity() < 5)
                validation_sums.reserve(5);
            device_diag_original.resize(layout.diag_element_count);
            device_off_diag_original.resize(layout.off_diag_element_count);
            device_rhs_original.resize(layout.rhs_element_count);
            validation_sums.resize(5);
        }
    }

    void upload_mappings_once(const std::vector<IndexT>& old_to_chain,
                              const std::vector<IndexT>& chain_to_old)
    {
        if(mappings_uploaded && device_old_to_chain.size() == old_to_chain.size()
           && device_chain_to_old.size() == chain_to_old.size())
            return;
        device_old_to_chain.resize(old_to_chain.size());
        device_chain_to_old.resize(chain_to_old.size());
        if(!old_to_chain.empty())
            device_old_to_chain.view().copy_from(old_to_chain.data());
        if(!chain_to_old.empty())
            device_chain_to_old.view().copy_from(chain_to_old.data());
        mappings_uploaded = true;
    }

    bool factor_and_solve(cudaStream_t stream)
    {
        const bool created = ensure_plan();
        socu_native::factor_and_solve_inplace_async(
            plan.get(),
            device_diag.data(),
            device_off_diag.data(),
            device_rhs.data(),
            socu_native::LaunchOptions{stream});
        return created;
    }

    void snapshot_matrix(cudaStream_t stream)
    {
        const auto diag_bytes = layout.diag_element_count * sizeof(Scalar);
        if(diag_bytes)
        {
            SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(device_diag_original.data(),
                                                   device_diag.data(),
                                                   diag_bytes,
                                                   cudaMemcpyDeviceToDevice,
                                                   stream));
        }

        const auto off_diag_bytes = layout.off_diag_element_count * sizeof(Scalar);
        if(off_diag_bytes)
        {
            SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(device_off_diag_original.data(),
                                                   device_off_diag.data(),
                                                   off_diag_bytes,
                                                   cudaMemcpyDeviceToDevice,
                                                   stream));
        }
    }

  private:
    bool ensure_plan()
    {
        if(plan != nullptr)
            return false;
        plan.reset(socu_native::create_solver_plan<Scalar>(shape, options));
        return true;
    }
#endif
};

namespace
{
namespace fs = std::filesystem;

#if UIPC_WITH_SOCU_NATIVE
#ifndef SOCU_NATIVE_DEFAULT_MATHDX_MANIFEST_PATH
#define SOCU_NATIVE_DEFAULT_MATHDX_MANIFEST_PATH ""
#endif

#ifndef SOCU_NATIVE_SOURCE_DIR
#define SOCU_NATIVE_SOURCE_DIR ""
#endif
#endif

template <typename T>
double norm2(const std::vector<T>& values)
{
    double sum = 0.0;
    for(const T value : values)
    {
        const double v = static_cast<double>(value);
        sum += v * v;
    }
    return std::sqrt(sum);
}

template <typename T>
double dot(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), 0.0);
}

#if UIPC_WITH_SOCU_NATIVE
fs::path default_mathdx_manifest_path()
{
    return fs::path{SOCU_NATIVE_DEFAULT_MATHDX_MANIFEST_PATH};
}
#endif

const Json* selected_candidate_json(const Json& report)
{
    if(report.contains("selected") && report["selected"].is_object())
        return &report["selected"];
    return &report;
}

const Json* ordering_json(const Json& candidate)
{
    if(candidate.contains("ordering") && candidate["ordering"].is_object())
        return &candidate["ordering"];
    if(candidate.contains("block_size"))
        return &candidate;
    return nullptr;
}

const Json* metrics_json(const Json& report, const Json& candidate)
{
    if(candidate.contains("metrics") && candidate["metrics"].is_object())
        return &candidate["metrics"];
    if(report.contains("metrics") && report["metrics"].is_object())
        return &report["metrics"];
    return nullptr;
}

bool required_array(const Json& json, std::string_view key, std::size_t expected_size)
{
    auto it = json.find(key);
    return it != json.end() && it->is_array() && it->size() == expected_size;
}

bool has_basic_ordering_schema(const Json& ordering)
{
    if(!ordering.contains("block_size")
       || !(ordering["block_size"].is_number_unsigned()
            || ordering["block_size"].is_number_integer()))
        return false;
    if(!ordering.contains("chain_to_old") || !ordering["chain_to_old"].is_array())
        return false;

    const auto atom_count = ordering["chain_to_old"].size();
    return required_array(ordering, "old_to_chain", atom_count)
           && required_array(ordering, "atom_to_block", atom_count)
           && required_array(ordering, "atom_block_offset", atom_count)
           && required_array(ordering, "atom_dof_count", atom_count)
           && ordering.contains("block_to_atom_range")
           && ordering["block_to_atom_range"].is_array()
           && !ordering["block_to_atom_range"].empty();
}

[[noreturn]] void throw_gate_failure(const SocuApproxGateReport& report)
{
    throw Exception{fmt::format("SocuApproxSolver gate failed: reason={}, detail={}",
                                to_string(report.reason),
                                report.detail)};
}

SocuApproxGateReport make_failure(SocuApproxGateReason reason,
                                  std::string          detail,
                                  std::string          ordering_report_path = {})
{
    SocuApproxGateReport report;
    report.reason               = reason;
    report.detail               = std::move(detail);
    report.ordering_report_path = std::move(ordering_report_path);
    if constexpr(std::is_same_v<GlobalLinearSystem::SolveScalar, float>)
        report.dtype = "float32";
    else
        report.dtype = "float64";
    return report;
}

fs::path absolute_workspace_path(std::string_view workspace, const std::string& path)
{
    fs::path p{path};
    if(p.is_relative())
        p = fs::absolute(fs::path{workspace} / p);
    return p;
}

bool read_size_t_field(const Json& json, const char* key, SizeT& value)
{
    auto it = json.find(key);
    if(it == json.end()
       || !(it->is_number_unsigned() || it->is_number_integer()))
        return false;
    value = it->get<SizeT>();
    return true;
}

bool parse_atom_dof_count(const Json& ordering, SizeT& dof_count, std::string& detail)
{
    dof_count = 0;
    auto it = ordering.find("atom_dof_count");
    if(it == ordering.end() || !it->is_array())
    {
        detail = "ordering atom_dof_count must be an array";
        return false;
    }

    for(const Json& value : *it)
    {
        if(!(value.is_number_unsigned() || value.is_number_integer()))
        {
            detail = "ordering atom_dof_count contains a non-integer value";
            return false;
        }
        dof_count += value.get<SizeT>();
    }
    return true;
}

bool parse_size_t_array(const Json&             ordering,
                        const char*            key,
                        std::vector<SizeT>&    values,
                        std::string&           detail)
{
    values.clear();
    auto it = ordering.find(key);
    if(it == ordering.end() || !it->is_array())
    {
        detail = fmt::format("ordering {} must be an array", key);
        return false;
    }

    values.reserve(it->size());
    for(const Json& value : *it)
    {
        if(!(value.is_number_unsigned() || value.is_number_integer()))
        {
            detail = fmt::format("ordering {} contains a non-integer value", key);
            return false;
        }
        values.push_back(value.get<SizeT>());
    }
    return true;
}

bool parse_block_layouts(const Json&                         ordering,
                         std::vector<SocuApproxBlockLayout>& blocks,
                         std::string&                       detail)
{
    blocks.clear();
    const auto& block_ranges = ordering.at("block_to_atom_range");
    SizeT       dof_offset   = 0;

    for(const Json& entry : block_ranges)
    {
        if(!entry.is_object())
        {
            detail = "ordering block_to_atom_range entries must be objects";
            return false;
        }

        SocuApproxBlockLayout block;
        if(!read_size_t_field(entry, "block", block.block)
           || !read_size_t_field(entry, "chain_begin", block.chain_begin)
           || !read_size_t_field(entry, "chain_end", block.chain_end)
           || !read_size_t_field(entry, "dof_count", block.dof_count))
        {
            detail =
                "ordering block_to_atom_range entries must contain block, chain_begin, chain_end, and dof_count";
            return false;
        }

        if(block.chain_end < block.chain_begin)
        {
            detail = "ordering block_to_atom_range has chain_end < chain_begin";
            return false;
        }

        block.dof_offset = dof_offset;
        dof_offset += block.dof_count;
        blocks.push_back(block);
    }

    return !blocks.empty();
}

bool validate_atom_inverse_mapping(const Json& ordering, std::string& detail)
{
    std::vector<SizeT> chain_to_old;
    std::vector<SizeT> old_to_chain;
    if(!parse_size_t_array(ordering, "chain_to_old", chain_to_old, detail)
       || !parse_size_t_array(ordering, "old_to_chain", old_to_chain, detail))
    {
        return false;
    }

    if(chain_to_old.size() != old_to_chain.size())
    {
        detail = "ordering old_to_chain size must match chain_to_old size";
        return false;
    }

    for(SizeT old = 0; old < old_to_chain.size(); ++old)
    {
        if(old_to_chain[old] >= chain_to_old.size())
        {
            detail = fmt::format(
                "ordering old_to_chain[{}]={} is out of chain range [0,{})",
                old,
                old_to_chain[old],
                chain_to_old.size());
            return false;
        }
    }

    for(SizeT chain = 0; chain < chain_to_old.size(); ++chain)
    {
        const SizeT old = chain_to_old[chain];
        if(old >= old_to_chain.size())
        {
            detail = fmt::format(
                "ordering chain_to_old[{}]={} is out of old atom range [0,{})",
                chain,
                old,
                old_to_chain.size());
            return false;
        }
        if(old_to_chain[old] != chain)
        {
            detail = fmt::format(
                "ordering old_to_chain is not the inverse of chain_to_old at chain {}: "
                "chain_to_old[{}]={}, old_to_chain[{}]={}",
                chain,
                chain,
                old,
                old,
                old_to_chain[old]);
            return false;
        }
    }

    return true;
}

class OrderingStructuredChainProvider final : public StructuredChainProvider
{
  public:
    OrderingStructuredChainProvider(SizeT                       block_size,
                                    std::vector<StructuredDofSlot> slots,
                                    double                      block_utilization)
        : m_slots(std::move(slots))
    {
        m_shape.horizon                  = m_slots.empty() ? 0 : (m_slots.back().block + 1);
        m_shape.block_size               = block_size;
        m_shape.nrhs                     = 1;
        m_shape.symmetric_positive_definite = true;
        m_quality.block_utilization      = block_utilization;
        for(const auto& slot : m_slots)
        {
            if(slot.is_padding)
                ++m_quality.padding_dof_count;
            else
                ++m_quality.active_dof_count;
        }
    }

    bool is_available() const override { return !m_slots.empty(); }
    StructuredChainShape shape() const override { return m_shape; }
    span<const StructuredDofSlot> dof_slots() const override { return m_slots; }
    StructuredQualityReport quality_report() const override { return m_quality; }
    void assemble_chain() override {}

  private:
    StructuredChainShape          m_shape;
    StructuredQualityReport       m_quality;
    std::vector<StructuredDofSlot> m_slots;
};

class HostStructuredDryRunSink
{
  public:
    virtual ~HostStructuredDryRunSink() = default;

    virtual void add_rhs(SizeT block, SizeT lane, double value) = 0;

    virtual void add_hessian(SizeT block_i,
                             SizeT lane_i,
                             SizeT block_j,
                             SizeT lane_j,
                             double value,
                             double weight) = 0;

    virtual void mark_off_band_drop(SizeT block_i,
                                    SizeT lane_i,
                                    SizeT block_j,
                                    SizeT lane_j,
                                    double value,
                                    double weight) = 0;
};

class CpuStructuredDryRunSink final : public HostStructuredDryRunSink
{
  public:
    CpuStructuredDryRunSink(SizeT block_size, std::vector<SocuApproxBlockLayout> blocks)
        : m_block_size(block_size)
        , m_blocks(std::move(blocks))
        , m_rhs(m_blocks.size() * block_size, 0.0)
        , m_diag(m_blocks.size() * block_size * block_size, 0.0)
        , m_first_offdiag(m_blocks.size() > 1
                              ? (m_blocks.size() - 1) * block_size * block_size
                              : 0,
                          0.0)
    {
        for(const auto& block : m_blocks)
        {
            if(block.block >= m_blocks.size())
                continue;
            for(SizeT lane = block.dof_count; lane < m_block_size; ++lane)
                m_diag[diag_index(block.block, lane, lane)] = 1.0;
        }
    }

    void add_rhs(SizeT block, SizeT lane, double value) override
    {
        if(!valid_lane(block, lane))
            return;
        m_rhs[block * m_block_size + lane] += value;
        m_rhs_abs_sum += std::abs(value);
    }

    void add_hessian(SizeT block_i,
                     SizeT lane_i,
                     SizeT block_j,
                     SizeT lane_j,
                     double value,
                     double weight) override
    {
        if(!valid_lane(block_i, lane_i) || !valid_lane(block_j, lane_j))
        {
            mark_off_band_drop(block_i, lane_i, block_j, lane_j, value, weight);
            return;
        }

        const SizeT distance =
            block_i > block_j ? block_i - block_j : block_j - block_i;
        if(distance > 1)
        {
            mark_off_band_drop(block_i, lane_i, block_j, lane_j, value, weight);
            return;
        }

        if(distance == 0)
        {
            m_diag[diag_index(block_i, lane_i, lane_j)] += value;
            ++m_diag_write_count;
            m_diag_contact_abs_sum += std::abs(value);
            return;
        }

        const bool ij_is_forward = block_i < block_j;
        const SizeT left_block   = ij_is_forward ? block_i : block_j;
        const SizeT row          = ij_is_forward ? lane_i : lane_j;
        const SizeT col          = ij_is_forward ? lane_j : lane_i;
        m_first_offdiag[offdiag_index(left_block, row, col)] += value;
        ++m_first_offdiag_write_count;
        m_first_offdiag_contact_abs_sum += std::abs(value);
    }

    void mark_off_band_drop(SizeT,
                            SizeT,
                            SizeT,
                            SizeT,
                            double value,
                            double weight) override
    {
        ++m_off_band_drop_count;
        m_off_band_drop_abs_sum += std::abs(value) * std::max(1.0, std::abs(weight));
    }

    void finalize(SocuApproxDryRunReport& report) const
    {
        report.rhs_scalar_count = m_rhs.size();
        report.diag_scalar_count = m_diag.size();
        report.first_offdiag_scalar_count = m_first_offdiag.size();
        report.rhs_nonzero_count = nonzero_count(m_rhs);
        report.diag_nonzero_count = nonzero_count(m_diag);
        report.first_offdiag_nonzero_count = nonzero_count(m_first_offdiag);
        report.first_offdiag_nonzero_index_sum = nonzero_index_sum(m_first_offdiag);
        report.structured_diag_write_count = m_diag_write_count;
        report.structured_first_offdiag_write_count = m_first_offdiag_write_count;
        report.structured_off_band_drop_count = m_off_band_drop_count;
        report.structured_diag_contact_abs_sum = m_diag_contact_abs_sum;
        report.structured_first_offdiag_contact_abs_sum =
            m_first_offdiag_contact_abs_sum;
        report.structured_off_band_drop_abs_sum = m_off_band_drop_abs_sum;
        report.rhs_abs_sum = m_rhs_abs_sum;
    }

    template <typename T>
    std::vector<T> rhs_as() const
    {
        return convert<T>(m_rhs);
    }

    template <typename T>
    std::vector<T> diag_as() const
    {
        return convert<T>(m_diag);
    }

    template <typename T>
    std::vector<T> off_diag_as(std::size_t full_scalar_count) const
    {
        std::vector<T> values(full_scalar_count, T{0});
        const std::size_t copy_count =
            std::min<std::size_t>(values.size(), m_first_offdiag.size());
        for(std::size_t i = 0; i < copy_count; ++i)
            values[i] = static_cast<T>(m_first_offdiag[i]);
        return values;
    }

  private:
    bool valid_lane(SizeT block, SizeT lane) const
    {
        return block < m_blocks.size() && lane < m_block_size;
    }

    SizeT diag_index(SizeT block, SizeT row, SizeT col) const
    {
        return (block * m_block_size + row) * m_block_size + col;
    }

    SizeT offdiag_index(SizeT left_block, SizeT row, SizeT col) const
    {
        return (left_block * m_block_size + row) * m_block_size + col;
    }

    static SizeT nonzero_count(const std::vector<double>& values)
    {
        return static_cast<SizeT>(
            std::count_if(values.begin(), values.end(), [](double v)
                          { return v != 0.0; }));
    }

    static SizeT nonzero_index_sum(const std::vector<double>& values)
    {
        SizeT sum = 0;
        for(SizeT i = 0; i < values.size(); ++i)
        {
            if(values[i] != 0.0)
                sum += i;
        }
        return sum;
    }

    template <typename T>
    static std::vector<T> convert(const std::vector<double>& values)
    {
        std::vector<T> converted(values.size());
        std::transform(values.begin(),
                       values.end(),
                       converted.begin(),
                       [](double value) { return static_cast<T>(value); });
        return converted;
    }

    SizeT m_block_size = 0;
    std::vector<SocuApproxBlockLayout> m_blocks;
    std::vector<double> m_rhs;
    std::vector<double> m_diag;
    std::vector<double> m_first_offdiag;
    SizeT m_diag_write_count = 0;
    SizeT m_first_offdiag_write_count = 0;
    SizeT m_off_band_drop_count = 0;
    double m_diag_contact_abs_sum = 0.0;
    double m_first_offdiag_contact_abs_sum = 0.0;
    double m_off_band_drop_abs_sum = 0.0;
    double m_rhs_abs_sum = 0.0;
};

bool build_ordering_provider(const Json&                         ordering,
                             SizeT                               block_size,
                             const std::vector<SocuApproxBlockLayout>& blocks,
                             std::unique_ptr<StructuredChainProvider>& provider,
                             SizeT&                              padding_slot_count,
                             std::string&                        detail)
{
    std::vector<SizeT> chain_to_old;
    std::vector<SizeT> atom_to_block;
    std::vector<SizeT> atom_block_offset;
    std::vector<SizeT> atom_dof_count;
    if(!parse_size_t_array(ordering, "chain_to_old", chain_to_old, detail)
       || !parse_size_t_array(ordering, "atom_to_block", atom_to_block, detail)
       || !parse_size_t_array(ordering, "atom_block_offset", atom_block_offset, detail)
       || !parse_size_t_array(ordering, "atom_dof_count", atom_dof_count, detail))
    {
        return false;
    }

    if(atom_to_block.size() != chain_to_old.size()
       || atom_block_offset.size() != chain_to_old.size()
       || atom_dof_count.size() != chain_to_old.size())
    {
        detail = "ordering mapping arrays have inconsistent atom counts";
        return false;
    }

    std::vector<SizeT> old_dof_offsets(atom_dof_count.size(), 0);
    SizeT              old_dof_offset = 0;
    for(SizeT old = 0; old < atom_dof_count.size(); ++old)
    {
        old_dof_offsets[old] = old_dof_offset;
        old_dof_offset += atom_dof_count[old];
    }

    std::vector<SizeT> valid_lanes_per_block(blocks.size(), 0);
    for(const auto& block : blocks)
    {
        if(block.block >= valid_lanes_per_block.size() || block.dof_count > block_size)
        {
            detail = "ordering block layout is inconsistent with block_size";
            return false;
        }
        valid_lanes_per_block[block.block] = block.dof_count;
    }

    std::vector<StructuredDofSlot> slots;
    slots.reserve(blocks.size() * block_size);
    for(SizeT chain = 0; chain < chain_to_old.size(); ++chain)
    {
        const SizeT old = chain_to_old[chain];
        if(old >= atom_dof_count.size())
        {
            detail = "ordering chain_to_old contains an out-of-range atom";
            return false;
        }

        const SizeT block = atom_to_block[old];
        const SizeT lane_begin = atom_block_offset[old];
        const SizeT dofs = atom_dof_count[old];
        if(block >= blocks.size() || lane_begin + dofs > block_size)
        {
            detail = "ordering atom block/lane mapping exceeds block_size";
            return false;
        }

        for(SizeT local_dof = 0; local_dof < dofs; ++local_dof)
        {
            const SizeT lane = lane_begin + local_dof;
            slots.push_back(StructuredDofSlot{
                .old_dof = static_cast<IndexT>(old_dof_offsets[old] + local_dof),
                .chain_dof = static_cast<IndexT>(block * block_size + lane),
                .block = block,
                .lane = lane,
                .is_padding = false,
                .scatter_write = true});
        }
    }

    padding_slot_count = 0;
    for(SizeT block = 0; block < valid_lanes_per_block.size(); ++block)
    {
        for(SizeT lane = valid_lanes_per_block[block]; lane < block_size; ++lane)
        {
            slots.push_back(StructuredDofSlot{.old_dof = -1,
                                             .chain_dof = static_cast<IndexT>(
                                                 block * block_size + lane),
                                             .block = block,
                                             .lane = lane,
                                             .is_padding = true,
                                             .scatter_write = false});
            ++padding_slot_count;
        }
    }

    const double padded_lanes =
        static_cast<double>(valid_lanes_per_block.size() * block_size);
    const double utilization =
        padded_lanes > 0.0
            ? static_cast<double>(old_dof_offset) / padded_lanes
            : 0.0;
    provider = std::make_unique<OrderingStructuredChainProvider>(
        block_size,
        std::move(slots),
        utilization);
    return provider->is_available();
}

bool validate_dof_coverage(span<const StructuredDofSlot> slots,
                           SizeT                         dof_count,
                           SizeT                         chain_scalar_count,
                           std::vector<IndexT>&          old_to_chain,
                           std::vector<IndexT>&          chain_to_old,
                           StructuredQualityReport&      quality,
                           std::string&                  detail)
{
    constexpr IndexT Missing = -2;
    old_to_chain.assign(dof_count, Missing);
    chain_to_old.assign(chain_scalar_count, Missing);
    quality.active_dof_count          = 0;
    quality.padding_dof_count         = 0;
    quality.duplicate_old_dof_count   = 0;
    quality.duplicate_chain_dof_count = 0;
    quality.missing_old_dof_count     = 0;
    quality.missing_chain_dof_count   = 0;
    quality.complete_dof_coverage     = false;

    for(const StructuredDofSlot& slot : slots)
    {
        if(slot.chain_dof < 0
           || static_cast<SizeT>(slot.chain_dof) >= chain_scalar_count)
        {
            detail = fmt::format("structured slot chain_dof {} is out of range [0,{})",
                                 slot.chain_dof,
                                 chain_scalar_count);
            return false;
        }

        const SizeT chain = static_cast<SizeT>(slot.chain_dof);
        if(chain_to_old[chain] != Missing)
            ++quality.duplicate_chain_dof_count;

        if(slot.is_padding)
        {
            ++quality.padding_dof_count;
            chain_to_old[chain] = -1;
            continue;
        }

        if(slot.old_dof < 0 || static_cast<SizeT>(slot.old_dof) >= dof_count)
        {
            detail = fmt::format("structured slot old_dof {} is out of range [0,{})",
                                 slot.old_dof,
                                 dof_count);
            return false;
        }

        ++quality.active_dof_count;
        const SizeT old = static_cast<SizeT>(slot.old_dof);
        if(old_to_chain[old] != Missing)
            ++quality.duplicate_old_dof_count;
        old_to_chain[old] = slot.chain_dof;
        chain_to_old[chain] = slot.old_dof;
    }

    for(IndexT value : old_to_chain)
    {
        if(value == Missing)
            ++quality.missing_old_dof_count;
    }
    for(IndexT value : chain_to_old)
    {
        if(value == Missing)
            ++quality.missing_chain_dof_count;
    }

    quality.complete_dof_coverage =
        quality.active_dof_count == dof_count
        && quality.duplicate_old_dof_count == 0
        && quality.duplicate_chain_dof_count == 0
        && quality.missing_old_dof_count == 0
        && quality.missing_chain_dof_count == 0;

    if(!quality.complete_dof_coverage)
    {
        detail = fmt::format("structured coverage invalid: active_dofs={}, expected_dofs={}, "
                             "padding_dofs={}, duplicate_old={}, duplicate_chain={}, "
                             "missing_old={}, missing_chain={}",
                             quality.active_dof_count,
                             dof_count,
                             quality.padding_dof_count,
                             quality.duplicate_old_dof_count,
                             quality.duplicate_chain_dof_count,
                             quality.missing_old_dof_count,
                             quality.missing_chain_dof_count);
        return false;
    }

    return true;
}

#if UIPC_WITH_SOCU_NATIVE
template <typename StoreScalar, typename SolveScalar>
void initialize_structured_workspace(
    cudaStream_t                         stream,
    StructuredChainShape                 shape,
    GlobalLinearSystem::CDenseVectorView b,
    muda::BufferView<SolveScalar>        diag,
    muda::BufferView<SolveScalar>        off_diag,
    muda::BufferView<SolveScalar>        rhs,
    muda::BufferView<SolveScalar>        rhs_original,
    muda::CBufferView<IndexT>            chain_to_old,
    double                               damping_shift)
{
    const auto diag_bytes = diag.size() * sizeof(SolveScalar);
    const auto off_bytes  = off_diag.size() * sizeof(SolveScalar);
    const auto rhs_bytes  = rhs.size() * sizeof(SolveScalar);
    if(diag_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(diag.data(), 0, diag_bytes, stream));
    if(off_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(off_diag.data(), 0, off_bytes, stream));
    if(rhs_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(rhs.data(), 0, rhs_bytes, stream));

    const SizeT chain_scalar_count = shape.horizon * shape.block_size;
    muda::ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(static_cast<int>(chain_scalar_count),
               [shape,
                b = b.cviewer().name("global_b"),
                diag = diag.viewer().name("structured_diag"),
                rhs = rhs.viewer().name("structured_rhs"),
                chain_to_old = chain_to_old.cviewer().name("chain_to_old"),
                damping_shift = static_cast<SolveScalar>(damping_shift)] __device__(int chain) mutable
               {
                   const SizeT block = static_cast<SizeT>(chain) / shape.block_size;
                   const SizeT lane  = static_cast<SizeT>(chain) % shape.block_size;
                   const SizeT diag_index =
                       (block * shape.block_size + lane) * shape.block_size + lane;

                   if(damping_shift != SolveScalar{0})
                       diag(diag_index) += damping_shift;

                   const IndexT old = chain_to_old(chain);
                   if(old >= 0)
                   {
                       rhs(chain) = static_cast<SolveScalar>(b(old));
                   }
                   else
                   {
                       diag(diag_index) += SolveScalar{1};
                   }
               });

    if(rhs_bytes && rhs_original.data() != nullptr)
    {
        SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(rhs_original.data(),
                                               rhs.data(),
                                               rhs_bytes,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
    }
}

template <typename SolveScalar>
void validate_structured_direction(cudaStream_t                  stream,
                                   StructuredChainShape          shape,
                                   muda::CBufferView<SolveScalar> diag,
                                   muda::CBufferView<SolveScalar> first_offdiag,
                                   muda::CBufferView<SolveScalar> rhs_original,
                                   muda::CBufferView<SolveScalar> solution,
                                   muda::CBufferView<IndexT>     chain_to_old,
                                   muda::BufferView<double>      sums)
{
    const auto sum_bytes = sums.size() * sizeof(double);
    if(sum_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(sums.data(), 0, sum_bytes, stream));

    const SizeT chain_scalar_count = shape.horizon * shape.block_size;
    const SizeT offdiag_scalar_count = first_offdiag.size();
    muda::ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(static_cast<int>(chain_scalar_count),
               [shape,
                offdiag_scalar_count,
                diag = diag.cviewer().name("diag"),
                first_offdiag = first_offdiag.cviewer().name("first_offdiag"),
                rhs = rhs_original.cviewer().name("rhs_original"),
                x = solution.cviewer().name("solution"),
                chain_to_old = chain_to_old.cviewer().name("chain_to_old"),
                sums = sums.viewer().name("sums")] __device__(int chain) mutable
               {
                   if(chain_to_old(chain) < 0)
                       return;

                   const SizeT block = static_cast<SizeT>(chain) / shape.block_size;
                   const SizeT lane  = static_cast<SizeT>(chain) % shape.block_size;
                   double      Ax    = 0.0;

                   for(SizeT col = 0; col < shape.block_size; ++col)
                   {
                       const SizeT col_chain = block * shape.block_size + col;
                       const SizeT index =
                           (block * shape.block_size + lane) * shape.block_size + col;
                       Ax += static_cast<double>(diag(index))
                             * static_cast<double>(x(col_chain));
                   }

                   if(block + 1 < shape.horizon)
                   {
                       for(SizeT col = 0; col < shape.block_size; ++col)
                       {
                           const SizeT right_chain = (block + 1) * shape.block_size + col;
                           const SizeT index =
                               (block * shape.block_size + lane) * shape.block_size + col;
                           if(index < offdiag_scalar_count)
                               Ax += static_cast<double>(first_offdiag(index))
                                     * static_cast<double>(x(right_chain));
                       }
                   }

                   if(block > 0)
                   {
                       const SizeT left_block = block - 1;
                       for(SizeT col = 0; col < shape.block_size; ++col)
                       {
                           const SizeT left_chain = left_block * shape.block_size + col;
                           const SizeT index =
                               (left_block * shape.block_size + col) * shape.block_size + lane;
                           if(index < offdiag_scalar_count)
                               Ax += static_cast<double>(first_offdiag(index))
                                     * static_cast<double>(x(left_chain));
                       }
                   }

                   const double rhs_i = static_cast<double>(rhs(chain));
                   const double x_i   = static_cast<double>(x(chain));
                   const double res   = Ax - rhs_i;
                   muda::atomic_add(sums.data() + 0, rhs_i * rhs_i);
                   muda::atomic_add(sums.data() + 1, x_i * x_i);
                   muda::atomic_add(sums.data() + 2, rhs_i * x_i);
                   muda::atomic_add(sums.data() + 3, res * res);
               });
}

template <typename SolveScalar>
void scatter_structured_solution(cudaStream_t                       stream,
                                 muda::CBufferView<SolveScalar>     solution,
                                 muda::CBufferView<IndexT>          old_to_chain,
                                 GlobalLinearSystem::SolveDenseVectorView x)
{
    muda::ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(static_cast<int>(old_to_chain.size()),
               [solution = solution.cviewer().name("solution"),
                old_to_chain = old_to_chain.cviewer().name("old_to_chain"),
                x = x.viewer().name("x")] __device__(int old) mutable
               {
                   const IndexT chain = old_to_chain(old);
                   if(chain >= 0)
                       x(old) = solution(chain);
               });
}
#endif

SizeT optional_size_t(const Json& json, const char* key)
{
    SizeT value = 0;
    if(read_size_t_field(json, key, value))
        return value;
    return 0;
}

double optional_double(const Json& json, const char* key, double fallback = 0.0)
{
    auto it = json.find(key);
    if(it == json.end() || !it->is_number())
        return fallback;
    return it->get<double>();
}

template <typename T>
std::string socu_dtype_name()
{
    if constexpr(std::is_same_v<T, float>)
        return "float32";
    else
        return "float64";
}

#if UIPC_WITH_SOCU_NATIVE
std::string to_report_string(socu_native::SolverBackend backend)
{
    switch(backend)
    {
        case socu_native::SolverBackend::NativeProof:
            return "native_proof";
        case socu_native::SolverBackend::NativePerf:
            return "native_perf";
        case socu_native::SolverBackend::CpuEigen:
            return "cpu_eigen";
    }
    return "unknown";
}

std::string to_report_string(socu_native::PerfBackend backend)
{
    switch(backend)
    {
        case socu_native::PerfBackend::Auto:
            return "auto";
        case socu_native::PerfBackend::Native:
            return "native";
        case socu_native::PerfBackend::CublasLt:
            return "cublaslt";
        case socu_native::PerfBackend::MathDx:
            return "mathdx";
    }
    return "unknown";
}

std::string to_report_string(socu_native::MathMode mode)
{
    switch(mode)
    {
        case socu_native::MathMode::Auto:
            return "auto";
        case socu_native::MathMode::Strict:
            return "strict";
        case socu_native::MathMode::TF32:
            return "tf32";
    }
    return "unknown";
}

std::string to_report_string(socu_native::GraphMode mode)
{
    switch(mode)
    {
        case socu_native::GraphMode::Off:
            return "off";
        case socu_native::GraphMode::On:
            return "on";
        case socu_native::GraphMode::Auto:
            return "auto";
    }
    return "unknown";
}

std::string mathdx_bundle_key(std::string_view prefix,
                              std::string_view dtype,
                              SizeT            block_size)
{
    return fmt::format("{}_{}_n{}", prefix, dtype, block_size);
}

fs::path manifest_relative_path(const fs::path& manifest_path, const std::string& path)
{
    fs::path p{path};
    if(p.is_relative())
        p = manifest_path.parent_path() / p;
    return p;
}

bool artifact_ref_ready(const Json&     artifact,
                        const fs::path& manifest_path,
                        std::string&    detail)
{
    if(!artifact.is_object())
    {
        detail = "MathDx artifact entry is not an object";
        return false;
    }
    const auto symbol = artifact.value("symbol", std::string{});
    const auto lto    = artifact.value("lto", std::string{});
    if(symbol.empty() || lto.empty())
    {
        detail = "MathDx artifact is missing symbol or lto";
        return false;
    }
    const auto lto_path = manifest_relative_path(manifest_path, lto);
    if(!fs::is_regular_file(lto_path))
    {
        detail = fmt::format("MathDx artifact lto '{}' is missing", lto_path.string());
        return false;
    }
    const auto fatbin = artifact.value("fatbin", std::string{});
    if(!fatbin.empty())
    {
        const auto fatbin_path = manifest_relative_path(manifest_path, fatbin);
        if(!fs::is_regular_file(fatbin_path))
        {
            detail =
                fmt::format("MathDx artifact fatbin '{}' is missing", fatbin_path.string());
            return false;
        }
    }
    return true;
}

template <typename Scalar>
bool validate_mathdx_manifest(const fs::path& manifest_path,
                              SizeT          block_size,
                              SocuApproxGateReport& gate,
                              std::string& detail)
{
    gate.mathdx_manifest_path = manifest_path.string();
    gate.mathdx_runtime_cache_dir = (manifest_path.parent_path() / "runtime").string();

    std::ifstream ifs{manifest_path};
    if(!ifs)
    {
        detail = fmt::format("MathDx manifest '{}' cannot be opened",
                             manifest_path.string());
        return false;
    }
    Json manifest = Json::parse(ifs, nullptr, false);
    if(manifest.is_discarded() || !manifest.is_object())
    {
        detail = "MathDx manifest is not valid JSON";
        return false;
    }
    if(!manifest.value("mathdx_enabled", false))
    {
        detail = "MathDx manifest reports mathdx_enabled=false";
        return false;
    }

    int device = 0;
    SOCU_NATIVE_CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    SOCU_NATIVE_CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    const int device_arch = prop.major * 10 + prop.minor;
    const int manifest_arch = manifest.value("arch", 0);
    if(manifest_arch != device_arch)
    {
        detail = fmt::format("MathDx manifest arch mismatch: manifest arch={}, device arch={}",
                             manifest_arch,
                             device_arch);
        return false;
    }

    const std::string dtype = socu_dtype_name<Scalar>();
    const Json* runtime_backend = nullptr;
    auto runtime_it = manifest.find("runtime_backend");
    if(runtime_it != manifest.end() && runtime_it->is_object())
        runtime_backend = &*runtime_it;
    if(!runtime_backend)
    {
        detail = "MathDx manifest is missing runtime_backend";
        return false;
    }

    auto validate_bundle = [&](const std::string& key,
                               std::initializer_list<const char*> required_keys) -> bool
    {
        auto bundle_it = runtime_backend->find(key);
        if(bundle_it == runtime_backend->end() || !bundle_it->is_object())
        {
            detail = fmt::format("MathDx manifest is missing runtime bundle '{}'", key);
            return false;
        }
        const auto cusolverdx_fatbin =
            bundle_it->value("cusolverdx_fatbin", std::string{});
        if(!cusolverdx_fatbin.empty()
           && !fs::is_regular_file(manifest_relative_path(manifest_path,
                                                          cusolverdx_fatbin)))
        {
            detail = fmt::format("MathDx runtime bundle '{}' has a missing cusolverdx_fatbin",
                                 key);
            return false;
        }
        for(const char* required_key : required_keys)
        {
            auto artifact_it = bundle_it->find(required_key);
            if(artifact_it == bundle_it->end())
            {
                detail = fmt::format("MathDx runtime bundle '{}' is missing '{}'",
                                     key,
                                     required_key);
                return false;
            }
            if(!artifact_ref_ready(*artifact_it, manifest_path, detail))
                return false;
        }
        return true;
    };

    const std::string factor_key = mathdx_bundle_key("factor", dtype, block_size);
    if(!validate_bundle(factor_key, {"potrf", "trsm_llnn"}))
        return false;

    auto factor_bundle = runtime_backend->find(factor_key);
    auto rltn_it       = factor_bundle->find("trsm_rltn_candidates");
    if(rltn_it == factor_bundle->end() || !rltn_it->is_array() || rltn_it->empty())
    {
        detail =
            fmt::format("MathDx runtime bundle '{}' is missing trsm_rltn_candidates",
                        factor_key);
        return false;
    }
    for(const Json& candidate : *rltn_it)
    {
        if(!artifact_ref_ready(candidate, manifest_path, detail))
            return false;
    }

    const std::string solve_key = mathdx_bundle_key("solve", dtype, block_size);
    if(!validate_bundle(solve_key,
                        {"potrs", "trsm_lower_rhs_1", "trsm_upper_rhs_1"}))
    {
        return false;
    }

    if(!fs::is_directory(manifest_path.parent_path()))
    {
        detail = "MathDx manifest parent directory is unavailable";
        return false;
    }

    gate.mathdx_manifest_ok = true;
    gate.mathdx_artifacts_ok = true;
    return true;
}
#endif

void apply_structured_contributions(const Json& contact, HostStructuredDryRunSink& sink)
{
    auto it = contact.find("structured_hessian_contributions");
    if(it == contact.end())
        it = contact.find("hessian_contributions");
    if(it == contact.end() || !it->is_array())
        return;

    for(const Json& entry : *it)
    {
        if(!entry.is_object())
            continue;

        SizeT block_i = 0;
        SizeT lane_i  = 0;
        SizeT block_j = 0;
        SizeT lane_j  = 0;
        if(!read_size_t_field(entry, "block_i", block_i)
           || !read_size_t_field(entry, "lane_i", lane_i)
           || !read_size_t_field(entry, "block_j", block_j)
           || !read_size_t_field(entry, "lane_j", lane_j))
        {
            continue;
        }

        const double value  = optional_double(entry, "value", 1.0);
        const double weight = optional_double(entry, "weight", 1.0);
        sink.add_hessian(block_i, lane_i, block_j, lane_j, value, weight);
    }
}

void load_contact_report(SocuApproxDryRunReport& dry_run, HostStructuredDryRunSink& sink)
{
    if(dry_run.contact_report_path.empty())
        return;

    std::ifstream ifs{dry_run.contact_report_path};
    if(!ifs)
        throw Exception{fmt::format("SocuApproxSolver contact report '{}' cannot be opened",
                                    dry_run.contact_report_path)};

    Json report = Json::parse(ifs, nullptr, false);
    if(report.is_discarded() || !report.is_object())
        throw Exception{fmt::format("SocuApproxSolver contact report '{}' is not valid JSON",
                                    dry_run.contact_report_path)};

    const Json& contact =
        report.contains("contact") && report["contact"].is_object() ? report["contact"] : report;
    dry_run.near_band_contact_count =
        optional_size_t(contact, "near_band_contact_count");
    dry_run.off_band_contact_count =
        optional_size_t(contact, "off_band_contact_count");
    dry_run.near_band_contribution_count =
        optional_size_t(contact, "near_band_contribution_count");
    dry_run.off_band_contribution_count =
        optional_size_t(contact, "off_band_contribution_count");
    dry_run.absorbed_hessian_contribution_count =
        optional_size_t(contact, "estimated_absorbed_hessian_contribution_count");
    dry_run.dropped_hessian_contribution_count =
        optional_size_t(contact, "estimated_dropped_contribution_count");
    dry_run.contribution_near_band_ratio =
        optional_double(contact, "contribution_near_band_ratio");
    dry_run.contribution_off_band_ratio =
        optional_double(contact, "contribution_off_band_ratio");
    dry_run.weighted_near_band_ratio =
        optional_double(contact, "weighted_near_band_ratio");
    dry_run.weighted_off_band_ratio =
        optional_double(contact, "weighted_off_band_ratio");

    apply_structured_contributions(contact, sink);
}

Json to_json(const SocuApproxDryRunReport& report)
{
    Json blocks = Json::array();
    for(const auto& block : report.blocks)
    {
        blocks.push_back(Json{{"block", block.block},
                              {"chain_begin", block.chain_begin},
                              {"chain_end", block.chain_end},
                              {"dof_offset", block.dof_offset},
                              {"dof_count", block.dof_count}});
    }

    return Json{{"solver", "socu_approx"},
                {"milestone",
                 report.mode == "structured_strict_solve" ? 7 : 5},
                {"mode", report.mode},
                {"packed", report.packed},
                {"ordering_report", report.ordering_report_path},
                {"contact_report", report.contact_report_path},
                {"block_size", report.block_size},
                {"chain_atom_count", report.chain_atom_count},
                {"ordering_dof_count", report.ordering_dof_count},
                {"structured_slot_count", report.structured_slot_count},
                {"padding_slot_count", report.padding_slot_count},
                {"block_utilization", report.block_utilization},
                {"gates",
                 {{"min_block_utilization", report.min_block_utilization},
                  {"min_near_band_ratio", report.min_near_band_ratio},
                  {"max_off_band_ratio", report.max_off_band_ratio},
                  {"max_off_band_drop_norm_ratio",
                   report.max_off_band_drop_norm_ratio},
                  {"complete_dof_coverage", report.complete_dof_coverage},
                  {"coverage_active_dof_count",
                   report.coverage_active_dof_count},
                  {"coverage_padding_dof_count",
                   report.coverage_padding_dof_count}}},
                {"layout",
                 {{"block_count", report.block_count},
                  {"diag_block_count", report.diag_block_count},
                  {"first_offdiag_block_count", report.first_offdiag_block_count},
                  {"active_rhs_scalar_count", report.active_rhs_scalar_count},
                  {"rhs_scalar_count", report.rhs_scalar_count},
                  {"diag_scalar_count", report.diag_scalar_count},
                  {"first_offdiag_scalar_count", report.first_offdiag_scalar_count},
                  {"rhs_nonzero_count", report.rhs_nonzero_count},
                  {"diag_nonzero_count", report.diag_nonzero_count},
                  {"first_offdiag_nonzero_count",
                   report.first_offdiag_nonzero_count},
                  {"first_offdiag_nonzero_index_sum",
                   report.first_offdiag_nonzero_index_sum},
                  {"blocks", blocks}}},
                {"contact",
                 {{"near_band_contact_count", report.near_band_contact_count},
                  {"off_band_contact_count", report.off_band_contact_count},
                  {"near_band_contribution_count", report.near_band_contribution_count},
                  {"off_band_contribution_count", report.off_band_contribution_count},
                  {"absorbed_hessian_contribution_count",
                   report.absorbed_hessian_contribution_count},
                  {"dropped_hessian_contribution_count",
                   report.dropped_hessian_contribution_count},
                  {"contribution_near_band_ratio",
                   report.contribution_near_band_ratio},
                  {"contribution_off_band_ratio",
                   report.contribution_off_band_ratio},
                  {"weighted_near_band_ratio", report.weighted_near_band_ratio},
                  {"weighted_off_band_ratio", report.weighted_off_band_ratio},
                  {"structured_diag_write_count",
                   report.structured_diag_write_count},
                  {"structured_first_offdiag_write_count",
                   report.structured_first_offdiag_write_count},
                  {"structured_off_band_drop_count",
                   report.structured_off_band_drop_count},
                  {"structured_diag_contact_abs_sum",
                   report.structured_diag_contact_abs_sum},
                  {"structured_first_offdiag_contact_abs_sum",
                   report.structured_first_offdiag_contact_abs_sum},
                  {"structured_off_band_drop_abs_sum",
                   report.structured_off_band_drop_abs_sum},
                  {"rhs_abs_sum", report.rhs_abs_sum}}},
                {"timing",
                 {{"dry_run_pack_time_ms", report.dry_run_pack_time_ms},
                  {"socu_factor_solve_time_ms", report.socu_factor_solve_time_ms},
                  {"scatter_time_ms", report.scatter_time_ms},
                  {"stream_source", report.stream_source},
                  {"plan_created_this_solve", report.plan_created_this_solve},
                  {"debug_timing_enabled", report.debug_timing_enabled}}},
                {"solve",
                 {{"damping_shift", report.damping_shift},
                  {"surrogate_residual", report.surrogate_residual},
                  {"surrogate_relative_residual",
                   report.surrogate_relative_residual},
                  {"descent_dot", report.descent_dot},
                  {"gradient_norm", report.gradient_norm},
                  {"direction_norm", report.direction_norm},
                  {"direction_min_abs_threshold",
                   report.direction_min_abs_threshold},
                  {"direction_min_rel_threshold",
                   report.direction_min_rel_threshold},
                  {"rhs_sign_convention", report.rhs_sign_convention},
                  {"debug_validation_enabled", report.debug_validation_enabled},
                  {"report_each_solve", report.report_each_solve}}},
                {"line_search",
                 {{"feedback_available", report.line_search_feedback_available},
                  {"accepted", report.line_search_accepted},
                  {"hit_max_iter", report.line_search_hit_max_iter},
                  {"iteration_count", report.line_search_iteration_count},
                  {"accepted_alpha", report.line_search_accepted_alpha},
                  {"reject_streak", report.line_search_reject_streak}}},
                {"status",
                 {{"direction_available", report.direction_available},
                  {"reason", report.status_reason},
                  {"detail", report.status_detail}}}};
}

void write_dry_run_report(const SocuApproxDryRunReport& report)
{
    fs::path report_path{report.report_path};
    fs::create_directories(report_path.parent_path());
    std::ofstream ofs{report_path};
    if(!ofs)
        throw Exception{fmt::format("SocuApproxSolver dry-run report '{}' cannot be written",
                                    report.report_path)};
    ofs << to_json(report).dump(2);
}
}  // namespace

std::string_view to_string(SocuApproxGateReason reason) noexcept
{
    switch(reason)
    {
    case SocuApproxGateReason::None:
        return "none";
    case SocuApproxGateReason::SocuDisabled:
        return "socu_disabled";
    case SocuApproxGateReason::OrderingMissing:
        return "ordering_missing";
    case SocuApproxGateReason::OrderingReportInvalid:
        return "ordering_report_invalid";
    case SocuApproxGateReason::UnsupportedPrecisionContract:
        return "unsupported_precision_contract";
    case SocuApproxGateReason::UnsupportedBlockSize:
        return "unsupported_block_size";
    case SocuApproxGateReason::SocuMathDxUnsupported:
        return "socu_mathdx_unsupported";
    case SocuApproxGateReason::SocuRuntimeArtifactUnavailable:
        return "socu_runtime_artifact_unavailable";
    case SocuApproxGateReason::OrderingQualityTooLow:
        return "ordering_quality_too_low";
    case SocuApproxGateReason::ContactOffBandRatioTooHigh:
        return "contact_off_band_ratio_too_high";
    case SocuApproxGateReason::StructuredProviderMissing:
        return "structured_provider_missing";
    case SocuApproxGateReason::StructuredCoverageInvalid:
        return "structured_coverage_invalid";
    case SocuApproxGateReason::StructuredSubsystemUnsupported:
        return "structured_subsystem_not_supported";
    case SocuApproxGateReason::StubNoDirection:
        return "socu_approx_stub_no_direction";
    case SocuApproxGateReason::DirectionInvalid:
        return "direction_invalid";
    case SocuApproxGateReason::SocuRuntimeError:
        return "socu_runtime_error";
    case SocuApproxGateReason::M8ContactRuntimeNotSupported:
        return "M8_contact_runtime_not_supported";
    case SocuApproxGateReason::LineSearchRejected:
        return "line_search_rejected";
    }
    return "unknown";
}

SocuApproxSolver::~SocuApproxSolver() = default;

void SocuApproxSolver::do_build(BuildInfo& info)
{
    auto&      config      = world().scene().config();
    auto       solver_attr = config.find<std::string>("linear_system/solver");
    const std::string solver_name =
        solver_attr ? solver_attr->view()[0] : std::string{"fused_pcg"};
    if(solver_name != "socu_approx")
    {
        throw SimSystemException("SocuApproxSolver unused");
    }

#if !UIPC_WITH_SOCU_NATIVE
    m_gate_report = make_failure(
        SocuApproxGateReason::SocuDisabled,
        "socu_native is disabled or not available; configure with UIPC_WITH_SOCU_NATIVE=AUTO or ON and initialize external/socu-native-cuda");
    throw_gate_failure(m_gate_report);
#endif

    require<GlobalLinearSystem>();

    auto mode_attr =
        config.find<std::string>("linear_system/socu_approx/mode");
    m_mode = mode_attr ? mode_attr->view()[0] : std::string{"solve"};
    if(m_mode != "dry_run" && m_mode != "solve")
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingReportInvalid,
            fmt::format("linear_system/socu_approx/mode must be 'dry_run' or 'solve', got '{}'",
                        m_mode));
        throw_gate_failure(m_gate_report);
    }

    auto debug_validation_attr =
        config.find<IndexT>("linear_system/socu_approx/debug_validation");
    auto debug_timing_attr =
        config.find<IndexT>("linear_system/socu_approx/debug_timing");
    auto report_each_solve_attr =
        config.find<IndexT>("linear_system/socu_approx/report_each_solve");
    m_debug_validation =
        debug_validation_attr && debug_validation_attr->view()[0] != 0;
    m_debug_timing = debug_timing_attr && debug_timing_attr->view()[0] != 0;
    m_report_each_solve =
        report_each_solve_attr && report_each_solve_attr->view()[0] != 0;

    auto ordering_report_attr =
        config.find<std::string>("linear_system/socu_approx/ordering_report");
    std::string ordering_report =
        ordering_report_attr ? ordering_report_attr->view()[0] : std::string{};

    if(ordering_report.empty())
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingMissing,
            "linear_system/socu_approx/ordering_report is empty");
        throw_gate_failure(m_gate_report);
    }

    fs::path ordering_report_path{ordering_report};
    if(ordering_report_path.is_relative())
        ordering_report_path = fs::absolute(fs::path{workspace()} / ordering_report_path);

    std::ifstream ifs{ordering_report_path};
    if(!ifs)
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingMissing,
                                     fmt::format("ordering report '{}' cannot be opened",
                                                 ordering_report_path.string()),
                                     ordering_report_path.string());
        throw_gate_failure(m_gate_report);
    }

    Json report = Json::parse(ifs, nullptr, false);
    if(report.is_discarded() || !report.is_object())
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingReportInvalid,
                                     fmt::format("ordering report '{}' is not valid JSON",
                                                 ordering_report_path.string()),
                                     ordering_report_path.string());
        throw_gate_failure(m_gate_report);
    }

    const Json* candidate = selected_candidate_json(report);
    const Json* ordering  = ordering_json(*candidate);
    if(!ordering || !has_basic_ordering_schema(*ordering))
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingReportInvalid,
                                     "ordering report is missing the required ordering mapping schema",
                                     ordering_report_path.string());
        throw_gate_failure(m_gate_report);
    }

    m_gate_report.ordering_report_path = ordering_report_path.string();
    m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
    if(m_gate_report.block_size != 32 && m_gate_report.block_size != 64)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::UnsupportedBlockSize,
            fmt::format("ordering block_size must be 32 or 64, got {}",
                        m_gate_report.block_size),
            ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    if constexpr(!std::is_same_v<ActivePolicy::StoreScalar, ActivePolicy::SolveScalar>)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::UnsupportedPrecisionContract,
            "socu_approx strict structured solve only accepts StoreScalar == SolveScalar",
            ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    std::string                       ordering_detail;
    std::vector<SocuApproxBlockLayout> block_layouts;
    SizeT                             ordering_dof_count = 0;
    if(!parse_atom_dof_count(*ordering, ordering_dof_count, ordering_detail)
       || !parse_block_layouts(*ordering, block_layouts, ordering_detail))
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingReportInvalid,
                                     ordering_detail.empty()
                                         ? "ordering report has an empty block layout"
                                         : ordering_detail,
                                     ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    std::unique_ptr<StructuredChainProvider> provider;
    SizeT                                    padding_slot_count = 0;
    if(!build_ordering_provider(*ordering,
                                m_gate_report.block_size,
                                block_layouts,
                                provider,
                                padding_slot_count,
                                ordering_detail))
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingReportInvalid,
                                     ordering_detail.empty()
                                         ? "ordering report cannot build a structured provider"
                                         : ordering_detail,
                                     ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    StructuredQualityReport build_coverage = {};
    build_coverage.block_utilization = provider->quality_report().block_utilization;
    std::vector<IndexT> build_old_to_chain;
    std::vector<IndexT> build_chain_to_old;
    const SizeT         chain_scalar_count =
        block_layouts.size() * m_gate_report.block_size;
    if(!validate_dof_coverage(provider->dof_slots(),
                              ordering_dof_count,
                              chain_scalar_count,
                              build_old_to_chain,
                              build_chain_to_old,
                              build_coverage,
                              ordering_detail))
    {
        m_gate_report = make_failure(SocuApproxGateReason::StructuredCoverageInvalid,
                                     ordering_detail,
                                     ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        m_gate_report.block_utilization = provider->quality_report().block_utilization;
        m_gate_report.coverage_active_dof_count = build_coverage.active_dof_count;
        m_gate_report.coverage_padding_dof_count = build_coverage.padding_dof_count;
        m_gate_report.complete_dof_coverage = false;
        throw_gate_failure(m_gate_report);
    }

    if(!validate_atom_inverse_mapping(*ordering, ordering_detail))
    {
        m_gate_report = make_failure(SocuApproxGateReason::OrderingReportInvalid,
                                     ordering_detail,
                                     ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    if(const Json* metrics = metrics_json(report, *candidate))
    {
        if(metrics->contains("near_band_ratio"))
            m_gate_report.near_band_ratio = metrics->at("near_band_ratio").get<double>();
        if(metrics->contains("off_band_ratio"))
            m_gate_report.off_band_ratio = metrics->at("off_band_ratio").get<double>();
        if(metrics->contains("off_band_drop_norm_ratio"))
            m_gate_report.off_band_drop_norm_ratio =
                metrics->at("off_band_drop_norm_ratio").get<double>();
    }

    m_gate_report.block_utilization =
        provider->quality_report().block_utilization;

    auto min_near_attr =
        config.find<Float>("linear_system/socu_approx/min_near_band_ratio");
    auto max_off_attr =
        config.find<Float>("linear_system/socu_approx/max_off_band_ratio");
    auto min_util_attr =
        config.find<Float>("linear_system/socu_approx/min_block_utilization");
    auto max_drop_attr =
        config.find<Float>("linear_system/socu_approx/max_off_band_drop_norm_ratio");
    const double min_near_band_ratio =
        min_near_attr ? static_cast<double>(min_near_attr->view()[0]) : 0.90;
    const double max_off_band_ratio =
        max_off_attr ? static_cast<double>(max_off_attr->view()[0]) : 0.10;
    const double min_block_utilization =
        min_util_attr ? static_cast<double>(min_util_attr->view()[0]) : 0.65;
    const double max_off_band_drop_norm_ratio =
        max_drop_attr ? static_cast<double>(max_drop_attr->view()[0]) : 0.05;
    if(m_gate_report.near_band_ratio < min_near_band_ratio
       || m_gate_report.off_band_ratio > max_off_band_ratio
       || m_gate_report.off_band_drop_norm_ratio > max_off_band_drop_norm_ratio)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingQualityTooLow,
            fmt::format("ordering quality rejected: near_band_ratio={}, off_band_ratio={}, "
                        "off_band_drop_norm_ratio={}, min_near_band_ratio={}, "
                        "max_off_band_ratio={}, max_off_band_drop_norm_ratio={}",
                        m_gate_report.near_band_ratio,
                        m_gate_report.off_band_ratio,
                        m_gate_report.off_band_drop_norm_ratio,
                        min_near_band_ratio,
                        max_off_band_ratio,
                        max_off_band_drop_norm_ratio),
            ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }
    if(m_gate_report.block_utilization < min_block_utilization)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingQualityTooLow,
            fmt::format("ordering block utilization rejected: block_utilization={}, "
                        "min_block_utilization={}",
                        m_gate_report.block_utilization,
                        min_block_utilization),
            ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    auto dry_run_report_attr =
        config.find<std::string>("linear_system/socu_approx/dry_run_report");
    std::string dry_run_report =
        dry_run_report_attr ? dry_run_report_attr->view()[0] : std::string{};
    fs::path dry_run_report_path =
        dry_run_report.empty()
            ? fs::absolute(fs::path{workspace()} / "socu_approx_dry_run_report.json")
            : absolute_workspace_path(workspace(), dry_run_report);

    auto contact_report_attr =
        config.find<std::string>("linear_system/socu_approx/contact_report");
    std::string contact_report =
        contact_report_attr ? contact_report_attr->view()[0] : std::string{};
    if(m_mode == "solve" && !contact_report.empty())
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingReportInvalid,
            "linear_system/socu_approx/contact_report is dry-run/debug only and is not accepted in solve mode",
            ordering_report_path.string());
        m_gate_report.block_size = ordering->at("block_size").get<SizeT>();
        throw_gate_failure(m_gate_report);
    }

    m_dry_run_report                         = SocuApproxDryRunReport{};
    m_dry_run_report.mode =
        m_mode == "solve" ? "structured_strict_solve" : "structured_dry_run";
    m_dry_run_report.report_path             = dry_run_report_path.string();
    m_dry_run_report.ordering_report_path    = ordering_report_path.string();
    m_dry_run_report.contact_report_path =
        contact_report.empty() ? std::string{}
                               : absolute_workspace_path(workspace(), contact_report).string();
    m_dry_run_report.block_size         = m_gate_report.block_size;
    m_dry_run_report.block_count        = block_layouts.size();
    m_dry_run_report.chain_atom_count   = ordering->at("chain_to_old").size();
    m_dry_run_report.ordering_dof_count = ordering_dof_count;
    m_dry_run_report.structured_slot_count = provider->dof_slots().size();
    m_dry_run_report.padding_slot_count = padding_slot_count;
    m_dry_run_report.block_utilization =
        provider->quality_report().block_utilization;
    m_dry_run_report.min_block_utilization = min_block_utilization;
    m_dry_run_report.min_near_band_ratio = min_near_band_ratio;
    m_dry_run_report.max_off_band_ratio = max_off_band_ratio;
    m_dry_run_report.max_off_band_drop_norm_ratio =
        max_off_band_drop_norm_ratio;
    m_dry_run_report.debug_validation_enabled = m_debug_validation;
    m_dry_run_report.debug_timing_enabled = m_debug_timing;
    m_dry_run_report.report_each_solve = m_report_each_solve;
    m_dry_run_report.coverage_active_dof_count =
        provider->quality_report().active_dof_count;
    m_dry_run_report.coverage_padding_dof_count =
        provider->quality_report().padding_dof_count;
    m_dry_run_report.blocks             = std::move(block_layouts);
    m_dof_slots.assign(provider->dof_slots().begin(), provider->dof_slots().end());

    m_gate_report.passed = true;
    m_gate_report.reason = SocuApproxGateReason::None;
    m_gate_report.detail =
        m_mode == "solve"
            ? "M7 strict ABD-only structured solve gate passed"
            : "M5 structured dry-run gate passed; solve direction is intentionally unavailable";
    m_gate_report.dtype = socu_dtype_name<GlobalLinearSystem::SolveScalar>();
    m_gate_report.coverage_active_dof_count =
        provider->quality_report().active_dof_count;
    m_gate_report.coverage_padding_dof_count =
        provider->quality_report().padding_dof_count;

    if(m_mode == "solve")
    {
#if UIPC_WITH_SOCU_NATIVE
        const auto manifest_path = default_mathdx_manifest_path();
        if(manifest_path.empty() || !fs::is_regular_file(manifest_path))
        {
            m_gate_report = make_failure(
                SocuApproxGateReason::SocuRuntimeArtifactUnavailable,
                fmt::format("MathDx manifest is missing; expected '{}'",
                            manifest_path.string()),
                ordering_report_path.string());
            m_gate_report.block_size = m_dry_run_report.block_size;
            throw_gate_failure(m_gate_report);
        }

        std::string manifest_detail;
        try
        {
            if(!validate_mathdx_manifest<Runtime::Scalar>(manifest_path,
                                                          m_dry_run_report.block_size,
                                                          m_gate_report,
                                                          manifest_detail))
            {
                m_gate_report = make_failure(
                    SocuApproxGateReason::SocuRuntimeArtifactUnavailable,
                    manifest_detail,
                    ordering_report_path.string());
                m_gate_report.block_size = m_dry_run_report.block_size;
                m_gate_report.mathdx_manifest_path = manifest_path.string();
                throw_gate_failure(m_gate_report);
            }
        }
        catch(const std::exception& e)
        {
            m_gate_report = make_failure(
                SocuApproxGateReason::SocuRuntimeArtifactUnavailable,
                fmt::format("MathDx manifest preflight failed: {}", e.what()),
                ordering_report_path.string());
            m_gate_report.block_size = m_dry_run_report.block_size;
            m_gate_report.mathdx_manifest_path = manifest_path.string();
            throw_gate_failure(m_gate_report);
        }

        m_gate_report.mathdx_prebuilt_cubin_ok = m_gate_report.mathdx_artifacts_ok;
        m_gate_report.debug_validation_enabled = m_debug_validation;
        m_gate_report.debug_timing_enabled = m_debug_timing;
        int device_count = 0;
        const cudaError_t device_query = cudaGetDeviceCount(&device_count);
        if(device_query != cudaSuccess || device_count == 0)
        {
            cudaGetLastError();
            m_gate_report = make_failure(
                SocuApproxGateReason::SocuMathDxUnsupported,
                "no CUDA device is available for socu_approx",
                ordering_report_path.string());
            m_gate_report.block_size = m_dry_run_report.block_size;
            throw_gate_failure(m_gate_report);
        }

        socu_native::ProblemShape shape{
            static_cast<int>(m_dry_run_report.block_count),
            static_cast<int>(m_dry_run_report.block_size),
            1};
        socu_native::SolverPlanOptions options;
        options.backend      = socu_native::SolverBackend::NativePerf;
        options.perf_backend = socu_native::PerfBackend::MathDx;
        options.math_mode    = socu_native::MathMode::Auto;
        options.graph_mode   = socu_native::GraphMode::Off;

        const auto capability =
            socu_native::query_solver_capability<Runtime::Scalar>(
                shape,
                socu_native::SolverOperation::FactorAndSolve,
                options);
        m_gate_report.resolved_backend =
            to_report_string(capability.resolved_backend);
        m_gate_report.resolved_perf_backend =
            to_report_string(capability.resolved_perf_backend);
        m_gate_report.resolved_math_mode =
            to_report_string(capability.resolved_math_mode);
        m_gate_report.resolved_graph_mode =
            to_report_string(capability.resolved_graph_mode);
        if(!capability.supported
           || capability.resolved_backend != socu_native::SolverBackend::NativePerf
           || capability.resolved_perf_backend != socu_native::PerfBackend::MathDx)
        {
            m_gate_report = make_failure(
                SocuApproxGateReason::SocuMathDxUnsupported,
                fmt::format("socu_native MathDx capability rejected: {}",
                            capability.reason),
                ordering_report_path.string());
            m_gate_report.block_size = m_dry_run_report.block_size;
            throw_gate_failure(m_gate_report);
        }

        try
        {
            m_runtime = std::make_unique<Runtime>(shape, options);
        }
        catch(const std::exception& e)
        {
            m_gate_report = make_failure(
                SocuApproxGateReason::SocuRuntimeError,
                fmt::format("socu_native plan creation failed: {}", e.what()),
                ordering_report_path.string());
            m_gate_report.block_size = m_dry_run_report.block_size;
            throw_gate_failure(m_gate_report);
        }
#else
        m_gate_report = make_failure(
            SocuApproxGateReason::SocuDisabled,
            "socu_native is disabled or not available");
        throw_gate_failure(m_gate_report);
#endif
    }

    logger::info("SocuApproxSolver {} enabled: block_size={}, blocks={}, report='{}'",
                 m_mode == "solve" ? "M7 strict solve" : "M5 dry-run",
                 m_dry_run_report.block_size,
                 m_dry_run_report.block_count,
                 m_dry_run_report.report_path);
}

void SocuApproxSolver::prepare_structured_chain(
    GlobalLinearSystem::StructuredAssemblyInfo& info)
{
    if(m_mode != "solve")
        return;

#if !UIPC_WITH_SOCU_NATIVE
    throw Exception{"SocuApproxSolver structured solve reached without socu_native support"};
#else
    UIPC_ASSERT(m_runtime != nullptr,
                "SocuApproxSolver solve mode requires initialized runtime.");

    auto reject_streak_attr =
        world().scene().config().find<IndexT>(
            "linear_system/socu_approx/max_line_search_reject_streak");
    const IndexT reject_streak_threshold =
        reject_streak_attr ? reject_streak_attr->view()[0] : 1;
    if(reject_streak_threshold > 0
       && static_cast<IndexT>(m_line_search_reject_streak) >= reject_streak_threshold)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::LineSearchRejected,
            fmt::format("previous socu_approx direction hit line-search max_iter "
                        "{} consecutive time(s); threshold={}",
                        m_line_search_reject_streak,
                        reject_streak_threshold),
            m_gate_report.ordering_report_path);
        m_gate_report.block_size = m_dry_run_report.block_size;
        m_dry_run_report.status_reason =
            std::string{to_string(SocuApproxGateReason::LineSearchRejected)};
        m_dry_run_report.status_detail = m_gate_report.detail;
        write_dry_run_report(m_dry_run_report);
        throw_gate_failure(m_gate_report);
    }

    const SizeT chain_scalar_count =
        m_runtime->shape.horizon * m_runtime->shape.n;
    StructuredQualityReport coverage = {};
    coverage.block_utilization = m_dry_run_report.block_utilization;
    std::string coverage_detail;
    if(!validate_dof_coverage(span<const StructuredDofSlot>{m_dof_slots},
                              static_cast<SizeT>(info.b().size()),
                              chain_scalar_count,
                              m_host_old_to_chain,
                              m_host_chain_to_old,
                              coverage,
                              coverage_detail))
    {
        m_gate_report = make_failure(SocuApproxGateReason::StructuredCoverageInvalid,
                                     coverage_detail,
                                     m_gate_report.ordering_report_path);
        m_gate_report.block_size = m_dry_run_report.block_size;
        m_gate_report.block_utilization = m_dry_run_report.block_utilization;
        m_gate_report.coverage_active_dof_count = coverage.active_dof_count;
        m_gate_report.coverage_padding_dof_count = coverage.padding_dof_count;
        m_gate_report.complete_dof_coverage = false;
        m_dry_run_report.status_reason =
            std::string{to_string(SocuApproxGateReason::StructuredCoverageInvalid)};
        m_dry_run_report.status_detail = coverage_detail;
        write_dry_run_report(m_dry_run_report);
        throw_gate_failure(m_gate_report);
    }

    m_gate_report.coverage_active_dof_count = coverage.active_dof_count;
    m_gate_report.coverage_padding_dof_count = coverage.padding_dof_count;
    m_gate_report.complete_dof_coverage = true;
    m_dry_run_report.coverage_active_dof_count = coverage.active_dof_count;
    m_dry_run_report.coverage_padding_dof_count = coverage.padding_dof_count;
    m_dry_run_report.complete_dof_coverage = true;

    auto damping_attr =
        world().scene().config().find<Float>("linear_system/socu_approx/damping_shift");
    m_dry_run_report.damping_shift =
        damping_attr ? static_cast<double>(damping_attr->view()[0]) : 0.0;
    if(m_dry_run_report.damping_shift < 0.0)
    {
        m_dry_run_report.status_reason =
            std::string{to_string(SocuApproxGateReason::DirectionInvalid)};
        m_dry_run_report.status_detail =
            "linear_system/socu_approx/damping_shift must be non-negative";
        write_dry_run_report(m_dry_run_report);
        throw Exception{fmt::format("SocuApproxSolver direction invalid: {}",
                                    m_dry_run_report.status_detail)};
    }

    const cudaStream_t stream = system().stream();
    m_runtime->reserve(m_debug_validation);
    m_runtime->upload_mappings_once(m_host_old_to_chain, m_host_chain_to_old);

    initialize_structured_workspace<GlobalLinearSystem::StoreScalar, Runtime::Scalar>(
        stream,
        StructuredChainShape{static_cast<SizeT>(m_runtime->shape.horizon),
                             static_cast<SizeT>(m_runtime->shape.n),
                             static_cast<SizeT>(m_runtime->shape.nrhs),
                             true},
        info.b(),
        m_runtime->device_diag.view(),
        m_runtime->device_off_diag.view(),
        m_runtime->device_rhs.view(),
        m_runtime->device_rhs_original.view(),
        m_runtime->device_chain_to_old.view(),
        m_dry_run_report.damping_shift);

    info.set_workspace(
        StructuredChainShape{static_cast<SizeT>(m_runtime->shape.horizon),
                             static_cast<SizeT>(m_runtime->shape.n),
                             static_cast<SizeT>(m_runtime->shape.nrhs),
                             true},
        span<const StructuredDofSlot>{m_dof_slots},
        m_runtime->device_diag.view(),
        m_runtime->device_off_diag.view(),
        m_runtime->device_rhs.view(),
        m_runtime->device_old_to_chain.view(),
        m_runtime->device_chain_to_old.view(),
        stream);

    m_dry_run_report.packed = true;
    m_dry_run_report.active_rhs_scalar_count = info.b().size();
    m_dry_run_report.rhs_scalar_count = m_runtime->layout.rhs_element_count;
    m_dry_run_report.diag_scalar_count = m_runtime->layout.diag_element_count;
    m_dry_run_report.first_offdiag_scalar_count =
        m_runtime->layout.off_diag_element_count;
    m_dry_run_report.diag_block_count = m_runtime->layout.diag_block_count;
    m_dry_run_report.first_offdiag_block_count =
        m_runtime->layout.off_diag_block_count;
    m_dry_run_report.stream_source = "mixed_backend_current_stream";
#endif
}

void SocuApproxSolver::finalize_structured_chain(
    GlobalLinearSystem::StructuredAssemblyInfo& info)
{
    if(m_mode != "solve")
        return;
    m_dry_run_report.structured_diag_write_count =
        info.contact_diag_write_count();
    m_dry_run_report.structured_first_offdiag_write_count =
        info.contact_first_offdiag_write_count();
    m_dry_run_report.near_band_contact_count = info.near_band_contact_count();
    m_dry_run_report.off_band_contact_count = info.off_band_contact_count();
    m_dry_run_report.near_band_contribution_count =
        info.near_band_contribution_count();
    m_dry_run_report.off_band_contribution_count =
        info.off_band_contribution_count();
    m_dry_run_report.absorbed_hessian_contribution_count =
        info.near_band_contribution_count();
    m_dry_run_report.dropped_hessian_contribution_count =
        info.off_band_contribution_count();

    const SizeT total_contact_count =
        info.near_band_contact_count() + info.off_band_contact_count();
    if(total_contact_count > 0)
    {
        m_dry_run_report.contribution_near_band_ratio =
            static_cast<double>(info.near_band_contact_count())
            / static_cast<double>(total_contact_count);
        m_dry_run_report.contribution_off_band_ratio =
            static_cast<double>(info.off_band_contact_count())
            / static_cast<double>(total_contact_count);
    }

    const SizeT total_contribution_count =
        info.near_band_contribution_count() + info.off_band_contribution_count();
    if(total_contribution_count > 0)
    {
        m_dry_run_report.weighted_near_band_ratio =
            static_cast<double>(info.near_band_contribution_count())
            / static_cast<double>(total_contribution_count);
        m_dry_run_report.weighted_off_band_ratio =
            static_cast<double>(info.off_band_contribution_count())
            / static_cast<double>(total_contribution_count);
    }

    if(info.off_band_contribution_count() > 0)
    {
        m_gate_report =
            make_failure(SocuApproxGateReason::ContactOffBandRatioTooHigh,
                         fmt::format("runtime contact generated {} off-band "
                                     "structured contribution(s); first M8 "
                                     "strict solve only absorbs near-band contact",
                                     info.off_band_contribution_count()),
                         m_gate_report.ordering_report_path);
        m_gate_report.block_size = m_dry_run_report.block_size;
        m_gate_report.near_band_ratio =
            m_dry_run_report.weighted_near_band_ratio;
        m_gate_report.off_band_ratio =
            m_dry_run_report.weighted_off_band_ratio;
        m_dry_run_report.direction_available = false;
        m_dry_run_report.status_reason =
            std::string{to_string(SocuApproxGateReason::ContactOffBandRatioTooHigh)};
        m_dry_run_report.status_detail = m_gate_report.detail;
        write_dry_run_report(m_dry_run_report);
        throw_gate_failure(m_gate_report);
    }
#if UIPC_WITH_SOCU_NATIVE
    UIPC_ASSERT(m_runtime != nullptr,
                "SocuApproxSolver solve mode requires initialized runtime.");
    if(m_debug_validation)
        m_runtime->snapshot_matrix(info.stream());
#endif
}

void SocuApproxSolver::notify_line_search_result(
    const GlobalLinearSystem::LineSearchFeedback& feedback)
{
    m_last_line_search_feedback = feedback;
    m_has_line_search_feedback  = true;
    if(feedback.hit_max_iter || !feedback.accepted)
        ++m_line_search_reject_streak;
    else
        m_line_search_reject_streak = 0;

    m_dry_run_report.line_search_feedback_available = true;
    m_dry_run_report.line_search_accepted = feedback.accepted;
    m_dry_run_report.line_search_hit_max_iter = feedback.hit_max_iter;
    m_dry_run_report.line_search_iteration_count = feedback.iteration_count;
    m_dry_run_report.line_search_accepted_alpha = feedback.accepted_alpha;
    m_dry_run_report.line_search_reject_streak = m_line_search_reject_streak;

    if((m_report_each_solve || m_debug_validation || m_debug_timing)
       && !m_dry_run_report.report_path.empty())
        write_dry_run_report(m_dry_run_report);
}

void SocuApproxSolver::debug_validate_direction(cudaStream_t stream)
{
#if UIPC_WITH_SOCU_NATIVE
    using SolveScalar = GlobalLinearSystem::SolveScalar;
    UIPC_ASSERT(m_runtime != nullptr,
                "SocuApproxSolver solve mode requires initialized runtime.");

    std::array<double, 5> validation_sums{};
    validate_structured_direction<Runtime::Scalar>(
        stream,
        StructuredChainShape{static_cast<SizeT>(m_runtime->shape.horizon),
                             static_cast<SizeT>(m_runtime->shape.n),
                             static_cast<SizeT>(m_runtime->shape.nrhs),
                             true},
        m_runtime->device_diag_original.view(),
        m_runtime->device_off_diag_original.view(),
        m_runtime->device_rhs_original.view(),
        m_runtime->device_rhs.view(),
        m_runtime->device_chain_to_old.view(),
        m_runtime->validation_sums.view());
    SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(validation_sums.data(),
                                           m_runtime->validation_sums.data(),
                                           validation_sums.size() * sizeof(double),
                                           cudaMemcpyDeviceToHost,
                                           stream));
    cudaEvent_t validation_done = nullptr;
    SOCU_NATIVE_CHECK_CUDA(cudaEventCreate(&validation_done));
    SOCU_NATIVE_CHECK_CUDA(cudaEventRecord(validation_done, stream));
    SOCU_NATIVE_CHECK_CUDA(cudaEventSynchronize(validation_done));
    SOCU_NATIVE_CHECK_CUDA(cudaEventDestroy(validation_done));

    m_dry_run_report.gradient_norm = std::sqrt(validation_sums[0]);
    m_dry_run_report.direction_norm = std::sqrt(validation_sums[1]);
    m_dry_run_report.descent_dot = -validation_sums[2];
    m_dry_run_report.surrogate_residual = std::sqrt(validation_sums[3]);
    m_dry_run_report.surrogate_relative_residual =
        m_dry_run_report.surrogate_residual
        / std::max(1.0, m_dry_run_report.gradient_norm);

    auto eta_attr =
        world().scene().config().find<Float>("linear_system/socu_approx/descent_eta");
    auto residual_attr =
        world().scene().config().find<Float>("linear_system/socu_approx/max_relative_residual");
    auto p_min_abs_attr =
        world().scene().config().find<Float>("linear_system/socu_approx/p_min_abs");
    auto p_min_rel_attr =
        world().scene().config().find<Float>("linear_system/socu_approx/p_min_rel");
    const double descent_eta =
        eta_attr ? static_cast<double>(eta_attr->view()[0]) : 1e-8;
    const double max_relative_residual =
        residual_attr ? static_cast<double>(residual_attr->view()[0]) : 1e-3;
    const double p_min_abs_default =
        sizeof(SolveScalar) == sizeof(float) ? 1e-10 : 1e-14;
    const double p_min_abs =
        p_min_abs_attr ? static_cast<double>(p_min_abs_attr->view()[0])
                       : p_min_abs_default;
    const double p_min_rel =
        p_min_rel_attr ? static_cast<double>(p_min_rel_attr->view()[0]) : 1e-12;
    m_dry_run_report.direction_min_abs_threshold = p_min_abs;
    m_dry_run_report.direction_min_rel_threshold = p_min_rel;
    m_dry_run_report.rhs_sign_convention = "rhs_is_global_b";

    const double p_threshold =
        std::max(p_min_abs, p_min_rel * m_dry_run_report.gradient_norm);
    const bool finite =
        std::isfinite(m_dry_run_report.surrogate_residual)
        && std::isfinite(m_dry_run_report.surrogate_relative_residual)
        && std::isfinite(m_dry_run_report.descent_dot)
        && std::isfinite(m_dry_run_report.gradient_norm)
        && std::isfinite(m_dry_run_report.direction_norm);
    const bool nonzero =
        m_dry_run_report.gradient_norm > 0.0
        && m_dry_run_report.direction_norm > p_threshold;
    const bool descent =
        m_dry_run_report.descent_dot
        < -descent_eta * m_dry_run_report.gradient_norm
               * m_dry_run_report.direction_norm;
    const bool residual_ok =
        m_dry_run_report.surrogate_relative_residual <= max_relative_residual;
    const bool zero_rhs_converged =
        finite
        && m_dry_run_report.gradient_norm <= p_min_abs
        && m_dry_run_report.direction_norm <= p_threshold
        && residual_ok;

    if(!zero_rhs_converged && (!finite || !nonzero || !descent || !residual_ok))
    {
        m_dry_run_report.direction_available = false;
        m_dry_run_report.status_reason =
            std::string{to_string(SocuApproxGateReason::DirectionInvalid)};
        m_dry_run_report.status_detail =
            fmt::format("direction validation failed: finite={}, nonzero={}, descent={}, "
                        "residual_ok={}, g_dot_p={}, g_norm={}, p_norm={}, "
                        "p_threshold={}, rel_residual={}, max_relative_residual={}",
                        finite,
                        nonzero,
                        descent,
                        residual_ok,
                        m_dry_run_report.descent_dot,
                        m_dry_run_report.gradient_norm,
                        m_dry_run_report.direction_norm,
                        p_threshold,
                        m_dry_run_report.surrogate_relative_residual,
                        max_relative_residual);
        write_dry_run_report(m_dry_run_report);
        throw Exception{fmt::format("SocuApproxSolver direction invalid: {}",
                                    m_dry_run_report.status_detail)};
    }
#else
    (void)stream;
#endif
}

void SocuApproxSolver::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    using StoreScalar = GlobalLinearSystem::StoreScalar;
    using SolveScalar = GlobalLinearSystem::SolveScalar;

    if(m_mode == "dry_run")
    {
        CpuStructuredDryRunSink sink{m_dry_run_report.block_size,
                                     m_dry_run_report.blocks};
        std::vector<StoreScalar> host_rhs(info.b().size());

        {
            Timer timer{"SocuApprox Dry Run Pack"};
            auto  start = std::chrono::steady_clock::now();

            m_dry_run_report.packed = true;
            m_dry_run_report.active_rhs_scalar_count = info.b().size();
            m_dry_run_report.diag_block_count = m_dry_run_report.blocks.size();
            m_dry_run_report.first_offdiag_block_count =
                m_dry_run_report.blocks.empty() ? 0 : m_dry_run_report.blocks.size() - 1;

            if(!host_rhs.empty())
                info.b().buffer_view().copy_to(host_rhs.data());

            for(const StructuredDofSlot& slot : m_dof_slots)
            {
                if(slot.is_padding || slot.old_dof < 0)
                    continue;
                const auto old_dof = static_cast<SizeT>(slot.old_dof);
                if(old_dof < host_rhs.size())
                    sink.add_rhs(slot.block,
                                 slot.lane,
                                 static_cast<double>(host_rhs[old_dof]));
            }

            load_contact_report(m_dry_run_report, sink);
            sink.finalize(m_dry_run_report);

            auto stop = std::chrono::steady_clock::now();
            m_dry_run_report.dry_run_pack_time_ms =
                std::chrono::duration<double, std::milli>(stop - start).count();
        }

        m_dry_run_report.direction_available = false;
        m_dry_run_report.status_reason =
            std::string{to_string(SocuApproxGateReason::StubNoDirection)};
        m_dry_run_report.status_detail =
            "M5 structured dry-run pack completed, but no socu solve direction is available";
        write_dry_run_report(m_dry_run_report);

        m_gate_report.reason = SocuApproxGateReason::StubNoDirection;
        m_gate_report.detail = m_dry_run_report.status_detail;
        info.x().buffer_view().fill(static_cast<SolveScalar>(0));
        info.iter_count(0);
        logger::warn("SocuApproxSolver M5 dry-run completed without a solve direction; "
                     "returning a zero direction. report='{}'",
                     m_dry_run_report.report_path);
        return;
    }

#if !UIPC_WITH_SOCU_NATIVE
    throw Exception{"SocuApproxSolver solve mode reached without socu_native support"};
#else
    UIPC_ASSERT(m_runtime != nullptr,
                "SocuApproxSolver solve mode requires initialized runtime.");

    const cudaStream_t stream = system().stream();

    try
    {
        m_dry_run_report.plan_created_this_solve =
            m_runtime->factor_and_solve(stream);
    }
    catch(const std::exception& e)
    {
        m_dry_run_report.direction_available = false;
        m_dry_run_report.status_reason =
            std::string{to_string(SocuApproxGateReason::SocuRuntimeError)};
        m_dry_run_report.status_detail =
            fmt::format("socu_native factor_and_solve failed: {}", e.what());
        write_dry_run_report(m_dry_run_report);
        throw Exception{fmt::format("SocuApproxSolver solve failed: {}",
                                    m_dry_run_report.status_detail)};
    }

    if(m_debug_validation)
        debug_validate_direction(stream);

    scatter_structured_solution<Runtime::Scalar>(
        stream,
        m_runtime->device_rhs.view(),
        m_runtime->device_old_to_chain.view(),
        info.x());

    m_dry_run_report.direction_available = true;
    m_dry_run_report.status_reason = std::string{to_string(SocuApproxGateReason::None)};
    m_dry_run_report.status_detail =
        "M7 strict ABD-only structured direction solved and scattered";
    m_gate_report.reason = SocuApproxGateReason::None;
    m_gate_report.detail = m_dry_run_report.status_detail;
    if(m_report_each_solve || m_debug_validation || m_debug_timing)
        write_dry_run_report(m_dry_run_report);

    info.iter_count(1);
    logger::info("SocuApproxSolver M7 strict solve launched on mixed backend stream: "
                 "debug_validation={}, report='{}'",
                 m_debug_validation,
                 m_dry_run_report.report_path);
#endif
}
}  // namespace uipc::backend::cuda_mixed
