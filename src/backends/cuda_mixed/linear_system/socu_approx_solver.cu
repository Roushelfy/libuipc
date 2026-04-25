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
#include <algorithm>
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
    muda::DeviceBuffer<Scalar> device_rhs;

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

    void reserve()
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
    }

    void upload(const std::vector<Scalar>& diag,
                const std::vector<Scalar>& off_diag,
                const std::vector<Scalar>& rhs,
                cudaStream_t stream)
    {
        if(diag.size() != layout.diag_element_count
           || off_diag.size() != layout.off_diag_element_count
           || rhs.size() != layout.rhs_element_count)
        {
            throw Exception{fmt::format(
                "SocuApproxSolver runtime buffer size mismatch: diag {}/{}, off_diag {}/{}, rhs {}/{}",
                diag.size(),
                layout.diag_element_count,
                off_diag.size(),
                layout.off_diag_element_count,
                rhs.size(),
                layout.rhs_element_count)};
        }

        reserve();
        SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(device_diag.data(),
                                               diag.data(),
                                               diag.size() * sizeof(Scalar),
                                               cudaMemcpyHostToDevice,
                                               stream));
        SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(device_off_diag.data(),
                                               off_diag.data(),
                                               off_diag.size() * sizeof(Scalar),
                                               cudaMemcpyHostToDevice,
                                               stream));
        SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(device_rhs.data(),
                                               rhs.data(),
                                               rhs.size() * sizeof(Scalar),
                                               cudaMemcpyHostToDevice,
                                               stream));
    }

    void factor_and_solve(cudaStream_t stream)
    {
        ensure_plan();
        socu_native::factor_and_solve_inplace_async(
            plan.get(),
            device_diag.data(),
            device_off_diag.data(),
            device_rhs.data(),
            socu_native::LaunchOptions{stream});
    }

    std::vector<Scalar> download_solution(cudaStream_t stream) const
    {
        std::vector<Scalar> solution(layout.rhs_element_count);
        SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(solution.data(),
                                               device_rhs.data(),
                                               solution.size() * sizeof(Scalar),
                                               cudaMemcpyDeviceToHost,
                                               stream));
        SOCU_NATIVE_CHECK_CUDA(cudaStreamSynchronize(stream));
        return solution;
    }

  private:
    void ensure_plan()
    {
        if(plan != nullptr)
            return;
        plan.reset(socu_native::create_solver_plan<Scalar>(shape, options));
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

fs::path discover_warp_so()
{
    if(const char* env = std::getenv("SOCU_NATIVE_WARP_SO"))
    {
        const fs::path candidate{env};
        if(fs::is_regular_file(candidate))
            return candidate;
    }

    const auto source_dir = fs::path{SOCU_NATIVE_SOURCE_DIR};
    if(source_dir.empty())
        return {};

    const auto venv_root = source_dir / ".venv" / "lib";
    if(!fs::is_directory(venv_root))
        return {};

    for(const auto& entry : fs::directory_iterator(venv_root))
    {
        if(!entry.is_directory())
            continue;
        if(entry.path().filename().string().rfind("python", 0) != 0)
            continue;
        const auto candidate = entry.path() / "site-packages" / "warp" / "bin" / "warp.so";
        if(fs::is_regular_file(candidate))
            return candidate;
    }
    return {};
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
    }

    bool is_available() const override { return !m_slots.empty(); }
    StructuredChainShape shape() const override { return m_shape; }
    span<const StructuredDofSlot> dof_slots() const override { return m_slots; }
    StructuredQualityReport quality_report() const override { return m_quality; }
    void assemble_chain(StructuredAssemblySink&) override {}

  private:
    StructuredChainShape          m_shape;
    StructuredQualityReport       m_quality;
    std::vector<StructuredDofSlot> m_slots;
};

class CpuStructuredDryRunSink final : public StructuredAssemblySink
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
        const SizeT row          = ij_is_forward ? lane_j : lane_i;
        const SizeT col          = ij_is_forward ? lane_i : lane_j;
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
        report.structured_diag_write_count = m_diag_write_count;
        report.structured_first_offdiag_write_count = m_first_offdiag_write_count;
        report.structured_off_band_drop_count = m_off_band_drop_count;
        report.structured_diag_contact_abs_sum = m_diag_contact_abs_sum;
        report.structured_first_offdiag_contact_abs_sum =
            m_first_offdiag_contact_abs_sum;
        report.structured_off_band_drop_abs_sum = m_off_band_drop_abs_sum;
        report.rhs_abs_sum = m_rhs_abs_sum;
    }

    void apply_damping(double shift)
    {
        for(SizeT block = 0; block < m_blocks.size(); ++block)
        {
            for(SizeT lane = 0; lane < m_block_size; ++lane)
                m_diag[diag_index(block, lane, lane)] += shift;
        }
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

void apply_structured_contributions(const Json& contact, StructuredAssemblySink& sink)
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

void load_contact_report(SocuApproxDryRunReport& dry_run, StructuredAssemblySink& sink)
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
                 report.mode == "structured_surrogate_solve" ? 7 : 5},
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
                  {"scatter_time_ms", report.scatter_time_ms}}},
                {"solve",
                 {{"damping_shift", report.damping_shift},
                  {"surrogate_residual", report.surrogate_residual},
                  {"surrogate_relative_residual",
                   report.surrogate_relative_residual},
                  {"descent_dot", report.descent_dot},
                  {"gradient_norm", report.gradient_norm},
                  {"direction_norm", report.direction_norm}}},
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
    case SocuApproxGateReason::StubNoDirection:
        return "socu_approx_stub_no_direction";
    case SocuApproxGateReason::DirectionInvalid:
        return "direction_invalid";
    case SocuApproxGateReason::SocuRuntimeError:
        return "socu_runtime_error";
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
            "M4 socu_approx skeleton only accepts StoreScalar == SolveScalar",
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

    if(const Json* metrics = metrics_json(report, *candidate))
    {
        if(metrics->contains("near_band_ratio"))
            m_gate_report.near_band_ratio = metrics->at("near_band_ratio").get<double>();
        if(metrics->contains("off_band_ratio"))
            m_gate_report.off_band_ratio = metrics->at("off_band_ratio").get<double>();
    }

    auto min_near_attr =
        config.find<Float>("linear_system/socu_approx/min_near_band_ratio");
    auto max_off_attr =
        config.find<Float>("linear_system/socu_approx/max_off_band_ratio");
    const double min_near_band_ratio =
        min_near_attr ? static_cast<double>(min_near_attr->view()[0]) : 0.0;
    const double max_off_band_ratio =
        max_off_attr ? static_cast<double>(max_off_attr->view()[0]) : 1.0;
    if(m_gate_report.near_band_ratio < min_near_band_ratio
       || m_gate_report.off_band_ratio > max_off_band_ratio)
    {
        m_gate_report = make_failure(
            SocuApproxGateReason::OrderingQualityTooLow,
            fmt::format("ordering quality rejected: near_band_ratio={}, off_band_ratio={}, "
                        "min_near_band_ratio={}, max_off_band_ratio={}",
                        m_gate_report.near_band_ratio,
                        m_gate_report.off_band_ratio,
                        min_near_band_ratio,
                        max_off_band_ratio),
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

    m_dry_run_report                         = SocuApproxDryRunReport{};
    m_dry_run_report.mode =
        m_mode == "solve" ? "structured_surrogate_solve" : "structured_dry_run";
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
    m_dry_run_report.blocks             = std::move(block_layouts);
    m_dof_slots.assign(provider->dof_slots().begin(), provider->dof_slots().end());

    m_gate_report.passed = true;
    m_gate_report.reason = SocuApproxGateReason::None;
    m_gate_report.detail =
        m_mode == "solve"
            ? "M7 structured surrogate solve gate passed"
            : "M5 structured dry-run gate passed; solve direction is intentionally unavailable";

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

        const auto warp_so = discover_warp_so();
        if(warp_so.empty())
        {
            m_gate_report = make_failure(
                SocuApproxGateReason::SocuRuntimeArtifactUnavailable,
                "Warp native library is missing; set SOCU_NATIVE_WARP_SO",
                ordering_report_path.string());
            m_gate_report.block_size = m_dry_run_report.block_size;
            throw_gate_failure(m_gate_report);
        }

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
                 m_mode == "solve" ? "M7 surrogate solve" : "M5 dry-run",
                 m_dry_run_report.block_size,
                 m_dry_run_report.block_count,
                 m_dry_run_report.report_path);
}

void SocuApproxSolver::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    using StoreScalar = GlobalLinearSystem::StoreScalar;
    using SolveScalar = GlobalLinearSystem::SolveScalar;

    CpuStructuredDryRunSink sink{m_dry_run_report.block_size,
                                 m_dry_run_report.blocks};
    std::vector<StoreScalar> host_rhs(info.b().size());
    std::vector<SolveScalar> structured_rhs;
    std::vector<SolveScalar> structured_diag;
    std::vector<SolveScalar> structured_off_diag;

    {
        Timer timer{"SocuApprox Dry Run Pack"};
        auto  start = std::chrono::steady_clock::now();

        m_dry_run_report.packed                    = true;
        m_dry_run_report.active_rhs_scalar_count   = info.b().size();
        m_dry_run_report.diag_block_count          = m_dry_run_report.blocks.size();
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

        if(m_mode == "solve")
        {
            auto damping_attr =
                world().scene().config().find<Float>("linear_system/socu_approx/damping_shift");
            m_dry_run_report.damping_shift =
                damping_attr ? static_cast<double>(damping_attr->view()[0]) : 1.0;
            if(!(m_dry_run_report.damping_shift > 0.0))
            {
                m_dry_run_report.status_reason =
                    std::string{to_string(SocuApproxGateReason::DirectionInvalid)};
                m_dry_run_report.status_detail =
                    "linear_system/socu_approx/damping_shift must be positive";
                write_dry_run_report(m_dry_run_report);
                throw Exception{fmt::format("SocuApproxSolver direction invalid: {}",
                                            m_dry_run_report.status_detail)};
            }
            sink.apply_damping(m_dry_run_report.damping_shift);
        }

        sink.finalize(m_dry_run_report);

#if UIPC_WITH_SOCU_NATIVE
        if(m_mode == "solve")
        {
            UIPC_ASSERT(m_runtime != nullptr,
                        "SocuApproxSolver solve mode requires initialized runtime.");
            structured_rhs = sink.rhs_as<SolveScalar>();
            structured_diag = sink.diag_as<SolveScalar>();
            structured_off_diag =
                sink.off_diag_as<SolveScalar>(m_runtime->layout.off_diag_element_count);
        }
#endif

        auto stop = std::chrono::steady_clock::now();
        m_dry_run_report.dry_run_pack_time_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
    }

    if(m_mode == "dry_run")
    {
        m_dry_run_report.direction_available = false;
        m_dry_run_report.status_reason =
            std::string{to_string(SocuApproxGateReason::StubNoDirection)};
        m_dry_run_report.status_detail =
            "M5 structured dry-run pack completed, but no socu solve direction is available yet";
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

#if UIPC_WITH_SOCU_NATIVE
    std::vector<SolveScalar> structured_solution;
    try
    {
        Timer timer{"SocuApprox Factor Solve"};
        auto  start = std::chrono::steady_clock::now();
        cudaStream_t stream = nullptr;
        m_runtime->upload(structured_diag, structured_off_diag, structured_rhs, stream);
        m_runtime->factor_and_solve(stream);
        SOCU_NATIVE_CHECK_CUDA(cudaStreamSynchronize(stream));
        structured_solution = m_runtime->download_solution(stream);
        auto stop = std::chrono::steady_clock::now();
        m_dry_run_report.socu_factor_solve_time_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
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

    m_dry_run_report.surrogate_residual =
        socu_native::residual_norm(structured_diag,
                                   structured_off_diag,
                                   structured_rhs,
                                   structured_solution,
                                   m_runtime->shape);
    m_dry_run_report.gradient_norm = norm2(structured_rhs);
    m_dry_run_report.direction_norm = norm2(structured_solution);
    m_dry_run_report.surrogate_relative_residual =
        m_dry_run_report.surrogate_residual
        / std::max(1.0, m_dry_run_report.gradient_norm);
    m_dry_run_report.descent_dot = -dot(structured_rhs, structured_solution);

    auto eta_attr =
        world().scene().config().find<Float>("linear_system/socu_approx/descent_eta");
    auto residual_attr =
        world().scene().config().find<Float>("linear_system/socu_approx/max_relative_residual");
    const double descent_eta =
        eta_attr ? static_cast<double>(eta_attr->view()[0]) : 1e-12;
    const double max_relative_residual =
        residual_attr ? static_cast<double>(residual_attr->view()[0]) : 1e-4;

    const bool finite =
        std::isfinite(m_dry_run_report.surrogate_residual)
        && std::isfinite(m_dry_run_report.surrogate_relative_residual)
        && std::isfinite(m_dry_run_report.descent_dot)
        && std::isfinite(m_dry_run_report.gradient_norm)
        && std::isfinite(m_dry_run_report.direction_norm);
    const bool nonzero =
        m_dry_run_report.gradient_norm > 0.0 && m_dry_run_report.direction_norm > 0.0;
    const bool descent =
        m_dry_run_report.descent_dot
        < -descent_eta * m_dry_run_report.gradient_norm
               * m_dry_run_report.direction_norm;
    const bool residual_ok =
        m_dry_run_report.surrogate_relative_residual <= max_relative_residual;

    if(!finite || !nonzero || !descent || !residual_ok)
    {
        m_dry_run_report.direction_available = false;
        m_dry_run_report.status_reason =
            std::string{to_string(SocuApproxGateReason::DirectionInvalid)};
        m_dry_run_report.status_detail =
            fmt::format("direction validation failed: finite={}, nonzero={}, descent={}, "
                        "residual_ok={}, g_dot_p={}, g_norm={}, p_norm={}, rel_residual={}, "
                        "max_relative_residual={}",
                        finite,
                        nonzero,
                        descent,
                        residual_ok,
                        m_dry_run_report.descent_dot,
                        m_dry_run_report.gradient_norm,
                        m_dry_run_report.direction_norm,
                        m_dry_run_report.surrogate_relative_residual,
                        max_relative_residual);
        write_dry_run_report(m_dry_run_report);
        throw Exception{fmt::format("SocuApproxSolver direction invalid: {}",
                                    m_dry_run_report.status_detail)};
    }

    {
        Timer timer{"SocuApprox Scatter Direction"};
        auto  start = std::chrono::steady_clock::now();
        std::vector<SolveScalar> host_x(info.x().size(), SolveScalar{0});
        for(const StructuredDofSlot& slot : m_dof_slots)
        {
            if(!slot.scatter_write || slot.is_padding || slot.old_dof < 0
               || slot.chain_dof < 0)
                continue;
            const auto old_dof = static_cast<SizeT>(slot.old_dof);
            const auto chain_dof = static_cast<SizeT>(slot.chain_dof);
            if(old_dof < host_x.size() && chain_dof < structured_solution.size())
                host_x[old_dof] = structured_solution[chain_dof];
        }
        if(!host_x.empty())
            info.x().buffer_view().copy_from(host_x.data());
        auto stop = std::chrono::steady_clock::now();
        m_dry_run_report.scatter_time_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
    }

    m_dry_run_report.direction_available = true;
    m_dry_run_report.status_reason = std::string{to_string(SocuApproxGateReason::None)};
    m_dry_run_report.status_detail = "M7 structured surrogate direction solved and scattered";
    write_dry_run_report(m_dry_run_report);

    m_gate_report.reason = SocuApproxGateReason::None;
    m_gate_report.detail = m_dry_run_report.status_detail;
    info.iter_count(1);

    logger::info("SocuApproxSolver M7 solve completed: rel_residual={}, g_dot_p={}, "
                 "pack_ms={}, solve_ms={}, scatter_ms={}, report='{}'",
                 m_dry_run_report.surrogate_relative_residual,
                 m_dry_run_report.descent_dot,
                 m_dry_run_report.dry_run_pack_time_ms,
                 m_dry_run_report.socu_factor_solve_time_ms,
                 m_dry_run_report.scatter_time_ms,
                 m_dry_run_report.report_path);
#else
    throw Exception{"SocuApproxSolver solve mode reached without socu_native support"};
#endif
}
}  // namespace uipc::backend::cuda_mixed
