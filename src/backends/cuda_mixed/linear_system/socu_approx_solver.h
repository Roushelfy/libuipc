#pragma once

#include <linear_system/linear_solver.h>
#include <linear_system/socu_approx_report.h>
#include <linear_system/structured_chain_provider.h>
#include <uipc/common/json.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace uipc::backend::cuda_mixed
{
struct SocuApproxRuntime;

class SocuApproxSolver : public LinearSolver
{
  public:
    using LinearSolver::LinearSolver;
    ~SocuApproxSolver();

    virtual AssemblyRequirements assembly_requirements() const override;

    const SocuApproxGateReport& gate_report() const noexcept { return m_gate_report; }
    const SocuApproxSolveReport& solve_report() const noexcept
    {
        return m_report;
    }

  protected:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) override;
    virtual void prepare_structured_chain(
        GlobalLinearSystem::StructuredAssemblyInfo& info) override;
    virtual void finalize_structured_chain(
        GlobalLinearSystem::StructuredAssemblyInfo& info) override;
    virtual void notify_line_search_result(
        const GlobalLinearSystem::LineSearchFeedback& feedback) override;

  private:
    using Runtime = SocuApproxRuntime;

    void validate_direction_light(cudaStream_t stream);
    void debug_validate_direction(cudaStream_t stream);
    void dump_structured_matrix(const GlobalLinearSystem::StructuredAssemblyInfo& info);
    void dump_problem_file(const GlobalLinearSystem::StructuredAssemblyInfo& info);
    bool install_ordering_report(const Json&                  report,
                                 const std::filesystem::path& ordering_report_path,
                                 bool                         throw_on_failure,
                                 std::string*                 failure_detail);
    void apply_pending_runtime_reorder();
    void begin_runtime_reorder_collection(cudaStream_t stream);
    void finalize_runtime_reorder_collection();

    SocuApproxGateReport  m_gate_report;
    SocuApproxSolveReport m_report;
    std::vector<StructuredDofSlot> m_dof_slots;
    std::vector<IndexT> m_host_old_to_chain;
    std::vector<IndexT> m_host_chain_to_old;
    std::vector<IndexT> m_host_old_dof_to_atom;
    std::vector<IndexT> m_host_atom_dof_count;
    SizeT       m_line_search_reject_streak = 0;
    GlobalLinearSystem::LineSearchFeedback m_last_line_search_feedback;
    bool        m_has_line_search_feedback = false;
    bool        m_debug_validation = false;
    bool        m_debug_timing = false;
    bool        m_debug_dump_structured_matrix = false;
    bool        m_debug_dump_problem_file = false;
    bool        m_debug_compare_full_sparse = false;
    bool        m_report_each_solve = false;
    bool        m_report_counters_enabled = false;
    bool        m_allows_structured_offdiag = true;
    bool        m_runtime_reorder_collecting = false;
    bool        m_runtime_reorder_pending = false;
    SizeT       m_runtime_reorder_frame_interval = 0;
    SizeT       m_runtime_reorder_edge_capacity_config = 0;
    SizeT       m_runtime_reorder_edge_capacity = 0;
    SizeT       m_runtime_reorder_collecting_frame = static_cast<SizeT>(-1);
    SizeT       m_runtime_reorder_last_applied_frame = static_cast<SizeT>(-1);
    SizeT       m_runtime_reorder_last_prepared_frame = static_cast<SizeT>(-1);
    std::string m_ordering_orderer = "rcm";
    std::string m_ordering_block_size = "64";
    std::string m_structured_scope = "multi_provider";
    double      m_damping_shift = 0.0;
    double      m_descent_eta = 1e-8;
    double      m_max_relative_residual = 1e-4;
    double      m_direction_min_abs = 0.0;
    double      m_direction_min_rel = 1e-12;
    double      m_rhs_zero_abs = 0.0;
    IndexT      m_max_line_search_reject_streak = 1;

    std::unique_ptr<Runtime> m_runtime;
};
}  // namespace uipc::backend::cuda_mixed
