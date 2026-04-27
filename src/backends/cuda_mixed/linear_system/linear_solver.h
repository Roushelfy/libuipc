#pragma once
#include <sim_system.h>
#include <linear_system/global_linear_system.h>
#include <linear_system/assembly_mode.h>
#include <string_view>

namespace uipc::backend::cuda_mixed
{
class LinearSolver : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    struct AssemblyRequirements
    {
        bool needs_dof_extent       = true;
        bool needs_gradient_b       = true;
        bool needs_full_sparse_A    = true;
        bool needs_structured_chain = false;
        bool allows_structured_offdiag = false;
        bool needs_preconditioner   = true;
        NewtonAssemblyMode assembly_mode = NewtonAssemblyMode::FullSparse;
    };

    class BuildInfo
    {
      public:
    };

    virtual AssemblyRequirements assembly_requirements() const
    {
        return AssemblyRequirements{};
    }

    virtual std::string_view iteration_counter_name() const { return {}; }

    GlobalLinearSystem& system() const noexcept { return *m_system; }

    virtual void prepare_structured_chain(
        GlobalLinearSystem::StructuredAssemblyInfo& info)
    {
    }

    virtual void finalize_structured_chain(
        GlobalLinearSystem::StructuredAssemblyInfo& info)
    {
    }

    virtual void notify_line_search_result(
        const GlobalLinearSystem::LineSearchFeedback& feedback)
    {
    }

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) = 0;
    GlobalLinearSystem* m_system = nullptr;

  private:
    friend class GlobalLinearSystem;

    virtual void do_build() final override;
    void solve(GlobalLinearSystem::SolvingInfo& info) { do_solve(info); }
};
}  // namespace uipc::backend::cuda_mixed
