#pragma once

#include <linear_system/linear_solver.h>
#include <linear_system/socu_approx_report.h>

namespace uipc::backend::cuda_mixed
{
class SocuApproxSolver : public LinearSolver
{
  public:
    using LinearSolver::LinearSolver;

    virtual AssemblyRequirements assembly_requirements() const override
    {
        AssemblyRequirements requirements;
        requirements.full_sparse_matrix = false;
        requirements.preconditioner     = false;
        return requirements;
    }

    const SocuApproxGateReport& gate_report() const noexcept { return m_gate_report; }

  protected:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) override;

  private:
    SocuApproxGateReport m_gate_report;
};
}  // namespace uipc::backend::cuda_mixed
