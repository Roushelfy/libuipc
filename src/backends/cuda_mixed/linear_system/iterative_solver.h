#pragma once
#include <linear_system/linear_solver.h>
#include <muda/ext/linear_system.h>
#include <mixed_precision/policy.h>

namespace uipc::backend::cuda_mixed
{
class IterativeSolver : public LinearSolver
{
  public:
    using LinearSolver::LinearSolver;
    virtual AssemblyRequirements assembly_requirements() const override
    {
        AssemblyRequirements requirements;
        requirements.needs_dof_extent       = true;
        requirements.needs_gradient_b       = true;
        requirements.needs_full_sparse_A    = true;
        requirements.needs_structured_chain = false;
        requirements.needs_preconditioner   = true;
        requirements.assembly_mode = NewtonAssemblyMode::FullSparse;
        return requirements;
    }

  protected:
    virtual void do_build(BuildInfo& info) = 0;

    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) = 0;


    /**********************************************************************************************
    * Util functions for derived classes
    ***********************************************************************************************/

    void spmv(ActivePolicy::PcgIterScalar             a,
              muda::CDenseVectorView<ActivePolicy::PcgAuxScalar> x,
              ActivePolicy::PcgIterScalar             b,
              muda::DenseVectorView<ActivePolicy::PcgAuxScalar> y);
    void spmv(muda::CDenseVectorView<ActivePolicy::PcgAuxScalar> x,
              muda::DenseVectorView<ActivePolicy::PcgAuxScalar> y);
    void apply_preconditioner(
        muda::DenseVectorView<ActivePolicy::PcgAuxScalar> z,
        muda::CDenseVectorView<ActivePolicy::PcgAuxScalar> r,
        muda::CVarView<IndexT>                             converged);
    bool accuracy_statisfied(muda::DenseVectorView<ActivePolicy::PcgAuxScalar> r);
    muda::LinearSystemContext& ctx() const;

  private:
    friend class GlobalLinearSystem;
};
}  // namespace uipc::backend::cuda_mixed
