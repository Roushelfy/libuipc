#pragma once
#include <sim_system.h>
#include <muda/ext/linear_system.h>
#include <linear_system/global_linear_system.h>
#include <mixed_precision/policy.h>

namespace uipc::backend::cuda_mixed
{
class IterativeSolver : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

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
    GlobalLinearSystem* m_system;

    virtual void do_build() final override;

    void solve(GlobalLinearSystem::SolvingInfo& info);
};
}  // namespace uipc::backend::cuda_mixed
