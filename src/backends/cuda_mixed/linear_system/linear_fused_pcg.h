#pragma once
#include <linear_system/iterative_solver.h>
#include <muda/buffer/device_var.h>

namespace uipc::backend::cuda_mixed
{
// Fused PCG: keeps dot-product scalars (rz, pAp, rz_new) on device
// to eliminate per-iteration host synchronizations.  The update kernels read
// alpha = rz/pAp and beta = rz_new/rz directly from device memory.
// SpMV and dot(p,Ap) are fused into a single kernel pass.
// Convergence is checked every `check_interval` iterations via a single D2H copy.
class LinearFusedPCG : public IterativeSolver
{
  public:
    using IterativeSolver::IterativeSolver;

  protected:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) override;

  private:
    using PcgScalar = ActivePolicy::PcgAuxScalar;
    using StoreScalar = ActivePolicy::StoreScalar;
    using SolveScalar = ActivePolicy::SolveScalar;
    using IterScalar = ActivePolicy::PcgIterScalar;
    using DeviceDenseVector = muda::DeviceDenseVector<PcgScalar>;

    SizeT fused_pcg(muda::DenseVectorView<SolveScalar>  x,
                    muda::CDenseVectorView<StoreScalar> b,
                    SizeT                         max_iter);

    DeviceDenseVector r;
    DeviceDenseVector z;
    DeviceDenseVector p;
    DeviceDenseVector Ap;

    muda::DeviceVar<IterScalar> d_rz;
    muda::DeviceVar<IterScalar> d_pAp;
    muda::DeviceVar<IterScalar> d_rz_new;
    muda::DeviceVar<IndexT>     d_converged;

    double max_iter_ratio  = 2.0;
    double global_tol_rate = 1e-4;
    double reserve_ratio   = 1.5;
    SizeT check_interval  = 5;
};
}  // namespace uipc::backend::cuda_mixed

