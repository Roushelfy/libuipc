#pragma once
#include <linear_system/iterative_solver.h>
#include <muda/buffer/device_var.h>

namespace uipc::backend::cuda_mixed
{
class LinearPCG : public IterativeSolver
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
    using PcgIterScalar = ActivePolicy::PcgIterScalar;
    using DeviceDenseVector = muda::DeviceDenseVector<PcgScalar>;
    using DeviceBCOOMatrix  = muda::DeviceBCOOMatrix<StoreScalar, 3>;
    using DeviceBSRMatrix   = muda::DeviceBSRMatrix<StoreScalar, 3>;

    SizeT pcg(muda::DenseVectorView<SolveScalar> x,
              muda::CDenseVectorView<StoreScalar> b,
              SizeT max_iter);
    void dump_r_z(SizeT k);
    void dump_p_Ap(SizeT k);
    void check_rz_nan_inf(SizeT k, PcgIterScalar rz);

    DeviceDenseVector z;   // preconditioned residual
    DeviceDenseVector r;   // residual
    DeviceDenseVector p;   // search direction
    DeviceDenseVector Ap;  // A*p
    muda::DeviceVar<IndexT> d_converged_false;

    double max_iter_ratio  = 2.0;
    double global_tol_rate = 1e-4;
    double reserve_ratio   = 1.5;

    bool        need_debug_dump = false;
    std::string debug_dump_path;
};
}  // namespace uipc::backend::cuda_mixed
