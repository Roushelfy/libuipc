#pragma once
#include <linear_system/iterative_solver.h>
#include <muda/buffer/device_var.h>
#include <vector>

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
    struct PcgSample
    {
        SizeT  iter         = 0;
        double norm_r       = 0.0;
        double rz_ratio     = 0.0;
        double alpha        = 0.0;
        double beta         = 0.0;
        bool   nan_inf_flag = false;
    };

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

    bool  telemetry_pcg_enable      = false;
    SizeT telemetry_sample_every_iter = 10;
    std::vector<PcgSample> m_pcg_samples;
    std::vector<SizeT>     m_iter_history;
};
}  // namespace uipc::backend::cuda_mixed
