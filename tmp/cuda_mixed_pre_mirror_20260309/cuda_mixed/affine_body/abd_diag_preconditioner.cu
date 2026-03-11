#include <linear_system/local_preconditioner.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <muda/ext/eigen/inverse.h>
#include <kernel_cout.h>
#include <mixed_precision/policy.h>

namespace uipc::backend::cuda_mixed
{
class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;
    using PrecondScalar =
        std::conditional_t<ActivePolicy::preconditioner_no_double_intermediate, float, double>;
    using PrecondMat12x12 = Eigen::Matrix<PrecondScalar, 12, 12>;
    using PrecondVec12    = Eigen::Matrix<PrecondScalar, 12, 1>;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    muda::DeviceBuffer<PrecondMat12x12> diag_inv;

    virtual void do_build(BuildInfo& info) override
    {
        auto& global_linear_system = require<GlobalLinearSystem>();
        abd_linear_subsystem       = &require<ABDLinearSubsystem>();

        info.connect(abd_linear_subsystem);
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        using namespace muda;

        auto diag_hessian = abd_linear_subsystem->diag_hessian();
        diag_inv.resize(diag_hessian.size());

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [diag_hessian = diag_hessian.viewer().name("diag_hessian"),
                    diag_inv = diag_inv.viewer().name("diag_inv")] __device__(int i) mutable
                   {
                       if constexpr(ActivePolicy::preconditioner_no_double_intermediate)
                       {
                           PrecondMat12x12 H = diag_hessian(i).template cast<PrecondScalar>();
                           diag_inv(i) = muda::eigen::inverse(H);
                       }
                       else
                       {
                           diag_inv(i) = muda::eigen::inverse(diag_hessian(i));
                       }
                   });
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        using namespace muda;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [r = info.r().viewer().name("r"),
                    z = info.z().viewer().name("z"),
                    diag_inv = diag_inv.viewer().name("diag_inv")] __device__(int i) mutable
                   {
                       PrecondVec12 r_p =
                           r.segment<12>(i * 12).as_eigen().template cast<PrecondScalar>();
                       auto z_p = diag_inv(i) * r_p;
                       z.segment<12>(i * 12).as_eigen() =
                           z_p.template cast<ActivePolicy::PcgAuxScalar>();
                   });
    }
};

REGISTER_SIM_SYSTEM(ABDDiagPreconditioner);
}  // namespace uipc::backend::cuda_mixed
