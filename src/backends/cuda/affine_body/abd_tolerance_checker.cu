#include <newton_tolerance/newton_tolerance_checker.h>
#include <affine_body/affine_body_dynamics.h>
#include <uipc/geometry/attribute_slot.h>
#include <muda/cub/device/device_reduce.h>

namespace uipc::backend::cuda
{
class ABDToleranceChecker final : public NewtonToleranceChecker
{
  public:
    using NewtonToleranceChecker::NewtonToleranceChecker;

    SimSystemSlot<AffineBodyDynamics>       affine_body_dynamics;
    S<const geometry::AttributeSlot<Float>> dt_attr;
    Float                                   transrate_tol = 0.0;
    Float                                   abs_tol       = 0.0;
    muda::DeviceBuffer<Float>               per_body_residual;
    muda::DeviceVar<Float>                  reduced_residual;
    Float                                   h_residual = 0.0;

    // Inherited via NewtonToleranceChecker
    void do_build(BuildInfo& info) override
    {
        affine_body_dynamics = require<AffineBodyDynamics>();
        auto& config         = world().scene().config();
        dt_attr              = config.find<Float>("dt");
        UIPC_ASSERT(dt_attr, "Scene config must have a 'dt' attribute.");
        auto transrate_tol_attr = config.find<Float>("newton/transrate_tol");
        transrate_tol           = transrate_tol_attr->view()[0];
    }

    void do_init(InitInfo& info) override {}

    void do_pre_newton(PreNewtonInfo& info) override {}

    void do_check(CheckResultInfo& info) override
    {
        abs_tol  = transrate_tol * dt_attr->view()[0];
        auto dqs = affine_body_dynamics->dqs();
        using namespace muda;

        if(dqs.size() == 0)
        {
            h_residual = 0.0;
            info.converged(true);
            return;
        }

        per_body_residual.resize(dqs.size());

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(dqs.size(),
                   [dqs = dqs.cviewer().name("dqs"),
                    per_body_residual =
                        per_body_residual.viewer().name("per_body_residual")] __device__(int I) mutable
                   {
                       const Vector12& dq      = dqs(I);
                       Float           max_val = 0.0;
                       // the first 3 components are translation, ignore
                       // the rest 9 components are rotation/scaling/shear
                       for(IndexT i = 3; i < 12; ++i)
                       {
                           Float a = abs(dq[i]);
                           if(a > max_val)
                               max_val = a;
                       }
                       per_body_residual(I) = max_val;
                   });

        DeviceReduce().Max(per_body_residual.data(),
                           reduced_residual.data(),
                           per_body_residual.size());
        h_residual = reduced_residual;
        info.converged(h_residual <= abs_tol);
    }

    std::string do_report() override
    {
        return fmt::format("Residual/AbsTol: {}/{}", h_residual, abs_tol);
    }
};

REGISTER_SIM_SYSTEM(ABDToleranceChecker);
}  // namespace uipc::backend::cuda