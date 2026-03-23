#include <time_integrator/bdf2_flag.h>
#include <finite_element/finite_element_kinetic.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
class FiniteElementBDF2Kinetic final : public FiniteElementKinetic
{
  public:
    static constexpr Float beta     = 4.0 / 9.0;  // BDF2 beta coefficient
    static constexpr Float inv_beta = Float{1} / beta;

    using FiniteElementKinetic::FiniteElementKinetic;

    virtual void do_build(BuildInfo& info) override
    {
        require<BDF2Flag>();
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.xs().size(),
                   [is_fixed = info.is_fixed().cviewer().name("is_fixed"),
                    xs       = info.xs().cviewer().name("xs"),
                    x_tildes = info.x_tildes().cviewer().name("x_tildes"),
                    masses   = info.masses().cviewer().name("masses"),
                    Ks       = info.energies().viewer().name("kinetic_energy"),
                    inv_beta = inv_beta] __device__(int i) mutable
                   {
                       auto& K = Ks(i);
                       if(is_fixed(i))
                       {
                           K = 0.0;
                       }
                       else
                       {
                           const Vector3& x       = xs(i);
                           const Vector3& x_tilde = x_tildes(i);
                           Float          M       = masses(i);
                           Vector3        dx      = x - x_tilde;
                           K                      = (inv_beta * Float{0.5}) * M * dx.dot(dx);
                       }
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        using StoreScalar = FEMLinearSubsystem::StoreScalar;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.xs().size(),
                   [is_fixed      = info.is_fixed().cviewer().name("is_fixed"),
                    xs            = info.xs().cviewer().name("xs"),
                    x_tildes      = info.x_tildes().viewer().name("x_tildes"),
                    masses        = info.masses().cviewer().name("masses"),
                    G3s           = info.gradients().viewer().name("G3s"),
                    H3x3s         = info.hessians().viewer().name("H3x3s"),
                    gradient_only = info.gradient_only(),
                    inv_beta      = inv_beta] __device__(int i) mutable
                   {
                       auto& m       = masses(i);
                       auto& x       = xs(i);
                       auto& x_tilde = x_tildes(i);

                       Vector3 G;

                       if(is_fixed(i))
                       {
                           G = Vector3::Zero();
                       }
                       else
                       {
                           G = inv_beta * m * (x - x_tilde);
                       }

                       G3s(i).write(i, downcast_gradient<StoreScalar>(G));

                       if(gradient_only)
                           return;

                       Matrix3x3 H = (inv_beta * masses(i)) * Matrix3x3::Identity();
                       H3x3s(i).write(i, i, downcast_hessian<StoreScalar>(H));
                   });
    }
};

REGISTER_SIM_SYSTEM(FiniteElementBDF2Kinetic);
}  // namespace uipc::backend::cuda_mixed
