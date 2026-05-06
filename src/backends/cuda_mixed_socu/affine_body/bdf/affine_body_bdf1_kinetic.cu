#include <affine_body/affine_body_kinetic.h>
#include <time_integrator/bdf1_flag.h>
#include <muda/ext/eigen/evd.h>
#include <kernel_cout.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
class AffineBodyBDF1Kinetic final : public AffineBodyKinetic
{
  public:
    using AffineBodyKinetic::AffineBodyKinetic;

    virtual void do_build(BuildInfo& info) override
    {
        // need BDF1 flag for BDF1 time integration
        require<BDF1Flag>();
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.qs().size(),
                   [is_fixed   = info.is_fixed().cviewer().name("is_fixed"),
                    is_dynamic = info.is_dynamic().cviewer().name("is_dynamic"),
                    ext_kinetic = info.external_kinetic().cviewer().name("ext_kinetic"),
                    qs        = info.qs().cviewer().name("qs"),
                    q_prevs   = info.q_prevs().cviewer().name("q_tildes"),
                    q_tildes  = info.q_tildes().cviewer().name("q_tildes"),
                    gravities = info.gravities().cviewer().name("gravities"),
                    masses    = info.masses().cviewer().name("masses"),
                   Ks = info.energies().viewer().name("kinetic_energy")] __device__(int i) mutable
                   {
                       using Alu = ActivePolicy::AluScalar;
                       using Energy = ActivePolicy::EnergyScalar;
                       auto& K = Ks(i);
                       if(is_fixed(i) || ext_kinetic(i))
                       {
                           K = Energy{0};
                       }
                       else
                       {
                           const auto& q       = qs(i);
                           const auto& q_tilde = q_tildes(i);
                           const auto& M       = masses(i);
                           Eigen::Matrix<Alu, 12, 1> dq_alu =
                               (q - q_tilde).template cast<Alu>();
                           Eigen::Matrix<Alu, 12, 12> M_alu =
                               M.template to_mat_t<Alu>();
                           Alu K_alu = safe_cast<Alu>(0.5)
                                       * dq_alu.dot((M_alu * dq_alu).eval());
                           K = safe_cast<Energy>(K_alu);
                       }
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.qs().size(),
                   [is_fixed   = info.is_fixed().cviewer().name("is_fixed"),
                    is_dynamic = info.is_dynamic().cviewer().name("is_dynamic"),
                    qs         = info.qs().cviewer().name("qs"),
                    q_prevs    = info.q_prevs().cviewer().name("q_tildes"),
                    q_tildes   = info.q_tildes().cviewer().name("q_tildes"),
                    gravities  = info.gravities().cviewer().name("gravities"),
                    masses     = info.masses().cviewer().name("masses"),
                    hessians   = info.hessians().viewer().name("hessians"),
                    gradients  = info.gradients().viewer().name("gradients"),
                    dt         = info.dt(),
                    gradient_only = info.gradient_only(),
                   cout = KernelCout::viewer()] __device__(int i) mutable
                   {
                       using Alu = ActivePolicy::AluScalar;
                       using Store = ActivePolicy::StoreScalar;
                       const auto& q       = qs(i);
                       const auto& q_prev  = q_prevs(i);
                       const auto& q_tilde = q_tildes(i);
                       auto&       G       = gradients(i);
                       const auto& M       = masses(i);

                       Eigen::Matrix<Alu, 12, 12> M_alu =
                           M.template to_mat_t<Alu>();
                       Eigen::Matrix<Alu, 12, 1> G_alu =
                           (M_alu * (q - q_tilde).template cast<Alu>()).eval();


                       if(is_fixed(i))
                       {
                           G_alu.setZero();
                       }
                       G = downcast_gradient<Store>(G_alu);

                       // cout << "KG(" << i << "): " << G.transpose().eval() << "\n";

                       if(gradient_only)
                           return;

                       hessians(i) = downcast_hessian<Store>(M_alu);
                   });
    }
};

REGISTER_SIM_SYSTEM(AffineBodyBDF1Kinetic);
}  // namespace uipc::backend::cuda_mixed
