#include <affine_body/affine_body_kinetic.h>
#include <time_integrator/bdf2_flag.h>
#include <kernel_cout.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
class AffineBodyBDF2Kinetic final : public AffineBodyKinetic
{
  public:
    static constexpr Float beta     = 4.0 / 9.0;  // BDF2 beta coefficient
    static constexpr Float inv_beta = Float{1} / beta;

    using AffineBodyKinetic::AffineBodyKinetic;

    virtual void do_build(BuildInfo& info) override
    {
        require<BDF2Flag>();
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.qs().size(),
                   [is_fixed    = info.is_fixed().cviewer().name("is_fixed"),
                    ext_kinetic = info.external_kinetic().cviewer().name("ext_kinetic"),
                    qs          = info.qs().cviewer().name("qs"),
                    q_tildes    = info.q_tildes().cviewer().name("q_tildes"),
                    masses      = info.masses().cviewer().name("masses"),
                    Ks          = info.energies().viewer().name("kinetic_energy"),
                    inv_beta    = inv_beta] __device__(int i) mutable
                   {
                       using Alu = ActivePolicy::AluScalar;
                       auto& K   = Ks(i);
                       if(is_fixed(i) || ext_kinetic(i))
                       {
                           K = 0.0;
                       }
                       else
                       {
                           const auto& q       = qs(i);
                           const auto& q_tilde = q_tildes(i);
                           const auto& M       = masses(i);
                           Eigen::Matrix<Alu, 12, 1> dq_alu =
                               (q - q_tilde).template cast<Alu>();
                           Eigen::Matrix<Alu, 12, 12> M_alu =
                               M.to_mat().template cast<Alu>();
                           const Alu inv_beta_alu = safe_cast<Alu>(inv_beta);
                           Alu K_alu = safe_cast<Alu>(0.5) * inv_beta_alu
                                       * dq_alu.dot((M_alu * dq_alu).eval());
                           K = safe_cast<Float>(K_alu);
                       }
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.qs().size(),
                   [is_fixed      = info.is_fixed().cviewer().name("is_fixed"),
                    qs            = info.qs().cviewer().name("qs"),
                    q_tildes      = info.q_tildes().cviewer().name("q_tildes"),
                    masses        = info.masses().cviewer().name("masses"),
                    hessians      = info.hessians().viewer().name("hessians"),
                    gradients     = info.gradients().viewer().name("gradients"),
                    gradient_only = info.gradient_only(),
                    inv_beta      = inv_beta] __device__(int i) mutable
                   {
                       using Alu = ActivePolicy::AluScalar;
                       const auto& q       = qs(i);
                       const auto& q_tilde = q_tildes(i);
                       const auto& M       = masses(i);
                       auto&       G       = gradients(i);

                       Eigen::Matrix<Alu, 12, 12> M_alu =
                           M.to_mat().template cast<Alu>();
                       const Alu inv_beta_alu = safe_cast<Alu>(inv_beta);
                       Eigen::Matrix<Alu, 12, 1> G_alu =
                           (inv_beta_alu * M_alu * (q - q_tilde).template cast<Alu>()).eval();

                       if(is_fixed(i))
                       {
                           G_alu.setZero();
                       }
                       G = downcast_gradient<typename Vector12::Scalar>(G_alu);

                       if(gradient_only)
                           return;

                       hessians(i) =
                           downcast_hessian<typename Matrix12x12::Scalar>(inv_beta_alu * M_alu);
                   });
    }
};

REGISTER_SIM_SYSTEM(AffineBodyBDF2Kinetic);
}  // namespace uipc::backend::cuda_mixed
