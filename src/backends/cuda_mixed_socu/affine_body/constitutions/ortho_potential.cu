#include <affine_body/affine_body_constitution.h>
#include <affine_body/constitutions/ortho_potential_function.h>
#include <utils/make_spd.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>


namespace uipc::backend::cuda_mixed
{
class OrthoPotential final : public AffineBodyConstitution
{
  public:
    static constexpr U64 ConstitutionUID = 1ull;

    using AffineBodyConstitution::AffineBodyConstitution;

    vector<Float> h_kappas;

    muda::DeviceBuffer<Float> kappas;

    virtual void do_build(AffineBodyConstitution::BuildInfo& info) override {}

    U64 get_uid() const override { return ConstitutionUID; }

    void do_init(AffineBodyDynamics::FilteredInfo& info) override
    {
        using ForEachInfo = AffineBodyDynamics::ForEachInfo;

        // find out constitution coefficients
        h_kappas.resize(info.body_count());
        auto geo_slots = world().scene().geometries();

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc)
            { return sc.instances().find<Float>("kappa")->view(); },
            [&](const ForEachInfo& I, Float kappa)
            {
                auto bodyI      = I.global_index();
                h_kappas[bodyI] = kappa;
            });

        auto async_copy = []<typename T>(span<T> src, muda::DeviceBuffer<T>& dst)
        {
            muda::BufferLaunch().resize<T>(dst, src.size());
            muda::BufferLaunch().copy<T>(dst.view(), src.data());
        };

        async_copy(span{h_kappas}, kappas);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        using Alu = ActivePolicy::AluScalar;

        auto body_count = info.qs().size();

        namespace AOP = sym::abd_ortho_potential;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(body_count,
                   [shape_energies = info.energies().viewer().name("energies"),
                    qs             = info.qs().cviewer().name("qs"),
                    kappas         = kappas.cviewer().name("kappas"),
                    volumes        = info.volumes().cviewer().name("volumes"),
                    dt             = info.dt()] __device__(int i) mutable
                   {
                       auto& q      = qs(i);
                       auto& volume = volumes(i);
                       Eigen::Matrix<Alu, 12, 1> q_alu = q.template cast<Alu>();
                       const Alu kappa   = safe_cast<Alu>(kappas(i));
                       const Alu Vdt2    = safe_cast<Alu>(volume * dt * dt);

                       Alu E = Alu{0};
                       AOP::E(E, kappa, q_alu);

                       shape_energies(i) = safe_cast<ActivePolicy::EnergyScalar>(E * Vdt2);
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        auto N             = info.qs().size();
        auto gradient_only = info.gradient_only();
        using Alu          = ActivePolicy::AluScalar;
        using Store        = ActivePolicy::StoreScalar;

        namespace AOP = sym::abd_ortho_potential;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(N,
                   [qs      = info.qs().cviewer().name("qs"),
                    volumes = info.volumes().cviewer().name("volumes"),
                    gradients = info.gradients().viewer().name("shape_gradients"),
                    body_hessian = info.hessians().viewer().name("shape_hessian"),
                    kappas = kappas.cviewer().name("kappas"),
                   dt     = info.dt(),
                   gradient_only] __device__(int i) mutable
                   {
                       Eigen::Matrix<Store, 12, 12> H = Eigen::Matrix<Store, 12, 12>::Zero();
                       Eigen::Matrix<Store, 12, 1>  G = Eigen::Matrix<Store, 12, 1>::Zero();

                       const auto& q      = qs(i);
                       Eigen::Matrix<Alu, 12, 1> q_alu = q.template cast<Alu>();
                       const Alu   kappa  = safe_cast<Alu>(kappas(i));
                       const auto& volume = volumes(i);

                       const Alu Vdt2 = safe_cast<Alu>(volume * dt * dt);

                       Eigen::Matrix<Alu, 9, 1> G9_alu;
                       AOP::dEdq(G9_alu, kappa, q_alu);
                       G.template segment<9>(3) = downcast_gradient<Store>(G9_alu * Vdt2);
                       gradients(i)    = G;

                       if(gradient_only)
                           return;

                       Eigen::Matrix<Alu, 9, 9> H9x9_alu;
                       AOP::ddEddq(H9x9_alu, kappa, q_alu);
                       make_spd(H9x9_alu);

                       H.template block<9, 9>(3, 3) = downcast_hessian<Store>(H9x9_alu * Vdt2);
                       body_hessian(i)     = H;
                   });
    }
};

REGISTER_SIM_SYSTEM(OrthoPotential);
}  // namespace uipc::backend::cuda_mixed
