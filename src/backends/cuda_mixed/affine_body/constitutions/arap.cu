#include <sim_system.h>
#include <affine_body/constitutions/arap_function.h>
#include <affine_body/affine_body_constitution.h>
#include <affine_body/affine_body_dynamics.h>
#include <uipc/common/enumerate.h>
#include <affine_body/abd_energy.h>
#include <muda/cub/device/device_reduce.h>
#include <muda/ext/eigen/svd.h>
#include <utils/make_spd.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
class ARAP final : public AffineBodyConstitution
{
  public:
    static constexpr U64 ConstitutionUID = 2ull;

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

        //SizeT bodyI = 0;
        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc)
            { return sc.instances().find<Float>("kappa")->view(); },
            [&](const ForEachInfo& I, Float kappa)
            { h_kappas[I.global_index()] = kappa; });

        _build_on_device();
    }

    void _build_on_device()
    {
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
        namespace abd_arap = sym::abd_arap;
        using Alu          = ActivePolicy::AluScalar;
        using Store        = ActivePolicy::StoreScalar;

        auto body_count = info.qs().size();

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
                       const Alu  kappa = safe_cast<Alu>(kappas(i));

                       const Alu Vdt2 = safe_cast<Alu>(volume * dt * dt);

                       Alu E = Alu{0};
                       abd_arap::E(E, kappa, q_alu);

                       shape_energies(i) = safe_cast<ActivePolicy::EnergyScalar>(E * Vdt2);
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace abd_arap = sym::abd_arap;
        using Alu          = ActivePolicy::AluScalar;
        using Store        = ActivePolicy::StoreScalar;

        auto N             = info.qs().size();
        auto gradient_only = info.gradient_only();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(N,
                   [qs      = info.qs().cviewer().name("qs"),
                    volumes = info.volumes().cviewer().name("volumes"),
                    gradients = info.gradients().viewer().name("shape_gradients"),
                    hessians = info.hessians().viewer().name("shape_hessian"),
                    kappas   = kappas.cviewer().name("kappas"),
                   dt       = info.dt(),
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
                       abd_arap::dEdq(G9_alu, kappa, q_alu);
                       G.template segment<9>(3) = downcast_gradient<Store>(G9_alu * Vdt2);
                       gradients(i)    = G;

                       if(gradient_only)
                           return;

                       Eigen::Matrix<Alu, 9, 9> H9x9_alu;
                       abd_arap::ddEddq(H9x9_alu, kappa, q_alu);
                       H.template block<9, 9>(3, 3) = downcast_hessian<Store>(H9x9_alu * Vdt2);

                       hessians(i) = H;
                   });
    }
};

REGISTER_SIM_SYSTEM(ARAP);
}  // namespace uipc::backend::cuda_mixed
