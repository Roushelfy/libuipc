#include <finite_element/fem_3d_constitution.h>
#include <finite_element/constitutions/arap_function.h>
#include <finite_element/fem_utils.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <Eigen/Dense>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
class ARAP3D final : public FEM3DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 9;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using FEM3DConstitution::FEM3DConstitution;

    vector<Float> h_kappas;

    muda::DeviceBuffer<Float> kappas;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(kappas.size());
        info.gradient_count(kappas.size() * StencilSize);
        if(info.gradient_only())
            return;
        info.hessian_count(kappas.size() * HalfHessianSize);
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        h_kappas.resize(N);

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc) -> auto
            {
                auto kappa = sc.tetrahedra().find<Float>("kappa");
                UIPC_ASSERT(kappa, "Can't find attribute `kappa` on tetrahedra, why can it happen?");
                return kappa->view();
            },
            [&](const ForEachInfo& I, Float kappa)
            { h_kappas[I.global_index()] = kappa; });

        kappas.resize(N);
        kappas.view().copy_from(h_kappas.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace ARAP = sym::arap_3d;
        using Alu      = ActivePolicy::AluScalar;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [kappas   = kappas.cviewer().name("mus"),
                    energies = info.energies().viewer().name("energies"),
                    indices  = info.indices().viewer().name("indices"),
                    xs       = info.xs().viewer().name("xs"),
                    Dm_invs  = info.Dm_invs().viewer().name("Dm_invs"),
                    volumes  = info.rest_volumes().viewer().name("volumes"),
                    dt       = info.dt()] __device__(int I)
                   {
                       const Vector4i&  tet    = indices(I);
                       const Matrix3x3& Dm_inv = Dm_invs(I);

                       Eigen::Matrix<Alu, 3, 1> x0 = xs(tet(0)).template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 1> x1 = xs(tet(1)).template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 1> x2 = xs(tet(2)).template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 1> x3 = xs(tet(3)).template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 3> Dm_inv_alu =
                           Dm_inv.template cast<Alu>();

                       auto F = fem::F<Alu>(x0, x1, x2, x3, Dm_inv_alu);

                       Alu E = Alu{0};
                       const Alu kt2 = safe_cast<Alu>(kappas(I) * dt * dt);
                       const Alu v   = safe_cast<Alu>(volumes(I));

                       ARAP::E(E, kt2, v, F);
                       energies(I) = safe_cast<Float>(E);
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace ARAP = sym::arap_3d;
        using Alu      = ActivePolicy::AluScalar;
        using Store    = ActivePolicy::StoreScalar;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [kappas  = kappas.cviewer().name("mus"),
                    indices = info.indices().viewer().name("indices"),
                    xs      = info.xs().viewer().name("xs"),
                    Dm_invs = info.Dm_invs().viewer().name("Dm_invs"),
                    G3s     = info.gradients().viewer().name("gradients"),
                    H3x3s   = info.hessians().viewer().name("hessians"),
                    volumes = info.rest_volumes().viewer().name("volumes"),
                    dt      = info.dt(),
                    gradient_only = info.gradient_only()] __device__(int I) mutable
                   {
                       const Vector4i&  tet    = indices(I);
                       const Matrix3x3& Dm_inv = Dm_invs(I);

                       Eigen::Matrix<Alu, 3, 1> x0 = xs(tet(0)).template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 1> x1 = xs(tet(1)).template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 1> x2 = xs(tet(2)).template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 1> x3 = xs(tet(3)).template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 3> Dm_inv_alu =
                           Dm_inv.template cast<Alu>();

                       auto F = fem::F<Alu>(x0, x1, x2, x3, Dm_inv_alu);

                       const Alu kt2 = safe_cast<Alu>(kappas(I) * dt * dt);
                       const Alu v   = safe_cast<Alu>(volumes(I));

                       Eigen::Matrix<Alu, 9, 1> dEdF;
                       ARAP::dEdF(dEdF, kt2, v, F);

                       Eigen::Matrix<Alu, 9, 12> dFdx = fem::dFdx<Alu>(Dm_inv_alu);
                       Eigen::Matrix<Alu, 12, 1> G12_alu  = dFdx.transpose() * dEdF;
                       auto G12_store = downcast_gradient<Store>(G12_alu);

                       DoubletVectorAssembler DVA{G3s};
                       DVA.segment<StencilSize>(I * StencilSize).write(tet, G12_store);

                       if(gradient_only)
                           return;

                       Eigen::Matrix<Alu, 9, 9> ddEddF;
                       ARAP::ddEddF(ddEddF, kt2, v, F);
                       make_spd(ddEddF);
                       Eigen::Matrix<Alu, 12, 12> H12x12_alu = dFdx.transpose() * ddEddF * dFdx;
                       auto H12x12_store = downcast_hessian<Store>(H12x12_alu);
                       TripletMatrixAssembler TMA{H3x3s};
                       TMA.half_block<StencilSize>(I * HalfHessianSize).write(tet, H12x12_store);
                   });
    }
};

REGISTER_SIM_SYSTEM(ARAP3D);
}  // namespace uipc::backend::cuda_mixed
