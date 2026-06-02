#include <contact_system/rcc_bonded_pt_virtual_tet_reporter.h>

#include <finite_element/fem_utils.h>
#include <finite_element/matrix_utils.h>
#include <inter_primitive_effect_system/constitutions/soft_vertex_triangle_stitch_function.h>
#include <muda/ext/eigen/eigen_core_cxx20.h>
#include <muda/launch/parallel_for.h>
#include <sim_engine.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>

namespace uipc::backend
{
template <>
class SimSystemCreator<cuda::RCCBondedPTVirtualTetReporter>
{
  public:
    static U<cuda::RCCBondedPTVirtualTetReporter> create(cuda::SimEngine& engine)
    {
        auto& config = engine.world().scene().config();
        auto  enabled_attr = config.find<IndexT>("rcc_bonded_pt_enabled");
        if(!enabled_attr || enabled_attr->view()[0] == 0)
            return nullptr;

        auto contact_enable_attr = config.find<IndexT>("contact/enable");
        const bool contact_enable =
            contact_enable_attr && contact_enable_attr->view()[0] != 0;

        auto& types = engine.world().scene().constitution_tabular().types();
        const bool has_inter_primitive_constitution =
            types.find(std::string{builtin::InterPrimitive}) != types.end();

        if(contact_enable || has_inter_primitive_constitution)
            return make_unique<cuda::RCCBondedPTVirtualTetReporter>(engine);
        return nullptr;
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(RCCBondedPTVirtualTetReporter);

namespace
{
struct RCCBondedPTVirtualTetEval
{
    Float       energy = 0.0;
    Vector12   gradient = Vector12::Zero();
    Matrix12x12 hessian = Matrix12x12::Zero();
};

template <typename PositionViewer>
MUDA_GENERIC RCCBondedPTVirtualTetEval eval_virtual_tet(
    const Vector4i& tet,
    muda::CBufferView<Vector3> positions_view,
    PositionViewer positions,
    const Matrix3x3& Dm_inv,
    Float rest_volume,
    Float mu,
    Float lambda,
    Float dt)
{
    namespace SVTS = sym::soft_vertex_triangle_stitch;

    RCCBondedPTVirtualTetEval out;
    if(rest_volume <= 0.0 || mu <= 0.0 || lambda <= 0.0)
        return out;

    const IndexT n = static_cast<IndexT>(positions_view.size());
    if(tet[0] < 0 || tet[1] < 0 || tet[2] < 0 || tet[3] < 0 || tet[0] >= n
       || tet[1] >= n || tet[2] >= n || tet[3] >= n)
        return out;

    Vector3 x0 = positions(tet[0]);
    Vector3 x1 = positions(tet[1]);
    Vector3 x2 = positions(tet[2]);
    Vector3 x3 = positions(tet[3]);

    Matrix3x3 F = fem::F(x0, x1, x2, x3, Dm_inv);
    Vector9   vec_F = flatten(F);
    const Float Vdt2 = rest_volume * dt * dt;

    Float E_val = 0.0;
    SVTS::E(E_val, mu, lambda, vec_F);
    out.energy = E_val * Vdt2;

    Vector9 dEdVecF;
    SVTS::dEdVecF(dEdVecF, mu, lambda, vec_F);
    dEdVecF *= Vdt2;

    Matrix9x12 dFdx = fem::dFdx(Dm_inv);
    out.gradient = dFdx.transpose() * dEdVecF;

    Matrix9x9 ddEddVecF;
    SVTS::ddEddVecF(ddEddVecF, mu, lambda, vec_F);
    ddEddVecF *= Vdt2;
    make_spd(ddEddVecF);
    out.hessian = dFdx.transpose() * ddEddVecF * dFdx;

    return out;
}
}  // namespace

void RCCBondedPTVirtualTetReporter::Impl::set_material(Float mu,
                                                       Float lambda) noexcept
{
    m_mu = mu;
    m_lambda = lambda;
}

Float RCCBondedPTVirtualTetReporter::Impl::mu() const noexcept
{
    return m_mu;
}

Float RCCBondedPTVirtualTetReporter::Impl::lambda() const noexcept
{
    return m_lambda;
}

bool RCCBondedPTVirtualTetReporter::Impl::active() const noexcept
{
    return m_mu > 0.0 && m_lambda > 0.0;
}

SizeT RCCBondedPTVirtualTetReporter::Impl::energy_count(
    muda::CBufferView<Vector4i> topos) const noexcept
{
    return active() ? topos.size() : 0;
}

SizeT RCCBondedPTVirtualTetReporter::Impl::gradient_count(
    muda::CBufferView<Vector4i> topos) const noexcept
{
    return active() ? StencilSize * topos.size() : 0;
}

SizeT RCCBondedPTVirtualTetReporter::Impl::hessian_count(
    muda::CBufferView<Vector4i> topos,
    bool gradient_only) const noexcept
{
    return active() && !gradient_only ? HalfHessianSize * topos.size() : 0;
}

void RCCBondedPTVirtualTetReporter::Impl::compute_energy(
    muda::CBufferView<Vector4i> topos,
    muda::CBufferView<Matrix3x3> dm_inv,
    muda::CBufferView<Float> rest_volume,
    muda::CBufferView<Vector3> positions,
    Float dt,
    muda::BufferView<Float> energies) const
{
    if(!active() || topos.size() == 0)
        return;

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(topos.size(),
               [topos = topos.viewer().name("rcc_bonded_pt_topos"),
                positions_view = positions,
                positions = positions.viewer().name("positions"),
                dm_inv = dm_inv.viewer().name("rcc_bonded_pt_dm_inv"),
                rest_volume =
                    rest_volume.viewer().name("rcc_bonded_pt_rest_volume"),
                energies = energies.viewer().name("rcc_bonded_pt_Es"),
                mu = m_mu,
                lambda = m_lambda,
                dt] __device__(int I) mutable
               {
                   auto eval = eval_virtual_tet(topos(I),
                                                positions_view,
                                                positions,
                                                dm_inv(I),
                                                rest_volume(I),
                                                mu,
                                                lambda,
                                                dt);
                   energies(I) = eval.energy;
               });
}

void RCCBondedPTVirtualTetReporter::Impl::compute_dense_energy_gradient_hessian(
    muda::CBufferView<Vector4i> topos,
    muda::CBufferView<Matrix3x3> dm_inv,
    muda::CBufferView<Float> rest_volume,
    muda::CBufferView<Vector3> positions,
    Float dt,
    muda::BufferView<Float> energies,
    muda::BufferView<Vector12> gradients,
    muda::BufferView<Matrix12x12> hessians) const
{
    if(!active() || topos.size() == 0)
        return;

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(topos.size(),
               [topos = topos.viewer().name("rcc_bonded_pt_topos"),
                positions_view = positions,
                positions = positions.viewer().name("positions"),
                dm_inv = dm_inv.viewer().name("rcc_bonded_pt_dm_inv"),
                rest_volume =
                    rest_volume.viewer().name("rcc_bonded_pt_rest_volume"),
                energies = energies.viewer().name("rcc_bonded_pt_dense_Es"),
                gradients =
                    gradients.viewer().name("rcc_bonded_pt_dense_Gs"),
                hessians = hessians.viewer().name("rcc_bonded_pt_dense_Hs"),
                mu = m_mu,
                lambda = m_lambda,
                dt] __device__(int I) mutable
               {
                   auto eval = eval_virtual_tet(topos(I),
                                                positions_view,
                                                positions,
                                                dm_inv(I),
                                                rest_volume(I),
                                                mu,
                                                lambda,
                                                dt);
                   energies(I) = eval.energy;
                   gradients(I) = eval.gradient;
                   hessians(I) = eval.hessian;
               });
}

void RCCBondedPTVirtualTetReporter::Impl::assemble(
    muda::CBufferView<Vector4i> topos,
    muda::CBufferView<Matrix3x3> dm_inv,
    muda::CBufferView<Float> rest_volume,
    muda::CBufferView<Vector3> positions,
    Float dt,
    muda::DoubletVectorView<Float, 3> gradients,
    muda::TripletMatrixView<Float, 3> hessians,
    bool gradient_only) const
{
    if(!active() || topos.size() == 0)
        return;

    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(topos.size(),
               [topos = topos.viewer().name("rcc_bonded_pt_topos"),
                positions_view = positions,
                positions = positions.viewer().name("positions"),
                dm_inv = dm_inv.viewer().name("rcc_bonded_pt_dm_inv"),
                rest_volume =
                    rest_volume.viewer().name("rcc_bonded_pt_rest_volume"),
                G3s = gradients.viewer().name("rcc_bonded_pt_Gs"),
                H3x3s = hessians.viewer().name("rcc_bonded_pt_H3x3s"),
                mu = m_mu,
                lambda = m_lambda,
                dt,
                gradient_only] __device__(int I) mutable
               {
                   const Vector4i tet = topos(I);
                   auto eval = eval_virtual_tet(tet,
                                                positions_view,
                                                positions,
                                                dm_inv(I),
                                                rest_volume(I),
                                                mu,
                                                lambda,
                                                dt);

                   DoubletVectorAssembler VA{G3s};
                   VA.segment<StencilSize>(I * StencilSize).write(tet,
                                                                  eval.gradient);

                   if(gradient_only)
                       return;

                   TripletMatrixAssembler MA{H3x3s};
                   MA.half_block<StencilSize>(I * HalfHessianSize)
                       .write(tet, eval.hessian);
               });
}

void RCCBondedPTVirtualTetReporter::do_build(DyTopoEffectReporter::BuildInfo&)
{
    m_impl.bonded_pt_system = find<RCCBondedPTSystem>();
    m_impl.global_vertex_manager = require<GlobalVertexManager>();
    m_impl.dt_attr = world().scene().config().find<Float>("dt");
    UIPC_ASSERT(m_impl.dt_attr, "Scene config must have a 'dt' attribute.");

    auto mu_attr = world().scene().config().find<Float>("rcc_bonded_pt_mu");
    auto lambda_attr =
        world().scene().config().find<Float>("rcc_bonded_pt_lambda");
    m_impl.set_material(mu_attr ? mu_attr->view()[0] : 0.0,
                        lambda_attr ? lambda_attr->view()[0] : 0.0);
}

void RCCBondedPTVirtualTetReporter::do_report_energy_extent(
    GlobalDyTopoEffectManager::EnergyExtentInfo& info)
{
    if(!m_impl.bonded_pt_system || !m_impl.bonded_pt_system->enabled())
        return;
    info.energy_count(m_impl.energy_count(m_impl.bonded_pt_system->locked_topos()));
}

void RCCBondedPTVirtualTetReporter::do_report_gradient_hessian_extent(
    GlobalDyTopoEffectManager::GradientHessianExtentInfo& info)
{
    if(!m_impl.bonded_pt_system || !m_impl.bonded_pt_system->enabled())
        return;

    const auto topos = m_impl.bonded_pt_system->locked_topos();
    info.gradient_count(m_impl.gradient_count(topos));
    info.hessian_count(m_impl.hessian_count(topos, info.gradient_only()));
}

void RCCBondedPTVirtualTetReporter::do_assemble(
    GlobalDyTopoEffectManager::GradientHessianInfo& info)
{
    if(!m_impl.bonded_pt_system || !m_impl.bonded_pt_system->enabled())
        return;

    const Float dt = m_impl.dt_attr->view()[0];
    m_impl.assemble(m_impl.bonded_pt_system->locked_topos(),
                    m_impl.bonded_pt_system->locked_dm_inv(),
                    m_impl.bonded_pt_system->locked_rest_volume(),
                    m_impl.global_vertex_manager->positions(),
                    dt,
                    info.gradients(),
                    info.hessians(),
                    info.gradient_only());
}

void RCCBondedPTVirtualTetReporter::do_compute_energy(
    GlobalDyTopoEffectManager::EnergyInfo& info)
{
    if(!m_impl.bonded_pt_system || !m_impl.bonded_pt_system->enabled())
        return;

    const Float dt = m_impl.dt_attr->view()[0];
    m_impl.compute_energy(m_impl.bonded_pt_system->locked_topos(),
                          m_impl.bonded_pt_system->locked_dm_inv(),
                          m_impl.bonded_pt_system->locked_rest_volume(),
                          m_impl.global_vertex_manager->positions(),
                          dt,
                          info.energies());
}

EnergyComponentFlags RCCBondedPTVirtualTetReporter::component_flags()
{
    return EnergyComponentFlags::Complement;
}
}  // namespace uipc::backend::cuda
