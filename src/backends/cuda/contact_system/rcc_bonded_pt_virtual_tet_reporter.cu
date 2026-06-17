#include <contact_system/rcc_bonded_pt_virtual_tet_reporter.h>

#include <affine_body/constitutions/ortho_potential_function.h>
#include <finite_element/fem_utils.h>
#include <finite_element/matrix_utils.h>
#include <muda/ext/eigen/eigen_core_cxx20.h>
#include <muda/launch/parallel_for.h>
#include <sim_engine.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>

#include <string>

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
    Vector12    gradient = Vector12::Zero();
    // NOT default-zeroed: a 12x12 zero-init is 144 doubles wasted on the
    // energy-only and gradient-only paths. Only EVAL_EGH fills it (and the
    // degenerate early-returns zero it for that mode).
    Matrix12x12 hessian;
};

// Evaluation mode (compile-time): energy only / energy+gradient / full.
// compute_energy needs only E; assemble's gradient_only Newton iterations
// need only E+G. Computing the 9x9 Hessian + make_spd + dFdx^T H dFdx on
// those paths is pure waste (the bonded reporter was ~21% of GPU time, with
// compute_energy alone 14.7% while computing a Hessian it never reads).
enum RCCEvalMode
{
    EVAL_E   = 0,
    EVAL_EG  = 1,
    EVAL_EGH = 2
};

MUDA_GENERIC Vector12 abd_q_from_F(const Matrix3x3& F)
{
    Vector12 q = Vector12::Zero();
    for(IndexT row = 0; row < 3; ++row)
        for(IndexT col = 0; col < 3; ++col)
            q[3 + row * 3 + col] = F(row, col);
    return q;
}

MUDA_GENERIC Vector9 abd_row_gradient_to_column_gradient(const Vector9& row_g)
{
    Vector9 col_g;
    for(IndexT row = 0; row < 3; ++row)
        for(IndexT col = 0; col < 3; ++col)
            col_g[col * 3 + row] = row_g[row * 3 + col];
    return col_g;
}

MUDA_GENERIC Matrix9x9 abd_row_hessian_to_column_hessian(const Matrix9x9& row_H)
{
    Matrix9x9 col_H;
    for(IndexT row_a = 0; row_a < 3; ++row_a)
    {
        for(IndexT col_a = 0; col_a < 3; ++col_a)
        {
            const IndexT a_col = col_a * 3 + row_a;
            const IndexT a_row = row_a * 3 + col_a;
            for(IndexT row_b = 0; row_b < 3; ++row_b)
            {
                for(IndexT col_b = 0; col_b < 3; ++col_b)
                {
                    const IndexT b_col = col_b * 3 + row_b;
                    const IndexT b_row = row_b * 3 + col_b;
                    col_H(a_col, b_col) = row_H(a_row, b_row);
                }
            }
        }
    }
    return col_H;
}

// Energy-only evaluation returning a bare Float — deliberately does NOT
// construct RCCBondedPTVirtualTetEval, whose Matrix12x12 hessian member is
// 1152 bytes/thread and spills to local memory, tanking occupancy of the
// energy kernel (line-search hot path) even when the Hessian is never used.
template <typename PositionViewer>
MUDA_GENERIC Float eval_virtual_tet_energy(
    const Vector4i& tet,
    muda::CBufferView<Vector3> positions_view,
    PositionViewer positions,
    const Matrix3x3& Dm_inv,
    Float rest_volume,
    Float kappa,
    Float dt)
{
    namespace AOP = sym::abd_ortho_potential;
    if(rest_volume <= 0.0 || kappa <= 0.0)
        return 0.0;
    const IndexT n = static_cast<IndexT>(positions_view.size());
    if(tet[0] < 0 || tet[1] < 0 || tet[2] < 0 || tet[3] < 0 || tet[0] >= n
       || tet[1] >= n || tet[2] >= n || tet[3] >= n)
        return 0.0;
    Matrix3x3 F = fem::F(positions(tet[0]), positions(tet[1]),
                         positions(tet[2]), positions(tet[3]), Dm_inv);
    Vector12  q = abd_q_from_F(F);
    Float E_val = 0.0;
    AOP::E(E_val, kappa, q);
    return E_val * (rest_volume * dt * dt);
}

template <int MODE, typename PositionViewer>
MUDA_GENERIC RCCBondedPTVirtualTetEval eval_virtual_tet(
    const Vector4i& tet,
    muda::CBufferView<Vector3> positions_view,
    PositionViewer positions,
    const Matrix3x3& Dm_inv,
    Float rest_volume,
    Float kappa,
    Float dt)
{
    namespace AOP = sym::abd_ortho_potential;

    RCCBondedPTVirtualTetEval out;
    auto degenerate = [&]() -> RCCBondedPTVirtualTetEval&
    {
        // gradient is Zero() by default; zero the hessian only when this mode
        // actually fills/reads it.
        if constexpr(MODE >= EVAL_EGH)
            out.hessian.setZero();
        return out;
    };
    if(rest_volume <= 0.0 || kappa <= 0.0)
        return degenerate();

    const IndexT n = static_cast<IndexT>(positions_view.size());
    if(tet[0] < 0 || tet[1] < 0 || tet[2] < 0 || tet[3] < 0 || tet[0] >= n
       || tet[1] >= n || tet[2] >= n || tet[3] >= n)
        return degenerate();

    Vector3 x0 = positions(tet[0]);
    Vector3 x1 = positions(tet[1]);
    Vector3 x2 = positions(tet[2]);
    Vector3 x3 = positions(tet[3]);

    Matrix3x3 F = fem::F(x0, x1, x2, x3, Dm_inv);
    Vector12  q = abd_q_from_F(F);
    const Float Vdt2 = rest_volume * dt * dt;

    Float E_val = 0.0;
    AOP::E(E_val, kappa, q);
    out.energy = E_val * Vdt2;

    if constexpr(MODE >= EVAL_EG)
    {
        Vector9 dEdVecF_row;
        AOP::dEdq(dEdVecF_row, kappa, q);
        Vector9 dEdVecF = abd_row_gradient_to_column_gradient(dEdVecF_row);
        dEdVecF *= Vdt2;

        Matrix9x12 dFdx = fem::dFdx(Dm_inv);
        out.gradient = dFdx.transpose() * dEdVecF;

        if constexpr(MODE >= EVAL_EGH)
        {
            Matrix9x9 ddEddVecF_row;
            AOP::ddEddq(ddEddVecF_row, kappa, q);
            Matrix9x9 ddEddVecF = abd_row_hessian_to_column_hessian(ddEddVecF_row);
            ddEddVecF *= Vdt2;
            make_spd(ddEddVecF);
            out.hessian = dFdx.transpose() * ddEddVecF * dFdx;
        }
    }

    return out;
}
}  // namespace

void RCCBondedPTVirtualTetReporter::Impl::set_material(Float kappa) noexcept
{
    m_kappa = kappa;
}

Float RCCBondedPTVirtualTetReporter::Impl::kappa() const noexcept
{
    return m_kappa;
}

bool RCCBondedPTVirtualTetReporter::Impl::active() const noexcept
{
    return m_kappa > 0.0;
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
                kappa = m_kappa,
                dt] __device__(int I) mutable
               {
                   energies(I) = eval_virtual_tet_energy(topos(I),
                                                positions_view,
                                                positions,
                                                dm_inv(I),
                                                rest_volume(I),
                                                kappa,
                                                dt);
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
                kappa = m_kappa,
                dt] __device__(int I) mutable
               {
                   auto eval = eval_virtual_tet<EVAL_EGH>(topos(I),
                                                positions_view,
                                                positions,
                                                dm_inv(I),
                                                rest_volume(I),
                                                kappa,
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
                kappa = m_kappa,
                dt,
                gradient_only] __device__(int I) mutable
               {
                   const Vector4i tet = topos(I);
                   // gradient_only Newton iterations skip the Hessian entirely
                   // (no ddEddq / make_spd / dFdx^T H dFdx).
                   if(gradient_only)
                   {
                       auto eval = eval_virtual_tet<EVAL_EG>(tet,
                                                positions_view, positions,
                                                dm_inv(I), rest_volume(I),
                                                kappa, dt);
                       DoubletVectorAssembler VA{G3s};
                       VA.segment<StencilSize>(I * StencilSize)
                           .write(tet, eval.gradient);
                       return;
                   }

                   auto eval = eval_virtual_tet<EVAL_EGH>(tet,
                                                positions_view, positions,
                                                dm_inv(I), rest_volume(I),
                                                kappa, dt);
                   DoubletVectorAssembler VA{G3s};
                   VA.segment<StencilSize>(I * StencilSize).write(tet,
                                                                  eval.gradient);
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

    auto model_attr =
        world().scene().config().find<std::string>("rcc_bonded_pt_energy_model");
    const std::string model =
        model_attr ? model_attr->view()[0] : std::string{"abd_ortho"};
    UIPC_ASSERT(model == "abd_ortho",
                "Unsupported rcc_bonded_pt_energy_model '{}'. "
                "The production bonded PT reporter currently supports only "
                "'abd_ortho'.",
                model);

    auto kappa_attr =
        world().scene().config().find<Float>("rcc_bonded_pt_kappa");
    m_impl.set_material(kappa_attr ? kappa_attr->view()[0] : Float{1e8});
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
