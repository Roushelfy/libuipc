#include <contact_system/rcc_bonded_pt_virtual_tet_reporter.h>

#include <Eigen/Dense>  // full Matrix (block/cross) for the reimplemented SNH energy
#include <affine_body/constitutions/ortho_potential_function.h>
#include <finite_element/fem_utils.h>
#include <finite_element/matrix_utils.h>
#include <muda/ext/eigen/eigen_core_cxx20.h>
#include <muda/ext/eigen/log_proxy.h>
#include <muda/launch/parallel_for.h>
#include <uipc/core/rcc_bonded_pt_oracle.h>
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

// ===========================================================================
// Energy policies — the ONLY material-specific part of the virtual tet. Each
// maps F -> (E, dEdVecF[9, column vec(F)], ddEddVecF[9x9, column vec(F)]). The
// shared geometry/assembly (fem::F, dFdx^T H dFdx, make_spd, scatter, bare-Float
// energy) is templated on the policy below. Compile-time dispatch keeps each
// kernel branch-free with model-specific register allocation.
//
//   AbdOrtho:        material = kappa;  E(q = abd_q_from_F(F)); row->column.
//   StableNeoHookean material = mu,lambda;  F-based; flatten (same convention
//                    as stable_neo_hookean_3d.cu's fem::dFdx triple product).
// ===========================================================================
struct AbdOrthoEnergy
{
    struct Mat
    {
        Float kappa = 0.0;
    };
    MUDA_GENERIC static bool inactive(const Mat& m) { return m.kappa <= 0.0; }
    MUDA_GENERIC static Float E(const Matrix3x3& F, const Mat& m)
    {
        namespace AOP = sym::abd_ortho_potential;
        Vector12 q     = abd_q_from_F(F);
        Float    E_val = 0.0;
        AOP::E(E_val, m.kappa, q);
        return E_val;
    }
    MUDA_GENERIC static Vector9 dEdVecF(const Matrix3x3& F, const Mat& m)
    {
        namespace AOP = sym::abd_ortho_potential;
        Vector12 q = abd_q_from_F(F);
        Vector9  row;
        AOP::dEdq(row, m.kappa, q);
        return abd_row_gradient_to_column_gradient(row);
    }
    MUDA_GENERIC static Matrix9x9 ddEddVecF(const Matrix3x3& F, const Mat& m)
    {
        namespace AOP = sym::abd_ortho_potential;
        Vector12  q = abd_q_from_F(F);
        Matrix9x9 row;
        AOP::ddEddq(row, m.kappa, q);
        return abd_row_hessian_to_column_hessian(row);
    }
};

// Stable Neo-Hookean (Smith et al. 2018), reimplemented inline here rather than
// reusing sym::stable_neo_hookean_3d: that header's `.inl` uses `g1.block<...>`
// on a TEMPLATE-dependent matrix without `.template`, which fails to parse when
// instantiated from this file's policy-templated context. These methods are
// concrete (Float), so Eigen's member templates need no `.template`. The closed
// form matches detail/stable_neo_hookean_3d.inl; the CPU oracle has its own
// independent copy and the finite-difference test guards both.
struct StableNeoHookeanEnergy
{
    struct Mat
    {
        Float mu     = 0.0;
        Float lambda = 0.0;
    };
    MUDA_GENERIC static bool inactive(const Mat& m)
    {
        return m.mu <= 0.0 || m.lambda <= 0.0;
    }
    MUDA_GENERIC static Float E(const Matrix3x3& F, const Mat& m)
    {
        const Float J     = F.determinant();
        const Float Ic    = F.squaredNorm();
        const Float alpha = 1.0 + 0.75 * m.mu / m.lambda;
        return 0.5 * m.lambda * (J - alpha) * (J - alpha)
               + 0.5 * m.mu * (Ic - 3.0) - 0.5 * m.mu * log(Ic + 1.0);
    }
    MUDA_GENERIC static Vector9 dEdVecF(const Matrix3x3& F, const Mat& m)
    {
        const Float J  = F.determinant();
        const Float Ic = F.squaredNorm();
        Matrix3x3   pJpF;
        pJpF(0, 0) = F(1, 1) * F(2, 2) - F(1, 2) * F(2, 1);
        pJpF(0, 1) = F(1, 2) * F(2, 0) - F(1, 0) * F(2, 2);
        pJpF(0, 2) = F(1, 0) * F(2, 1) - F(1, 1) * F(2, 0);
        pJpF(1, 0) = F(2, 1) * F(0, 2) - F(2, 2) * F(0, 1);
        pJpF(1, 1) = F(2, 2) * F(0, 0) - F(2, 0) * F(0, 2);
        pJpF(1, 2) = F(2, 0) * F(0, 1) - F(2, 1) * F(0, 0);
        pJpF(2, 0) = F(0, 1) * F(1, 2) - F(1, 1) * F(0, 2);
        pJpF(2, 1) = F(0, 2) * F(1, 0) - F(0, 0) * F(1, 2);
        pJpF(2, 2) = F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0);
        Matrix3x3 PEPF = m.mu * (1.0 - 1.0 / (Ic + 1.0)) * F
                         + (m.lambda * (J - 1.0 - 0.75 * m.mu / m.lambda)) * pJpF;
        return flatten(PEPF);
    }
    MUDA_GENERIC static Matrix9x9 ddEddVecF(const Matrix3x3& F, const Mat& m)
    {
        const Float J  = F.determinant();
        const Float Ic = F.squaredNorm();

        Matrix9x9 H1 = 2.0 * Matrix9x9::Identity();

        Vector9 g1;
        g1.segment<3>(0) = 2.0 * F.col(0);
        g1.segment<3>(3) = 2.0 * F.col(1);
        g1.segment<3>(6) = 2.0 * F.col(2);

        Vector9 gJ;
        gJ.segment<3>(0) = F.col(1).cross(F.col(2));
        gJ.segment<3>(3) = F.col(2).cross(F.col(0));
        gJ.segment<3>(6) = F.col(0).cross(F.col(1));

        Matrix3x3 f0hat, f1hat, f2hat;
        f0hat << 0, -F(2, 0), F(1, 0), F(2, 0), 0, -F(0, 0), -F(1, 0), F(0, 0), 0;
        f1hat << 0, -F(2, 1), F(1, 1), F(2, 1), 0, -F(0, 1), -F(1, 1), F(0, 1), 0;
        f2hat << 0, -F(2, 2), F(1, 2), F(2, 2), 0, -F(0, 2), -F(1, 2), F(0, 2), 0;

        Matrix9x9 HJ;
        HJ.block<3, 3>(0, 0) = Matrix3x3::Zero();
        HJ.block<3, 3>(0, 3) = -f2hat;
        HJ.block<3, 3>(0, 6) = f1hat;
        HJ.block<3, 3>(3, 0) = f2hat;
        HJ.block<3, 3>(3, 3) = Matrix3x3::Zero();
        HJ.block<3, 3>(3, 6) = -f0hat;
        HJ.block<3, 3>(6, 0) = -f1hat;
        HJ.block<3, 3>(6, 3) = f0hat;
        HJ.block<3, 3>(6, 6) = Matrix3x3::Zero();

        return (Ic * m.mu) / (2.0 * (Ic + 1.0)) * H1
               + m.lambda * (J - 1.0 - (3.0 * m.mu) / (4.0 * m.lambda)) * HJ
               + (m.mu / (2.0 * (Ic + 1.0) * (Ic + 1.0))) * g1 * g1.transpose()
               + m.lambda * gJ * gJ.transpose();
    }
};

// Energy-only evaluation returning a bare Float — deliberately does NOT
// construct RCCBondedPTVirtualTetEval, whose Matrix12x12 hessian member is
// 1152 bytes/thread and spills to local memory, tanking occupancy of the
// energy kernel (line-search hot path) even when the Hessian is never used.
template <typename Policy, typename PositionViewer>
MUDA_GENERIC Float eval_virtual_tet_energy(
    const Vector4i& tet,
    muda::CBufferView<Vector3> positions_view,
    PositionViewer positions,
    const Matrix3x3& Dm_inv,
    Float rest_volume,
    const typename Policy::Mat& mat,
    Float dt)
{
    if(rest_volume <= 0.0 || Policy::inactive(mat))
        return 0.0;
    const IndexT n = static_cast<IndexT>(positions_view.size());
    if(tet[0] < 0 || tet[1] < 0 || tet[2] < 0 || tet[3] < 0 || tet[0] >= n
       || tet[1] >= n || tet[2] >= n || tet[3] >= n)
        return 0.0;
    Matrix3x3 F = fem::F(positions(tet[0]), positions(tet[1]),
                         positions(tet[2]), positions(tet[3]), Dm_inv);
    return Policy::E(F, mat) * (rest_volume * dt * dt);
}

template <int MODE, typename Policy, typename PositionViewer>
MUDA_GENERIC RCCBondedPTVirtualTetEval eval_virtual_tet(
    const Vector4i& tet,
    muda::CBufferView<Vector3> positions_view,
    PositionViewer positions,
    const Matrix3x3& Dm_inv,
    Float rest_volume,
    const typename Policy::Mat& mat,
    Float dt)
{
    RCCBondedPTVirtualTetEval out;
    auto degenerate = [&]() -> RCCBondedPTVirtualTetEval&
    {
        // gradient is Zero() by default; zero the hessian only when this mode
        // actually fills/reads it.
        if constexpr(MODE >= EVAL_EGH)
            out.hessian.setZero();
        return out;
    };
    if(rest_volume <= 0.0 || Policy::inactive(mat))
        return degenerate();

    const IndexT n = static_cast<IndexT>(positions_view.size());
    if(tet[0] < 0 || tet[1] < 0 || tet[2] < 0 || tet[3] < 0 || tet[0] >= n
       || tet[1] >= n || tet[2] >= n || tet[3] >= n)
        return degenerate();

    Matrix3x3 F = fem::F(positions(tet[0]), positions(tet[1]),
                         positions(tet[2]), positions(tet[3]), Dm_inv);
    const Float Vdt2 = rest_volume * dt * dt;

    out.energy = Policy::E(F, mat) * Vdt2;

    if constexpr(MODE >= EVAL_EG)
    {
        Vector9 dEdVecF = Policy::dEdVecF(F, mat);
        dEdVecF *= Vdt2;

        Matrix9x12 dFdx = fem::dFdx(Dm_inv);
        out.gradient = dFdx.transpose() * dEdVecF;

        if constexpr(MODE >= EVAL_EGH)
        {
            Matrix9x9 ddEddVecF = Policy::ddEddVecF(F, mat);
            ddEddVecF *= Vdt2;
            make_spd(ddEddVecF);
            out.hessian = dFdx.transpose() * ddEddVecF * dFdx;
        }
    }

    return out;
}

// ---- policy-templated kernel launchers (one ParallelFor each; the C++-level
// model switch in the Impl methods picks the instantiation -> no per-thread
// branch). ----
template <typename Policy>
void run_compute_energy(const typename Policy::Mat&   mat,
                        muda::CBufferView<Vector4i>   topos,
                        muda::CBufferView<Matrix3x3>  dm_inv,
                        muda::CBufferView<Float>      rest_volume,
                        muda::CBufferView<Vector3>    positions,
                        Float                         dt,
                        muda::BufferView<Float>       energies)
{
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(topos.size(),
               [topos = topos.viewer().name("rcc_bonded_pt_topos"),
                positions_view = positions,
                positions = positions.viewer().name("positions"),
                dm_inv = dm_inv.viewer().name("rcc_bonded_pt_dm_inv"),
                rest_volume = rest_volume.viewer().name("rcc_bonded_pt_rest_volume"),
                energies = energies.viewer().name("rcc_bonded_pt_Es"),
                mat,
                dt] __device__(int I) mutable
               {
                   energies(I) = eval_virtual_tet_energy<Policy>(
                       topos(I), positions_view, positions, dm_inv(I),
                       rest_volume(I), mat, dt);
               });
}

template <typename Policy>
void run_compute_dense(const typename Policy::Mat&     mat,
                       muda::CBufferView<Vector4i>     topos,
                       muda::CBufferView<Matrix3x3>    dm_inv,
                       muda::CBufferView<Float>        rest_volume,
                       muda::CBufferView<Vector3>      positions,
                       Float                           dt,
                       muda::BufferView<Float>         energies,
                       muda::BufferView<Vector12>      gradients,
                       muda::BufferView<Matrix12x12>   hessians)
{
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(topos.size(),
               [topos = topos.viewer().name("rcc_bonded_pt_topos"),
                positions_view = positions,
                positions = positions.viewer().name("positions"),
                dm_inv = dm_inv.viewer().name("rcc_bonded_pt_dm_inv"),
                rest_volume = rest_volume.viewer().name("rcc_bonded_pt_rest_volume"),
                energies = energies.viewer().name("rcc_bonded_pt_dense_Es"),
                gradients = gradients.viewer().name("rcc_bonded_pt_dense_Gs"),
                hessians = hessians.viewer().name("rcc_bonded_pt_dense_Hs"),
                mat,
                dt] __device__(int I) mutable
               {
                   auto eval = eval_virtual_tet<EVAL_EGH, Policy>(
                       topos(I), positions_view, positions, dm_inv(I),
                       rest_volume(I), mat, dt);
                   energies(I)  = eval.energy;
                   gradients(I) = eval.gradient;
                   hessians(I)  = eval.hessian;
               });
}

template <typename Policy>
void run_assemble(const typename Policy::Mat&        mat,
                  muda::CBufferView<Vector4i>        topos,
                  muda::CBufferView<Matrix3x3>       dm_inv,
                  muda::CBufferView<Float>           rest_volume,
                  muda::CBufferView<Vector3>         positions,
                  Float                              dt,
                  muda::DoubletVectorView<Float, 3>  gradients,
                  muda::TripletMatrixView<Float, 3>  hessians,
                  bool                               gradient_only)
{
    using namespace muda;
    constexpr SizeT StencilSize =
        RCCBondedPTVirtualTetReporter::StencilSize;
    constexpr SizeT HalfHessianSize =
        RCCBondedPTVirtualTetReporter::HalfHessianSize;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(topos.size(),
               [topos = topos.viewer().name("rcc_bonded_pt_topos"),
                positions_view = positions,
                positions = positions.viewer().name("positions"),
                dm_inv = dm_inv.viewer().name("rcc_bonded_pt_dm_inv"),
                rest_volume = rest_volume.viewer().name("rcc_bonded_pt_rest_volume"),
                G3s = gradients.viewer().name("rcc_bonded_pt_Gs"),
                H3x3s = hessians.viewer().name("rcc_bonded_pt_H3x3s"),
                mat,
                dt,
                gradient_only] __device__(int I) mutable
               {
                   const Vector4i tet = topos(I);
                   if(gradient_only)
                   {
                       auto eval = eval_virtual_tet<EVAL_EG, Policy>(
                           tet, positions_view, positions, dm_inv(I),
                           rest_volume(I), mat, dt);
                       DoubletVectorAssembler VA{G3s};
                       VA.template segment<StencilSize>(I * StencilSize)
                           .write(tet, eval.gradient);
                       return;
                   }

                   auto eval = eval_virtual_tet<EVAL_EGH, Policy>(
                       tet, positions_view, positions, dm_inv(I),
                       rest_volume(I), mat, dt);
                   DoubletVectorAssembler VA{G3s};
                   VA.template segment<StencilSize>(I * StencilSize)
                       .write(tet, eval.gradient);
                   TripletMatrixAssembler MA{H3x3s};
                   MA.template half_block<StencilSize>(I * HalfHessianSize)
                       .write(tet, eval.hessian);
               });
}
}  // namespace

void RCCBondedPTVirtualTetReporter::Impl::set_material(Float kappa) noexcept
{
    m_energy_model = uipc::core::RCCBondedPTVirtualTetEnergyModel::ABDOrtho;
    m_kappa        = kappa;
}

void RCCBondedPTVirtualTetReporter::Impl::set_material_neohookean(Float mu,
                                                                  Float lambda) noexcept
{
    m_energy_model =
        uipc::core::RCCBondedPTVirtualTetEnergyModel::StableNeoHookean;
    m_mu     = mu;
    m_lambda = lambda;
}

Float RCCBondedPTVirtualTetReporter::Impl::kappa() const noexcept
{
    return m_kappa;
}

bool RCCBondedPTVirtualTetReporter::Impl::active() const noexcept
{
    using EM = uipc::core::RCCBondedPTVirtualTetEnergyModel;
    return m_energy_model == EM::StableNeoHookean
               ? (m_mu > 0.0 && m_lambda > 0.0)
               : (m_kappa > 0.0);
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

    using EM = uipc::core::RCCBondedPTVirtualTetEnergyModel;
    if(m_energy_model == EM::StableNeoHookean)
        run_compute_energy<StableNeoHookeanEnergy>(
            StableNeoHookeanEnergy::Mat{m_mu, m_lambda}, topos, dm_inv,
            rest_volume, positions, dt, energies);
    else
        run_compute_energy<AbdOrthoEnergy>(AbdOrthoEnergy::Mat{m_kappa}, topos,
                                           dm_inv, rest_volume, positions, dt,
                                           energies);
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

    using EM = uipc::core::RCCBondedPTVirtualTetEnergyModel;
    if(m_energy_model == EM::StableNeoHookean)
        run_compute_dense<StableNeoHookeanEnergy>(
            StableNeoHookeanEnergy::Mat{m_mu, m_lambda}, topos, dm_inv,
            rest_volume, positions, dt, energies, gradients, hessians);
    else
        run_compute_dense<AbdOrthoEnergy>(AbdOrthoEnergy::Mat{m_kappa}, topos,
                                          dm_inv, rest_volume, positions, dt,
                                          energies, gradients, hessians);
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

    using EM = uipc::core::RCCBondedPTVirtualTetEnergyModel;
    if(m_energy_model == EM::StableNeoHookean)
        run_assemble<StableNeoHookeanEnergy>(
            StableNeoHookeanEnergy::Mat{m_mu, m_lambda}, topos, dm_inv,
            rest_volume, positions, dt, gradients, hessians, gradient_only);
    else
        run_assemble<AbdOrthoEnergy>(AbdOrthoEnergy::Mat{m_kappa}, topos, dm_inv,
                                     rest_volume, positions, dt, gradients,
                                     hessians, gradient_only);
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
    UIPC_ASSERT(model == "abd_ortho" || model == "stable_neo_hookean",
                "Unsupported rcc_bonded_pt_energy_model '{}'. "
                "Supported: 'abd_ortho' (kappa) or 'stable_neo_hookean' "
                "(young+poisson).",
                model);

    if(model == "stable_neo_hookean")
    {
        // NeoHookean material from Young's modulus E + Poisson ratio nu:
        //   mu     = E / (2 (1+nu))           (shear / second Lamé)
        //   lambda = E nu / ((1+nu)(1-2 nu))  (first Lamé)
        auto young_attr =
            world().scene().config().find<Float>("rcc_bonded_pt_neohookean_young");
        auto poisson_attr =
            world().scene().config().find<Float>("rcc_bonded_pt_neohookean_poisson");
        const Float E  = young_attr ? young_attr->view()[0] : Float{5e7};
        const Float nu = poisson_attr ? poisson_attr->view()[0] : Float{0.45};
        const Float mu = E / (2.0 * (1.0 + nu));
        const Float lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        m_impl.set_material_neohookean(mu, lambda);
    }
    else
    {
        auto kappa_attr =
            world().scene().config().find<Float>("rcc_bonded_pt_kappa");
        m_impl.set_material(kappa_attr ? kappa_attr->view()[0] : Float{1e8});
    }
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
