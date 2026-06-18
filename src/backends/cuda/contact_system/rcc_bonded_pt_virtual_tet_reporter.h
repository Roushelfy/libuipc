#pragma once

#include <contact_system/rcc_bonded_pt_system.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <muda/buffer/device_buffer.h>
#include <uipc/core/rcc_bonded_pt_oracle.h>  // RCCBondedPTVirtualTetEnergyModel

namespace uipc::backend::cuda
{
class RCCBondedPTVirtualTetReporter final : public DyTopoEffectReporter
{
  public:
    using DyTopoEffectReporter::DyTopoEffectReporter;

    static constexpr SizeT StencilSize = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    class Impl
    {
      public:
        // ABDOrtho material (also resets the model to ABDOrtho).
        void set_material(Float kappa) noexcept;
        // StableNeoHookean material (Lamé mu, lambda; sets the model).
        void set_material_neohookean(Float mu, Float lambda) noexcept;
        Float kappa() const noexcept;
        bool active() const noexcept;

        SizeT energy_count(muda::CBufferView<Vector4i> topos) const noexcept;
        SizeT gradient_count(muda::CBufferView<Vector4i> topos) const noexcept;
        SizeT hessian_count(muda::CBufferView<Vector4i> topos,
                            bool gradient_only) const noexcept;

        void compute_energy(muda::CBufferView<Vector4i> topos,
                            muda::CBufferView<Matrix3x3> dm_inv,
                            muda::CBufferView<Float> rest_volume,
                            muda::CBufferView<Vector3> positions,
                            Float dt,
                            muda::BufferView<Float> energies) const;

        void compute_dense_energy_gradient_hessian(
            muda::CBufferView<Vector4i> topos,
            muda::CBufferView<Matrix3x3> dm_inv,
            muda::CBufferView<Float> rest_volume,
            muda::CBufferView<Vector3> positions,
            Float dt,
            muda::BufferView<Float> energies,
            muda::BufferView<Vector12> gradients,
            muda::BufferView<Matrix12x12> hessians) const;

        void assemble(muda::CBufferView<Vector4i> topos,
                      muda::CBufferView<Matrix3x3> dm_inv,
                      muda::CBufferView<Float> rest_volume,
                      muda::CBufferView<Vector3> positions,
                      Float dt,
                      muda::DoubletVectorView<Float, 3> gradients,
                      muda::TripletMatrixView<Float, 3> hessians,
                      bool gradient_only) const;

        SimSystemSlot<RCCBondedPTSystem> bonded_pt_system;
        SimSystemSlot<GlobalVertexManager> global_vertex_manager;
        S<const geometry::AttributeSlot<Float>> dt_attr;

      private:
        uipc::core::RCCBondedPTVirtualTetEnergyModel m_energy_model =
            uipc::core::RCCBondedPTVirtualTetEnergyModel::ABDOrtho;
        Float m_kappa  = 0.0;  // ABDOrtho material
        Float m_mu     = 0.0;  // StableNeoHookean Lamé (shear)
        Float m_lambda = 0.0;  // StableNeoHookean Lamé (dilational)
    };

  private:
    virtual void do_build(DyTopoEffectReporter::BuildInfo&) override;
    virtual void do_report_energy_extent(
        GlobalDyTopoEffectManager::EnergyExtentInfo& info) override;
    virtual void do_report_gradient_hessian_extent(
        GlobalDyTopoEffectManager::GradientHessianExtentInfo& info) override;
    virtual void do_assemble(GlobalDyTopoEffectManager::GradientHessianInfo& info) override;
    virtual void do_compute_energy(GlobalDyTopoEffectManager::EnergyInfo& info) override;
    virtual EnergyComponentFlags component_flags() override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
