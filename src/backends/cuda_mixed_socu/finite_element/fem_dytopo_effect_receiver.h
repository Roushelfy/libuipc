#pragma once
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <finite_element/finite_element_vertex_reporter.h>

namespace uipc::backend::cuda_mixed
{
class FEMDyTopoEffectReceiver final : public DyTopoEffectReceiver
{
  public:
    using DyTopoEffectReceiver::DyTopoEffectReceiver;
    using StoreScalar = GlobalDyTopoEffectManager::StoreScalar;

    class Impl
    {
      public:
        void receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info);

        FiniteElementVertexReporter* finite_element_vertex_reporter = nullptr;

        muda::CDoubletVectorView<StoreScalar, 3> gradients;
        muda::CTripletMatrixView<StoreScalar, 3> hessians;
    };

    auto gradients() const noexcept { return m_impl.gradients; }
    auto hessians() const noexcept { return m_impl.hessians; }

  protected:
    virtual void do_build(DyTopoEffectReceiver::BuildInfo& info) override;

  private:
    friend class FEMLinearSubsystem;

    virtual void do_report(GlobalDyTopoEffectManager::ClassifyInfo& info) override;
    virtual void do_receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info) override;
    Impl m_impl;
};
}  // namespace uipc::backend::cuda_mixed
