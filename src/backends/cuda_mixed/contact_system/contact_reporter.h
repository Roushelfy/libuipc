#pragma once
#include <sim_system.h>
#include <contact_system/global_contact_manager.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>

namespace uipc::backend::cuda_mixed
{
class ContactReporter : public DyTopoEffectReporter
{
  public:
    using DyTopoEffectReporter::DyTopoEffectReporter;
    using StoreScalar = GlobalDyTopoEffectManager::StoreScalar;
    using EnergyScalar = GlobalDyTopoEffectManager::EnergyScalar;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

    class Impl
    {
      public:
        muda::CBufferView<EnergyScalar>    energies;
        muda::CDoubletVectorView<StoreScalar, 3> gradients;
        muda::CTripletMatrixView<StoreScalar, 3> hessians;
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_init(InitInfo& info);

  private:
    friend class GlobalContactManager;
    void         init();  // only be called by GlobalContactManager
    virtual void do_build(DyTopoEffectReporter::BuildInfo&) final override;

    virtual EnergyComponentFlags component_flags() final override
    {
        return EnergyComponentFlags::Contact;
    }
    SizeT m_index = ~0ull;
    Impl  m_impl;
};
}  // namespace uipc::backend::cuda_mixed
