#pragma once
#include <sim_system.h>
#include <mixed_precision/policy.h>
#include <optional>

namespace uipc::backend::cuda_mixed
{
class LineSearchReporter;

class LineSearcher : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    using EnergyScalar = ActivePolicy::EnergyScalar;

    class RecordInfo
    {
      public:
    };

    class StepInfo
    {
      public:
        Float alpha;
    };

    class ComputeEnergyInfo
    {
      public:
        ComputeEnergyInfo(LineSearcher* impl) noexcept;
        Float dt() noexcept;
        void  energy(EnergyScalar e) noexcept;
        bool  is_initial() noexcept;


      private:
        friend class LineSearcher;
        LineSearcher*               m_impl       = nullptr;
        std::optional<EnergyScalar> m_energy     = std::nullopt;
        bool                        m_is_initial = false;
    };

    void add_reporter(LineSearchReporter* reporter);

    SizeT max_iter() const noexcept;

  protected:
    void do_build() override;

  private:
    friend class SimEngine;
    void  init();                           // only be called by SimEngine
    void  record_start_point();             // only be called by SimEngine
    void  step_forward(Float alpha);        // only be called by SimEngine
    EnergyScalar compute_energy(bool is_initial);  // only be called by SimEngine

    SimSystemSlotCollection<LineSearchReporter> m_reporters;

    vector<EnergyScalar> m_energy_values;
    bool                 m_report_energy = false;
    std::stringstream    m_report_stream;
    Float                m_dt       = 0.0;
    IndexT               m_max_iter = 64;
};
}  // namespace uipc::backend::cuda_mixed

