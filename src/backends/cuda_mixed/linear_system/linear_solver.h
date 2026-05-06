#pragma once
#include <sim_system.h>
#include <linear_system/global_linear_system.h>
#include <string_view>

namespace uipc::backend::cuda_mixed
{
class LinearSolver : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    struct AssemblyRequirements
    {
        bool full_sparse_matrix = true;
        bool preconditioner     = false;
    };

    class BuildInfo
    {
      public:
    };

    virtual AssemblyRequirements assembly_requirements() const
    {
        return AssemblyRequirements{};
    }

    virtual std::string_view iteration_counter_name() const { return {}; }

    GlobalLinearSystem& system() const noexcept { return *m_system; }

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_solve(GlobalLinearSystem::SolvingInfo& info) = 0;
    GlobalLinearSystem* m_system = nullptr;

  private:
    friend class GlobalLinearSystem;

    virtual void do_build() final override;
    void solve(GlobalLinearSystem::SolvingInfo& info) { do_solve(info); }
};
}  // namespace uipc::backend::cuda_mixed
