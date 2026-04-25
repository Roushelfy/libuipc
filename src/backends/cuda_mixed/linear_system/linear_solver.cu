#include <linear_system/linear_solver.h>

namespace uipc::backend::cuda_mixed
{
void LinearSolver::do_build()
{
    m_system = &require<GlobalLinearSystem>();

    BuildInfo info;
    do_build(info);

    m_system->add_solver(this);
}

}  // namespace uipc::backend::cuda_mixed
