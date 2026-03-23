#include <linear_system/iterative_solver.h>
#include <linear_system/global_linear_system.h>
namespace uipc::backend::cuda_mixed
{
void IterativeSolver::do_build()
{
    m_system = &require<GlobalLinearSystem>();

    BuildInfo info;
    do_build(info);

    m_system->add_solver(this);
}

void IterativeSolver::spmv(ActivePolicy::PcgIterScalar                 a,
                           muda::CDenseVectorView<ActivePolicy::PcgAuxScalar> x,
                           ActivePolicy::PcgIterScalar                 b,
                           muda::DenseVectorView<ActivePolicy::PcgAuxScalar>  y)
{
    m_system->m_impl.spmv(a, x, b, y);
}

void IterativeSolver::spmv(muda::CDenseVectorView<ActivePolicy::PcgAuxScalar> x,
                           muda::DenseVectorView<ActivePolicy::PcgAuxScalar> y)
{
    using IterScalar = ActivePolicy::PcgIterScalar;
    spmv(IterScalar{1}, x, IterScalar{0}, y);
}

void IterativeSolver::apply_preconditioner(
    muda::DenseVectorView<ActivePolicy::PcgAuxScalar>  z,
    muda::CDenseVectorView<ActivePolicy::PcgAuxScalar> r,
    muda::CVarView<IndexT>                             converged)
{
    m_system->m_impl.apply_preconditioner(z, r, converged);
}

bool IterativeSolver::accuracy_statisfied(
    muda::DenseVectorView<ActivePolicy::PcgAuxScalar> r)
{
    return m_system->m_impl.accuracy_statisfied(r);
}

muda::LinearSystemContext& IterativeSolver::ctx() const
{
    return m_system->m_impl.ctx;
}

void IterativeSolver::solve(GlobalLinearSystem::SolvingInfo& info)
{
    do_solve(info);
}
}  // namespace uipc::backend::cuda_mixed
