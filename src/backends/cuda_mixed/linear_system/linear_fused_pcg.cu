#include <linear_system/linear_fused_pcg.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <uipc/common/timer.h>
#include <cmath>

namespace uipc::backend::cuda_mixed
{
REGISTER_SIM_SYSTEM(LinearFusedPCG);

void LinearFusedPCG::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();

    auto solver_attr = config.find<std::string>("linear_system/solver");
    std::string solver_name = solver_attr ? solver_attr->view()[0] : std::string{"fused_pcg"};
    if(solver_name != "fused_pcg")
    {
        throw SimSystemException("LinearFusedPCG unused");
    }

    auto& global_linear_system = require<GlobalLinearSystem>();

    max_iter_ratio = 2;

    auto tol_rate_attr = config.find<double>("linear_system/tol_rate");
    if(tol_rate_attr)
        global_tol_rate = tol_rate_attr->view()[0];

    auto check_attr = config.find<IndexT>("linear_system/check_interval");
    if(check_attr)
        check_interval = check_attr->view()[0];

    logger::info("LinearFusedPCG: max_iter_ratio = {}, tol_rate = {}, check_interval = {}",
                 max_iter_ratio,
                 global_tol_rate,
                 check_interval);
}

void LinearFusedPCG::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    auto x = info.x();
    auto b = info.b();

    x.buffer_view().fill(static_cast<SolveScalar>(0));

    auto N = x.size();
    if(r.capacity() < N)
    {
        auto M = reserve_ratio * N;
        r.reserve(M);
        z.reserve(M);
        p.reserve(M);
        Ap.reserve(M);
    }

    r.resize(N);
    z.resize(N);
    p.resize(N);
    Ap.resize(N);

    auto iter = fused_pcg(x, b, static_cast<SizeT>(max_iter_ratio * static_cast<double>(b.size())));

    info.iter_count(iter);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC T abs_scalar(T v)
{
    return v >= T{0} ? v : -v;
}

// d_result = x^T * y  (cublas-free, device-only)
template <typename VecScalar, typename IterScalar>
void fused_dot(muda::CDenseVectorView<VecScalar> x,
               muda::CDenseVectorView<VecScalar> y,
               muda::VarView<IterScalar>         d_result)
{
    using namespace muda;

    cudaMemsetAsync(d_result.data(), 0, sizeof(IterScalar));

    constexpr int block_dim = 256;
    constexpr int warp_size = 32;
    int           n         = x.size();
    int           block_count = (n + block_dim - 1) / block_dim;

    Launch(block_count, block_dim)
        .file_line(__FILE__, __LINE__)
        .apply(
               [x        = x.cviewer().name("x"),
                y        = y.cviewer().name("y"),
                d_result = d_result.data(),
                n] __device__() mutable
               {
                   int i = blockIdx.x * blockDim.x + threadIdx.x;
                   IterScalar val =
                       (i < n) ? static_cast<IterScalar>(x(i)) * static_cast<IterScalar>(y(i))
                               : IterScalar{0};

                   for(int offset = warp_size / 2; offset > 0; offset /= 2)
                       val += __shfl_down_sync(0xFFFFFFFF, val, offset);

                   if((threadIdx.x & (warp_size - 1)) == 0)
                       atomicAdd(d_result, val);
               });
}

// x += alpha*p, r -= alpha*Ap  (alpha = d_rz/d_pAp on device)
template <typename SolveScalar, typename PcgScalar, typename IterScalar>
void fused_update_xr(muda::CVarView<IterScalar>          d_rz,
                     muda::CVarView<IterScalar>          d_pAp,
                     muda::DenseVectorView<SolveScalar>  x,
                     muda::CDenseVectorView<PcgScalar>   p,
                     muda::DenseVectorView<PcgScalar>    r,
                     muda::CDenseVectorView<PcgScalar>   Ap)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(r.size(),
               [d_rz  = d_rz.cviewer().name("d_rz"),
                d_pAp = d_pAp.cviewer().name("d_pAp"),
                x     = x.viewer().name("x"),
                p     = p.cviewer().name("p"),
                r     = r.viewer().name("r"),
                Ap    = Ap.cviewer().name("Ap")] __device__(int i) mutable
               {
                   IterScalar alpha = *d_rz / *d_pAp;
                   x(i) += static_cast<SolveScalar>(alpha * static_cast<IterScalar>(p(i)));
                   r(i) -= static_cast<PcgScalar>(alpha * static_cast<IterScalar>(Ap(i)));
               });
}

// p = z + beta*p (beta = rz_new/rz on device). Skips update when abs(rz_new) <= rz_tol. No d_rz write.
template <typename PcgScalar, typename IterScalar>
void fused_update_p(muda::CVarView<IterScalar>       d_rz_new,
                    muda::CVarView<IterScalar>       d_rz,
                    muda::DenseVectorView<PcgScalar> p,
                    muda::CDenseVectorView<PcgScalar> z,
                    IterScalar                       rz_tol)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(p.size(),
               [d_rz_new = d_rz_new.cviewer().name("d_rz_new"),
                d_rz     = d_rz.cviewer().name("d_rz"),
                p        = p.viewer().name("p"),
                z        = z.cviewer().name("z"),
                rz_tol] __device__(int i) mutable
               {
                   IterScalar rz_new = *d_rz_new;
                   if(abs_scalar(rz_new) <= rz_tol)
                       return;
                   IterScalar rz_old = *d_rz;
                   IterScalar beta   = rz_new / rz_old;
                   p(i)              = z(i) + static_cast<PcgScalar>(
                                          beta * static_cast<IterScalar>(p(i)));
               });
}

// d_rz = d_rz_new when not converged (single-thread write).
template <typename IterScalar>
void fused_swap_rz(muda::CVarView<IterScalar> d_rz_new,
                   muda::VarView<IterScalar>  d_rz,
                   IterScalar                 rz_tol)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(1,
               [d_rz_new = d_rz_new.cviewer().name("d_rz_new"),
                d_rz     = d_rz.viewer().name("d_rz"),
                rz_tol] __device__(int) mutable
               {
                   IterScalar rz_new = *d_rz_new;
                   if(abs_scalar(rz_new) > rz_tol)
                       *d_rz = rz_new;
               });
}

template <typename PcgScalar, typename StoreScalar>
void initialize_residual_from_rhs(muda::DenseVectorView<PcgScalar>    r,
                                  muda::CDenseVectorView<StoreScalar> b)
{
    using namespace muda;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(r.size(),
               [r = r.viewer().name("r"), b = b.cviewer().name("b")] __device__(int i) mutable
               { r(i) = static_cast<PcgScalar>(b(i)); });
}

SizeT LinearFusedPCG::fused_pcg(muda::DenseVectorView<SolveScalar>  x,
                                muda::CDenseVectorView<StoreScalar> b,
                                SizeT                         max_iter)
{
    Timer pcg_timer{"FusedPCG"};

    SizeT k = 0;
    d_converged = 0;

    // r = b - A*x, but x0 = 0 so r = b
    initialize_residual_from_rhs<PcgScalar, StoreScalar>(r.view(), b);

    // z = P^{-1} * r
    {
        Timer timer{"Apply Preconditioner"};
        apply_preconditioner(z, r, d_converged.view());
    }

    // p = z
    p = z;

    // rz = r^T * z
    fused_dot<PcgScalar, IterScalar>(r.cview(), z.cview(), d_rz.view());
    IterScalar rz_host = d_rz;
    IterScalar abs_rz0 = abs_scalar(rz_host);

    if(accuracy_statisfied(r) && abs_rz0 == IterScalar{0})
        return 0;

    IterScalar rz_tol = static_cast<IterScalar>(global_tol_rate) * abs_rz0;

    for(k = 1; k < max_iter; ++k)
    {
        // Ap = A * p,  pAp = p^T * Ap
        {
            Timer timer{"SpMV"};
            spmv(p.cview(), Ap.view());
            fused_dot<PcgScalar, IterScalar>(p.cview(), Ap.cview(), d_pAp.view());
        }

        // alpha = rz / pAp,  x += alpha * p,  r -= alpha * Ap
        fused_update_xr<SolveScalar, PcgScalar, IterScalar>(
            d_rz.view(), d_pAp.view(), x, p.cview(), r.view(), Ap.cview());

        // z = P^{-1} * r
        {
            Timer timer{"Apply Preconditioner"};
            apply_preconditioner(z, r, d_converged.view());
        }

        // rz_new = r^T * z
        fused_dot<PcgScalar, IterScalar>(r.cview(), z.cview(), d_rz_new.view());

        if(k % check_interval == 0)
        {
            if(accuracy_statisfied(r))
            {
                IterScalar rz_new_host = d_rz_new;
                if(!std::isfinite(static_cast<double>(rz_new_host))) [[unlikely]]
                {
                    auto norm_r = ctx().norm(r.cview());
                    auto norm_z = ctx().norm(z.cview());
                    UIPC_ASSERT(false,
                                "Frame {}, Newton {}, FusedPCG Iter {}: r^T*z = {}, "
                                "norm(r) = {}, norm(z) = {}.",
                                engine().frame(),
                                engine().newton_iter(),
                                k,
                                rz_new_host,
                                norm_r,
                                norm_z);
                }
                if(abs_scalar(rz_new_host) <= rz_tol)
                    break;
            }
        }

        // p = z + beta * p (skip when abs(rz_new) <= rz_tol), then rz = rz_new and convergence flag.
        fused_update_p<PcgScalar, IterScalar>(
            d_rz_new.view(), d_rz.view(), p.view(), z.cview(), rz_tol);
        fused_swap_rz<IterScalar>(d_rz_new.view(), d_rz.view(), rz_tol);
    }

    return k;
}
}  // namespace uipc::backend::cuda_mixed

