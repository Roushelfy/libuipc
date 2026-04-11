#include <linear_system/linear_fused_pcg.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <uipc/common/timer.h>
#include <cub/warp/warp_reduce.cuh>

namespace uipc::backend::cuda_mixed
{
REGISTER_SIM_SYSTEM(LinearFusedPCG);

void LinearFusedPCG::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();

    auto        solver_attr = config.find<std::string>("linear_system/solver");
    std::string solver_name =
        solver_attr ? solver_attr->view()[0] : std::string{"fused_pcg"};
    if(solver_name != "fused_pcg")
    {
        throw SimSystemException("LinearFusedPCG unused");
    }

    auto& global_linear_system = require<GlobalLinearSystem>();

    max_iter_ratio = 2;

    auto tol_rate_attr = config.find<Float>("linear_system/tol_rate");
    global_tol_rate    = tol_rate_attr->view()[0];

    auto check_attr = config.find<IndexT>("linear_system/check_interval");
    if(check_attr)
        check_interval = check_attr->view()[0];

    auto dump_attr = config.find<IndexT>("extras/debug/dump_linear_pcg");
    if(dump_attr && dump_attr->view()[0] != 0)
        logger::warn(
            "LinearFusedPCG: extras/debug/dump_linear_pcg is enabled but "
            "fused_pcg does not support PCG vector dumps. "
            "Set linear_system/solver to \"linear_pcg\" to use this feature.");

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

    auto iter = fused_pcg(x, b, max_iter_ratio * b.size());

    info.iter_count(iter);
}

void LinearFusedPCG::check_init_rz_nan_inf(IterScalar rz)
{
    if(!std::isfinite(static_cast<double>(rz))) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        bool r_bad  = !std::isfinite(norm_r);
        auto hint = r_bad ? "gradient assembling produced NaN values, likely due to error in formula implementation" :
                            "preconditioner failed, likely due to inverse matrix calculation failure";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, FusedPCG Init: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

void LinearFusedPCG::check_iter_rz_nan_inf(IterScalar rz_new,
                                           SizeT      k,
                                           IterScalar rz,
                                           IterScalar pAp,
                                           IterScalar alpha,
                                           IterScalar beta)
{
    if(!std::isfinite(static_cast<double>(rz_new))) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        bool r_ok   = std::isfinite(norm_r);
        bool z_bad  = !std::isfinite(norm_z);
        auto hint = (r_ok && z_bad) ?
                        "preconditioner failed, likely due to inverse matrix calculation failure" :
                        "PCG iteration diverged";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, FusedPCG Iter {}: rz_new = {}, rz = {}, pAp = {}, alpha = {}, beta = {}, "
                    "norm(r) = {}, norm(z) = {}. Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    k,
                    rz_new,
                    rz,
                    pAp,
                    alpha,
                    beta,
                    norm_r,
                    norm_z,
                    hint);
    }
}

// d_result = x^T * y  (cublas-free, device-only, CUB warp reduction)
template <typename VecScalar, typename IterScalar>
void fused_dot(muda::CDenseVectorView<VecScalar> x,
               muda::CDenseVectorView<VecScalar> y,
               muda::VarView<IterScalar>         d_result)
{
    using namespace muda;

    cudaMemsetAsync(d_result.data(), 0, sizeof(IterScalar));

    constexpr int block_dim = 256;
    constexpr int warp_size = 32;
    constexpr int num_warps = block_dim / warp_size;
    int           n         = x.size();
    int           block_count = (n + block_dim - 1) / block_dim;

    Launch(block_count, block_dim)
        .file_line(__FILE__, __LINE__)
        .apply(
            [x        = x.cviewer().name("x"),
             y        = y.cviewer().name("y"),
             d_result = d_result.viewer().name("d_result"),
             n] __device__() mutable
            {
                using WarpReduce = cub::WarpReduce<IterScalar, warp_size>;
                __shared__ typename WarpReduce::TempStorage temp_storage[num_warps];

                int        i   = blockIdx.x * blockDim.x + threadIdx.x;
                IterScalar val = (i < n) ?
                                     static_cast<IterScalar>(x(i))
                                         * static_cast<IterScalar>(y(i)) :
                                     IterScalar(0);

                int        warp_id  = threadIdx.x / warp_size;
                int        lane_id  = threadIdx.x & (warp_size - 1);
                IterScalar warp_sum = WarpReduce(temp_storage[warp_id]).Sum(val);

                if(lane_id == 0)
                    muda::atomic_add(d_result.data(), warp_sum);
            });
}

// Same as linear_pcg update_xr: alpha = rz/pAp, x += alpha*p, r -= alpha*Ap.
template <typename SolveScalar, typename PcgScalar, typename IterScalar>
void fused_update_xr(muda::CVarView<IterScalar>          d_rz,
                     muda::CVarView<IterScalar>          d_pAp,
                     muda::CVarView<IndexT>              d_converged,
                     muda::DenseVectorView<SolveScalar>  x,
                     muda::CDenseVectorView<PcgScalar>   p,
                     muda::DenseVectorView<PcgScalar>    r,
                     muda::CDenseVectorView<PcgScalar>   Ap)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(r.size(),
               [d_rz        = d_rz.cviewer().name("d_rz"),
                d_pAp       = d_pAp.cviewer().name("d_pAp"),
                d_converged = d_converged.cviewer().name("d_converged"),
                x           = x.viewer().name("x"),
                p           = p.cviewer().name("p"),
                r           = r.viewer().name("r"),
                Ap          = Ap.cviewer().name("Ap")] __device__(int i) mutable
               {
                   if(*d_converged != 0)
                       return;
                   IterScalar alpha = *d_rz / *d_pAp;
                   x(i) += static_cast<SolveScalar>(alpha * static_cast<IterScalar>(p(i)));
                   r(i) -= static_cast<PcgScalar>(alpha * static_cast<IterScalar>(Ap(i)));
               });
}

// Same as linear_pcg update_p: beta = rz_new/rz, p = z + beta*p.
template <typename PcgScalar, typename IterScalar>
void fused_update_p(muda::CVarView<IterScalar>       d_rz_new,
                    muda::CVarView<IterScalar>       d_rz,
                    muda::CVarView<IndexT>           d_converged,
                    muda::DenseVectorView<PcgScalar> p,
                    muda::CDenseVectorView<PcgScalar> z)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(p.size(),
               [d_rz_new    = d_rz_new.cviewer().name("d_rz_new"),
                d_rz        = d_rz.cviewer().name("d_rz"),
                d_converged = d_converged.cviewer().name("d_converged"),
                p           = p.viewer().name("p"),
                z           = z.cviewer().name("z")] __device__(int i) mutable
               {
                   if(*d_converged != 0)
                       return;
                   IterScalar beta = *d_rz_new / *d_rz;
                   p(i)            = z(i)
                          + static_cast<PcgScalar>(beta * static_cast<IterScalar>(p(i)));
               });
}

// d_rz = d_rz_new when not converged (single-thread write).
template <typename IterScalar>
void fused_swap_rz(muda::CVarView<IterScalar> d_rz_new,
                   muda::VarView<IterScalar>  d_rz,
                   muda::CVarView<IndexT>     d_converged)
{
    using namespace muda;

    Launch()
        .file_line(__FILE__, __LINE__)
        .apply(
            [d_rz_new    = d_rz_new.cviewer().name("d_rz_new"),
             d_rz        = d_rz.viewer().name("d_rz"),
             d_converged = d_converged.cviewer().name("d_converged")] __device__() mutable
            {
                if(*d_converged != 0)
                    return;
                *d_rz = *d_rz_new;
            });
}

template <typename IterScalar>
void fused_update_converged(muda::VarView<IterScalar> d_rz_new,
                            muda::VarView<IndexT>     d_converged,
                            IterScalar                rz_tol)
{
    using namespace muda;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(1,
               [d_rz_new    = d_rz_new.cviewer().name("d_rz_new"),
                d_converged = d_converged.viewer().name("d_converged"),
                rz_tol] __device__(int) mutable
               {
                   IterScalar rz_new = *d_rz_new;
                   IterScalar abs_rz_new = rz_new >= IterScalar{0} ? rz_new : -rz_new;
                   *d_converged          = abs_rz_new <= rz_tol ? 1 : 0;
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
                                SizeT                               max_iter)
{
    Timer pcg_timer{"FusedPCG"};

    SizeT k     = 0;
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
    IterScalar rz_host0 = d_rz;
    check_init_rz_nan_inf(rz_host0);
    IterScalar abs_rz0 = static_cast<IterScalar>(std::abs(static_cast<double>(rz_host0)));

    if(abs_rz0 == IterScalar{0})
        return 0;

    IterScalar rz_tol = static_cast<IterScalar>(global_tol_rate) * abs_rz0;
    SizeT effective_check_interval = check_interval > 0 ? check_interval : SizeT{1};

    for(k = 1; k < max_iter; ++k)
    {
        Timer iter_timer{"PCG Iteration"};
        bool do_check = (k % effective_check_interval == 0) || (k + 1 == max_iter);

        // Ap = A * p,  pAp = p^T * Ap
        {
            Timer timer{"SpMV"};
            spmv(p.cview(), Ap.view());
            fused_dot<PcgScalar, IterScalar>(p.cview(), Ap.cview(), d_pAp.view());
        }

        // alpha = rz / pAp,  x += alpha * p,  r -= alpha * Ap
        fused_update_xr<SolveScalar, PcgScalar, IterScalar>(
            d_rz.view(), d_pAp.view(), d_converged.view(), x, p.cview(), r.view(), Ap.cview());

        // z = P^{-1} * r
        {
            Timer timer{"Apply Preconditioner"};
            apply_preconditioner(z, r, d_converged.view());
        }

        // rz_new = r^T * z, keep convergence flag on device for preconditioner skip.
        fused_dot<PcgScalar, IterScalar>(r.cview(), z.cview(), d_rz_new.view());
        fused_update_converged(d_rz_new.view(), d_converged.view(), rz_tol);

        if(do_check)
        {
            IterScalar rz_host_cur = d_rz;
            IterScalar pAp_host    = d_pAp;
            IterScalar rz_new_host = d_rz_new;
            IterScalar alpha_host  = rz_host_cur / pAp_host;
            IterScalar beta_host   = rz_new_host / rz_host_cur;
            check_iter_rz_nan_inf(rz_new_host, k, rz_host_cur, pAp_host, alpha_host, beta_host);
            if((std::abs(static_cast<double>(rz_new_host)) / static_cast<double>(abs_rz0))
               <= global_tol_rate)
                break;
        }

        // p = z + beta * p, then rz = rz_new.
        fused_update_p<PcgScalar, IterScalar>(
            d_rz_new.view(), d_rz.view(), d_converged.view(), p.view(), z.cview());
        fused_swap_rz<IterScalar>(d_rz_new.view(), d_rz.view(), d_converged.view());
    }

    return k;
}
}  // namespace uipc::backend::cuda_mixed
