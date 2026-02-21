#include <linear_system/linear_pcg.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <cuda_device/builtin.h>
#include <utils/matrix_market.h>
#include <backends/common/backend_path_tool.h>
#include <algorithm>
#include <cmath>
namespace uipc::backend::cuda_mixed
{
REGISTER_SIM_SYSTEM(LinearPCG);

void LinearPCG::do_build(BuildInfo& info)
{
    auto& global_linear_system = require<GlobalLinearSystem>();

    // TODO: get info from the scene, now we just use the default value
    max_iter_ratio = 2;

    auto& config = world().scene().config();

    auto tol_rate_attr = config.find<double>("linear_system/tol_rate");
    if(tol_rate_attr)
        global_tol_rate = tol_rate_attr->view()[0];

    auto dump_attr  = config.find<IndexT>("extras/debug/dump_linear_pcg");
    need_debug_dump = dump_attr ? dump_attr->view()[0] : false;

    auto telemetry_enable_attr = config.find<IndexT>("extras/telemetry/enable");
    auto telemetry_pcg_attr    = config.find<IndexT>("extras/telemetry/pcg/enable");
    auto sample_every_attr = config.find<IndexT>("extras/telemetry/pcg/sample_every_iter");

    const bool telemetry_enable =
        telemetry_enable_attr && (telemetry_enable_attr->view()[0] != 0);
    telemetry_pcg_enable =
        telemetry_enable && telemetry_pcg_attr && (telemetry_pcg_attr->view()[0] != 0);

    if(sample_every_attr)
    {
        const auto sample_every = sample_every_attr->view()[0];
        telemetry_sample_every_iter =
            sample_every > 0 ? static_cast<SizeT>(sample_every) : SizeT{1};
    }

    logger::info("LinearPCG: max_iter_ratio = {}, tol_rate = {}, debug_dump = {}",
                 max_iter_ratio,
                 global_tol_rate,
                 need_debug_dump);
}

void LinearPCG::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    auto x = info.x();
    auto b = info.b();

    x.buffer_view().fill(0);
    if(telemetry_pcg_enable)
        m_pcg_samples.clear();

    auto N = x.size();
    if(z.capacity() < N)
    {
        auto M = reserve_ratio * N;
        z.reserve(M);
        p.reserve(M);
        r.reserve(M);
        Ap.reserve(M);
    }

    z.resize(N);
    p.resize(N);
    r.resize(N);
    Ap.resize(N);

    r0 = r;

    auto iter = pcg(x, b, static_cast<SizeT>(max_iter_ratio * static_cast<double>(b.size())));

    info.iter_count(iter);
    m_iter_history.push_back(info.iter_count());
}

void LinearPCG::dump_r_z(SizeT k)
{

    auto path_tool     = BackendPathTool(workspace());
    auto output_path   = path_tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
    auto output_path_r = fmt::format("{}r.{}.{}.{}.mtx",
                                     output_path.string(),
                                     engine().frame(),
                                     engine().newton_iter(),
                                     k);

    export_vector_market(output_path_r, r.cview());
    logger::info("Dumped PCG r to {}", output_path_r);

    auto output_path_z = fmt::format("{}z.{}.{}.{}.mtx",
                                     output_path.string(),
                                     engine().frame(),
                                     engine().newton_iter(),
                                     k);

    export_vector_market(fmt::format("{}z.{}.{}.{}.mtx",
                                     output_path.string(),
                                     engine().frame(),
                                     engine().newton_iter(),
                                     k),
                         z.cview());

    logger::info("Dumped PCG z to {}", output_path_z);
}

void LinearPCG::dump_p_Ap(SizeT k)
{
    auto path_tool     = BackendPathTool(workspace());
    auto output_folder = path_tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");

    auto output_path_p = fmt::format("{}p.{}.{}.{}.mtx",
                                     output_folder.string(),
                                     engine().frame(),
                                     engine().newton_iter(),
                                     k);

    export_vector_market(output_path_p, p.cview());
    logger::info("Dumped PCG p to {}", output_path_p);

    auto output_path_Ap = fmt::format("{}Ap.{}.{}.{}.mtx",
                                      output_folder.string(),
                                      engine().frame(),
                                      engine().newton_iter(),
                                      k);
    export_vector_market(output_path_Ap, Ap.cview());
    logger::info("Dumped PCG Ap to {}", output_path_Ap);
}

void LinearPCG::check_rz_nan_inf(SizeT k)
{
    double rz = static_cast<double>(ctx().dot(r.cview(), z.cview()));
    if(std::isnan(rz) || !std::isfinite(rz))
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        UIPC_ASSERT(!std::isnan(rz) && std::isfinite(rz),
                    "Frame {}, Newton: {}, Iteration {}: Residual is {}, norm(r) = {}, norm(z) = {}",
                    engine().frame(),
                    engine().newton_iter(),
                    k,
                    rz,
                    norm_r,
                    norm_z);
    }
}

void update_xr(double                                       alpha,
               muda::DenseVectorView<ActivePolicy::SolveScalar>  x,
               muda::CDenseVectorView<ActivePolicy::PcgAuxScalar> p,
               muda::DenseVectorView<ActivePolicy::PcgAuxScalar>  r,
               muda::CDenseVectorView<ActivePolicy::PcgAuxScalar> Ap)
{
    using namespace muda;
    using PcgScalar = ActivePolicy::PcgAuxScalar;

    // Fused update of x and r for better performance
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(r.size(),
               [alpha = alpha,
                x     = x.viewer().name("x"),
                p     = p.cviewer().name("p"),
                r     = r.viewer().name("r"),
                Ap    = Ap.cviewer().name("Ap")] __device__(int i) mutable
               {
                   x(i) += alpha * static_cast<double>(p(i));
                   r(i) -= static_cast<PcgScalar>(alpha * static_cast<double>(Ap(i)));
               });
}

void update_p(muda::DenseVectorView<ActivePolicy::PcgAuxScalar> p,
              muda::CDenseVectorView<ActivePolicy::PcgAuxScalar> z,
              double beta)
{
    using namespace muda;
    using PcgScalar = ActivePolicy::PcgAuxScalar;

    // Simple axpby
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(p.size(),
               [p = p.viewer().name("p"), z = z.cviewer().name("z"), beta = beta] __device__(
                   int i) mutable
               {
                   p(i) = z(i) + static_cast<PcgScalar>(beta * static_cast<double>(p(i)));
               });
}

void initialize_residual_from_rhs(muda::DenseVectorView<ActivePolicy::PcgAuxScalar> r,
                                  muda::CDenseVectorView<ActivePolicy::StoreScalar> b)
{
    using namespace muda;
    using PcgScalar = ActivePolicy::PcgAuxScalar;

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(r.size(),
               [r = r.viewer().name("r"), b = b.cviewer().name("b")] __device__(int i) mutable
               { r(i) = static_cast<PcgScalar>(b(i)); });
}

SizeT LinearPCG::pcg(muda::DenseVectorView<SolveScalar> x,
                     muda::CDenseVectorView<StoreScalar> b,
                     SizeT max_iter)
{
    SizeT k = 0;
    // r = b - A * x
    {
        initialize_residual_from_rhs(r.view(), b);

        // x == 0, so we don't need to do the following
        // r = - A * x + r
        //spmv(-1.0, x.as_const(), 1.0, r.view());
    }

    double alpha = 0.0, beta = 0.0, rz = 0.0, abs_rz0 = 0.0;

    // z = P * r (apply preconditioner)
    apply_preconditioner(z, r);

    if(need_debug_dump) [[unlikely]]
        dump_r_z(k);

    // p = z
    p = z;

    // init rz
    // rz = r^T * z
    rz = static_cast<double>(ctx().dot(r.cview(), z.cview()));
    check_rz_nan_inf(k);

    abs_rz0 = std::abs(rz);

    auto maybe_sample = [&](SizeT iter, double alpha_v, double beta_v, double rz_v)
    {
        if(!telemetry_pcg_enable)
            return;
        if(telemetry_sample_every_iter == 0)
            return;
        if(iter % telemetry_sample_every_iter != 0)
            return;

        PcgSample sample;
        sample.iter     = iter;
        sample.norm_r   = static_cast<double>(ctx().norm(r.cview()));
        sample.rz_ratio = (abs_rz0 > 0.0) ?
                              static_cast<double>(std::abs(rz_v) / abs_rz0) :
                              0.0;
        sample.alpha = alpha_v;
        sample.beta  = beta_v;
        sample.nan_inf_flag =
            !(std::isfinite(sample.norm_r) && std::isfinite(sample.rz_ratio)
              && std::isfinite(sample.alpha) && std::isfinite(sample.beta));
        m_pcg_samples.push_back(sample);
    };

    maybe_sample(k, 0.0, 0.0, rz);

    // check convergence
    if(accuracy_statisfied(r) && abs_rz0 == 0.0)
        return 0;

    for(k = 1; k < max_iter; ++k)
    {
        spmv(p.cview(), Ap.view());

        if(need_debug_dump) [[unlikely]]
            dump_p_Ap(k);

        // alpha = rz / p^T * Ap
        alpha = rz / static_cast<double>(ctx().dot(p.cview(), Ap.cview()));

        // x = x + alpha * p
        // r = r - alpha * Ap
        update_xr(alpha, x, p.cview(), r.view(), Ap.cview());

        // z = P * r (apply preconditioner)
        apply_preconditioner(z, r);

        if(need_debug_dump) [[unlikely]]
            dump_r_z(k);

        // rz_new = r^T * z
        double rz_new = static_cast<double>(ctx().dot(r.cview(), z.cview()));
        check_rz_nan_inf(k);

        // check convergence
        if(accuracy_statisfied(r) && std::abs(rz_new) <= global_tol_rate * abs_rz0)
        {
            maybe_sample(k, alpha, 0.0, rz_new);
            break;
        }

        // beta = rz_new / rz
        beta = rz_new / rz;

        // p = z + beta * p
        update_p(p.view(), z.cview(), beta);

        maybe_sample(k, alpha, beta, rz_new);

        rz = rz_new;
    }

    return k;
}
}  // namespace uipc::backend::cuda_mixed
