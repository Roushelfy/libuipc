#include <benchmark/benchmark.h>

#include <string>

#include <tcg/registry.h>
#include <tcg/runner.h>

namespace
{
using namespace uipc::tensor_core_gemm;

std::string mode_suffix(Mode mode)
{
    switch(mode)
    {
        case Mode::Fp64RefNoTc:
            return "Fp64";
        case Mode::Fp32NoTc:
            return "Fp32";
        case Mode::Tc32Tf32:
            return "Tc32";
    }
    return "Fp64";
}

std::string layout_prefix(GemmLayoutVariant variant)
{
    return variant == GemmLayoutVariant::Raw ? "Raw" : "Padded";
}

std::string benchmark_name(GemmLayoutVariant          variant,
                           Mode                       mode,
                           const RegisteredGemmShape& shape,
                           int                        batch_count)
{
    return "BM_Gemm" + layout_prefix(variant) + mode_suffix(mode) + "/"
           + std::to_string(shape.logical_shape.m) + "/"
           + std::to_string(shape.logical_shape.n) + "/"
           + std::to_string(shape.logical_shape.k) + "/"
           + std::to_string(batch_count);
}

template <typename PreparedT>
void run_bench(benchmark::State& state,
               GemmCaseData&     data,
               PreparedT&        prepared,
               BackendContext&   context,
               RunOutcome (*execute)(BackendContext&, PreparedT&, bool))
{
    const auto warm = execute(context, prepared, true);
    if(warm.status != RunStatus::Ok)
    {
        state.SkipWithError(warm.message.c_str());
        return;
    }

    const auto metrics = ::uipc::tensor_core_lab::compare_matrix_batches(
        data.reference_fp64,
        download_output(prepared),
        data.spec.m,
        data.spec.n,
        data.spec.physical_m,
        data.spec.physical_n,
        data.spec.batch_count);

    state.counters["rel_error"] = metrics.rel_fro;
    state.counters["rel_fro"]   = metrics.rel_fro;
    state.counters["abs_linf"]  = metrics.abs_linf;
    state.counters["nan_inf_count"] = metrics.nan_inf_count;
    state.counters["impl_path_id"] = static_cast<double>(static_cast<int>(warm.trace.impl_path));
    state.counters["tensor_core_requested"] =
        warm.trace.tensor_core_requested ? 1.0 : 0.0;
    state.counters["tensor_core_verified_id"] =
        static_cast<double>(static_cast<int>(warm.trace.tensor_core_verified));

    for(auto _ : state)
    {
        const auto out = execute(context, prepared, true);
        if(out.status != RunStatus::Ok)
        {
            state.SkipWithError(out.message.c_str());
            break;
        }
        state.SetIterationTime(out.elapsed_ms / 1000.0);
    }
}

void bench_gemm(benchmark::State&          state,
                Mode                       mode,
                const RegisteredGemmShape& shape,
                GemmLayoutVariant          variant,
                int                        batch_count)
{
    auto data =
        make_gemm_case(shape.logical_shape, shape.shape_group, variant, batch_count, 41);
    ensure_fp64_reference(data);

    BackendContext context(mode);
    if(!context.is_supported())
    {
        state.SkipWithError(context.unsupported_reason().c_str());
        return;
    }

    if(mode == Mode::Fp64RefNoTc)
    {
        auto prepared = prepare_f64_case(data);
        run_bench(state, data, prepared, context, execute_gemm_case);
    }
    else
    {
        auto prepared = prepare_f32_case(data);
        run_bench(state, data, prepared, context, execute_gemm_case);
    }
}

bool register_benchmarks()
{
    for(const auto& shape : all_registered_shapes())
    {
        for(const auto variant : {GemmLayoutVariant::Raw, GemmLayoutVariant::Padded})
        {
            for(const int batch_count : full_batches(shape))
            {
                for(const auto mode :
                    {Mode::Fp64RefNoTc, Mode::Fp32NoTc, Mode::Tc32Tf32})
                {
                    const auto name = benchmark_name(variant, mode, shape, batch_count);
                    benchmark::RegisterBenchmark(
                        name.c_str(),
                        [shape, variant, batch_count, mode](benchmark::State& state)
                        { bench_gemm(state, mode, shape, variant, batch_count); })
                        ->UseManualTime();
                }
            }
        }
    }
    return true;
}

const bool kRegistered = register_benchmarks();
}  // namespace
