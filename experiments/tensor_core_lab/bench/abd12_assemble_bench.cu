#include <benchmark/benchmark.h>

#include <cmath>

#include <tcl/case_spec.h>
#include <tcl/runner.h>

namespace
{
using namespace uipc::tensor_core_lab;

template <typename PreparedT>
void run_bench(benchmark::State& state,
               const ContractionCaseData& data,
               PreparedT&                  prepared,
               BackendContext&             context,
               RunOutcome (*execute)(BackendContext&, PreparedT&, bool))
{
    const auto warm = execute(context, prepared, true);
    if(warm.status != RunStatus::Ok)
    {
        state.SkipWithError(warm.message.c_str());
        return;
    }

    const auto output = download_output(prepared);
    const auto metrics = compare_square_batches(data.reference,
                                                output,
                                                data.spec.shape.logical_rows,
                                                data.spec.shape.physical_rows,
                                                data.spec.batch_count);

    state.counters["rel_error"]      = metrics.rel_fro;
    state.counters["rel_fro"]        = metrics.rel_fro;
    state.counters["abs_linf"]       = metrics.abs_linf;
    state.counters["nan_inf_count"]  = metrics.nan_inf_count;
    state.counters["symmetry_error"] = metrics.symmetry_error;
    state.counters["impl_path_id"]   = static_cast<double>(static_cast<int>(warm.trace.impl_path));
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

void bench_abd12_assemble(benchmark::State& state, Mode mode)
{
    const int    batch    = static_cast<int>(state.range(0));
    const int    cond_exp = static_cast<int>(state.range(1));
    const double cond     = std::pow(10.0, static_cast<double>(cond_exp));
    const auto   data     = make_abd12_assemble_case("bench_abd12_assemble", batch, 43, cond);

    BackendContext context(mode);
    if(!context.is_supported())
    {
        state.SkipWithError(context.unsupported_reason().c_str());
        return;
    }

    if(mode == Mode::Fp64RefNoTc)
    {
        auto prepared = prepare_f64_case(data);
        run_bench(state, data, prepared, context, execute_abd12_assemble_case);
    }
    else
    {
        auto prepared = prepare_f32_case(data);
        run_bench(state, data, prepared, context, execute_abd12_assemble_case);
    }
}

void BM_Abd12AssembleFp64(benchmark::State& state)
{
    bench_abd12_assemble(state, Mode::Fp64RefNoTc);
}
void BM_Abd12AssembleFp32(benchmark::State& state)
{
    bench_abd12_assemble(state, Mode::Fp32NoTc);
}
void BM_Abd12AssembleTc32(benchmark::State& state)
{
    bench_abd12_assemble(state, Mode::Tc32Tf32);
}
}  // namespace

BENCHMARK(BM_Abd12AssembleFp64)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
BENCHMARK(BM_Abd12AssembleFp32)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
BENCHMARK(BM_Abd12AssembleTc32)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
