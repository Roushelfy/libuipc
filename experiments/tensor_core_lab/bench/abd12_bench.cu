#include <benchmark/benchmark.h>

#include <cmath>

#include <tcl/case_spec.h>
#include <tcl/runner.h>

namespace
{
using namespace uipc::tensor_core_lab;

template <typename PreparedT, typename DownloadFn, typename MetricFn, typename ExecFn>
void run_spd_bench(benchmark::State& state,
                   PreparedT&        prepared,
                   ExecFn            execute,
                   DownloadFn        download,
                   MetricFn          compute_metrics)
{
    const auto warm = execute(true);
    if(warm.status != RunStatus::Ok)
    {
        state.SkipWithError(warm.message.c_str());
        return;
    }

    const auto metrics = compute_metrics(download(prepared));
    state.counters["rel_error"]      = metrics.rel_fro;
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
        const auto out = execute(true);
        if(out.status != RunStatus::Ok)
        {
            state.SkipWithError(out.message.c_str());
            break;
        }
        state.SetIterationTime(out.elapsed_ms / 1000.0);
    }
}

void bench_abd12(benchmark::State& state, Mode mode, OpKind op)
{
    const int    batch    = static_cast<int>(state.range(0));
    const int    cond_exp = static_cast<int>(state.range(1));
    const double cond     = std::pow(10.0, static_cast<double>(cond_exp));
    const auto   data     = make_abd12_case("bench_abd12", batch, 43, cond);

    BackendContext context(mode);
    if(!context.is_supported())
    {
        state.SkipWithError(context.unsupported_reason().c_str());
        return;
    }

    if(mode == Mode::Fp64RefNoTc)
    {
        auto prepared = prepare_f64_case(data);
        if(op == OpKind::Abd12Factorize)
        {
            run_spd_bench(
                state,
                prepared,
                [&](bool measure) { return execute_spd_factorize_case(context, prepared, measure); },
                [](const auto& p) { return download_factor(p); },
                [&](const std::vector<double>& output)
                {
                    return factorization_reconstruction_metrics(data.matrix,
                                                                output,
                                                                data.spec.shape.logical_rows,
                                                                data.spec.shape.physical_rows,
                                                                data.spec.batch_count);
                });
        }
        else if(op == OpKind::Abd12Inverse)
        {
            run_spd_bench(
                state,
                prepared,
                [&](bool measure) { return execute_spd_inverse_case(context, prepared, measure); },
                [](const auto& p) { return download_inverse(p); },
                [&](const std::vector<double>& output)
                {
                    return compare_square_batches(data.inverse_reference,
                                                  output,
                                                  data.spec.shape.logical_rows,
                                                  data.spec.shape.physical_rows,
                                                  data.spec.batch_count);
                });
        }
        else
        {
            run_spd_bench(
                state,
                prepared,
                [&](bool measure) { return execute_spd_solve_case(context, prepared, measure); },
                [](const auto& p) { return download_solution(p); },
                [&](const std::vector<double>& output)
                {
                    return compare_vector_batches(data.solution_reference,
                                                  output,
                                                  data.spec.shape.logical_rows,
                                                  data.spec.shape.physical_rows,
                                                  data.spec.batch_count);
                });
        }
        return;
    }

    auto prepared = prepare_f32_case(data);
    if(op == OpKind::Abd12Factorize)
    {
        run_spd_bench(
            state,
            prepared,
            [&](bool measure) { return execute_spd_factorize_case(context, prepared, measure); },
            [](const auto& p) { return download_factor(p); },
            [&](const std::vector<double>& output)
            {
                return factorization_reconstruction_metrics(data.matrix,
                                                            output,
                                                            data.spec.shape.logical_rows,
                                                            data.spec.shape.physical_rows,
                                                            data.spec.batch_count);
            });
    }
    else if(op == OpKind::Abd12Inverse)
    {
        run_spd_bench(
            state,
            prepared,
            [&](bool measure) { return execute_spd_inverse_case(context, prepared, measure); },
            [](const auto& p) { return download_inverse(p); },
            [&](const std::vector<double>& output)
            {
                return compare_square_batches(data.inverse_reference,
                                              output,
                                              data.spec.shape.logical_rows,
                                              data.spec.shape.physical_rows,
                                              data.spec.batch_count);
            });
    }
    else
    {
        run_spd_bench(
            state,
            prepared,
            [&](bool measure) { return execute_spd_solve_case(context, prepared, measure); },
            [](const auto& p) { return download_solution(p); },
            [&](const std::vector<double>& output)
            {
                return compare_vector_batches(data.solution_reference,
                                              output,
                                              data.spec.shape.logical_rows,
                                              data.spec.shape.physical_rows,
                                              data.spec.batch_count);
            });
    }
}

void BM_Abd12FactorFp64(benchmark::State& state)
{
    bench_abd12(state, Mode::Fp64RefNoTc, OpKind::Abd12Factorize);
}
void BM_Abd12FactorFp32(benchmark::State& state)
{
    bench_abd12(state, Mode::Fp32NoTc, OpKind::Abd12Factorize);
}
void BM_Abd12FactorTc32(benchmark::State& state)
{
    bench_abd12(state, Mode::Tc32Tf32, OpKind::Abd12Factorize);
}
void BM_Abd12InverseFp64(benchmark::State& state)
{
    bench_abd12(state, Mode::Fp64RefNoTc, OpKind::Abd12Inverse);
}
void BM_Abd12InverseFp32(benchmark::State& state)
{
    bench_abd12(state, Mode::Fp32NoTc, OpKind::Abd12Inverse);
}
void BM_Abd12InverseTc32(benchmark::State& state)
{
    bench_abd12(state, Mode::Tc32Tf32, OpKind::Abd12Inverse);
}
void BM_Abd12SolveFp64(benchmark::State& state)
{
    bench_abd12(state, Mode::Fp64RefNoTc, OpKind::Abd12Solve);
}
void BM_Abd12SolveFp32(benchmark::State& state)
{
    bench_abd12(state, Mode::Fp32NoTc, OpKind::Abd12Solve);
}
void BM_Abd12SolveTc32(benchmark::State& state)
{
    bench_abd12(state, Mode::Tc32Tf32, OpKind::Abd12Solve);
}
}  // namespace

BENCHMARK(BM_Abd12FactorFp64)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
BENCHMARK(BM_Abd12FactorFp32)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
BENCHMARK(BM_Abd12FactorTc32)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
BENCHMARK(BM_Abd12InverseFp64)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
BENCHMARK(BM_Abd12InverseFp32)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
BENCHMARK(BM_Abd12InverseTc32)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
BENCHMARK(BM_Abd12SolveFp64)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
BENCHMARK(BM_Abd12SolveFp32)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
BENCHMARK(BM_Abd12SolveTc32)->Args({4096, 2})->Args({32768, 4})->UseManualTime();
