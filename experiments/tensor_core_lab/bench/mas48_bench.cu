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

void bench_mas48(benchmark::State& state, Mode mode, OpKind op)
{
    const int    batch    = static_cast<int>(state.range(0));
    const int    cond_exp = static_cast<int>(state.range(1));
    const double cond     = std::pow(10.0, static_cast<double>(cond_exp));
    const auto   data     = make_mas48_case("bench_mas48", batch, 41, cond);

    BackendContext context(mode);
    if(!context.is_supported())
    {
        state.SkipWithError(context.unsupported_reason().c_str());
        return;
    }

    if(mode == Mode::Fp64RefNoTc)
    {
        auto prepared = prepare_f64_case(data);
        if(op == OpKind::Mas48Factorize)
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
        else if(op == OpKind::Mas48Inverse)
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
    if(op == OpKind::Mas48Factorize)
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
    else if(op == OpKind::Mas48Inverse)
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

void BM_Mas48FactorFp64(benchmark::State& state)
{
    bench_mas48(state, Mode::Fp64RefNoTc, OpKind::Mas48Factorize);
}
void BM_Mas48FactorFp32(benchmark::State& state)
{
    bench_mas48(state, Mode::Fp32NoTc, OpKind::Mas48Factorize);
}
void BM_Mas48FactorTc32(benchmark::State& state)
{
    bench_mas48(state, Mode::Tc32Tf32, OpKind::Mas48Factorize);
}
void BM_Mas48InverseFp64(benchmark::State& state)
{
    bench_mas48(state, Mode::Fp64RefNoTc, OpKind::Mas48Inverse);
}
void BM_Mas48InverseFp32(benchmark::State& state)
{
    bench_mas48(state, Mode::Fp32NoTc, OpKind::Mas48Inverse);
}
void BM_Mas48InverseTc32(benchmark::State& state)
{
    bench_mas48(state, Mode::Tc32Tf32, OpKind::Mas48Inverse);
}
void BM_Mas48SolveFp64(benchmark::State& state)
{
    bench_mas48(state, Mode::Fp64RefNoTc, OpKind::Mas48Solve);
}
void BM_Mas48SolveFp32(benchmark::State& state)
{
    bench_mas48(state, Mode::Fp32NoTc, OpKind::Mas48Solve);
}
void BM_Mas48SolveTc32(benchmark::State& state)
{
    bench_mas48(state, Mode::Tc32Tf32, OpKind::Mas48Solve);
}
}  // namespace

BENCHMARK(BM_Mas48FactorFp64)->Args({256, 2})->Args({2048, 4})->UseManualTime();
BENCHMARK(BM_Mas48FactorFp32)->Args({256, 2})->Args({2048, 4})->UseManualTime();
BENCHMARK(BM_Mas48FactorTc32)->Args({256, 2})->Args({2048, 4})->UseManualTime();
BENCHMARK(BM_Mas48InverseFp64)->Args({256, 2})->Args({2048, 4})->UseManualTime();
BENCHMARK(BM_Mas48InverseFp32)->Args({256, 2})->Args({2048, 4})->UseManualTime();
BENCHMARK(BM_Mas48InverseTc32)->Args({256, 2})->Args({2048, 4})->UseManualTime();
BENCHMARK(BM_Mas48SolveFp64)->Args({256, 2})->Args({2048, 4})->UseManualTime();
BENCHMARK(BM_Mas48SolveFp32)->Args({256, 2})->Args({2048, 4})->UseManualTime();
BENCHMARK(BM_Mas48SolveTc32)->Args({256, 2})->Args({2048, 4})->UseManualTime();
