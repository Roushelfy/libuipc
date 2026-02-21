#include <benchmark/benchmark.h>
#include <cstdint>

static std::uint64_t Fibonacci(std::uint64_t number)
{
    return number < 2 ? 1 : Fibonacci(number - 1) + Fibonacci(number - 2);
}

static void BM_Fibonacci(benchmark::State& state)
{
    const auto n = static_cast<std::uint64_t>(state.range(0));
    for(auto _ : state)
    {
        benchmark::DoNotOptimize(Fibonacci(n));
    }
}

BENCHMARK(BM_Fibonacci)->Arg(20)->Arg(25)->Arg(30)->Arg(35);
