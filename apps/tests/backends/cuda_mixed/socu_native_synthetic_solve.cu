#include <app/app.h>
#include <mixed_precision/policy.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#ifndef UIPC_WITH_SOCU_NATIVE
#define UIPC_WITH_SOCU_NATIVE 0
#endif

#if UIPC_WITH_SOCU_NATIVE
#include <socu_native/problem_generator.h>
#include <socu_native/solver.h>
#endif

namespace
{
#if UIPC_WITH_SOCU_NATIVE
using namespace uipc::backend::cuda_mixed;

#ifndef SOCU_NATIVE_DEFAULT_MATHDX_MANIFEST_PATH
#define SOCU_NATIVE_DEFAULT_MATHDX_MANIFEST_PATH ""
#endif

#ifndef SOCU_NATIVE_SOURCE_DIR
#define SOCU_NATIVE_SOURCE_DIR ""
#endif

struct SolverPlanDeleter
{
    void operator()(socu_native::SolverPlan* plan) const
    {
        socu_native::destroy_solver_plan(plan);
    }
};

struct StreamDeleter
{
    void operator()(cudaStream_t stream) const
    {
        if(stream != nullptr)
            cudaStreamDestroy(stream);
    }
};

template <typename T>
struct DeviceProblem
{
    socu_native::ProblemShape shape{};
    std::size_t               diag_count = 0;
    std::size_t               off_count  = 0;
    std::size_t               rhs_count  = 0;
    T*                        diag       = nullptr;
    T*                        off_diag   = nullptr;
    T*                        rhs        = nullptr;

    DeviceProblem() = default;
    DeviceProblem(const DeviceProblem&) = delete;
    DeviceProblem& operator=(const DeviceProblem&) = delete;

    DeviceProblem(DeviceProblem&& other) noexcept
        : shape(other.shape)
        , diag_count(other.diag_count)
        , off_count(other.off_count)
        , rhs_count(other.rhs_count)
        , diag(other.diag)
        , off_diag(other.off_diag)
        , rhs(other.rhs)
    {
        other.diag     = nullptr;
        other.off_diag = nullptr;
        other.rhs      = nullptr;
    }

    DeviceProblem& operator=(DeviceProblem&& other) noexcept
    {
        if(this == &other)
            return *this;
        cleanup();
        shape      = other.shape;
        diag_count = other.diag_count;
        off_count  = other.off_count;
        rhs_count  = other.rhs_count;
        diag       = other.diag;
        off_diag   = other.off_diag;
        rhs        = other.rhs;
        other.diag = nullptr;
        other.off_diag = nullptr;
        other.rhs = nullptr;
        return *this;
    }

    ~DeviceProblem()
    {
        cleanup();
    }

    void cleanup()
    {
        if(diag != nullptr)
            cudaFree(diag);
        if(off_diag != nullptr)
            cudaFree(off_diag);
        if(rhs != nullptr)
            cudaFree(rhs);
        diag     = nullptr;
        off_diag = nullptr;
        rhs      = nullptr;
    }
};

template <typename T>
DeviceProblem<T> make_device_problem(const socu_native::HostProblem<T>& host)
{
    DeviceProblem<T> device;
    device.shape      = host.shape;
    device.diag_count = host.diag.size();
    device.off_count  = host.off_diag.size();
    device.rhs_count  = host.rhs.size();

    SOCU_NATIVE_CHECK_CUDA(cudaMalloc(&device.diag, device.diag_count * sizeof(T)));
    SOCU_NATIVE_CHECK_CUDA(cudaMalloc(&device.off_diag, device.off_count * sizeof(T)));
    SOCU_NATIVE_CHECK_CUDA(cudaMalloc(&device.rhs, device.rhs_count * sizeof(T)));
    return device;
}

template <typename T>
void upload_problem(const socu_native::HostProblem<T>& host,
                    DeviceProblem<T>&                 device,
                    cudaStream_t                      stream)
{
    REQUIRE(device.diag_count == host.diag.size());
    REQUIRE(device.off_count == host.off_diag.size());
    REQUIRE(device.rhs_count == host.rhs.size());

    SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(device.diag,
                                           host.diag.data(),
                                           device.diag_count * sizeof(T),
                                           cudaMemcpyHostToDevice,
                                           stream));
    SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(device.off_diag,
                                           host.off_diag.data(),
                                           device.off_count * sizeof(T),
                                           cudaMemcpyHostToDevice,
                                           stream));
    SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(device.rhs,
                                           host.rhs.data(),
                                           device.rhs_count * sizeof(T),
                                           cudaMemcpyHostToDevice,
                                           stream));
}

template <typename T>
std::vector<T> download_rhs(const DeviceProblem<T>& device, cudaStream_t stream)
{
    std::vector<T> host(device.rhs_count);
    SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(host.data(),
                                           device.rhs,
                                           device.rhs_count * sizeof(T),
                                           cudaMemcpyDeviceToHost,
                                           stream));
    SOCU_NATIVE_CHECK_CUDA(cudaStreamSynchronize(stream));
    return host;
}

template <typename T>
double norm2(const std::vector<T>& values)
{
    double sum = 0.0;
    for(const T value : values)
    {
        const double v = static_cast<double>(value);
        sum += v * v;
    }
    return std::sqrt(sum);
}

template <typename T>
void require_descent_direction(const std::vector<T>& rhs,
                               const std::vector<T>& solution,
                               int                   block_size)
{
    REQUIRE(rhs.size() == solution.size());
    const double rhs_dot_solution =
        std::inner_product(rhs.begin(), rhs.end(), solution.begin(), 0.0);
    const double g_dot_p = -rhs_dot_solution;
    const double g_norm  = norm2(rhs);
    const double p_norm  = norm2(solution);
    INFO("block_size=" << block_size << " g_dot_p=" << g_dot_p
                       << " g_norm=" << g_norm << " p_norm=" << p_norm);
    REQUIRE(std::isfinite(g_dot_p));
    REQUIRE(std::isfinite(g_norm));
    REQUIRE(std::isfinite(p_norm));
    REQUIRE(g_norm > 0.0);
    REQUIRE(p_norm > 0.0);
    REQUIRE(g_dot_p < -1e-12 * g_norm * p_norm);
}

template <typename T>
void run_synthetic_solve_smoke(int block_size)
{
    const socu_native::ProblemShape shape{64, block_size, 1};
    const auto layout = socu_native::describe_problem_layout(shape);
    REQUIRE(layout.diag_element_count
            == static_cast<std::size_t>(shape.horizon) * shape.n * shape.n);
    REQUIRE(layout.rhs_element_count
            == static_cast<std::size_t>(shape.horizon) * shape.n * shape.nrhs);

    socu_native::SolverPlanOptions plan_options;
    plan_options.backend      = socu_native::SolverBackend::NativePerf;
    plan_options.perf_backend = socu_native::PerfBackend::MathDx;
    plan_options.math_mode    = socu_native::MathMode::Auto;
    plan_options.graph_mode   = socu_native::GraphMode::Off;

    const auto capability =
        socu_native::query_solver_capability<T>(shape,
                                                socu_native::SolverOperation::FactorAndSolve,
                                                plan_options);
    INFO("block_size=" << block_size << " capability.reason=" << capability.reason);
    REQUIRE(capability.supported);
    REQUIRE(capability.resolved_backend == socu_native::SolverBackend::NativePerf);
    REQUIRE(capability.resolved_perf_backend == socu_native::PerfBackend::MathDx);
    REQUIRE(capability.resolved_graph_mode == socu_native::GraphMode::Off);

    auto base = socu_native::generate_random_spd_block_tridiag<T>(
        shape,
        static_cast<std::uint64_t>(12000 + block_size));

    DeviceProblem<T> device = make_device_problem(base);
    cudaStream_t raw_stream = nullptr;
    SOCU_NATIVE_CHECK_CUDA(cudaStreamCreateWithFlags(&raw_stream, cudaStreamNonBlocking));
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter> stream(raw_stream);
    const socu_native::LaunchOptions launch_options{raw_stream};

    std::unique_ptr<socu_native::SolverPlan, SolverPlanDeleter> plan(
        socu_native::create_solver_plan<T>(shape, plan_options));

    for(int repeat = 0; repeat < 3; ++repeat)
    {
        upload_problem(base, device, raw_stream);
        socu_native::factor_and_solve_inplace_async(
            plan.get(),
            device.diag,
            device.off_diag,
            device.rhs,
            launch_options);
        SOCU_NATIVE_CHECK_CUDA(cudaStreamSynchronize(raw_stream));
        const auto solution = download_rhs(device, raw_stream);
        const double residual =
            socu_native::residual_norm(base.diag, base.off_diag, base.rhs, solution, shape);
        const double relative_residual = residual / std::max(1.0, norm2(base.rhs));
        const double residual_limit = std::is_same_v<T, float> ? 5e-3 : 5e-8;
        INFO("block_size=" << block_size << " repeat=" << repeat
                           << " residual=" << residual
                           << " relative_residual=" << relative_residual);
        REQUIRE(std::isfinite(residual));
        REQUIRE(relative_residual < residual_limit);
        require_descent_direction(base.rhs, solution, block_size);
    }

    SOCU_NATIVE_CHECK_CUDA(cudaGetLastError());
}

std::filesystem::path default_mathdx_manifest_path()
{
    return std::filesystem::path{SOCU_NATIVE_DEFAULT_MATHDX_MANIFEST_PATH};
}

#endif
}  // namespace

TEST_CASE("cuda_mixed_socu_native_synthetic_solve_smoke",
          "[cuda_mixed][contract][socu_native][m6]")
{
#if !UIPC_WITH_SOCU_NATIVE
    SKIP("socu_native is not enabled in this build");
#else
    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        SKIP("no CUDA device is available for the socu_native synthetic solve smoke");
    }

    using StoreScalar = ActivePolicy::StoreScalar;
    using SolveScalar = ActivePolicy::SolveScalar;
    if constexpr(!std::is_same_v<StoreScalar, SolveScalar>)
    {
        SKIP("M6 synthetic socu smoke requires StoreScalar == SolveScalar");
    }
    else
    {
        const auto manifest_path = default_mathdx_manifest_path();
        if(manifest_path.empty() || !std::filesystem::is_regular_file(manifest_path))
        {
            SKIP("MathDx manifest is missing; expected " << manifest_path.string());
        }

        run_synthetic_solve_smoke<StoreScalar>(32);
        run_synthetic_solve_smoke<StoreScalar>(64);
    }
#endif
}
