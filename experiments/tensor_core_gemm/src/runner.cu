#include <tcg/runner.h>

#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

#include "cuda_check.h"
#include "event_timer.h"

namespace uipc::tensor_core_gemm
{
namespace
{
using ::uipc::tensor_core_lab::EventTimer;
using ::uipc::tensor_core_lab::ImplPath;
using ::uipc::tensor_core_lab::TensorCoreVerification;

template <typename To, typename From>
std::vector<To> cast_copy(const std::vector<From>& src)
{
    std::vector<To> out(src.size());
    std::transform(src.begin(), src.end(), out.begin(), [](From v)
                   { return static_cast<To>(v); });
    return out;
}

size_t matrix_elements(int rows, int cols, int batch_count)
{
    return static_cast<size_t>(rows) * static_cast<size_t>(cols)
           * static_cast<size_t>(batch_count);
}
}  // namespace

PreparedGemmCaseF32 prepare_f32_case(const GemmCaseData& data)
{
    PreparedGemmCaseF32 out;
    out.spec = data.spec;

    const auto a_count = matrix_elements(data.spec.physical_m,
                                         data.spec.physical_k,
                                         data.spec.batch_count);
    const auto b_count = matrix_elements(data.spec.physical_k,
                                         data.spec.physical_n,
                                         data.spec.batch_count);
    const auto c_count = matrix_elements(data.spec.physical_m,
                                         data.spec.physical_n,
                                         data.spec.batch_count);

    out.a.resize(a_count);
    out.b.resize(b_count);
    out.c.resize(c_count);
    out.a.copy_from_host(cast_copy<float>(data.a_fp64));
    out.b.copy_from_host(cast_copy<float>(data.b_fp64));
    return out;
}

PreparedGemmCaseF64 prepare_f64_case(const GemmCaseData& data)
{
    PreparedGemmCaseF64 out;
    out.spec = data.spec;

    const auto a_count = matrix_elements(data.spec.physical_m,
                                         data.spec.physical_k,
                                         data.spec.batch_count);
    const auto b_count = matrix_elements(data.spec.physical_k,
                                         data.spec.physical_n,
                                         data.spec.batch_count);
    const auto c_count = matrix_elements(data.spec.physical_m,
                                         data.spec.physical_n,
                                         data.spec.batch_count);

    out.a.resize(a_count);
    out.b.resize(b_count);
    out.c.resize(c_count);
    out.a.copy_from_host(std::span<const double>(data.a_fp64.data(), data.a_fp64.size()));
    out.b.copy_from_host(std::span<const double>(data.b_fp64.data(), data.b_fp64.size()));
    return out;
}

std::vector<double> download_output(const PreparedGemmCaseF32& data)
{
    std::vector<float> host(data.c.size());
    data.c.copy_to_host(std::span<float>(host.data(), host.size()));
    return cast_copy<double>(host);
}

std::vector<double> download_output(const PreparedGemmCaseF64& data)
{
    std::vector<double> host(data.c.size());
    data.c.copy_to_host(std::span<double>(host.data(), host.size()));
    return host;
}

RunOutcome execute_gemm_case(BackendContext& context,
                             PreparedGemmCaseF32& data,
                             bool measure_time)
{
    try
    {
        const int       m        = data.spec.physical_m;
        const int       n        = data.spec.physical_n;
        const int       k        = data.spec.physical_k;
        const long long stride_a = static_cast<long long>(m) * static_cast<long long>(k);
        const long long stride_b = static_cast<long long>(k) * static_cast<long long>(n);
        const long long stride_c = static_cast<long long>(m) * static_cast<long long>(n);
        const float     alpha    = 1.0f;
        const float     beta     = 0.0f;

        context.reset_trace();
        if(context.mode() == Mode::Tc32Tf32)
            context.set_trace(ImplPath::TcBlas,
                              true,
                              TensorCoreVerification::BlockedByPermissions);
        else
            context.set_trace(
                ImplPath::Baseline, false, TensorCoreVerification::No);

        TCL_CUDA_CHECK(cudaMemset(data.c.data(), 0, data.c.size() * sizeof(float)));

        EventTimer timer;
        if(measure_time)
            timer.start();

        if(context.mode() == Mode::Tc32Tf32)
        {
            TCL_CUBLAS_CHECK(context.tc_blas_matmul_strided_batched(CUBLAS_OP_N,
                                                                    CUBLAS_OP_N,
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    &alpha,
                                                                    data.a.data(),
                                                                    m,
                                                                    stride_a,
                                                                    data.b.data(),
                                                                    k,
                                                                    stride_b,
                                                                    &beta,
                                                                    data.c.data(),
                                                                    m,
                                                                    stride_c,
                                                                    data.c.data(),
                                                                    m,
                                                                    stride_c,
                                                                    data.spec.batch_count));
        }
        else
        {
            TCL_CUBLAS_CHECK(context.gemm_strided_batched(CUBLAS_OP_N,
                                                          CUBLAS_OP_N,
                                                          m,
                                                          n,
                                                          k,
                                                          &alpha,
                                                          data.a.data(),
                                                          m,
                                                          stride_a,
                                                          data.b.data(),
                                                          k,
                                                          stride_b,
                                                          &beta,
                                                          data.c.data(),
                                                          m,
                                                          stride_c,
                                                          data.spec.batch_count));
        }

        double elapsed_ms = 0.0;
        if(measure_time)
            elapsed_ms = timer.stop_elapsed_ms();
        else
            TCL_CUDA_CHECK(cudaDeviceSynchronize());
        return {RunStatus::Ok, elapsed_ms, {}, {}, context.trace()};
    }
    catch(const std::exception& e)
    {
        return {RunStatus::Failure, 0.0, {}, e.what(), context.trace()};
    }
}

RunOutcome execute_gemm_case(BackendContext& context,
                             PreparedGemmCaseF64& data,
                             bool measure_time)
{
    try
    {
        const int       m        = data.spec.physical_m;
        const int       n        = data.spec.physical_n;
        const int       k        = data.spec.physical_k;
        const long long stride_a = static_cast<long long>(m) * static_cast<long long>(k);
        const long long stride_b = static_cast<long long>(k) * static_cast<long long>(n);
        const long long stride_c = static_cast<long long>(m) * static_cast<long long>(n);
        const double    alpha    = 1.0;
        const double    beta     = 0.0;

        context.reset_trace();
        context.set_trace(
            ImplPath::Baseline, false, TensorCoreVerification::No);
        TCL_CUDA_CHECK(cudaMemset(data.c.data(), 0, data.c.size() * sizeof(double)));

        EventTimer timer;
        if(measure_time)
            timer.start();

        TCL_CUBLAS_CHECK(context.gemm_strided_batched(CUBLAS_OP_N,
                                                      CUBLAS_OP_N,
                                                      m,
                                                      n,
                                                      k,
                                                      &alpha,
                                                      data.a.data(),
                                                      m,
                                                      stride_a,
                                                      data.b.data(),
                                                      k,
                                                      stride_b,
                                                      &beta,
                                                      data.c.data(),
                                                      m,
                                                      stride_c,
                                                      data.spec.batch_count));

        double elapsed_ms = 0.0;
        if(measure_time)
            elapsed_ms = timer.stop_elapsed_ms();
        else
            TCL_CUDA_CHECK(cudaDeviceSynchronize());
        return {RunStatus::Ok, elapsed_ms, {}, {}, context.trace()};
    }
    catch(const std::exception& e)
    {
        return {RunStatus::Failure, 0.0, {}, e.what(), context.trace()};
    }
}

void ensure_fp64_reference(GemmCaseData& data)
{
    if(!data.reference_fp64.empty())
        return;

    BackendContext context(Mode::Fp64RefNoTc);
    auto           prepared = prepare_f64_case(data);
    const auto     out      = execute_gemm_case(context, prepared, false);
    if(out.status != RunStatus::Ok)
        throw std::runtime_error("failed to build fp64 reference: " + out.message);
    data.reference_fp64 = download_output(prepared);
}

RunOutcome run_gemm_case(Mode mode, GemmCaseData& data, bool measure_time)
{
    try
    {
        ensure_fp64_reference(data);

        BackendContext context(mode);
        if(!context.is_supported())
            return {RunStatus::Unsupported, 0.0, {}, context.unsupported_reason()};

        if(mode == Mode::Fp64RefNoTc)
        {
            auto prepared = prepare_f64_case(data);
            auto out      = execute_gemm_case(context, prepared, measure_time);
            if(out.status == RunStatus::Ok)
                out.metrics = ::uipc::tensor_core_lab::compare_matrix_batches(
                    data.reference_fp64,
                    download_output(prepared),
                    data.spec.m,
                    data.spec.n,
                    data.spec.physical_m,
                    data.spec.physical_n,
                    data.spec.batch_count);
            return out;
        }

        auto prepared = prepare_f32_case(data);
        auto out      = execute_gemm_case(context, prepared, measure_time);
        if(out.status == RunStatus::Ok)
            out.metrics = ::uipc::tensor_core_lab::compare_matrix_batches(
                data.reference_fp64,
                download_output(prepared),
                data.spec.m,
                data.spec.n,
                data.spec.physical_m,
                data.spec.physical_n,
                data.spec.batch_count);
        return out;
    }
    catch(const std::exception& e)
    {
        return {RunStatus::Failure, 0.0, {}, e.what()};
    }
}
}  // namespace uipc::tensor_core_gemm
