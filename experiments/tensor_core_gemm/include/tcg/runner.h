#pragma once

#include <string>
#include <vector>

#include <tcl/backend_api.h>
#include <tcl/metrics.h>
#include <tcl/mode.h>

#include <tcg/gemm_case.h>

namespace uipc::tensor_core_gemm
{
using Mode = ::uipc::tensor_core_lab::Mode;
using RunMetrics = ::uipc::tensor_core_lab::MatrixMetrics;
using ExecutionTrace = ::uipc::tensor_core_lab::ExecutionTrace;
template <typename T>
using DeviceBuffer = ::uipc::tensor_core_lab::DeviceBuffer<T>;
using BackendContext = ::uipc::tensor_core_lab::BackendContext;

enum class RunStatus
{
    Ok,
    Unsupported,
    InvalidArgument,
    Failure
};

struct RunOutcome
{
    RunStatus      status = RunStatus::Ok;
    double         elapsed_ms = 0.0;
    RunMetrics     metrics;
    std::string    message;
    ExecutionTrace trace;
};

struct PreparedGemmCaseF32
{
    GemmCaseSpec     spec;
    DeviceBuffer<float> a;
    DeviceBuffer<float> b;
    DeviceBuffer<float> c;
};

struct PreparedGemmCaseF64
{
    GemmCaseSpec      spec;
    DeviceBuffer<double> a;
    DeviceBuffer<double> b;
    DeviceBuffer<double> c;
};

PreparedGemmCaseF32 prepare_f32_case(const GemmCaseData& data);
PreparedGemmCaseF64 prepare_f64_case(const GemmCaseData& data);

std::vector<double> download_output(const PreparedGemmCaseF32& data);
std::vector<double> download_output(const PreparedGemmCaseF64& data);

RunOutcome execute_gemm_case(BackendContext& context,
                             PreparedGemmCaseF32& data,
                             bool measure_time);
RunOutcome execute_gemm_case(BackendContext& context,
                             PreparedGemmCaseF64& data,
                             bool measure_time);

void ensure_fp64_reference(GemmCaseData& data);

RunOutcome run_gemm_case(Mode mode, GemmCaseData& data, bool measure_time = false);
}  // namespace uipc::tensor_core_gemm
