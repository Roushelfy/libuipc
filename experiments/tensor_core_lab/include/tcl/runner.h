#pragma once
#include <string>
#include <vector>

#include <tcl/backend_api.h>
#include <tcl/case_spec.h>
#include <tcl/metrics.h>

namespace uipc::tensor_core_lab
{
enum class RunStatus
{
    Ok,
    Unsupported,
    InvalidArgument,
    Failure
};

struct RunOutcome
{
    RunStatus     status     = RunStatus::Ok;
    double        elapsed_ms = 0.0;
    MatrixMetrics metrics;
    std::string   message;
    ExecutionTrace trace;
};

struct PreparedCaseF32
{
    CaseSpec          spec;
    bool              has_secondary = false;
    DeviceBuffer<float> left;
    DeviceBuffer<float> middle;
    DeviceBuffer<float> right;
    DeviceBuffer<float> aux;
    DeviceBuffer<float> temp0;
    DeviceBuffer<float> temp1;
    DeviceBuffer<float> output;
};

struct PreparedCaseF64
{
    CaseSpec           spec;
    bool               has_secondary = false;
    DeviceBuffer<double> left;
    DeviceBuffer<double> middle;
    DeviceBuffer<double> right;
    DeviceBuffer<double> aux;
    DeviceBuffer<double> temp0;
    DeviceBuffer<double> temp1;
    DeviceBuffer<double> output;
};

struct PreparedSpdCaseF32
{
    CaseSpec            spec;
    DeviceBuffer<float> source_matrix;
    DeviceBuffer<float> factor;
    DeviceBuffer<float> rhs;
    DeviceBuffer<float> inv_l;
    DeviceBuffer<float> inverse;
    DeviceBuffer<float> temp_vector;
    DeviceBuffer<float> solution;
};

struct PreparedSpdCaseF64
{
    CaseSpec             spec;
    DeviceBuffer<double> source_matrix;
    DeviceBuffer<double> factor;
    DeviceBuffer<double> rhs;
    DeviceBuffer<double> inv_l;
    DeviceBuffer<double> inverse;
    DeviceBuffer<double> temp_vector;
    DeviceBuffer<double> solution;
};

PreparedCaseF32 prepare_f32_case(const ContractionCaseData& data);
PreparedCaseF64 prepare_f64_case(const ContractionCaseData& data);

PreparedSpdCaseF32 prepare_f32_case(const SpdCaseData& data);
PreparedSpdCaseF64 prepare_f64_case(const SpdCaseData& data);

std::vector<double> download_output(const PreparedCaseF32& data);
std::vector<double> download_output(const PreparedCaseF64& data);
std::vector<double> download_factor(const PreparedSpdCaseF32& data);
std::vector<double> download_factor(const PreparedSpdCaseF64& data);
std::vector<double> download_inverse(const PreparedSpdCaseF32& data);
std::vector<double> download_inverse(const PreparedSpdCaseF64& data);
std::vector<double> download_solution(const PreparedSpdCaseF32& data);
std::vector<double> download_solution(const PreparedSpdCaseF64& data);

RunOutcome execute_fem12_case(BackendContext& context,
                              PreparedCaseF32& data,
                              bool             measure_time);
RunOutcome execute_fem12_case(BackendContext& context,
                              PreparedCaseF64& data,
                              bool             measure_time);

RunOutcome execute_joint24_case(BackendContext& context,
                                PreparedCaseF32& data,
                                bool             measure_time);
RunOutcome execute_joint24_case(BackendContext& context,
                                PreparedCaseF64& data,
                                bool             measure_time);

RunOutcome execute_spd_factorize_case(BackendContext&   context,
                                      PreparedSpdCaseF32& data,
                                      bool               measure_time);
RunOutcome execute_spd_factorize_case(BackendContext&   context,
                                      PreparedSpdCaseF64& data,
                                      bool               measure_time);

RunOutcome execute_spd_inverse_case(BackendContext&   context,
                                    PreparedSpdCaseF32& data,
                                    bool               measure_time);
RunOutcome execute_spd_inverse_case(BackendContext&   context,
                                    PreparedSpdCaseF64& data,
                                    bool               measure_time);

RunOutcome execute_spd_solve_case(BackendContext&   context,
                                  PreparedSpdCaseF32& data,
                                  bool               measure_time);
RunOutcome execute_spd_solve_case(BackendContext&   context,
                                  PreparedSpdCaseF64& data,
                                  bool               measure_time);

RunOutcome run_fem12_case(Mode mode,
                          const ContractionCaseData& data,
                          bool                       measure_time = false);
RunOutcome run_joint24_case(Mode mode,
                            const ContractionCaseData& data,
                            bool                       measure_time = false);
RunOutcome run_spd_case(Mode               mode,
                        OpKind             op_kind,
                        const SpdCaseData& data,
                        bool               measure_time = false);
}  // namespace uipc::tensor_core_lab
