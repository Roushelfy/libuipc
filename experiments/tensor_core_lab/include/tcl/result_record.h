#pragma once
#include <string>
#include <tcl/backend_api.h>
#include <tcl/metrics.h>
#include <tcl/mode.h>
#include <tcl/op_kind.h>

namespace uipc::tensor_core_lab
{
struct ResultRecord
{
    OpKind          op                 = OpKind::Fem12Assemble;
    Mode            mode               = Mode::Fp64RefNoTc;
    std::string     case_name;
    int             batch_count        = 0;
    double          condition_scale    = 0.0;
    double          elapsed_ms         = 0.0;
    MatrixMetrics metrics;
    ExecutionTrace trace;
};
}  // namespace uipc::tensor_core_lab
