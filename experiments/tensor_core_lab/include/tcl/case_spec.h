#pragma once
#include <string>
#include <vector>
#include <tcl/op_kind.h>
#include <tcl/tensor_shape.h>

namespace uipc::tensor_core_lab
{
struct CaseSpec
{
    std::string name;
    OpKind      op_kind         = OpKind::Fem12Assemble;
    int         batch_count     = 0;
    int         seed            = 0;
    double      condition_scale = 1.0;
    TensorShape shape;
};

struct ContractionCaseData
{
    CaseSpec            spec;
    std::vector<double> left;
    std::vector<double> middle;
    std::vector<double> right;
    std::vector<double> aux;
    std::vector<double> reference;
};

struct SpdCaseData
{
    CaseSpec            spec;
    std::vector<double> matrix;
    std::vector<double> rhs;
    std::vector<double> inverse_reference;
    std::vector<double> solution_reference;
};

ContractionCaseData make_fem12_case(const std::string& name,
                                    int                batch_count,
                                    int                seed,
                                    double             condition_scale);

ContractionCaseData make_joint24_case(const std::string& name,
                                      int                batch_count,
                                      int                seed,
                                      double             condition_scale);

ContractionCaseData make_abd12_assemble_case(const std::string& name,
                                             int                batch_count,
                                             int                seed,
                                             double             condition_scale);

SpdCaseData make_mas48_case(const std::string& name,
                            int                batch_count,
                            int                seed,
                            double             condition_scale);

SpdCaseData make_abd12_case(const std::string& name,
                            int                batch_count,
                            int                seed,
                            double             condition_scale);
}  // namespace uipc::tensor_core_lab
