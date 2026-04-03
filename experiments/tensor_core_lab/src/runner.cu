#include <tcl/runner.h>

#include <algorithm>
#include <vector>

namespace uipc::tensor_core_lab
{
namespace
{
template <typename To, typename From>
std::vector<To> cast_copy(const std::vector<From>& src)
{
    std::vector<To> out(src.size());
    std::transform(src.begin(), src.end(), out.begin(), [](From v)
                   { return static_cast<To>(v); });
    return out;
}
}  // namespace

PreparedCaseF32 prepare_f32_case(const ContractionCaseData& data)
{
    PreparedCaseF32 out;
    out.spec          = data.spec;
    out.has_secondary = !data.right.empty() && !data.aux.empty();

    const size_t matrix_elements = static_cast<size_t>(data.spec.shape.physical_rows)
                                   * data.spec.shape.physical_cols
                                   * static_cast<size_t>(data.spec.batch_count);

    out.left.resize(matrix_elements);
    out.middle.resize(matrix_elements);
    out.temp0.resize(matrix_elements);
    out.temp1.resize(matrix_elements);
    out.output.resize(matrix_elements);
    out.left.copy_from_host(cast_copy<float>(data.left));
    out.middle.copy_from_host(cast_copy<float>(data.middle));

    if(out.has_secondary)
    {
        out.right.resize(matrix_elements);
        out.aux.resize(matrix_elements);
        out.right.copy_from_host(cast_copy<float>(data.right));
        out.aux.copy_from_host(cast_copy<float>(data.aux));
    }

    return out;
}

PreparedCaseF64 prepare_f64_case(const ContractionCaseData& data)
{
    PreparedCaseF64 out;
    out.spec          = data.spec;
    out.has_secondary = !data.right.empty() && !data.aux.empty();

    const size_t matrix_elements = static_cast<size_t>(data.spec.shape.physical_rows)
                                   * data.spec.shape.physical_cols
                                   * static_cast<size_t>(data.spec.batch_count);

    out.left.resize(matrix_elements);
    out.middle.resize(matrix_elements);
    out.temp0.resize(matrix_elements);
    out.temp1.resize(matrix_elements);
    out.output.resize(matrix_elements);
    out.left.copy_from_host(
        std::span<const double>(data.left.data(), data.left.size()));
    out.middle.copy_from_host(
        std::span<const double>(data.middle.data(), data.middle.size()));

    if(out.has_secondary)
    {
        out.right.resize(matrix_elements);
        out.aux.resize(matrix_elements);
        out.right.copy_from_host(
            std::span<const double>(data.right.data(), data.right.size()));
        out.aux.copy_from_host(
            std::span<const double>(data.aux.data(), data.aux.size()));
    }

    return out;
}

PreparedSpdCaseF32 prepare_f32_case(const SpdCaseData& data)
{
    PreparedSpdCaseF32 out;
    out.spec = data.spec;

    const size_t matrix_elements = static_cast<size_t>(data.spec.shape.physical_rows)
                                   * data.spec.shape.physical_cols
                                   * static_cast<size_t>(data.spec.batch_count);
    const size_t vector_elements = static_cast<size_t>(data.spec.shape.physical_rows)
                                   * static_cast<size_t>(data.spec.batch_count);

    out.source_matrix.resize(matrix_elements);
    out.factor.resize(matrix_elements);
    out.inv_l.resize(matrix_elements);
    out.inverse.resize(matrix_elements);
    out.rhs.resize(vector_elements);
    out.temp_vector.resize(vector_elements);
    out.solution.resize(vector_elements);

    out.source_matrix.copy_from_host(cast_copy<float>(data.matrix));
    out.rhs.copy_from_host(cast_copy<float>(data.rhs));
    return out;
}

PreparedSpdCaseF64 prepare_f64_case(const SpdCaseData& data)
{
    PreparedSpdCaseF64 out;
    out.spec = data.spec;

    const size_t matrix_elements = static_cast<size_t>(data.spec.shape.physical_rows)
                                   * data.spec.shape.physical_cols
                                   * static_cast<size_t>(data.spec.batch_count);
    const size_t vector_elements = static_cast<size_t>(data.spec.shape.physical_rows)
                                   * static_cast<size_t>(data.spec.batch_count);

    out.source_matrix.resize(matrix_elements);
    out.factor.resize(matrix_elements);
    out.inv_l.resize(matrix_elements);
    out.inverse.resize(matrix_elements);
    out.rhs.resize(vector_elements);
    out.temp_vector.resize(vector_elements);
    out.solution.resize(vector_elements);

    out.source_matrix.copy_from_host(
        std::span<const double>(data.matrix.data(), data.matrix.size()));
    out.rhs.copy_from_host(std::span<const double>(data.rhs.data(), data.rhs.size()));
    return out;
}

std::vector<double> download_output(const PreparedCaseF32& data)
{
    std::vector<float> host(data.output.size());
    data.output.copy_to_host(std::span<float>(host.data(), host.size()));
    return cast_copy<double>(host);
}

std::vector<double> download_output(const PreparedCaseF64& data)
{
    std::vector<double> host(data.output.size());
    data.output.copy_to_host(std::span<double>(host.data(), host.size()));
    return host;
}

std::vector<double> download_factor(const PreparedSpdCaseF32& data)
{
    std::vector<float> host(data.factor.size());
    data.factor.copy_to_host(std::span<float>(host.data(), host.size()));
    return cast_copy<double>(host);
}

std::vector<double> download_factor(const PreparedSpdCaseF64& data)
{
    std::vector<double> host(data.factor.size());
    data.factor.copy_to_host(std::span<double>(host.data(), host.size()));
    return host;
}

std::vector<double> download_inverse(const PreparedSpdCaseF32& data)
{
    std::vector<float> host(data.inverse.size());
    data.inverse.copy_to_host(std::span<float>(host.data(), host.size()));
    return cast_copy<double>(host);
}

std::vector<double> download_inverse(const PreparedSpdCaseF64& data)
{
    std::vector<double> host(data.inverse.size());
    data.inverse.copy_to_host(std::span<double>(host.data(), host.size()));
    return host;
}

std::vector<double> download_solution(const PreparedSpdCaseF32& data)
{
    std::vector<float> host(data.solution.size());
    data.solution.copy_to_host(std::span<float>(host.data(), host.size()));
    return cast_copy<double>(host);
}

std::vector<double> download_solution(const PreparedSpdCaseF64& data)
{
    std::vector<double> host(data.solution.size());
    data.solution.copy_to_host(std::span<double>(host.data(), host.size()));
    return host;
}

RunOutcome run_fem12_case(Mode mode,
                          const ContractionCaseData& data,
                          bool                       measure_time)
{
    try
    {
        BackendContext context(mode);
        if(!context.is_supported())
            return {RunStatus::Unsupported, 0.0, {}, context.unsupported_reason()};

        RunOutcome out;
        if(mode == Mode::Fp64RefNoTc)
        {
            auto prepared = prepare_f64_case(data);
            out           = execute_fem12_case(context, prepared, measure_time);
            if(out.status == RunStatus::Ok)
                out.metrics = compare_square_batches(data.reference,
                                                     download_output(prepared),
                                                     data.spec.shape.logical_rows,
                                                     data.spec.shape.physical_rows,
                                                     data.spec.batch_count);
        }
        else
        {
            auto prepared = prepare_f32_case(data);
            out           = execute_fem12_case(context, prepared, measure_time);
            if(out.status == RunStatus::Ok)
                out.metrics = compare_square_batches(data.reference,
                                                     download_output(prepared),
                                                     data.spec.shape.logical_rows,
                                                     data.spec.shape.physical_rows,
                                                     data.spec.batch_count);
        }
        return out;
    }
    catch(const std::exception& e)
    {
        return {RunStatus::Failure, 0.0, {}, e.what()};
    }
}

RunOutcome run_joint24_case(Mode mode,
                            const ContractionCaseData& data,
                            bool                       measure_time)
{
    try
    {
        BackendContext context(mode);
        if(!context.is_supported())
            return {RunStatus::Unsupported, 0.0, {}, context.unsupported_reason()};

        RunOutcome out;
        if(mode == Mode::Fp64RefNoTc)
        {
            auto prepared = prepare_f64_case(data);
            out           = execute_joint24_case(context, prepared, measure_time);
            if(out.status == RunStatus::Ok)
                out.metrics = compare_square_batches(data.reference,
                                                     download_output(prepared),
                                                     data.spec.shape.logical_rows,
                                                     data.spec.shape.physical_rows,
                                                     data.spec.batch_count);
        }
        else
        {
            auto prepared = prepare_f32_case(data);
            out           = execute_joint24_case(context, prepared, measure_time);
            if(out.status == RunStatus::Ok)
                out.metrics = compare_square_batches(data.reference,
                                                     download_output(prepared),
                                                     data.spec.shape.logical_rows,
                                                     data.spec.shape.physical_rows,
                                                     data.spec.batch_count);
        }
        return out;
    }
    catch(const std::exception& e)
    {
        return {RunStatus::Failure, 0.0, {}, e.what()};
    }
}

RunOutcome run_spd_case(Mode               mode,
                        OpKind             op_kind,
                        const SpdCaseData& data,
                        bool               measure_time)
{
    try
    {
        BackendContext context(mode);
        if(!context.is_supported())
            return {RunStatus::Unsupported, 0.0, {}, context.unsupported_reason()};

        if(mode == Mode::Fp64RefNoTc)
        {
            auto prepared = prepare_f64_case(data);
            RunOutcome out;

            switch(op_kind)
            {
                case OpKind::Mas48Factorize:
                case OpKind::Abd12Factorize:
                    out = execute_spd_factorize_case(context, prepared, measure_time);
                    if(out.status == RunStatus::Ok)
                        out.metrics = factorization_reconstruction_metrics(
                            data.matrix,
                            download_factor(prepared),
                            data.spec.shape.logical_rows,
                            data.spec.shape.physical_rows,
                            data.spec.batch_count);
                    return out;
                case OpKind::Mas48Inverse:
                case OpKind::Abd12Inverse:
                    out = execute_spd_inverse_case(context, prepared, measure_time);
                    if(out.status == RunStatus::Ok)
                        out.metrics = compare_square_batches(
                            data.inverse_reference,
                            download_inverse(prepared),
                            data.spec.shape.logical_rows,
                            data.spec.shape.physical_rows,
                            data.spec.batch_count);
                    return out;
                case OpKind::Mas48Solve:
                case OpKind::Abd12Solve:
                    out = execute_spd_solve_case(context, prepared, measure_time);
                    if(out.status == RunStatus::Ok)
                        out.metrics = compare_vector_batches(
                            data.solution_reference,
                            download_solution(prepared),
                            data.spec.shape.logical_rows,
                            data.spec.shape.physical_rows,
                            data.spec.batch_count);
                    return out;
                default:
                    return {RunStatus::InvalidArgument,
                            0.0,
                            {},
                            "run_spd_case received non-SPD OpKind"};
            }
        }

        auto prepared = prepare_f32_case(data);
        RunOutcome out;
        switch(op_kind)
        {
            case OpKind::Mas48Factorize:
            case OpKind::Abd12Factorize:
                out = execute_spd_factorize_case(context, prepared, measure_time);
                if(out.status == RunStatus::Ok)
                    out.metrics = factorization_reconstruction_metrics(
                        data.matrix,
                        download_factor(prepared),
                        data.spec.shape.logical_rows,
                        data.spec.shape.physical_rows,
                        data.spec.batch_count);
                return out;
            case OpKind::Mas48Inverse:
            case OpKind::Abd12Inverse:
                out = execute_spd_inverse_case(context, prepared, measure_time);
                if(out.status == RunStatus::Ok)
                    out.metrics = compare_square_batches(data.inverse_reference,
                                                         download_inverse(prepared),
                                                         data.spec.shape.logical_rows,
                                                         data.spec.shape.physical_rows,
                                                         data.spec.batch_count);
                return out;
            case OpKind::Mas48Solve:
            case OpKind::Abd12Solve:
                out = execute_spd_solve_case(context, prepared, measure_time);
                if(out.status == RunStatus::Ok)
                    out.metrics = compare_vector_batches(data.solution_reference,
                                                         download_solution(prepared),
                                                         data.spec.shape.logical_rows,
                                                         data.spec.shape.physical_rows,
                                                         data.spec.batch_count);
                return out;
            default:
                return {RunStatus::InvalidArgument,
                        0.0,
                        {},
                        "run_spd_case received non-SPD OpKind"};
        }
    }
    catch(const std::exception& e)
    {
        return {RunStatus::Failure, 0.0, {}, e.what()};
    }
}
}  // namespace uipc::tensor_core_lab
