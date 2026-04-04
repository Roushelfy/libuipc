#include <catch2/catch_test_macros.hpp>

#include <tcg/registry.h>
#include <tcg/runner.h>

namespace
{
using namespace uipc::tensor_core_gemm;
using ::uipc::tensor_core_lab::compare_matrix_batches;
using ::uipc::tensor_core_lab::ImplPath;
using ::uipc::tensor_core_lab::TensorCoreVerification;

std::vector<GemmLayoutVariant> variants()
{
    return {GemmLayoutVariant::Raw, GemmLayoutVariant::Padded};
}

std::vector<double> cpu_reference(const GemmCaseData& data)
{
    std::vector<double> out(static_cast<size_t>(data.spec.physical_m)
                                * static_cast<size_t>(data.spec.physical_n)
                                * static_cast<size_t>(data.spec.batch_count),
                            0.0);

    for(int batch = 0; batch < data.spec.batch_count; ++batch)
    {
        const size_t    a_offset =
            static_cast<size_t>(batch) * static_cast<size_t>(data.spec.physical_m)
            * static_cast<size_t>(data.spec.physical_k);
        const size_t b_offset =
            static_cast<size_t>(batch) * static_cast<size_t>(data.spec.physical_k)
            * static_cast<size_t>(data.spec.physical_n);
        const size_t          c_offset =
            static_cast<size_t>(batch) * static_cast<size_t>(data.spec.physical_m)
            * static_cast<size_t>(data.spec.physical_n);
        for(int col = 0; col < data.spec.n; ++col)
        {
            for(int row = 0; row < data.spec.m; ++row)
            {
                double sum = 0.0;
                for(int inner = 0; inner < data.spec.k; ++inner)
                {
                    const auto a_index =
                        a_offset
                        + static_cast<size_t>(inner)
                              * static_cast<size_t>(data.spec.physical_m)
                        + static_cast<size_t>(row);
                    const auto b_index =
                        b_offset
                        + static_cast<size_t>(col)
                              * static_cast<size_t>(data.spec.physical_k)
                        + static_cast<size_t>(inner);
                    sum += data.a_fp64[a_index] * data.b_fp64[b_index];
                }
                out[c_offset
                    + static_cast<size_t>(col) * static_cast<size_t>(data.spec.physical_m)
                    + static_cast<size_t>(row)] = sum;
            }
        }
    }

    return out;
}

bool tc32_supported()
{
    BackendContext context(Mode::Tc32Tf32);
    return context.is_supported();
}

void validate_reference_against_cpu(GemmCaseData& data)
{
    ensure_fp64_reference(data);
    const auto cpu = cpu_reference(data);
    const auto metrics =
        compare_matrix_batches(cpu,
                               data.reference_fp64,
                               data.spec.m,
                               data.spec.n,
                               data.spec.physical_m,
                               data.spec.physical_n,
                               data.spec.batch_count);
    REQUIRE(metrics.nan_inf_count == 0);
    REQUIRE(metrics.rel_fro < 1.0e-12);
}
}  // namespace

TEST_CASE("uipc common gemm shapes match fp64 reference", "[tensor_core_gemm][uipc]")
{
    for(const auto& shape : uipc_common_shapes())
    {
        for(const auto variant : variants())
        {
            CAPTURE(shape.shape_tag, variant == GemmLayoutVariant::Raw ? "raw" : "padded");
            auto data =
                make_gemm_case(shape.logical_shape, shape.shape_group, variant, 32, 17);

            validate_reference_against_cpu(data);

            const auto fp64 = run_gemm_case(Mode::Fp64RefNoTc, data, false);
            REQUIRE(fp64.status == RunStatus::Ok);
            REQUIRE(fp64.metrics.nan_inf_count == 0);
            REQUIRE(fp64.metrics.rel_fro < 1.0e-12);
            REQUIRE(fp64.trace.impl_path == ImplPath::Baseline);
            REQUIRE(fp64.trace.tensor_core_requested == false);

            const auto fp32 = run_gemm_case(Mode::Fp32NoTc, data, false);
            REQUIRE(fp32.status == RunStatus::Ok);
            REQUIRE(fp32.metrics.nan_inf_count == 0);
            REQUIRE(fp32.metrics.rel_fro < 5.0e-5);
            REQUIRE(fp32.trace.impl_path == ImplPath::Baseline);
            REQUIRE(fp32.trace.tensor_core_requested == false);
            REQUIRE(fp32.trace.tensor_core_verified == TensorCoreVerification::No);

            const auto tc32 = run_gemm_case(Mode::Tc32Tf32, data, false);
            if(tc32_supported())
            {
                REQUIRE(tc32.status == RunStatus::Ok);
                REQUIRE(tc32.metrics.nan_inf_count == 0);
                REQUIRE(tc32.metrics.rel_fro < 5.0e-3);
                REQUIRE(tc32.trace.impl_path == ImplPath::TcBlas);
                REQUIRE(tc32.trace.tensor_core_requested == true);
                REQUIRE(tc32.trace.tensor_core_verified
                        == TensorCoreVerification::BlockedByPermissions);
            }
            else
            {
                REQUIRE(tc32.status == RunStatus::Unsupported);
            }
        }
    }
}

TEST_CASE("representative extra gemm shapes cover raw and padded", "[tensor_core_gemm][extra]")
{
    for(const auto& shape : representative_extra_shapes())
    {
        for(const auto variant : variants())
        {
            CAPTURE(shape.shape_tag, variant == GemmLayoutVariant::Raw ? "raw" : "padded");
            auto data =
                make_gemm_case(shape.logical_shape, shape.shape_group, variant, 32, 29);

            validate_reference_against_cpu(data);

            const auto fp32 = run_gemm_case(Mode::Fp32NoTc, data, false);
            REQUIRE(fp32.status == RunStatus::Ok);
            REQUIRE(fp32.metrics.nan_inf_count == 0);
            REQUIRE(fp32.metrics.rel_fro < 5.0e-5);

            const auto tc32 = run_gemm_case(Mode::Tc32Tf32, data, false);
            if(tc32_supported())
            {
                REQUIRE(tc32.status == RunStatus::Ok);
                REQUIRE(tc32.metrics.nan_inf_count == 0);
                REQUIRE(tc32.metrics.rel_fro < 5.0e-3);
            }
            else
            {
                REQUIRE(tc32.status == RunStatus::Unsupported);
            }
        }
    }
}
