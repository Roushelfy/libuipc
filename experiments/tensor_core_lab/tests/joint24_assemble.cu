#include <catch2/catch_all.hpp>

#include <tcl/case_spec.h>
#include <tcl/runner.h>

TEST_CASE("tensor_core_lab_joint24_assemble", "[tensor_core_lab][joint24]")
{
    using namespace uipc::tensor_core_lab;

    const auto data = make_joint24_case("unit_joint24", 24, 11, 1.0e4);

    const auto fp64 = run_joint24_case(Mode::Fp64RefNoTc, data);
    REQUIRE(fp64.status == RunStatus::Ok);
    CHECK(fp64.trace.impl_path == ImplPath::Baseline);
    CHECK_FALSE(fp64.trace.tensor_core_requested);
    CHECK(fp64.trace.tensor_core_verified == TensorCoreVerification::No);
    CHECK(fp64.metrics.rel_fro < 2.0e-8);
    CHECK(fp64.metrics.nan_inf_count == 0);
    CHECK(fp64.metrics.symmetry_error < 1.0e-10);

    const auto fp32 = run_joint24_case(Mode::Fp32NoTc, data);
    REQUIRE(fp32.status == RunStatus::Ok);
    CHECK(fp32.trace.impl_path == ImplPath::Baseline);
    CHECK_FALSE(fp32.trace.tensor_core_requested);
    CHECK(fp32.trace.tensor_core_verified == TensorCoreVerification::No);
    CHECK(fp32.metrics.rel_fro < 2.0e-4);
    CHECK(fp32.metrics.nan_inf_count == 0);
    CHECK(fp32.metrics.symmetry_error < 2.0e-3);

    const auto tc32 = run_joint24_case(Mode::Tc32Tf32, data);
    if(tc32.status == RunStatus::Unsupported)
    {
        SUCCEED("tc32_tf32 is unsupported on this GPU");
    }
    else
    {
        REQUIRE(tc32.status == RunStatus::Ok);
        CHECK(tc32.trace.impl_path == ImplPath::TcBlas);
        CHECK(tc32.trace.tensor_core_requested);
        CHECK(tc32.trace.tensor_core_verified
              == TensorCoreVerification::BlockedByPermissions);
        CHECK(tc32.metrics.rel_fro < 8.0e-4);
        CHECK(tc32.metrics.nan_inf_count == 0);
        CHECK(tc32.metrics.symmetry_error < 5.0);
    }
}
