#include <catch2/catch_all.hpp>

#include <tcl/case_spec.h>
#include <tcl/runner.h>

TEST_CASE("tensor_core_lab_mas48_factorize", "[tensor_core_lab][mas48]")
{
    using namespace uipc::tensor_core_lab;

    const auto data = make_mas48_case("unit_mas48_factorize", 256, 41, 1.0e2);

    const auto fp64 = run_spd_case(Mode::Fp64RefNoTc, OpKind::Mas48Factorize, data);
    REQUIRE(fp64.status == RunStatus::Ok);
    CHECK(fp64.trace.impl_path == ImplPath::Baseline);
    CHECK_FALSE(fp64.trace.tensor_core_requested);
    CHECK(fp64.trace.tensor_core_verified == TensorCoreVerification::No);
    CHECK(fp64.metrics.rel_fro < 1.0e-11);
    CHECK(fp64.metrics.nan_inf_count == 0);

    const auto fp32 = run_spd_case(Mode::Fp32NoTc, OpKind::Mas48Factorize, data);
    REQUIRE(fp32.status == RunStatus::Ok);
    CHECK(fp32.trace.impl_path == ImplPath::Baseline);
    CHECK_FALSE(fp32.trace.tensor_core_requested);
    CHECK(fp32.trace.tensor_core_verified == TensorCoreVerification::No);
    CHECK(fp32.metrics.rel_fro < 8.0e-4);
    CHECK(fp32.metrics.nan_inf_count == 0);

    const auto tc32 = run_spd_case(Mode::Tc32Tf32, OpKind::Mas48Factorize, data);
    if(tc32.status == RunStatus::Unsupported)
    {
        SUCCEED("tc32_tf32 is unsupported on this GPU");
    }
    else
    {
        REQUIRE(tc32.status == RunStatus::Ok);
        CHECK(tc32.trace.impl_path == ImplPath::Baseline);
        CHECK(tc32.trace.tensor_core_requested);
        CHECK(tc32.trace.tensor_core_verified
              == TensorCoreVerification::BlockedByPermissions);
        CHECK(tc32.metrics.rel_fro < 5.0e-4);
        CHECK(tc32.metrics.nan_inf_count == 0);
    }
}
