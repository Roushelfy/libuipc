#include <catch2/catch_all.hpp>

#include <tcl/backend_api.h>

TEST_CASE("tensor_core_lab_tc32_support_gate", "[tensor_core_lab][device]")
{
    using namespace uipc::tensor_core_lab;

    BackendContext tc(Mode::Tc32Tf32);
    const bool expected = supports_tf32_tensor_cores(tc.device_info());
    REQUIRE(tc.is_supported() == expected);
}
