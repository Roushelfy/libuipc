#include <catch2/catch_all.hpp>

#include <tcl/device_info.h>

TEST_CASE("tensor_core_lab_device_smoke", "[tensor_core_lab][smoke]")
{
    const auto info = uipc::tensor_core_lab::query_device_info();
    REQUIRE(info.device_id >= 0);
    REQUIRE(!info.name.empty());
    REQUIRE(info.sm > 0);
}
