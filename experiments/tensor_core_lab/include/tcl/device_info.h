#pragma once
#include <string>

namespace uipc::tensor_core_lab
{
struct DeviceInfo
{
    int         device_id = -1;
    int         major     = 0;
    int         minor     = 0;
    int         sm        = 0;
    std::string name;
};

DeviceInfo query_device_info();
bool       supports_tf32_tensor_cores(const DeviceInfo& info) noexcept;
}  // namespace uipc::tensor_core_lab
