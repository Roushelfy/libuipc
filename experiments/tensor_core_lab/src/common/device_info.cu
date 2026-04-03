#include <tcl/device_info.h>

#include <cuda_runtime.h>

namespace uipc::tensor_core_lab
{
DeviceInfo query_device_info()
{
    DeviceInfo info;

    int device_id = -1;
    if(cudaGetDevice(&device_id) != cudaSuccess)
        return info;

    cudaDeviceProp prop{};
    if(cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
        return info;

    info.device_id = device_id;
    info.major     = prop.major;
    info.minor     = prop.minor;
    info.sm        = prop.major * 10 + prop.minor;
    info.name      = prop.name;
    return info;
}

bool supports_tf32_tensor_cores(const DeviceInfo& info) noexcept
{
    return info.sm >= 80;
}
}  // namespace uipc::tensor_core_lab
