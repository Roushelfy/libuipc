#pragma once
#include <cstddef>
#include <cstdint>
#include <uipc/common/dllexport.h>

namespace uipc::common
{
struct MudaMemoryTrackerSnapshot
{
    std::uint64_t alloc_calls   = 0;
    std::uint64_t free_calls    = 0;
    std::uint64_t copy_calls    = 0;
    std::uint64_t set_calls     = 0;
    std::uint64_t resize_calls  = 0;
    std::uint64_t reserve_calls = 0;

    std::uint64_t alloc_bytes = 0;
    std::uint64_t copy_bytes  = 0;
    std::uint64_t set_bytes   = 0;
};

UIPC_CORE_API void muda_memory_tracker_reset() noexcept;
UIPC_CORE_API MudaMemoryTrackerSnapshot muda_memory_tracker_snapshot() noexcept;

UIPC_CORE_API void muda_memory_tracker_record_alloc(std::size_t byte_size) noexcept;
UIPC_CORE_API void muda_memory_tracker_record_free() noexcept;
UIPC_CORE_API void muda_memory_tracker_record_copy(std::size_t byte_size) noexcept;
UIPC_CORE_API void muda_memory_tracker_record_set(std::size_t byte_size) noexcept;
UIPC_CORE_API void muda_memory_tracker_record_resize() noexcept;
UIPC_CORE_API void muda_memory_tracker_record_reserve() noexcept;
}  // namespace uipc::common
