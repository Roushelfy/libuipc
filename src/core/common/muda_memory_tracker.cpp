#include <uipc/common/muda_memory_tracker.h>
#include <atomic>

namespace uipc::common
{
namespace
{
struct TrackerState
{
    std::atomic_uint64_t alloc_calls{0};
    std::atomic_uint64_t free_calls{0};
    std::atomic_uint64_t copy_calls{0};
    std::atomic_uint64_t set_calls{0};
    std::atomic_uint64_t resize_calls{0};
    std::atomic_uint64_t reserve_calls{0};

    std::atomic_uint64_t alloc_bytes{0};
    std::atomic_uint64_t copy_bytes{0};
    std::atomic_uint64_t set_bytes{0};
};

TrackerState& tracker_state() noexcept
{
    static TrackerState state;
    return state;
}
}  // namespace

void muda_memory_tracker_reset() noexcept
{
    auto& state = tracker_state();
    state.alloc_calls.store(0, std::memory_order_relaxed);
    state.free_calls.store(0, std::memory_order_relaxed);
    state.copy_calls.store(0, std::memory_order_relaxed);
    state.set_calls.store(0, std::memory_order_relaxed);
    state.resize_calls.store(0, std::memory_order_relaxed);
    state.reserve_calls.store(0, std::memory_order_relaxed);
    state.alloc_bytes.store(0, std::memory_order_relaxed);
    state.copy_bytes.store(0, std::memory_order_relaxed);
    state.set_bytes.store(0, std::memory_order_relaxed);
}

MudaMemoryTrackerSnapshot muda_memory_tracker_snapshot() noexcept
{
    auto& state = tracker_state();
    return MudaMemoryTrackerSnapshot{
        .alloc_calls   = state.alloc_calls.load(std::memory_order_relaxed),
        .free_calls    = state.free_calls.load(std::memory_order_relaxed),
        .copy_calls    = state.copy_calls.load(std::memory_order_relaxed),
        .set_calls     = state.set_calls.load(std::memory_order_relaxed),
        .resize_calls  = state.resize_calls.load(std::memory_order_relaxed),
        .reserve_calls = state.reserve_calls.load(std::memory_order_relaxed),
        .alloc_bytes   = state.alloc_bytes.load(std::memory_order_relaxed),
        .copy_bytes    = state.copy_bytes.load(std::memory_order_relaxed),
        .set_bytes     = state.set_bytes.load(std::memory_order_relaxed),
    };
}

void muda_memory_tracker_record_alloc(std::size_t byte_size) noexcept
{
    auto& state = tracker_state();
    state.alloc_calls.fetch_add(1, std::memory_order_relaxed);
    state.alloc_bytes.fetch_add(static_cast<std::uint64_t>(byte_size),
                                std::memory_order_relaxed);
}

void muda_memory_tracker_record_free() noexcept
{
    tracker_state().free_calls.fetch_add(1, std::memory_order_relaxed);
}

void muda_memory_tracker_record_copy(std::size_t byte_size) noexcept
{
    auto& state = tracker_state();
    state.copy_calls.fetch_add(1, std::memory_order_relaxed);
    state.copy_bytes.fetch_add(static_cast<std::uint64_t>(byte_size),
                               std::memory_order_relaxed);
}

void muda_memory_tracker_record_set(std::size_t byte_size) noexcept
{
    auto& state = tracker_state();
    state.set_calls.fetch_add(1, std::memory_order_relaxed);
    state.set_bytes.fetch_add(static_cast<std::uint64_t>(byte_size),
                              std::memory_order_relaxed);
}

void muda_memory_tracker_record_resize() noexcept
{
    tracker_state().resize_calls.fetch_add(1, std::memory_order_relaxed);
}

void muda_memory_tracker_record_reserve() noexcept
{
    tracker_state().reserve_calls.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace uipc::common
