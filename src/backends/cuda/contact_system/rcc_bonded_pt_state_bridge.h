#pragma once
#include <muda/buffer/device_buffer.h>
#include <type_define.h>
#include <uipc/core/rcc_bonded_pt_state.h>

namespace uipc::backend::cuda
{
struct RCCBondedPTDeviceEntry
{
    U64      key;
    Vector4i topo;
    Float    beta;
    IndexT   age;
    U32      release_flags;
};

class RCCBondedPTStateBridge
{
  public:
    void clear();
    void upload(const core::RCCBondedPTState& state);
    void replace_from_sorted_device_entries(
        muda::CBufferView<RCCBondedPTDeviceEntry> entries,
        const core::RCCBondedPTCounters& counters);
    core::RCCBondedPTState download() const;

    SizeT size() const noexcept;
    bool  empty() const noexcept;

    const core::RCCBondedPTCounters& counters() const noexcept;

    muda::CBufferView<U64>      locked_keys() const noexcept;
    muda::CBufferView<Vector4i> locked_topos() const noexcept;
    muda::CBufferView<Float>    locked_beta() const noexcept;
    muda::CBufferView<IndexT>   locked_age() const noexcept;
    muda::CBufferView<U32>      release_flags() const noexcept;

    muda::BufferView<Float>  locked_beta() noexcept;
    muda::BufferView<IndexT> locked_age() noexcept;
    muda::BufferView<U32>    release_flags() noexcept;

  private:
    muda::DeviceBuffer<U64>      m_locked_keys;
    muda::DeviceBuffer<Vector4i> m_locked_topos;
    muda::DeviceBuffer<Float>    m_locked_beta;
    muda::DeviceBuffer<IndexT>   m_locked_age;
    muda::DeviceBuffer<U32>      m_release_flags;
    core::RCCBondedPTCounters    m_counters;
};
}  // namespace uipc::backend::cuda
