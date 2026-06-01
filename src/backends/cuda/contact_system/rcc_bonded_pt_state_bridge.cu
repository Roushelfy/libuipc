#include <contact_system/rcc_bonded_pt_state_bridge.h>
#include <uipc/common/log.h>

namespace uipc::backend::cuda
{
namespace
{
template <typename T>
void copy_span_to_device(span<const T> host, muda::DeviceBuffer<T>& device)
{
    device.resize(host.size());
    if(!host.empty())
        device.view().copy_from(host.data());
}

template <typename T>
vector<T> copy_device_to_host(const muda::DeviceBuffer<T>& device)
{
    vector<T> host(device.size());
    if(!host.empty())
        device.view().copy_to(host.data());
    return host;
}
}  // namespace

void RCCBondedPTStateBridge::clear()
{
    m_locked_keys.resize(0);
    m_locked_topos.resize(0);
    m_locked_beta.resize(0);
    m_locked_age.resize(0);
    m_release_flags.resize(0);
    m_counters = {};
}

void RCCBondedPTStateBridge::upload(const core::RCCBondedPTState& state)
{
    UIPC_ASSERT(state.validate(),
                "RCC bonded PT state bridge received unzipped host buffers.");

    copy_span_to_device(state.locked_keys(), m_locked_keys);
    copy_span_to_device(state.locked_topos(), m_locked_topos);
    copy_span_to_device(state.locked_beta(), m_locked_beta);
    copy_span_to_device(state.locked_age(), m_locked_age);
    copy_span_to_device(state.release_flags(), m_release_flags);
    m_counters = state.counters();
}

core::RCCBondedPTState RCCBondedPTStateBridge::download() const
{
    const SizeT n = size();
    UIPC_ASSERT(m_locked_topos.size() == n && m_locked_beta.size() == n
                    && m_locked_age.size() == n && m_release_flags.size() == n,
                "RCC bonded PT state bridge has unzipped device buffers.");

    auto keys  = copy_device_to_host(m_locked_keys);
    auto topos = copy_device_to_host(m_locked_topos);
    auto beta  = copy_device_to_host(m_locked_beta);
    auto age   = copy_device_to_host(m_locked_age);
    auto flags = copy_device_to_host(m_release_flags);

    core::RCCBondedPTState state;
    state.reserve(n);
    for(SizeT i = 0; i < n; ++i)
    {
        state.push_locked(core::RCCBondedPTEntry{keys[i],
                                                 topos[i],
                                                 beta[i],
                                                 age[i],
                                                 flags[i]});
    }
    state.set_counters(m_counters);
    return state;
}

SizeT RCCBondedPTStateBridge::size() const noexcept
{
    return m_locked_keys.size();
}

bool RCCBondedPTStateBridge::empty() const noexcept
{
    return size() == 0;
}

const core::RCCBondedPTCounters& RCCBondedPTStateBridge::counters() const noexcept
{
    return m_counters;
}

muda::CBufferView<U64> RCCBondedPTStateBridge::locked_keys() const noexcept
{
    return m_locked_keys;
}

muda::CBufferView<Vector4i> RCCBondedPTStateBridge::locked_topos() const noexcept
{
    return m_locked_topos;
}

muda::CBufferView<Float> RCCBondedPTStateBridge::locked_beta() const noexcept
{
    return m_locked_beta;
}

muda::CBufferView<IndexT> RCCBondedPTStateBridge::locked_age() const noexcept
{
    return m_locked_age;
}

muda::CBufferView<U32> RCCBondedPTStateBridge::release_flags() const noexcept
{
    return m_release_flags;
}

muda::BufferView<Float> RCCBondedPTStateBridge::locked_beta() noexcept
{
    return m_locked_beta.view();
}

muda::BufferView<IndexT> RCCBondedPTStateBridge::locked_age() noexcept
{
    return m_locked_age.view();
}

muda::BufferView<U32> RCCBondedPTStateBridge::release_flags() noexcept
{
    return m_release_flags.view();
}
}  // namespace uipc::backend::cuda
