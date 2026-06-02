#include <contact_system/rcc_bonded_pt_state_bridge.h>
#include <contact_system/rcc_bonded_pt_lookup.h>
#include <muda/launch/parallel_for.h>
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
    m_locked_dm_inv.resize(0);
    m_locked_rest_volume.resize(0);
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
    copy_span_to_device(state.locked_dm_inv(), m_locked_dm_inv);
    copy_span_to_device(state.locked_rest_volume(), m_locked_rest_volume);
    m_counters = state.counters();
}

void RCCBondedPTStateBridge::replace_from_sorted_device_entries(
    muda::CBufferView<RCCBondedPTDeviceEntry> entries,
    const core::RCCBondedPTCounters& counters)
{
    using namespace muda;

    const SizeT n = entries.size();
    const SizeT prev_n = m_locked_keys.size();
    DeviceBuffer<U64>      prev_keys;
    DeviceBuffer<Matrix3x3> prev_dm_inv;
    DeviceBuffer<Float>     prev_rest_volume;
    if(prev_n > 0)
    {
        prev_keys.resize(prev_n);
        prev_dm_inv.resize(prev_n);
        prev_rest_volume.resize(prev_n);
        prev_keys.view().copy_from(m_locked_keys.view());
        prev_dm_inv.view().copy_from(m_locked_dm_inv.view());
        prev_rest_volume.view().copy_from(m_locked_rest_volume.view());
    }

    m_locked_keys.resize(n);
    m_locked_topos.resize(n);
    m_locked_beta.resize(n);
    m_locked_age.resize(n);
    m_release_flags.resize(n);
    m_locked_dm_inv.resize(n);
    m_locked_rest_volume.resize(n);

    if(n > 0)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(n,
                   [entries = entries.viewer().name("entries"),
                    keys    = m_locked_keys.view().viewer().name("locked_keys"),
                    topos   = m_locked_topos.view().viewer().name("locked_topos"),
                    beta    = m_locked_beta.view().viewer().name("locked_beta"),
                    age     = m_locked_age.view().viewer().name("locked_age"),
                    flags = m_release_flags.view().viewer().name("release_flags"),
                    prev_keys_view = prev_keys.view(),
                    prev_keys = prev_keys.view().viewer().name("prev_keys"),
                    prev_dm_inv = prev_dm_inv.view().viewer().name("prev_dm_inv"),
                    prev_rest_volume = prev_rest_volume.view().viewer().name("prev_rest_volume"),
                    dm_inv = m_locked_dm_inv.view().viewer().name("locked_dm_inv"),
                    rest_volume = m_locked_rest_volume.view().viewer().name("rest_volume")] __device__(int i) mutable
                   {
                       const auto entry = entries(i);
                       keys(i)         = entry.key;
                       topos(i)        = entry.topo;
                       beta(i)         = entry.beta;
                       age(i)          = entry.age;
                       flags(i)        = entry.release_flags;
                       const IndexT prev =
                           rcc_bonded_pt_lower_bound(prev_keys_view, entry.key);
                       if(prev < static_cast<IndexT>(prev_keys_view.size())
                          && prev_keys(prev) == entry.key)
                       {
                           dm_inv(i)      = prev_dm_inv(prev);
                           rest_volume(i) = prev_rest_volume(prev);
                       }
                       else
                       {
                           dm_inv(i)      = Matrix3x3::Identity();
                           rest_volume(i) = 0.0;
                       }
                   });
    }

    m_counters = counters;
}

core::RCCBondedPTState RCCBondedPTStateBridge::download() const
{
    const SizeT n = size();
    UIPC_ASSERT(m_locked_topos.size() == n && m_locked_beta.size() == n
                    && m_locked_age.size() == n && m_release_flags.size() == n,
                "RCC bonded PT state bridge has unzipped device buffers.");
    UIPC_ASSERT(m_locked_dm_inv.size() == n && m_locked_rest_volume.size() == n,
                "RCC bonded PT state bridge has unzipped rest-shape buffers.");

    auto keys  = copy_device_to_host(m_locked_keys);
    auto topos = copy_device_to_host(m_locked_topos);
    auto beta  = copy_device_to_host(m_locked_beta);
    auto age   = copy_device_to_host(m_locked_age);
    auto flags = copy_device_to_host(m_release_flags);
    auto dm_inv = copy_device_to_host(m_locked_dm_inv);
    auto rest_volume = copy_device_to_host(m_locked_rest_volume);

    core::RCCBondedPTState state;
    state.reserve(n);
    for(SizeT i = 0; i < n; ++i)
    {
        state.push_locked(core::RCCBondedPTEntry{keys[i],
                                                 topos[i],
                                                 beta[i],
                                                 age[i],
                                                 flags[i],
                                                 dm_inv[i],
                                                 rest_volume[i]});
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

muda::CBufferView<Matrix3x3> RCCBondedPTStateBridge::locked_dm_inv() const noexcept
{
    return m_locked_dm_inv;
}

muda::CBufferView<Float> RCCBondedPTStateBridge::locked_rest_volume() const noexcept
{
    return m_locked_rest_volume;
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

muda::BufferView<Matrix3x3> RCCBondedPTStateBridge::locked_dm_inv() noexcept
{
    return m_locked_dm_inv.view();
}

muda::BufferView<Float> RCCBondedPTStateBridge::locked_rest_volume() noexcept
{
    return m_locked_rest_volume.view();
}
}  // namespace uipc::backend::cuda
