#pragma once

#include <linear_system/socu_contact_plan_types.h>
#include <cuda_runtime_api.h>
#include <muda/buffer/buffer_view.h>
#include <muda/buffer/device_buffer.h>

namespace uipc::backend::cuda_mixed
{
struct SocuContactTopologyDeviceHash
{
    SizeT xor_hash = 0;
    SizeT sum_hash = 0;

    bool operator==(const SocuContactTopologyDeviceHash&) const noexcept = default;
};

struct SocuContactTopologyHashWorkspace
{
    muda::DeviceBuffer<unsigned long long> accum;
};

class SocuContactTopologyStampCache
{
  public:
    SocuContactTopologyStamp update(SocuContactTopologyStamp stamp) noexcept;
    SizeT epoch() const noexcept { return m_epoch; }
    const SocuContactTopologyStamp& last_stamp() const noexcept
    {
        return m_last_stamp;
    }

  private:
    SizeT                    m_epoch = 0;
    SocuContactTopologyStamp m_last_stamp;
};

SocuContactTopologyStamp socu_contact_topology_make_stamp_seed() noexcept;

void socu_contact_topology_hash_reset(
    SocuContactTopologyHashWorkspace& workspace,
    cudaStream_t                      stream);

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector4i>   view);

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector4i>   view,
                                    SizeT                         layout_token);

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector3i>   view);

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector3i>   view,
                                    SizeT                         layout_token);

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector2i>   view);

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector2i>   view,
                                    SizeT                         layout_token);

void socu_contact_topology_mix_unknown_source(
    SocuContactTopologyStamp& stamp,
    SizeT                     reporter_id,
    SizeT                     source_id,
    SizeT                     contact_count,
    SizeT                     layout_token = 0) noexcept;

void socu_contact_topology_finalize_metadata(
    SocuContactTopologyStamp& stamp,
    SizeT                     reporter_count) noexcept;

SocuContactTopologyDeviceHash socu_contact_topology_hash_finish(
    SocuContactTopologyHashWorkspace& workspace,
    cudaStream_t                      stream);

void socu_contact_topology_mix_device_hash(
    SocuContactTopologyStamp&           stamp,
    const SocuContactTopologyDeviceHash& device_hash) noexcept;
}  // namespace uipc::backend::cuda_mixed
