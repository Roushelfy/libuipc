#include <linear_system/socu_contact_topology_stamp.h>

#include <sim_system.h>
#include <fmt/format.h>
#include <muda/buffer/buffer_launch.h>

#include <array>

namespace uipc::backend::cuda_mixed
{
namespace
{
cudaStream_t launch_stream(cudaStream_t stream) noexcept
{
    return stream == cudaStreamLegacy ? nullptr : stream;
}

MUDA_DEVICE SizeT device_mix_contact_topology(SizeT hash, SizeT value) noexcept
{
    return socu_contact_mix_hash(hash, value);
}

SizeT source_item_tag(SizeT                   reporter_id,
                      SizeT                   source_id,
                      SocuContactSourceFamily family) noexcept
{
    SizeT tag = SocuContactPlanHashOffset;
    socu_contact_mix_in_place(tag, reporter_id);
    socu_contact_mix_in_place(tag, source_id);
    socu_contact_mix_in_place(tag, socu_contact_source_family_value(family));
    return tag;
}

void mix_source_metadata(SocuContactTopologyStamp& stamp,
                         SizeT                     reporter_id,
                         SizeT                     source_id,
                         SocuContactSourceFamily    family,
                         SizeT                     count,
                         SizeT                     content_hash,
                         SizeT                     layout_token) noexcept
{
    SocuContactTopologySource source;
    source.reporter_id   = reporter_id;
    source.source_id     = source_id;
    source.family        = family;
    source.contact_count = count;
    source.content_hash  = content_hash;
    source.layout_token  = layout_token;

    socu_contact_mix_in_place(stamp.layout_hash,
                              socu_contact_source_layout_hash(source));
    socu_contact_mix_in_place(stamp.content_hash,
                              socu_contact_source_content_hash(source));
    ++stamp.source_count;
}

template <typename ValueT>
__global__ void mix_contact_topology_view_kernel(
    muda::CBufferView<ValueT> view,
    SizeT                     item_tag,
    unsigned long long*       accum)
{
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= view.size())
        return;

    SizeT hash = SocuContactPlanHashOffset;
    hash       = device_mix_contact_topology(hash, item_tag);
    hash       = device_mix_contact_topology(hash, i);

    const auto item = view.data()[i];
    constexpr int ValueSize = ValueT::SizeAtCompileTime;
#pragma unroll
    for(int c = 0; c < ValueSize; ++c)
        hash = device_mix_contact_topology(hash, static_cast<SizeT>(item(c)));

    const auto h = static_cast<unsigned long long>(hash);
    atomicXor(accum, h);
    atomicAdd(accum + 1, h);
}

template <typename ValueT>
void mix_view_impl(SocuContactTopologyStamp& stamp,
                   SocuContactTopologyHashWorkspace& workspace,
                   cudaStream_t                  stream,
                   SizeT                         reporter_id,
                   SizeT                         source_id,
                   SocuContactSourceFamily       family,
                   muda::CBufferView<ValueT>     view,
                   SizeT                         layout_token)
{
    mix_source_metadata(stamp,
                        reporter_id,
                        source_id,
                        family,
                        view.size(),
                        0,
                        layout_token);
    if(view.size() == 0)
        return;

    constexpr int block_dim = 256;
    const auto    grid_dim =
        static_cast<unsigned int>((view.size() + block_dim - 1) / block_dim);
    mix_contact_topology_view_kernel<<<grid_dim,
                                       block_dim,
                                       0,
                                       launch_stream(stream)>>>(
        view,
        source_item_tag(reporter_id, source_id, family),
        workspace.accum.data());
}
}  // namespace

SocuContactTopologyStamp SocuContactTopologyStampCache::update(
    SocuContactTopologyStamp stamp) noexcept
{
    const bool changed =
        m_epoch == 0 || stamp.layout_hash != m_last_stamp.layout_hash
        || stamp.content_hash != m_last_stamp.content_hash
        || stamp.reporter_count != m_last_stamp.reporter_count
        || stamp.source_count != m_last_stamp.source_count
        || !(stamp.counts == m_last_stamp.counts);
    if(changed)
        ++m_epoch;

    stamp.epoch = m_epoch;
    m_last_stamp = stamp;
    return stamp;
}

SocuContactTopologyStamp socu_contact_topology_make_stamp_seed() noexcept
{
    SocuContactTopologyStamp stamp;
    stamp.layout_hash  = SocuContactPlanHashOffset;
    stamp.content_hash = SocuContactPlanHashOffset;
    return stamp;
}

void socu_contact_topology_hash_reset(
    SocuContactTopologyHashWorkspace& workspace,
    cudaStream_t                      stream)
{
    muda::BufferLaunch(stream).resize(workspace.accum, 2);
    muda::BufferLaunch(stream)
        .fill<unsigned long long>(workspace.accum.view(), 0ull);
}

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector4i>   view)
{
    socu_contact_topology_mix_view(stamp,
                                   workspace,
                                   stream,
                                   reporter_id,
                                   source_id,
                                   family,
                                   view,
                                   reinterpret_cast<SizeT>(view.data()));
}

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector4i>   view,
                                    SizeT                         layout_token)
{
    mix_view_impl(stamp,
                  workspace,
                  stream,
                  reporter_id,
                  source_id,
                  family,
                  view,
                  layout_token);
}

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector3i>   view)
{
    socu_contact_topology_mix_view(stamp,
                                   workspace,
                                   stream,
                                   reporter_id,
                                   source_id,
                                   family,
                                   view,
                                   reinterpret_cast<SizeT>(view.data()));
}

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector3i>   view,
                                    SizeT                         layout_token)
{
    mix_view_impl(stamp,
                  workspace,
                  stream,
                  reporter_id,
                  source_id,
                  family,
                  view,
                  layout_token);
}

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector2i>   view)
{
    socu_contact_topology_mix_view(stamp,
                                   workspace,
                                   stream,
                                   reporter_id,
                                   source_id,
                                   family,
                                   view,
                                   reinterpret_cast<SizeT>(view.data()));
}

void socu_contact_topology_mix_view(SocuContactTopologyStamp& stamp,
                                    SocuContactTopologyHashWorkspace& workspace,
                                    cudaStream_t                  stream,
                                    SizeT                         reporter_id,
                                    SizeT                         source_id,
                                    SocuContactSourceFamily       family,
                                    muda::CBufferView<Vector2i>   view,
                                    SizeT                         layout_token)
{
    mix_view_impl(stamp,
                  workspace,
                  stream,
                  reporter_id,
                  source_id,
                  family,
                  view,
                  layout_token);
}

void socu_contact_topology_mix_unknown_source(
    SocuContactTopologyStamp& stamp,
    SizeT                     reporter_id,
    SizeT                     source_id,
    SizeT                     contact_count,
    SizeT                     layout_token) noexcept
{
    mix_source_metadata(stamp,
                        reporter_id,
                        source_id,
                        SocuContactSourceFamily::Unknown,
                        contact_count,
                        0,
                        layout_token);
}

void socu_contact_topology_finalize_metadata(
    SocuContactTopologyStamp& stamp,
    SizeT                     reporter_count) noexcept
{
    stamp.reporter_count = reporter_count;
    socu_contact_mix_in_place(stamp.layout_hash, stamp.reporter_count);
    socu_contact_mix_in_place(stamp.layout_hash, stamp.source_count);
    socu_contact_mix_in_place(stamp.content_hash, stamp.reporter_count);
    socu_contact_mix_in_place(stamp.content_hash, stamp.source_count);
}

SocuContactTopologyDeviceHash socu_contact_topology_hash_finish(
    SocuContactTopologyHashWorkspace& workspace,
    cudaStream_t                      stream)
{
    std::array<unsigned long long, 2> host_hash{};
    const cudaStream_t                copy_stream = launch_stream(stream);
    auto error = cudaMemcpyAsync(host_hash.data(),
                                 workspace.accum.data(),
                                 host_hash.size() * sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost,
                                 copy_stream);
    if(error != cudaSuccess)
    {
        throw SimSystemException{fmt::format(
            "contact_topology_hash_copy_failed: {}",
            cudaGetErrorString(error))};
    }
    error = cudaStreamSynchronize(copy_stream);
    if(error != cudaSuccess)
    {
        throw SimSystemException{fmt::format(
            "contact_topology_hash_sync_failed: {}",
            cudaGetErrorString(error))};
    }

    return SocuContactTopologyDeviceHash{static_cast<SizeT>(host_hash[0]),
                                         static_cast<SizeT>(host_hash[1])};
}

void socu_contact_topology_mix_device_hash(
    SocuContactTopologyStamp&            stamp,
    const SocuContactTopologyDeviceHash& device_hash) noexcept
{
    socu_contact_mix_in_place(stamp.content_hash, device_hash.xor_hash);
    socu_contact_mix_in_place(stamp.content_hash, device_hash.sum_hash);
}
}  // namespace uipc::backend::cuda_mixed
