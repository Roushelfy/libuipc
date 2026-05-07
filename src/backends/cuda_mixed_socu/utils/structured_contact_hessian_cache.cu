#include <linear_system/socu_native_descriptors.h>
#include <utils/structured_contact_assembly_sink.h>

namespace uipc::backend::cuda_mixed
{
namespace
{
cudaStream_t normalize_muda_launch_stream(cudaStream_t stream) noexcept
{
    return stream == cudaStreamLegacy ? nullptr : stream;
}

MUDA_DEVICE bool socu_native_device_dof_range_active(const IndexT* old_to_chain,
                                                     SizeT         old_to_chain_size,
                                                     SizeT         horizon,
                                                     SizeT         block_size,
                                                     IndexT        old_dof,
                                                     IndexT        dof_count,
                                                     SizeT&        block,
                                                     SizeT&        lane) noexcept
{
    if(old_to_chain == nullptr || block_size == 0 || old_dof < 0 || dof_count <= 0)
        return false;
    if(static_cast<SizeT>(old_dof + dof_count) > old_to_chain_size)
        return false;

    const IndexT first_chain = old_to_chain[static_cast<SizeT>(old_dof)];
    if(first_chain < 0)
        return false;
    block = static_cast<SizeT>(first_chain) / block_size;
    lane  = static_cast<SizeT>(first_chain) % block_size;
    if(block >= horizon || lane + static_cast<SizeT>(dof_count) > block_size)
        return false;

    for(IndexT i = 1; i < dof_count; ++i)
    {
        const IndexT chain = old_to_chain[static_cast<SizeT>(old_dof + i)];
        if(chain != first_chain + i)
            return false;
    }
    return true;
}

MUDA_DEVICE SocuNativeVertexDescriptor socu_native_make_device_vertex_descriptor(
    SocuNativeDescriptorKind kind,
    bool                     fixed,
    IndexT                   old_dof,
    IndexT                   dof_count,
    IndexT                   abd_body,
    IndexT                   abd_j_index,
    IndexT                   epoch,
    const IndexT*            old_to_chain,
    SizeT                    old_to_chain_size,
    SizeT                    horizon,
    SizeT                    block_size) noexcept
{
    SocuNativeVertexDescriptor descriptor;
    descriptor.kind        = kind;
    descriptor.fixed       = fixed;
    descriptor.old_dof     = old_dof;
    descriptor.dof_count   = dof_count;
    descriptor.abd_body    = abd_body;
    descriptor.abd_j_index = abd_j_index;
    descriptor.epoch       = epoch;
    descriptor.active = socu_native_device_dof_range_active(old_to_chain,
                                                            old_to_chain_size,
                                                            horizon,
                                                            block_size,
                                                            old_dof,
                                                            dof_count,
                                                            descriptor.block,
                                                            descriptor.lane);
    return descriptor;
}

__global__ void rebuild_socu_native_vertex_descriptors_kernel(
    muda::BufferView<SocuNativeVertexDescriptor> descriptors,
    muda::CBufferView<IndexT> old_to_chain,
    SizeT                     horizon,
    SizeT                     block_size,
    IndexT                    epoch,
    IndexT                    fem_vertex_offset,
    IndexT                    fem_vertex_count,
    IndexT                    fem_old_dof_offset,
    muda::CBufferView<IndexT> fem_vertex_is_fixed,
    IndexT                    abd_vertex_offset,
    IndexT                    abd_vertex_count,
    IndexT                    abd_old_dof_offset,
    IndexT                    abd_body_count,
    muda::CBufferView<IndexT> abd_vertex_to_body,
    muda::CBufferView<IndexT> abd_body_is_fixed)
{
    const SizeT global =
        static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(global >= descriptors.size())
        return;

    SocuNativeVertexDescriptor descriptor{};
    const IndexT global_vertex = static_cast<IndexT>(global);

    if(fem_vertex_offset >= 0 && fem_vertex_count > 0
       && global_vertex >= fem_vertex_offset
       && global_vertex < fem_vertex_offset + fem_vertex_count)
    {
        const IndexT local = global_vertex - fem_vertex_offset;
        const bool fixed =
            static_cast<SizeT>(local) < fem_vertex_is_fixed.size()
                ? fem_vertex_is_fixed.data()[static_cast<SizeT>(local)] != 0
                : false;
        descriptor = socu_native_make_device_vertex_descriptor(
            SocuNativeDescriptorKind::Fem,
            fixed,
            fem_old_dof_offset + local * 3,
            3,
            -1,
            -1,
            epoch,
            old_to_chain.data(),
            old_to_chain.size(),
            horizon,
            block_size);
    }

    if(abd_vertex_offset >= 0 && abd_vertex_count > 0
       && global_vertex >= abd_vertex_offset
       && global_vertex < abd_vertex_offset + abd_vertex_count)
    {
        const IndexT local = global_vertex - abd_vertex_offset;
        if(static_cast<SizeT>(local) < abd_vertex_to_body.size())
        {
            const IndexT body = abd_vertex_to_body.data()[static_cast<SizeT>(local)];
            if(body >= 0 && body < abd_body_count)
            {
                const bool fixed =
                    static_cast<SizeT>(body) < abd_body_is_fixed.size()
                        ? abd_body_is_fixed.data()[static_cast<SizeT>(body)] != 0
                        : false;
                descriptor = socu_native_make_device_vertex_descriptor(
                    SocuNativeDescriptorKind::Abd,
                    fixed,
                    abd_old_dof_offset + body * 12,
                    12,
                    body,
                    local,
                    epoch,
                    old_to_chain.data(),
                    old_to_chain.size(),
                    horizon,
                    block_size);
            }
        }
    }

    descriptors.data()[global] = descriptor;
}

__global__ void replay_structured_contact_hessian_cache_kernel(
    StructuredContactAssemblySink<ActivePolicy::StoreScalar, ActivePolicy::SolveScalar>
        sink)
{
    const SizeT i = static_cast<SizeT>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= sink.hessian_cache.replay_count)
        return;

    const auto& record = sink.hessian_cache.records.data()[i];
    Eigen::Matrix<ActivePolicy::StoreScalar, 3, 3> H;
#pragma unroll
    for(IndexT r = 0; r < 3; ++r)
    {
#pragma unroll
        for(IndexT c = 0; c < 3; ++c)
            H(r, c) = record.H[r * 3 + c];
    }
    sink.write_contact_half_block(record.global_i,
                                  record.global_j,
                                  H,
                                  record.mirror_diag_block != 0);
}
}  // namespace

void rebuild_socu_native_vertex_descriptors(
    cudaStream_t                           stream,
    muda::BufferView<SocuNativeVertexDescriptor> descriptors,
    muda::CBufferView<IndexT>              old_to_chain,
    SizeT                                  horizon,
    SizeT                                  block_size,
    IndexT                                 epoch,
    IndexT                                 fem_vertex_offset,
    IndexT                                 fem_vertex_count,
    IndexT                                 fem_old_dof_offset,
    muda::CBufferView<IndexT>              fem_vertex_is_fixed,
    IndexT                                 abd_vertex_offset,
    IndexT                                 abd_vertex_count,
    IndexT                                 abd_old_dof_offset,
    IndexT                                 abd_body_count,
    muda::CBufferView<IndexT>              abd_vertex_to_body,
    muda::CBufferView<IndexT>              abd_body_is_fixed)
{
    if(descriptors.size() == 0)
        return;

    constexpr int block_dim = 256;
    const auto    grid_dim =
        static_cast<unsigned int>((descriptors.size() + block_dim - 1) / block_dim);
    const cudaStream_t launch_stream = normalize_muda_launch_stream(stream);
    rebuild_socu_native_vertex_descriptors_kernel<<<grid_dim,
                                                    block_dim,
                                                    0,
                                                    launch_stream>>>(
        descriptors,
        old_to_chain,
        horizon,
        block_size,
        epoch,
        fem_vertex_offset,
        fem_vertex_count,
        fem_old_dof_offset,
        fem_vertex_is_fixed,
        abd_vertex_offset,
        abd_vertex_count,
        abd_old_dof_offset,
        abd_body_count,
        abd_vertex_to_body,
        abd_body_is_fixed);
}

void replay_structured_contact_hessian_cache(
    cudaStream_t stream,
    StructuredContactAssemblySink<ActivePolicy::StoreScalar, ActivePolicy::SolveScalar>
        sink)
{
    if(!sink.hessian_cache.replay_valid())
        return;

    constexpr int block_dim = 256;
    const auto    grid_dim = static_cast<unsigned int>(
        (sink.hessian_cache.replay_count + block_dim - 1) / block_dim);
    const cudaStream_t launch_stream = normalize_muda_launch_stream(stream);
    replay_structured_contact_hessian_cache_kernel<<<grid_dim,
                                                     block_dim,
                                                     0,
                                                     launch_stream>>>(sink);
}
}  // namespace uipc::backend::cuda_mixed
