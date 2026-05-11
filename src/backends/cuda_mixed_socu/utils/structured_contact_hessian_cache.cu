#include <utils/structured_contact_assembly_sink.h>

namespace uipc::backend::cuda_mixed
{
namespace
{
cudaStream_t normalize_muda_launch_stream(cudaStream_t stream) noexcept
{
    return stream == cudaStreamLegacy ? nullptr : stream;
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
