#include <utils/structured_contact_assembly_sink.h>

#include <muda/launch/parallel_for.h>

namespace uipc::backend::cuda_mixed
{
void replay_structured_contact_hessian_cache(
    cudaStream_t stream,
    StructuredContactAssemblySink<ActivePolicy::StoreScalar, ActivePolicy::SolveScalar>
        sink)
{
    if(!sink.hessian_cache.replay_valid())
        return;

    muda::ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(static_cast<int>(sink.hessian_cache.replay_count),
               [sink,
                records = sink.hessian_cache.records.viewer().name("contact_hessian_cache")]
                   __device__(int i) mutable
               {
                   const auto& record = records(i);
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
               });
}
}  // namespace uipc::backend::cuda_mixed
