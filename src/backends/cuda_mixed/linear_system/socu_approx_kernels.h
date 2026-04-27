#pragma once

#include <linear_system/global_linear_system.h>
#include <linear_system/structured_chain_provider.h>

#include <cuda_runtime.h>
#include <muda/atomic.h>
#include <muda/launch/parallel_for.h>

#ifndef UIPC_WITH_SOCU_NATIVE
#define UIPC_WITH_SOCU_NATIVE 0
#endif

#if UIPC_WITH_SOCU_NATIVE
#include <socu_native/solver.h>
#endif

namespace uipc::backend::cuda_mixed::socu_approx
{
#if UIPC_WITH_SOCU_NATIVE
template <typename StoreScalar, typename SolveScalar>
void initialize_structured_workspace(
    cudaStream_t                         stream,
    StructuredChainShape                 shape,
    GlobalLinearSystem::CDenseVectorView b,
    muda::BufferView<SolveScalar>        diag,
    muda::BufferView<SolveScalar>        off_diag,
    muda::BufferView<SolveScalar>        rhs,
    muda::BufferView<SolveScalar>        rhs_original,
    muda::CBufferView<IndexT>            chain_to_old,
    double                               damping_shift)
{
    const auto diag_bytes = diag.size() * sizeof(SolveScalar);
    const auto off_bytes  = off_diag.size() * sizeof(SolveScalar);
    const auto rhs_bytes  = rhs.size() * sizeof(SolveScalar);
    if(diag_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(diag.data(), 0, diag_bytes, stream));
    if(off_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(off_diag.data(), 0, off_bytes, stream));
    if(rhs_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(rhs.data(), 0, rhs_bytes, stream));

    const SizeT chain_scalar_count = shape.horizon * shape.block_size;
    muda::ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(static_cast<int>(chain_scalar_count),
               [shape,
                b = b.cviewer().name("global_b"),
                diag = diag.viewer().name("structured_diag"),
                rhs = rhs.viewer().name("structured_rhs"),
                chain_to_old = chain_to_old.cviewer().name("chain_to_old"),
                damping_shift = static_cast<SolveScalar>(damping_shift)] __device__(int chain) mutable
               {
                   const SizeT block = static_cast<SizeT>(chain) / shape.block_size;
                   const SizeT lane  = static_cast<SizeT>(chain) % shape.block_size;
                   const SizeT diag_index =
                       (block * shape.block_size + lane) * shape.block_size + lane;

                   if(damping_shift != SolveScalar{0})
                       diag(diag_index) += damping_shift;

                   const IndexT old = chain_to_old(chain);
                   if(old >= 0)
                   {
                       rhs(chain) = static_cast<SolveScalar>(b(old));
                   }
                   else
                   {
                       diag(diag_index) += SolveScalar{1};
                   }
               });

    if(rhs_bytes && rhs_original.data() != nullptr)
    {
        SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(rhs_original.data(),
                                               rhs.data(),
                                               rhs_bytes,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
    }
}

template <typename SolveScalar>
void validate_structured_direction_light(cudaStream_t                  stream,
                                         StructuredChainShape          shape,
                                         muda::CBufferView<SolveScalar> rhs_original,
                                         muda::CBufferView<SolveScalar> solution,
                                         muda::CBufferView<IndexT>     chain_to_old,
                                         muda::BufferView<double>      sums)
{
    const auto sum_bytes = sums.size() * sizeof(double);
    if(sum_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(sums.data(), 0, sum_bytes, stream));

    const SizeT chain_scalar_count = shape.horizon * shape.block_size;
    muda::ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(static_cast<int>(chain_scalar_count),
               [rhs = rhs_original.cviewer().name("rhs_original"),
                x = solution.cviewer().name("solution"),
                chain_to_old = chain_to_old.cviewer().name("chain_to_old"),
                sums = sums.viewer().name("sums")] __device__(int chain) mutable
               {
                   if(chain_to_old(chain) < 0)
                       return;

                   const double rhs_i = static_cast<double>(rhs(chain));
                   const double x_i   = static_cast<double>(x(chain));
                   if(!isfinite(rhs_i))
                   {
                       muda::atomic_add(sums.data() + 3, 1.0);
                       muda::atomic_add(sums.data() + 4, 1.0);
                       return;
                   }

                   muda::atomic_add(sums.data() + 0, rhs_i * rhs_i);

                   if(!isfinite(x_i))
                   {
                       muda::atomic_add(sums.data() + 4, 1.0);
                       return;
                   }

                   muda::atomic_add(sums.data() + 1, x_i * x_i);
                   muda::atomic_add(sums.data() + 2, rhs_i * x_i);
               });
}

template <typename SolveScalar>
void validate_structured_direction(cudaStream_t                  stream,
                                   StructuredChainShape          shape,
                                   muda::CBufferView<SolveScalar> diag,
                                   muda::CBufferView<SolveScalar> first_offdiag,
                                   muda::CBufferView<SolveScalar> rhs_original,
                                   muda::CBufferView<SolveScalar> solution,
                                   muda::CBufferView<IndexT>     chain_to_old,
                                   muda::BufferView<double>      sums)
{
    const auto sum_bytes = sums.size() * sizeof(double);
    if(sum_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(sums.data(), 0, sum_bytes, stream));

    const SizeT chain_scalar_count = shape.horizon * shape.block_size;
    const SizeT offdiag_scalar_count = first_offdiag.size();
    muda::ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(static_cast<int>(chain_scalar_count),
               [shape,
                offdiag_scalar_count,
                diag = diag.cviewer().name("diag"),
                first_offdiag = first_offdiag.cviewer().name("first_offdiag"),
                rhs = rhs_original.cviewer().name("rhs_original"),
                x = solution.cviewer().name("solution"),
                chain_to_old = chain_to_old.cviewer().name("chain_to_old"),
                sums = sums.viewer().name("sums")] __device__(int chain) mutable
               {
                   if(chain_to_old(chain) < 0)
                       return;

                   const SizeT block = static_cast<SizeT>(chain) / shape.block_size;
                   const SizeT lane  = static_cast<SizeT>(chain) % shape.block_size;
                   double      Ax    = 0.0;

                   for(SizeT col = 0; col < shape.block_size; ++col)
                   {
                       const SizeT col_chain = block * shape.block_size + col;
                       const SizeT index =
                           (block * shape.block_size + lane) * shape.block_size + col;
                       Ax += static_cast<double>(diag(index))
                             * static_cast<double>(x(col_chain));
                   }

                   if(block + 1 < shape.horizon)
                   {
                       for(SizeT col = 0; col < shape.block_size; ++col)
                       {
                           const SizeT right_chain = (block + 1) * shape.block_size + col;
                           const SizeT index =
                               (block * shape.block_size + col) * shape.block_size + lane;
                           if(index < offdiag_scalar_count)
                               Ax += static_cast<double>(first_offdiag(index))
                                     * static_cast<double>(x(right_chain));
                       }
                   }

                   if(block > 0)
                   {
                       const SizeT left_block = block - 1;
                       for(SizeT col = 0; col < shape.block_size; ++col)
                       {
                           const SizeT left_chain = left_block * shape.block_size + col;
                           const SizeT index =
                               (left_block * shape.block_size + lane) * shape.block_size + col;
                           if(index < offdiag_scalar_count)
                               Ax += static_cast<double>(first_offdiag(index))
                                     * static_cast<double>(x(left_chain));
                       }
                   }

                   const double rhs_i = static_cast<double>(rhs(chain));
                   const double x_i   = static_cast<double>(x(chain));
                   const double res   = Ax - rhs_i;
                   muda::atomic_add(sums.data() + 0, rhs_i * rhs_i);
                   muda::atomic_add(sums.data() + 1, x_i * x_i);
                   muda::atomic_add(sums.data() + 2, rhs_i * x_i);
                   muda::atomic_add(sums.data() + 3, res * res);
               });
}

template <typename SolveScalar>
void scatter_structured_solution(cudaStream_t                       stream,
                                 muda::CBufferView<SolveScalar>     solution,
                                 muda::CBufferView<IndexT>          old_to_chain,
                                 GlobalLinearSystem::SolveDenseVectorView x)
{
    muda::ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(static_cast<int>(old_to_chain.size()),
               [solution = solution.cviewer().name("solution"),
                old_to_chain = old_to_chain.cviewer().name("old_to_chain"),
                x = x.viewer().name("x")] __device__(int old) mutable
               {
                   const IndexT chain = old_to_chain(old);
                   if(chain >= 0)
                       x(old) = solution(chain);
               });
}
#endif

}  // namespace uipc::backend::cuda_mixed::socu_approx
