#pragma once

#include <linear_system/global_linear_system.h>
#include <linear_system/socu_native_matrix_builder.h>
#include <linear_system/structured_chain_provider.h>

#include <cuda_runtime.h>
#include <muda/atomic.h>
#include <muda/launch/parallel_for.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#ifndef UIPC_WITH_SOCU_NATIVE
#define UIPC_WITH_SOCU_NATIVE 0
#endif

#if UIPC_WITH_SOCU_NATIVE
#include <socu_native/solver.h>
#endif

namespace uipc::backend::cuda_mixed::socu_approx
{
#if UIPC_WITH_SOCU_NATIVE
MUDA_DEVICE __forceinline__ void atomic_add_double(double* address,
                                                   double  value) noexcept
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    auto address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    auto old            = *address_as_ull;
    unsigned long long int assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while(assumed != old);
#else
    atomicAdd(address, value);
#endif
}

template <typename SolveScalar>
SocuNativeMatrixView<SolveScalar> make_socu_native_matrix_view(
    StructuredChainShape          shape,
    muda::BufferView<SolveScalar> diag,
    muda::BufferView<SolveScalar> off_diag,
    muda::BufferView<SolveScalar> rhs) noexcept
{
    const SizeT block_elements = shape.block_size * shape.block_size;
    const SizeT offdiag_block_count =
        block_elements == 0 ? SizeT{0} : off_diag.size() / block_elements;
    return SocuNativeMatrixView<SolveScalar>{diag,
                                             off_diag,
                                             rhs,
                                             {},
                                             shape.horizon,
                                             shape.block_size,
                                             shape.nrhs,
                                             shape.horizon > 0
                                                 ? shape.horizon - 1
                                                 : SizeT{0},
                                             offdiag_block_count};
}

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

template <typename StoreScalar, typename SolveScalar>
void initialize_socu_native_diag_rhs_workspace(
    cudaStream_t                                stream,
    StructuredChainShape                        shape,
    GlobalLinearSystem::CDenseVectorView        b,
    muda::BufferView<SolveScalar>               diag,
    muda::BufferView<SolveScalar>               off_diag,
    muda::BufferView<SolveScalar>               rhs,
    muda::BufferView<SolveScalar>               rhs_original,
    muda::CBufferView<IndexT>                   chain_to_old,
    muda::CBufferView<SocuNativeDofDescriptor>  dof_descriptors,
    double                                      damping_shift)
{
    if(static_cast<SizeT>(b.size()) > dof_descriptors.size())
        throw std::runtime_error(
            "SOCU native diagonal/RHS init requires one DoF descriptor per global RHS entry");

    const auto diag_bytes = diag.size() * sizeof(SolveScalar);
    const auto off_bytes  = off_diag.size() * sizeof(SolveScalar);
    const auto rhs_bytes  = rhs.size() * sizeof(SolveScalar);
    if(diag_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(diag.data(), 0, diag_bytes, stream));
    if(off_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(off_diag.data(), 0, off_bytes, stream));
    if(rhs_bytes)
        SOCU_NATIVE_CHECK_CUDA(cudaMemsetAsync(rhs.data(), 0, rhs_bytes, stream));

    auto native = make_socu_native_matrix_view(shape, diag, off_diag, rhs);
    const SizeT chain_scalar_count = shape.horizon * shape.block_size;
    muda::ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(static_cast<int>(chain_scalar_count),
               [shape,
                native,
                chain_to_old = chain_to_old.cviewer().name("chain_to_old"),
                damping_shift = static_cast<SolveScalar>(damping_shift)] __device__(int chain) mutable
               {
                   const SizeT block = static_cast<SizeT>(chain) / shape.block_size;
                   const SizeT lane  = static_cast<SizeT>(chain) % shape.block_size;

                   if(damping_shift != SolveScalar{0})
                       native.add_diag_scalar(block, lane, lane, damping_shift);

                   const IndexT old = chain_to_old(chain);
                   if(old < 0)
                       native.add_diag_scalar(block, lane, lane, SolveScalar{1});
               });

    muda::ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(b.size(),
               [native,
                b = b.cviewer().name("global_b"),
                dof_descriptors = dof_descriptors.cviewer().name(
                    "socu_native_dof_descriptors")] __device__(int old) mutable
               {
                   const auto value = static_cast<SolveScalar>(b(old));
                   native.add_rhs_scalar(dof_descriptors(old), SizeT{0}, value);
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
void compare_socu_native_diag_rhs_workspace(
    cudaStream_t                         stream,
    muda::CBufferView<SolveScalar>       reference_diag,
    muda::CBufferView<SolveScalar>       reference_off_diag,
    muda::CBufferView<SolveScalar>       reference_rhs,
    muda::CBufferView<SolveScalar>       actual_diag,
    muda::CBufferView<SolveScalar>       actual_off_diag,
    muda::CBufferView<SolveScalar>       actual_rhs,
    muda::BufferView<double>             diff_sums,
    muda::BufferView<IndexT>             mismatch_count,
    double                               abs_tolerance,
    double                               rel_tolerance)
{
    if(reference_diag.size() != actual_diag.size()
       || reference_off_diag.size() != actual_off_diag.size()
       || reference_rhs.size() != actual_rhs.size())
    {
        throw std::runtime_error(
            "SOCU native diagonal/RHS diff requires matching buffer sizes");
    }
    if(diff_sums.size() < 3 || mismatch_count.size() < 1)
        throw std::runtime_error(
            "SOCU native diagonal/RHS diff requires at least 3 sums and 1 status slot");

    SOCU_NATIVE_CHECK_CUDA(
        cudaMemsetAsync(diff_sums.data(),
                        0,
                        diff_sums.size() * sizeof(double),
                        stream));
    SOCU_NATIVE_CHECK_CUDA(
        cudaMemsetAsync(mismatch_count.data(),
                        0,
                        mismatch_count.size() * sizeof(IndexT),
                        stream));

    const SizeT diag_size = reference_diag.size();
    const SizeT off_size  = reference_off_diag.size();
    const SizeT rhs_size  = reference_rhs.size();
    const SizeT total_size = diag_size + off_size + rhs_size;
    muda::ParallelFor(256, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(static_cast<int>(total_size),
               [diag_size,
                off_size,
                abs_tolerance,
                rel_tolerance,
                reference_diag,
                reference_off_diag,
                reference_rhs,
                actual_diag,
                actual_off_diag,
                actual_rhs,
                diff_sums,
                mismatch_count] __device__(int linear) mutable
               {
                   const SizeT index = static_cast<SizeT>(linear);
                   double      reference = 0.0;
                   double      actual = 0.0;
                   SizeT       channel = 0;
                   if(index < diag_size)
                   {
                       reference = static_cast<double>(reference_diag[index]);
                       actual    = static_cast<double>(actual_diag[index]);
                       channel   = 0;
                   }
                   else if(index < diag_size + off_size)
                   {
                       const SizeT local = index - diag_size;
                       reference = static_cast<double>(reference_off_diag[local]);
                       actual    = static_cast<double>(actual_off_diag[local]);
                       channel   = 1;
                   }
                   else
                   {
                       const SizeT local = index - diag_size - off_size;
                       reference = static_cast<double>(reference_rhs[local]);
                       actual    = static_cast<double>(actual_rhs[local]);
                       channel   = 2;
                   }

                   const double diff = fabs(actual - reference);
                   const double scale = fmax(fabs(reference), fabs(actual));
                   atomic_add_double(diff_sums.data(channel), diff);
                   if(diff > abs_tolerance + rel_tolerance * scale)
                       muda::atomic_add(mismatch_count.data(0), IndexT{1});
               });
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
                       atomic_add_double(sums.data() + 3, 1.0);
                       atomic_add_double(sums.data() + 4, 1.0);
                       return;
                   }

                   atomic_add_double(sums.data() + 0, rhs_i * rhs_i);

                   if(!isfinite(x_i))
                   {
                       atomic_add_double(sums.data() + 4, 1.0);
                       return;
                   }

                   atomic_add_double(sums.data() + 1, x_i * x_i);
                   atomic_add_double(sums.data() + 2, rhs_i * x_i);
               });
}

template <typename SolveScalar>
void finalize_structured_direction_light_status(cudaStream_t             stream,
                                                muda::CBufferView<double> sums,
                                                muda::BufferView<IndexT> status,
                                                double direction_min_abs,
                                                double direction_min_rel,
                                                double rhs_zero_abs,
                                                double descent_eta)
{
    muda::ParallelFor(1, 0, stream)
        .file_line(__FILE__, __LINE__)
        .apply(1,
               [sums = sums.cviewer().name("sums"),
                status = status.viewer().name("status"),
                direction_min_abs,
                direction_min_rel,
                rhs_zero_abs,
                descent_eta] __device__(int) mutable
               {
                   const double gradient_norm = sqrt(sums(0));
                   const double direction_norm = sqrt(sums(1));
                   const double descent_dot = -sums(2);
                   const double p_threshold =
                       fmax(direction_min_abs, direction_min_rel * gradient_norm);
                   const double rhs_zero_threshold =
                       fmax(rhs_zero_abs, 1000.0 * direction_min_abs);
                   const double tiny_rhs_threshold =
                       sqrt(static_cast<double>(
                           std::numeric_limits<SolveScalar>::epsilon()));
                   const double near_zero_direction_rhs_threshold =
                       fmax(fmax(rhs_zero_threshold,
                                 10000.0 * direction_min_abs),
                            tiny_rhs_threshold);

                   const bool rhs_finite =
                       sums(3) == 0.0 && isfinite(gradient_norm);
                   if(rhs_finite && gradient_norm <= rhs_zero_threshold)
                   {
                       status(0) = 1;
                       return;
                   }

                   const bool finite =
                       sums(4) == 0.0 && isfinite(descent_dot)
                       && isfinite(gradient_norm) && isfinite(direction_norm);
                   const bool nonzero =
                       gradient_norm > 0.0 && direction_norm > p_threshold;
                   const bool descent =
                       descent_dot
                       < -descent_eta * gradient_norm * direction_norm;
                   const bool near_zero_direction =
                       finite && gradient_norm <= near_zero_direction_rhs_threshold
                       && direction_norm <= p_threshold;
                   if(near_zero_direction)
                   {
                       status(0) = 2;
                       return;
                   }

                   status(0) = (finite && nonzero && descent) ? 0 : 3;
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
                   atomic_add_double(sums.data() + 0, rhs_i * rhs_i);
                   atomic_add_double(sums.data() + 1, x_i * x_i);
                   atomic_add_double(sums.data() + 2, rhs_i * x_i);
                   atomic_add_double(sums.data() + 3, res * res);
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
