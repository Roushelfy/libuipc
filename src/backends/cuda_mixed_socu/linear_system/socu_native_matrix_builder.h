#pragma once

#include <uipc/common/type_define.h>

#include <cuda_runtime.h>
#include <muda/atomic.h>
#include <muda/buffer/buffer_view.h>
#include <muda/buffer/device_buffer.h>

#include <stdexcept>
#include <vector>

namespace uipc::backend::cuda_mixed
{
struct SocuNativeOffdiagLevel
{
    IndexT stride       = 0;
    IndexT block_offset = 0;
    IndexT block_count  = 0;
};

struct SocuNativeStorageLayout
{
    SizeT block_count               = 0;
    SizeT block_size                = 0;
    SizeT nrhs                      = 0;
    SizeT diag_block_count          = 0;
    SizeT first_offdiag_block_count = 0;
    SizeT offdiag_block_count       = 0;
    SizeT rhs_block_count           = 0;
    SizeT diag_element_count        = 0;
    SizeT offdiag_element_count     = 0;
    SizeT rhs_element_count         = 0;
    IndexT recursive_iterations     = 0;
    std::vector<SocuNativeOffdiagLevel> offdiag_levels;
};

inline IndexT socu_native_recursive_iterations(SizeT block_count)
{
    if(block_count == 0)
        throw std::invalid_argument("SOCU native storage requires at least one block");

    IndexT iterations = 0;
    while(block_count != 0)
    {
        ++iterations;
        block_count >>= 1;
    }
    return iterations;
}

inline IndexT socu_native_recursive_offdiag_count(SizeT block_count, SizeT stride)
{
    if(block_count == 0 || stride == 0)
        throw std::invalid_argument("SOCU native offdiag layout fields must be positive");
    const SizeT count = block_count / stride;
    return count == 0 ? IndexT{-1} : static_cast<IndexT>(count - 1);
}

inline SocuNativeStorageLayout
make_socu_native_storage_layout(SizeT block_count, SizeT block_size, SizeT nrhs = 1)
{
    if(block_count == 0 || block_size == 0 || nrhs == 0)
        throw std::invalid_argument("SOCU native storage dimensions must be positive");

    SocuNativeStorageLayout layout;
    layout.block_count               = block_count;
    layout.block_size                = block_size;
    layout.nrhs                      = nrhs;
    layout.diag_block_count          = block_count;
    layout.rhs_block_count           = block_count;
    layout.first_offdiag_block_count = block_count > 0 ? block_count - 1 : 0;
    layout.recursive_iterations      = socu_native_recursive_iterations(block_count);
    layout.offdiag_levels.reserve(static_cast<SizeT>(layout.recursive_iterations));

    IndexT block_offset = 0;
    SizeT  stride       = 1;
    for(IndexT level = 0; level < layout.recursive_iterations; ++level)
    {
        const IndexT block_level_count =
            socu_native_recursive_offdiag_count(block_count, stride);
        layout.offdiag_levels.push_back(
            SocuNativeOffdiagLevel{static_cast<IndexT>(stride),
                                   block_offset,
                                   block_level_count});
        if(block_level_count > 0)
            layout.offdiag_block_count += static_cast<SizeT>(block_level_count);
        block_offset += block_level_count;
        stride <<= 1;
    }

    const SizeT block_elements = block_size * block_size;
    layout.diag_element_count = layout.diag_block_count * block_elements;
    layout.offdiag_element_count = layout.offdiag_block_count * block_elements;
    layout.rhs_element_count = layout.rhs_block_count * block_size * nrhs;
    return layout;
}

inline SizeT socu_native_offdiag_storage_block(
    const SocuNativeStorageLayout& layout,
    SizeT                          level,
    SizeT                          level_block)
{
    if(level >= layout.offdiag_levels.size())
        throw std::out_of_range("SOCU native offdiag level out of range");

    const auto& level_layout = layout.offdiag_levels[level];
    if(level_block >= static_cast<SizeT>(level_layout.block_count))
        throw std::out_of_range("SOCU native offdiag block out of range");

    return static_cast<SizeT>(level_layout.block_offset) + level_block;
}

struct SocuNativeBlockMeta
{
    IndexT old_dof_begin     = -1;
    IndexT active_lane_count = 0;
    IndexT padding_lane_begin = 0;
    IndexT padding_lane_count = 0;
    IndexT ordering_epoch     = 0;
};

template <typename Scalar>
struct SocuNativeMatrixSnapshot
{
    SocuNativeStorageLayout      layout;
    std::vector<Scalar>          D;
    std::vector<Scalar>          E;
    std::vector<Scalar>          rhs;
    std::vector<SocuNativeBlockMeta> blocks;
};

template <typename Scalar>
struct SocuNativeMatrixView
{
    muda::BufferView<Scalar> D;
    muda::BufferView<Scalar> E;
    muda::BufferView<Scalar> rhs;
    muda::BufferView<SocuNativeBlockMeta> blocks;
    SizeT block_count               = 0;
    SizeT block_size                = 0;
    SizeT nrhs                      = 0;
    SizeT first_offdiag_block_count = 0;
    SizeT offdiag_block_count       = 0;

    MUDA_GENERIC bool valid() const noexcept
    {
        return D.data() != nullptr && rhs.data() != nullptr && block_count != 0
               && block_size != 0 && nrhs != 0;
    }

    MUDA_GENERIC SizeT diag_index(SizeT block, SizeT row, SizeT col) const noexcept
    {
        return (block * block_size + row) * block_size + col;
    }

    MUDA_GENERIC SizeT first_offdiag_index(SizeT left_block,
                                           SizeT row,
                                           SizeT col) const noexcept
    {
        return (left_block * block_size + row) * block_size + col;
    }

    MUDA_GENERIC SizeT offdiag_index(SizeT offdiag_block,
                                     SizeT row,
                                     SizeT col) const noexcept
    {
        return (offdiag_block * block_size + row) * block_size + col;
    }

    MUDA_GENERIC SizeT rhs_index(SizeT block, SizeT lane, SizeT rhs_col) const noexcept
    {
        return (block * block_size + lane) * nrhs + rhs_col;
    }

    MUDA_GENERIC bool valid_block_lane(SizeT block, SizeT lane) const noexcept
    {
        return block < block_count && lane < block_size;
    }

    MUDA_GENERIC bool valid_block_entry(SizeT block,
                                        SizeT row,
                                        SizeT col) const noexcept
    {
        return block < block_count && row < block_size && col < block_size;
    }

    MUDA_DEVICE void add_diag_scalar(SizeT block,
                                     SizeT row,
                                     SizeT col,
                                     Scalar value) const noexcept
    {
        if(!valid_block_entry(block, row, col))
            return;
        const SizeT index = diag_index(block, row, col);
        if(index < D.size())
            muda::atomic_add(D.data(index), value);
    }

    MUDA_DEVICE void add_first_offdiag_scalar(SizeT left_block,
                                              SizeT row,
                                              SizeT col,
                                              Scalar value) const noexcept
    {
        if(left_block >= first_offdiag_block_count || row >= block_size
           || col >= block_size)
            return;
        const SizeT index = first_offdiag_index(left_block, row, col);
        if(index < E.size())
            muda::atomic_add(E.data(index), value);
    }

    MUDA_DEVICE void add_offdiag_scalar(SizeT offdiag_block,
                                        SizeT row,
                                        SizeT col,
                                        Scalar value) const noexcept
    {
        if(offdiag_block >= offdiag_block_count || row >= block_size
           || col >= block_size)
            return;
        const SizeT index = offdiag_index(offdiag_block, row, col);
        if(index < E.size())
            muda::atomic_add(E.data(index), value);
    }

    MUDA_DEVICE void add_rhs_scalar(SizeT block,
                                    SizeT lane,
                                    SizeT rhs_col,
                                    Scalar value) const noexcept
    {
        if(!valid_block_lane(block, lane) || rhs_col >= nrhs)
            return;
        const SizeT index = rhs_index(block, lane, rhs_col);
        if(index < rhs.size())
            muda::atomic_add(rhs.data(index), value);
    }

    MUDA_DEVICE void add_diag_block3_row_major(SizeT block,
                                               SizeT row_begin,
                                               SizeT col_begin,
                                               const Scalar* values) const noexcept
    {
#pragma unroll
        for(SizeT row = 0; row < 3; ++row)
        {
#pragma unroll
            for(SizeT col = 0; col < 3; ++col)
            {
                add_diag_scalar(block,
                                row_begin + row,
                                col_begin + col,
                                values[row * 3 + col]);
            }
        }
    }

    MUDA_DEVICE void add_first_offdiag_block3_row_major(
        SizeT left_block,
        SizeT row_begin,
        SizeT col_begin,
        const Scalar* values) const noexcept
    {
#pragma unroll
        for(SizeT row = 0; row < 3; ++row)
        {
#pragma unroll
            for(SizeT col = 0; col < 3; ++col)
            {
                add_first_offdiag_scalar(left_block,
                                         row_begin + row,
                                         col_begin + col,
                                         values[row * 3 + col]);
            }
        }
    }

    MUDA_DEVICE void add_offdiag_block3_row_major(SizeT offdiag_block,
                                                  SizeT row_begin,
                                                  SizeT col_begin,
                                                  const Scalar* values) const noexcept
    {
#pragma unroll
        for(SizeT row = 0; row < 3; ++row)
        {
#pragma unroll
            for(SizeT col = 0; col < 3; ++col)
            {
                add_offdiag_scalar(offdiag_block,
                                   row_begin + row,
                                   col_begin + col,
                                   values[row * 3 + col]);
            }
        }
    }

    MUDA_DEVICE void write_block_meta(SizeT block,
                                      SocuNativeBlockMeta meta) const noexcept
    {
        if(block < blocks.size())
            *blocks.data(block) = meta;
    }
};

template <typename Scalar>
class SocuNativeMatrixBuilder
{
  public:
    using View     = SocuNativeMatrixView<Scalar>;
    using Snapshot = SocuNativeMatrixSnapshot<Scalar>;

    void reserve(SizeT block_count, SizeT block_size, SizeT nrhs = 1)
    {
        m_layout = make_socu_native_storage_layout(block_count, block_size, nrhs);
        m_D.resize(m_layout.diag_element_count);
        m_E.resize(m_layout.offdiag_element_count);
        m_rhs.resize(m_layout.rhs_element_count);
        m_blocks.resize(m_layout.block_count);
    }

    void clear(cudaStream_t stream = nullptr)
    {
        memset_async(m_D.data(), m_D.size(), stream);
        memset_async(m_E.data(), m_E.size(), stream);
        memset_async(m_rhs.data(), m_rhs.size(), stream);
    }

    const SocuNativeStorageLayout& layout() const noexcept { return m_layout; }
    SizeT block_count() const noexcept { return m_layout.block_count; }
    SizeT block_size() const noexcept { return m_layout.block_size; }
    SizeT nrhs() const noexcept { return m_layout.nrhs; }

    Scalar* diag_data() noexcept { return m_D.data(); }
    Scalar* offdiag_data() noexcept { return m_E.data(); }
    Scalar* rhs_data() noexcept { return m_rhs.data(); }
    const Scalar* diag_data() const noexcept { return m_D.data(); }
    const Scalar* offdiag_data() const noexcept { return m_E.data(); }
    const Scalar* rhs_data() const noexcept { return m_rhs.data(); }

    View view() noexcept
    {
        return View{m_D.view(),
                    m_E.view(),
                    m_rhs.view(),
                    m_blocks.view(),
                    m_layout.block_count,
                    m_layout.block_size,
                    m_layout.nrhs,
                    m_layout.first_offdiag_block_count,
                    m_layout.offdiag_block_count};
    }

    void set_block_metadata(const std::vector<SocuNativeBlockMeta>& metadata,
                            cudaStream_t stream = nullptr)
    {
        if(metadata.size() != m_layout.block_count)
            throw std::invalid_argument("SOCU native block metadata size mismatch");
        if(!metadata.empty())
        {
            const auto bytes = metadata.size() * sizeof(SocuNativeBlockMeta);
            const cudaError_t error = cudaMemcpyAsync(
                m_blocks.data(),
                metadata.data(),
                bytes,
                cudaMemcpyHostToDevice,
                stream);
            if(error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
        }
    }

    Snapshot snapshot(cudaStream_t stream = nullptr) const
    {
        Snapshot result;
        result.layout = m_layout;
        result.D.resize(m_D.size());
        result.E.resize(m_E.size());
        result.rhs.resize(m_rhs.size());
        result.blocks.resize(m_blocks.size());
        copy_to_host(result.D.data(), m_D.data(), m_D.size(), stream);
        copy_to_host(result.E.data(), m_E.data(), m_E.size(), stream);
        copy_to_host(result.rhs.data(), m_rhs.data(), m_rhs.size(), stream);
        copy_to_host(result.blocks.data(), m_blocks.data(), m_blocks.size(), stream);
        const cudaError_t error = cudaStreamSynchronize(stream);
        if(error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        return result;
    }

  private:
    template <typename T>
    static void memset_async(T* data, SizeT count, cudaStream_t stream)
    {
        if(data == nullptr || count == 0)
            return;
        const cudaError_t error = cudaMemsetAsync(data, 0, count * sizeof(T), stream);
        if(error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }

    template <typename T>
    static void copy_to_host(T* dst, const T* src, SizeT count, cudaStream_t stream)
    {
        if(count == 0)
            return;
        if(dst == nullptr || src == nullptr)
            throw std::runtime_error("SOCU native snapshot source or destination is null");
        const cudaError_t error =
            cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
        if(error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }

    SocuNativeStorageLayout m_layout;
    muda::DeviceBuffer<Scalar> m_D;
    muda::DeviceBuffer<Scalar> m_E;
    muda::DeviceBuffer<Scalar> m_rhs;
    muda::DeviceBuffer<SocuNativeBlockMeta> m_blocks;
};
}  // namespace uipc::backend::cuda_mixed
