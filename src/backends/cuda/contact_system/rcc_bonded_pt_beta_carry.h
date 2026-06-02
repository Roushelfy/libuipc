#pragma once

#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>
#include <type_define.h>

namespace uipc::backend::cuda
{
struct RCCBondedPTBetaCarryEntry
{
    U64   key = 0;
    Float beta = 0.0;
};

class RCCBondedPTBetaCarryScratch
{
  public:
    void merge_released_beta(muda::DeviceBuffer<U64>& prev_keys,
                             muda::DeviceBuffer<Float>& prev_beta,
                             muda::CBufferView<U64> released_keys,
                             muda::CBufferView<Float> released_beta);

  private:
    muda::DeviceBuffer<RCCBondedPTBetaCarryEntry> m_release_entries;
    muda::DeviceBuffer<RCCBondedPTBetaCarryEntry> m_filtered_release_entries;
    muda::DeviceBuffer<U64>                       m_merge_keys;
    muda::DeviceBuffer<Float>                     m_merge_beta;
    muda::DeviceVar<IndexT>                       m_filtered_count;
};
}  // namespace uipc::backend::cuda
