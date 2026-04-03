#pragma once

#include <cuda_runtime.h>

namespace uipc::tensor_core_lab
{
class EventTimer
{
  public:
    EventTimer();
    ~EventTimer();

    EventTimer(const EventTimer&)            = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    void   start();
    double stop_elapsed_ms();

  private:
    cudaEvent_t m_start = nullptr;
    cudaEvent_t m_stop  = nullptr;
};
}  // namespace uipc::tensor_core_lab
