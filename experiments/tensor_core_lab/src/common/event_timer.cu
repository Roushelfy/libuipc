#include "event_timer.h"

#include "cuda_check.h"

namespace uipc::tensor_core_lab
{
EventTimer::EventTimer()
{
    TCL_CUDA_CHECK(cudaEventCreate(&m_start));
    TCL_CUDA_CHECK(cudaEventCreate(&m_stop));
}

EventTimer::~EventTimer()
{
    if(m_stop)
        cudaEventDestroy(m_stop);
    if(m_start)
        cudaEventDestroy(m_start);
}

void EventTimer::start()
{
    TCL_CUDA_CHECK(cudaEventRecord(m_start));
}

double EventTimer::stop_elapsed_ms()
{
    TCL_CUDA_CHECK(cudaEventRecord(m_stop));
    TCL_CUDA_CHECK(cudaEventSynchronize(m_stop));

    float elapsed_ms = 0.0f;
    TCL_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, m_start, m_stop));
    return static_cast<double>(elapsed_ms);
}
}  // namespace uipc::tensor_core_lab
