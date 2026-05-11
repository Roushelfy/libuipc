#include <sim_engine.h>

#ifndef UIPC_CUDA_MIXED_SOCU_BUILD_AL_PIPELINE
#define UIPC_CUDA_MIXED_SOCU_BUILD_AL_PIPELINE 0
#endif

namespace uipc::backend::cuda_mixed
{
void SimEngine::do_advance()
{
    switch(m_pipeline_type)
    {
        case SimEngine::PipelineType::Basic:
            advance();
            break;
        case SimEngine::PipelineType::AugmentedLagrangian:
#if UIPC_CUDA_MIXED_SOCU_BUILD_AL_PIPELINE
            advance_AL();
#else
            UIPC_ERROR_WITH_LOCATION(
                "The AL-IPC pipeline is disabled in this cuda_mixed_socu build");
#endif
            break;
        default:
            UIPC_ERROR_WITH_LOCATION("Unknown pipeline type");
            break;
    }
}
}  // namespace uipc::backend::cuda_mixed
