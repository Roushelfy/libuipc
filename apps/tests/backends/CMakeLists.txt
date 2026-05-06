if(UIPC_WITH_CUDA_BACKEND)
    add_subdirectory(cuda)
endif()

if(UIPC_WITH_CUDA_MIXED_BACKEND)
    add_subdirectory(cuda_mixed)
endif()

if(UIPC_WITH_CUDA_MIXED_SOCU_BACKEND)
    add_subdirectory(cuda_mixed_socu)
endif()
