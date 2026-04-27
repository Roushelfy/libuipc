set(CMAKE_CUDA_ARCHITECTURES ${UIPC_CUDA_ARCHITECTURES})
uipc_info("CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}.")
enable_language(CUDA)

# ------------------------------------------------------------------------------
# CUDA Backend
# ------------------------------------------------------------------------------
uipc_add_backend(cuda_mixed)

# basic setup
target_link_libraries(cuda_mixed PUBLIC
    muda::muda
    cublas
    cusparse
    cusolver
    uipc_geometry
)

if(TARGET socu_native)
    target_link_libraries(cuda_mixed PUBLIC socu_native)
    target_compile_definitions(cuda_mixed PUBLIC UIPC_WITH_SOCU_NATIVE=1)
else()
    target_compile_definitions(cuda_mixed PUBLIC UIPC_WITH_SOCU_NATIVE=0)
endif()

# add CCCL include directory for CUDA 13.0 compatibility
if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    target_include_directories(cuda_mixed PUBLIC ${CUDAToolkit_INCLUDE_DIRS}/cccl)
endif()

target_compile_features(cuda_mixed PUBLIC cxx_std_20)
target_include_directories(cuda_mixed PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(cuda_mixed PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${UIPC_CUDA_ARCHITECTURES}
    CUDA_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
)

target_compile_options(cuda_mixed PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --diag-suppress=27,174,221,997,1388,1394,1866>
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>
)

string(TOUPPER "${UIPC_CUDA_MIXED_PRECISION_LEVEL}" _UIPC_MIXED_LEVEL_UPPER)
target_compile_definitions(cuda_mixed PRIVATE "UIPC_MIXED_LEVEL_${_UIPC_MIXED_LEVEL_UPPER}=1")
if(UIPC_WITH_NVTX)
    target_compile_definitions(cuda_mixed PRIVATE UIPC_WITH_NVTX=1)
endif()
if(UIPC_BUILD_TESTS)
    target_compile_definitions(cuda_mixed PRIVATE UIPC_ENABLE_MUDA_MEMORY_TRACKING=1)
endif()

# all files in this directory
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cu" "*.h" "*.hpp" "*.cpp" "*.inl")

target_sources(cuda_mixed PRIVATE ${SOURCES})
uipc_target_set_relative_source_file(cuda_mixed)

# ------------------------------------------------------------------------------
# setup source group for the IDE
# ------------------------------------------------------------------------------
file(GLOB_RECURSE SOURCE_GROUP_FILES CONFIGURE_DEPENDS "*.h" "*.hpp" "*.cpp" "*.cu" "*.inl")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${SOURCE_GROUP_FILES})
