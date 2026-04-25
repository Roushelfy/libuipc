enable_language(CUDA)
file(GLOB_RECURSE SOURCE "*.cpp" "*.h" "*.cu" "*.inl")
uipc_add_test(backend_cuda_mixed ${SOURCE})
target_link_libraries(backend_cuda_mixed PRIVATE uipc::backend::cuda_mixed)
target_include_directories(backend_cuda_mixed PRIVATE ${PROJECT_SOURCE_DIR}/src)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(UIPC_GITHUB_ACTIONS)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    else()
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
endif()

string(TOUPPER "${UIPC_CUDA_MIXED_PRECISION_LEVEL}" _UIPC_MIXED_LEVEL_UPPER)
target_compile_definitions(backend_cuda_mixed PRIVATE
    "UIPC_MIXED_LEVEL_${_UIPC_MIXED_LEVEL_UPPER}=1")

set_target_properties(backend_cuda_mixed PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    CUDA_INCLUDE_WHAT_YOU_USE ON
)

uipc_register_catch_test(backend_cuda_mixed_contract backend_cuda_mixed "[cuda_mixed][contract]")

if(BUILD_TESTING)
    add_test(
        NAME backend_cuda_mixed_source_contract_scan
        COMMAND ${UIPC_PYTHON_EXECUTABLE_PATH} ${CMAKE_CURRENT_LIST_DIR}/scan_precision_contracts.py
    )
    set_tests_properties(backend_cuda_mixed_source_contract_scan PROPERTIES
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
