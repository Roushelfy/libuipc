enable_language(CUDA)

find_package(Eigen3 REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS cublas cublasLt)

set(TCG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(TCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tensor_core_lab/include")
set(TCL_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tensor_core_lab/src/common")
set(TCL_BACKENDS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tensor_core_lab/src/backends")

set(TCG_SHARED_SOURCES
    ${TCL_COMMON_DIR}/device_info.cu
    ${TCL_COMMON_DIR}/event_timer.cu
    ${TCL_COMMON_DIR}/metrics.cpp
    ${TCL_COMMON_DIR}/rng.cpp
    ${TCL_BACKENDS_DIR}/fp64_ref.cu
    ${TCL_BACKENDS_DIR}/fp32_simt.cu
    ${TCL_BACKENDS_DIR}/tc32_tf32.cu
)

set(TCG_SHARED_HEADERS
    ${TCL_INCLUDE_DIR}/tcl/backend_api.h
    ${TCL_INCLUDE_DIR}/tcl/device_info.h
    ${TCL_INCLUDE_DIR}/tcl/metrics.h
    ${TCL_INCLUDE_DIR}/tcl/mode.h
    ${TCL_COMMON_DIR}/cuda_check.h
    ${TCL_COMMON_DIR}/event_timer.h
    ${TCL_COMMON_DIR}/rng.h
)

set(TCG_CORE_SOURCES
    src/generators/gemm_cases.cpp
    src/registry/shape_registry.cpp
    src/runner.cu
)

set(TCG_CORE_HEADERS
    include/tcg/gemm_case.h
    include/tcg/registry.h
    include/tcg/runner.h
)

add_library(tensor_core_gemm_core STATIC)
target_sources(tensor_core_gemm_core
    PRIVATE
        ${TCG_SHARED_SOURCES}
        ${TCG_SHARED_HEADERS}
        ${TCG_CORE_SOURCES}
        ${TCG_CORE_HEADERS}
)
target_include_directories(tensor_core_gemm_core
    PUBLIC
        ${TCG_INCLUDE_DIR}
        ${TCL_INCLUDE_DIR}
    PRIVATE
        ${TCL_COMMON_DIR}
)
target_link_libraries(tensor_core_gemm_core
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        Eigen3::Eigen
)
target_compile_features(tensor_core_gemm_core PUBLIC cxx_std_20)
set_target_properties(tensor_core_gemm_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${UIPC_CUDA_ARCHITECTURES}
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
uipc_target_set_output_directory(tensor_core_gemm_core)
uipc_target_set_relative_source_file(tensor_core_gemm_core)
set_property(TARGET tensor_core_gemm_core PROPERTY FOLDER "experiments/tensor_core_gemm")

add_executable(tensor_core_gemm_test
    src/cli/test_main.cpp
    tests/gemm_cases.cu
)
target_link_libraries(tensor_core_gemm_test PRIVATE tensor_core_gemm_core Catch2::Catch2)
target_include_directories(tensor_core_gemm_test
    PRIVATE
        ${TCG_INCLUDE_DIR}
        ${TCL_INCLUDE_DIR}
        ${TCL_COMMON_DIR}
)
target_compile_features(tensor_core_gemm_test PRIVATE cxx_std_20)
set_target_properties(tensor_core_gemm_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${UIPC_CUDA_ARCHITECTURES}
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    OUTPUT_NAME "tensor_core_gemm_test"
)
uipc_target_set_output_directory(tensor_core_gemm_test)
uipc_target_set_relative_source_file(tensor_core_gemm_test)
set_property(TARGET tensor_core_gemm_test PROPERTY FOLDER "experiments/tensor_core_gemm")

add_executable(tensor_core_gemm_bench
    src/cli/bench_main.cpp
    bench/gemm_bench.cu
)
target_link_libraries(tensor_core_gemm_bench PRIVATE tensor_core_gemm_core benchmark::benchmark)
target_include_directories(tensor_core_gemm_bench
    PRIVATE
        ${TCG_INCLUDE_DIR}
        ${TCL_INCLUDE_DIR}
        ${TCL_COMMON_DIR}
)
target_compile_features(tensor_core_gemm_bench PRIVATE cxx_std_20)
set_target_properties(tensor_core_gemm_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${UIPC_CUDA_ARCHITECTURES}
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    OUTPUT_NAME "tensor_core_gemm_bench"
)
uipc_target_set_output_directory(tensor_core_gemm_bench)
uipc_target_set_relative_source_file(tensor_core_gemm_bench)
set_property(TARGET tensor_core_gemm_bench PROPERTY FOLDER "experiments/tensor_core_gemm")
