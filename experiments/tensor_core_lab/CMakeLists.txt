enable_language(CUDA)

find_package(Eigen3 REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS cublas cublasLt)

set(TCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(TCL_HOST_ONLY_TEST_SOURCES
    tests/abd12_factorize_host.cpp
    tests/abd12_inverse_host.cpp
    tests/abd12_solve_host.cpp
    tests/mas48_factorize_host.cpp
    tests/mas48_inverse_host.cpp
    tests/mas48_solve_host.cpp
)
set(TCL_HOST_ONLY_BENCH_SOURCES
    bench/abd12_bench_host.cpp
    bench/mas48_bench_host.cpp
)

set(TCL_CORE_SOURCES
    src/common/device_info.cu
    src/common/event_timer.cu
    src/common/json_writer.cpp
    src/common/metrics.cpp
    src/common/padding.cpp
    src/common/rng.cpp
    src/backends/fp64_ref.cu
    src/backends/fp32_simt.cu
    src/backends/tc32_tf32.cu
    src/generators/abd12_assemble_cases.cpp
    src/generators/abd12_cases.cpp
    src/generators/fem12_local_cases.cpp
    src/generators/joint24_local_cases.cpp
    src/generators/mas48_cases.cpp
    src/registry/case_registry.cpp
    src/registry/op_registry.cpp
    src/runner.cu
    src/ops/abd12_assemble.cu
    src/ops/chol_factor.cu
    src/ops/fem12_assemble.cu
    src/ops/inverse_from_factor.cu
    src/ops/joint24_assemble.cu
    src/ops/solve_from_factor.cu
)

set(TCL_CORE_HEADERS
    include/tcl/backend_api.h
    include/tcl/case_spec.h
    include/tcl/device_info.h
    include/tcl/metrics.h
    include/tcl/mode.h
    include/tcl/op_kind.h
    include/tcl/registry.h
    include/tcl/result_record.h
    include/tcl/runner.h
    include/tcl/tensor_shape.h
    src/common/cuda_check.h
    src/common/event_timer.h
    src/common/json_writer.h
    src/common/padding.h
    src/common/rng.h
    src/ops/chol_ops.h
)

add_library(tensor_core_lab_core STATIC)
target_sources(tensor_core_lab_core PRIVATE ${TCL_CORE_SOURCES} ${TCL_CORE_HEADERS})
target_include_directories(tensor_core_lab_core PUBLIC ${TCL_INCLUDE_DIR})
target_link_libraries(tensor_core_lab_core
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        Eigen3::Eigen
)
target_compile_features(tensor_core_lab_core PUBLIC cxx_std_20)
set_target_properties(tensor_core_lab_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${UIPC_CUDA_ARCHITECTURES}
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
uipc_target_set_output_directory(tensor_core_lab_core)
uipc_target_set_relative_source_file(tensor_core_lab_core)
set_property(TARGET tensor_core_lab_core PROPERTY FOLDER "experiments/tensor_core_lab")

add_executable(tensor_core_lab_test
    src/cli/test_main.cpp
    ${TCL_HOST_ONLY_TEST_SOURCES}
    tests/smoke_device.cu
    tests/abd12_assemble.cu
    tests/fem12_assemble.cu
    tests/joint24_assemble.cu
    tests/unsupported_arch.cu
)
target_link_libraries(tensor_core_lab_test PRIVATE tensor_core_lab_core Catch2::Catch2)
target_include_directories(tensor_core_lab_test PRIVATE ${TCL_INCLUDE_DIR})
target_compile_features(tensor_core_lab_test PRIVATE cxx_std_20)
set_target_properties(tensor_core_lab_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${UIPC_CUDA_ARCHITECTURES}
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    OUTPUT_NAME "tensor_core_lab_test"
)
uipc_target_set_output_directory(tensor_core_lab_test)
uipc_target_set_relative_source_file(tensor_core_lab_test)
set_property(TARGET tensor_core_lab_test PROPERTY FOLDER "experiments/tensor_core_lab")

add_executable(tensor_core_lab_bench
    ${TCL_HOST_ONLY_BENCH_SOURCES}
    src/cli/bench_main.cpp
    bench/abd12_assemble_bench.cu
    bench/fem12_local_bench.cu
    bench/joint24_local_bench.cu
)
target_link_libraries(tensor_core_lab_bench PRIVATE tensor_core_lab_core benchmark::benchmark)
target_include_directories(tensor_core_lab_bench PRIVATE ${TCL_INCLUDE_DIR})
target_compile_features(tensor_core_lab_bench PRIVATE cxx_std_20)
set_target_properties(tensor_core_lab_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${UIPC_CUDA_ARCHITECTURES}
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    OUTPUT_NAME "tensor_core_lab_bench"
)
uipc_target_set_output_directory(tensor_core_lab_bench)
uipc_target_set_relative_source_file(tensor_core_lab_bench)
set_property(TARGET tensor_core_lab_bench PROPERTY FOLDER "experiments/tensor_core_lab")
