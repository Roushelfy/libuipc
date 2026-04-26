find_package(nlohmann_json REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(CUDAToolkit QUIET COMPONENTS cusolver cusparse)

set(SOL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(SOL_CORE_SOURCES
    src/contact.cpp
    src/graph.cpp
    src/io.cpp
    src/ordering.cpp
    src/presets.cpp
    src/reorder.cpp
)

set(SOL_CORE_HEADERS
    include/sol/contact.h
    include/sol/graph.h
    include/sol/io.h
    include/sol/ordering.h
    include/sol/presets.h
    include/sol/reorder.h
    include/sol/types.h
)

add_library(socu_ordering_lab_core STATIC)
set_target_properties(socu_ordering_lab_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_sources(socu_ordering_lab_core PRIVATE ${SOL_CORE_SOURCES} ${SOL_CORE_HEADERS})
target_include_directories(socu_ordering_lab_core PUBLIC ${SOL_INCLUDE_DIR})
target_link_libraries(socu_ordering_lab_core
    PUBLIC
        metis
        uipc::core
        uipc::geometry
        nlohmann_json::nlohmann_json
        fmt::fmt
)
if(CUDAToolkit_FOUND AND TARGET CUDA::cusolver AND TARGET CUDA::cusparse)
    target_compile_definitions(socu_ordering_lab_core PUBLIC SOL_HAS_CUSOLVER_RCM=1)
    target_link_libraries(socu_ordering_lab_core PUBLIC CUDA::cusolver CUDA::cusparse)
else()
    target_compile_definitions(socu_ordering_lab_core PUBLIC SOL_HAS_CUSOLVER_RCM=0)
endif()
target_compile_features(socu_ordering_lab_core PUBLIC cxx_std_20)
uipc_target_set_output_directory(socu_ordering_lab_core)
uipc_target_set_relative_source_file(socu_ordering_lab_core)
set_property(TARGET socu_ordering_lab_core PROPERTY FOLDER "experiments/socu_ordering_lab")

if(UIPC_BUILD_SOCU_ORDERING_LAB)
    find_package(Catch2 CONFIG REQUIRED)

    add_executable(socu_ordering_lab_test
        src/cli/test_main.cpp
        tests/contact_tests.cpp
        tests/graph_tests.cpp
        tests/io_tests.cpp
        tests/ordering_tests.cpp
        tests/reorder_tests.cpp
    )
    target_link_libraries(socu_ordering_lab_test PRIVATE socu_ordering_lab_core Catch2::Catch2)
    target_include_directories(socu_ordering_lab_test PRIVATE ${SOL_INCLUDE_DIR})
    target_compile_features(socu_ordering_lab_test PRIVATE cxx_std_20)
    set_target_properties(socu_ordering_lab_test PROPERTIES OUTPUT_NAME "socu_ordering_lab_test")
    uipc_target_set_output_directory(socu_ordering_lab_test)
    uipc_target_set_relative_source_file(socu_ordering_lab_test)
    set_property(TARGET socu_ordering_lab_test PROPERTY FOLDER "experiments/socu_ordering_lab")

    add_executable(socu_ordering_bench
        src/cli/bench_main.cpp
    )
    target_link_libraries(socu_ordering_bench PRIVATE socu_ordering_lab_core)
    target_include_directories(socu_ordering_bench PRIVATE ${SOL_INCLUDE_DIR})
    target_compile_features(socu_ordering_bench PRIVATE cxx_std_20)
    set_target_properties(socu_ordering_bench PROPERTIES OUTPUT_NAME "socu_ordering_bench")
    uipc_target_set_output_directory(socu_ordering_bench)
    uipc_target_set_relative_source_file(socu_ordering_bench)
    set_property(TARGET socu_ordering_bench PROPERTY FOLDER "experiments/socu_ordering_lab")

    add_test(NAME socu_ordering_lab COMMAND $<TARGET_FILE:socu_ordering_lab_test>)
endif()
