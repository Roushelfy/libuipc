set(MIXED_COMMON_SOURCES
    mixed_bench_utils.cpp
    mixed_scene_builders.cpp)

set(MIXED_COMMON_HEADERS
    mixed_bench_utils.h
    mixed_scene_builders.h)

uipc_add_benchmark(mixed_stage1)
target_sources(mixed_stage1
    PRIVATE
    mixed_stage1_benchmark.cpp
    ${MIXED_COMMON_SOURCES}
    ${MIXED_COMMON_HEADERS})
uipc_target_set_relative_source_file(mixed_stage1)

uipc_add_benchmark(mixed_stage2)
target_sources(mixed_stage2
    PRIVATE
    mixed_stage2_benchmark.cpp
    ${MIXED_COMMON_SOURCES}
    ${MIXED_COMMON_HEADERS})
uipc_target_set_relative_source_file(mixed_stage2)
