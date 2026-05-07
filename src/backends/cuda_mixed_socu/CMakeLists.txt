set(CMAKE_CUDA_ARCHITECTURES ${UIPC_CUDA_ARCHITECTURES})
uipc_info("CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}.")
enable_language(CUDA)

option(UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY
    "Build cuda_mixed_socu without legacy structured contact assembly translation units."
    OFF)

# ------------------------------------------------------------------------------
# CUDA Backend
# ------------------------------------------------------------------------------
uipc_add_backend(cuda_mixed_socu)

# basic setup
target_link_libraries(cuda_mixed_socu PUBLIC
    muda::muda
    cublas
    cusparse
    cusolver
    uipc_geometry
)

if(TARGET socu_native)
    target_link_libraries(cuda_mixed_socu PRIVATE socu_native)
    target_compile_definitions(cuda_mixed_socu PUBLIC UIPC_WITH_SOCU_NATIVE=1)
else()
    target_compile_definitions(cuda_mixed_socu PUBLIC UIPC_WITH_SOCU_NATIVE=0)
endif()

# add CCCL include directory for CUDA 13.0 compatibility
if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    target_include_directories(cuda_mixed_socu PUBLIC ${CUDAToolkit_INCLUDE_DIRS}/cccl)
endif()

target_compile_features(cuda_mixed_socu PUBLIC cxx_std_20)
target_include_directories(cuda_mixed_socu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(cuda_mixed_socu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES ${UIPC_CUDA_ARCHITECTURES}
    CUDA_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
)

target_compile_options(cuda_mixed_socu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --diag-suppress=27,174,221,997,1388,1394,1866>
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>
)

string(TOUPPER "${UIPC_CUDA_MIXED_PRECISION_LEVEL}" _UIPC_MIXED_LEVEL_UPPER)
target_compile_definitions(cuda_mixed_socu PRIVATE "UIPC_MIXED_LEVEL_${_UIPC_MIXED_LEVEL_UPPER}=1")
target_compile_definitions(cuda_mixed_socu PRIVATE UIPC_CUDA_MIXED_SOCU_ONLY=1)
if(UIPC_WITH_NVTX)
    target_compile_definitions(cuda_mixed_socu PRIVATE UIPC_WITH_NVTX=1)
endif()
if(UIPC_BUILD_TESTS)
    target_compile_definitions(cuda_mixed_socu PRIVATE UIPC_ENABLE_MUDA_MEMORY_TRACKING=1)
endif()

# all files in this directory
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cu" "*.h" "*.hpp" "*.cpp" "*.inl")

if(UIPC_CUDA_MIXED_WRECKING_BALL_MINIMAL_BUILD)
    set(_UIPC_CUDA_MIXED_SOURCE_COUNT_BEFORE ${SOURCES})
    list(LENGTH _UIPC_CUDA_MIXED_SOURCE_COUNT_BEFORE _UIPC_CUDA_MIXED_SOURCE_COUNT_BEFORE)

    set(_UIPC_CUDA_MIXED_MINIMAL_EXCLUDE_PATTERNS
        ".*/affine_body/constitutions/affine_body_.*joint.*\\.cu$"
        ".*/affine_body/constraints/(external_articulation_constraint|soft_transform_constraint|affine_body_.*joint_external_body_force_constraint)\\.cu$"
        ".*/affine_body/affine_body_.*joint_external_body_force\\.cu$"
        ".*/contact_system/al_.*\\.cu$"
        ".*/contact_system/adaptive_strategies/.*\\.cu$"
        ".*/coupling_system/.*\\.cu$"
        ".*/finite_element/constitutions/(discrete_shell_bending|hookean_spring_1d|kirchhoff_rod_bending|neo_hookean_shell_2d|plastic_discrete_shell_bending|strain_limiting_baraff_witkin_shell_2d|stress_plastic_discrete_shell_bending)\\.cu$"
        ".*/finite_element/constraints/soft_position_constraint\\.cu$"
        ".*/inter_primitive_effect_system/constitutions/.*\\.cu$")

    foreach(_UIPC_CUDA_MIXED_EXCLUDE_PATTERN IN LISTS _UIPC_CUDA_MIXED_MINIMAL_EXCLUDE_PATTERNS)
        list(FILTER SOURCES EXCLUDE REGEX "${_UIPC_CUDA_MIXED_EXCLUDE_PATTERN}")
    endforeach()

    list(LENGTH SOURCES _UIPC_CUDA_MIXED_SOURCE_COUNT_AFTER)
    uipc_info("cuda_mixed_socu wrecking_ball minimal build: ${_UIPC_CUDA_MIXED_SOURCE_COUNT_BEFORE} -> ${_UIPC_CUDA_MIXED_SOURCE_COUNT_AFTER} source entries")
endif()

if(UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY)
    set(_UIPC_CUDA_MIXED_SOURCE_COUNT_BEFORE_NATIVE_ONLY ${SOURCES})
    list(LENGTH _UIPC_CUDA_MIXED_SOURCE_COUNT_BEFORE_NATIVE_ONLY _UIPC_CUDA_MIXED_SOURCE_COUNT_BEFORE_NATIVE_ONLY)

    set(_UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY_CALLER_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/contact_system/contact_models/ipc_simplex_frictional_contact.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/contact_system/contact_models/ipc_simplex_normal_contact.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/contact_system/contact_models/ipc_vertex_half_plane_frictional_contact.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/contact_system/contact_models/ipc_vertex_half_plane_normal_contact.cu")
    set_property(
        SOURCE ${_UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY_CALLER_SOURCES}
        APPEND
        PROPERTY COMPILE_DEFINITIONS UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY=1)

    set(_UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY_EXCLUDE_PATTERNS
        ".*/contact_system/contact_models/.*_structured\\.cu$")

    foreach(_UIPC_CUDA_MIXED_EXCLUDE_PATTERN IN LISTS _UIPC_CUDA_MIXED_SOCU_NATIVE_ONLY_EXCLUDE_PATTERNS)
        list(FILTER SOURCES EXCLUDE REGEX "${_UIPC_CUDA_MIXED_EXCLUDE_PATTERN}")
    endforeach()

    list(LENGTH SOURCES _UIPC_CUDA_MIXED_SOURCE_COUNT_AFTER_NATIVE_ONLY)
    uipc_info("cuda_mixed_socu native-only build: ${_UIPC_CUDA_MIXED_SOURCE_COUNT_BEFORE_NATIVE_ONLY} -> ${_UIPC_CUDA_MIXED_SOURCE_COUNT_AFTER_NATIVE_ONLY} source entries")
endif()

target_sources(cuda_mixed_socu PRIVATE ${SOURCES})
uipc_target_set_relative_source_file(cuda_mixed_socu)

# ------------------------------------------------------------------------------
# setup source group for the IDE
# ------------------------------------------------------------------------------
file(GLOB_RECURSE SOURCE_GROUP_FILES CONFIGURE_DEPENDS "*.h" "*.hpp" "*.cpp" "*.cu" "*.inl")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/.." FILES ${SOURCE_GROUP_FILES})
