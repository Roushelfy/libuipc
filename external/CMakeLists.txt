# 2025.10.9
# We remove all submodules from git and manage them via vcpkg instead.
# For fast-start-development, we provide a helper function to initialize submodules.
# Once the dependency is stabilized, we will make it a vcpkg dependency and remove the submodule initialization.

# Submodule Usage 
# uipc_init_submodule(my_lib) # auto init submodule if not exist
# add_subdirectory(my_lib)    # include the submodule cmake project
# set_target_properties(my_lib PROPERTIES FOLDER "external") # optional, organize in IDEs

# GKlib + METIS (bundled from Stiff-GIPC for graph partitioning)
add_subdirectory(GKlib)
add_subdirectory(METIS)
set_target_properties(GKlib PROPERTIES FOLDER "external")
set_target_properties(metis PROPERTIES FOLDER "external")

if (UIPC_WITH_CUDA_BACKEND OR UIPC_WITH_CUDA_MIXED_BACKEND)
    uipc_init_submodule(muda)
    if(UIPC_WITH_NVTX)
        set(MUDA_WITH_NVTX3 ON CACHE BOOL "" FORCE)
    else()
        set(MUDA_WITH_NVTX3 OFF CACHE BOOL "" FORCE)
    endif()
    # Turn off example builds in muda
    set(MUDA_BUILD_EXAMPLE OFF CACHE BOOL "Build muda example" FORCE)
    if(UIPC_BUILD_PYTHON_WHEEL)
        # When building wheel, we don't need to install muda to the wheel package.
        add_subdirectory(muda EXCLUDE_FROM_ALL)
    else()
        add_subdirectory(muda)
    endif()
    add_library(muda::muda ALIAS muda)
    set_target_properties(muda PROPERTIES FOLDER "external")
endif()

if(UIPC_WITH_SOCU_NATIVE STREQUAL "ON" AND NOT UIPC_WITH_CUDA_MIXED_BACKEND)
    message(FATAL_ERROR "UIPC_WITH_SOCU_NATIVE=ON requires UIPC_WITH_CUDA_MIXED_BACKEND=ON")
endif()

if(UIPC_WITH_CUDA_MIXED_BACKEND AND NOT UIPC_WITH_SOCU_NATIVE STREQUAL "OFF")
    set(_uipc_socu_native_dir "${CMAKE_CURRENT_SOURCE_DIR}/socu-native-cuda")
    if(EXISTS "${_uipc_socu_native_dir}/CMakeLists.txt")
        set(_uipc_build_testing_before "${BUILD_TESTING}")
        set(BUILD_TESTING OFF CACHE BOOL "Enable testing." FORCE)

        if(NOT DEFINED SOCU_NATIVE_ENABLE_MATHDX)
            set(SOCU_NATIVE_ENABLE_MATHDX "AUTO"
                CACHE STRING "Enable MathDx backend in socu_native: AUTO|ON|OFF")
        endif()

        add_subdirectory(socu-native-cuda EXCLUDE_FROM_ALL)

        set(BUILD_TESTING "${_uipc_build_testing_before}" CACHE BOOL "Enable testing." FORCE)

        if(TARGET socu_native)
            set_target_properties(socu_native PROPERTIES FOLDER "external")
            if(TARGET socu_native_cli)
                set_target_properties(socu_native_cli PROPERTIES FOLDER "external")
            endif()
            message(STATUS "socu_native integration enabled from ${_uipc_socu_native_dir}")
        elseif(UIPC_WITH_SOCU_NATIVE STREQUAL "ON")
            message(FATAL_ERROR "UIPC_WITH_SOCU_NATIVE=ON, but socu_native target was not created")
        endif()
    elseif(UIPC_WITH_SOCU_NATIVE STREQUAL "ON")
        message(FATAL_ERROR
            "UIPC_WITH_SOCU_NATIVE=ON, but ${_uipc_socu_native_dir} is missing. "
            "Run git submodule update --init external/socu-native-cuda.")
    else()
        message(STATUS
            "socu_native integration skipped: ${_uipc_socu_native_dir} is missing "
            "(set UIPC_WITH_SOCU_NATIVE=ON to require it)")
    endif()
endif()
