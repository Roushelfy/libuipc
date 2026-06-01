include(uipc_utils)
file(GLOB BACKEND_COMMON_SOURCES "common/*.h" "common/*.cpp" "common/details/*.inl")

# ---------------------------------------------------------------------------
# A dummy target to group all backends
# ---------------------------------------------------------------------------
add_library(uipc_backends INTERFACE)
add_library(uipc::backends ALIAS uipc_backends)

set(UIPC_BACKENDS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# Auto-sync rebuilt backend .so files to the venv's pyuipc/_native directory,
# along with the runtime libraries loaded from the same directory.
# Without this, a `cmake --build --target cuda` re-link will leave the
# Python-side pyuipc loading a stale backend (the `pip install` step that
# would refresh it only runs when the pyuipc target itself rebuilds). Copying
# the dependent libuipc libraries as well avoids mixing a fresh backend with a
# stale core/geometry/constitution/io ABI in the installed venv. This detection
# is best-effort: if pyuipc isn't installed yet, the post-build step is skipped
# silently.
# ---------------------------------------------------------------------------
execute_process(
    COMMAND "${UIPC_PYTHON_EXECUTABLE_PATH}" -c
        "import os, sys\ntry:\n    import uipc\n    print(os.path.dirname(uipc.__file__) + '/_native', end='')\nexcept ImportError:\n    pass"
    OUTPUT_VARIABLE _UIPC_PYUIPC_NATIVE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(_UIPC_PYUIPC_NATIVE_DIR AND IS_DIRECTORY "${_UIPC_PYUIPC_NATIVE_DIR}")
    set(UIPC_PYUIPC_NATIVE_DIR "${_UIPC_PYUIPC_NATIVE_DIR}")
    uipc_info("Backend auto-sync enabled: rebuilt .so files will be copied to ${UIPC_PYUIPC_NATIVE_DIR}")
else()
    uipc_info("pyuipc not yet installed; backend auto-sync disabled (will activate after first `pip install`).")
endif()

# ---------------------------------------------------------------------------
# function to set some common properties for backends
# ---------------------------------------------------------------------------
function(uipc_add_backend name)
    uipc_info("Adding backend: [${name}]")
    if(UIPC_BUILD_TESTS) # for tests, we need to link against the shared library
        # uipc_info("UIPC_BUILD_TESTS=${UIPC_BUILD_TESTS}, so we build the backend [${name}] as shared library")
        add_library(${name} SHARED)
    else() # else, we just dynamically load the backend
        add_library(${name} MODULE)
    endif()

    # alias the target
    add_library(uipc::backend::${name} ALIAS ${name})
    uipc_target_add_include_files(${name})
    target_include_directories(${name} PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(${name} PRIVATE "-DUIPC_BACKEND_DIR=R\"(${UIPC_BACKENDS_SOURCE_DIR})\"")
    target_compile_definitions(${name} PRIVATE "-DUIPC_BACKEND_NAME=R\"(${name})\"")
    target_link_libraries(${name} PRIVATE uipc::core)
    target_compile_definitions(${name} PRIVATE UIPC_BACKEND_EXPORT_DLL=1)
    target_compile_features(${name} PRIVATE cxx_std_20)
    set_target_properties(${name} PROPERTIES OUTPUT_NAME "uipc_backend_${name}")
    # Loader (dylib.hpp) expects lib<name>.dylib on macOS; MODULE default is .so
    if(APPLE)
        set_target_properties(${name} PROPERTIES SUFFIX ".dylib")
    endif()

    # set some properties for the target
    uipc_target_set_output_directory(${name})
    uipc_target_set_rpath(${name})
    add_dependencies(uipc_backends ${name})
    set_property(GLOBAL APPEND PROPERTY UIPC_BACKEND_TARGETS ${name})

    # folder for IDE
    set_target_properties(${name} PROPERTIES FOLDER "uipc-backends")
    target_sources(${name} PRIVATE ${BACKEND_COMMON_SOURCES})

    # install the target
    uipc_install_target(${name})

    # Auto-sync rebuilt backend and pyuipc runtime libraries to the venv.
    # See header note above.
    if(UIPC_PYUIPC_NATIVE_DIR)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${name}>"
                "$<TARGET_FILE:uipc_core>"
                "$<TARGET_FILE:uipc_geometry>"
                "$<TARGET_FILE:uipc_constitution>"
                "$<TARGET_FILE:uipc_io>"
                "$<TARGET_FILE:uipc_sanity_check>"
                "${UIPC_PYUIPC_NATIVE_DIR}"
            COMMENT "[uipc] sync ${name} and libuipc runtime → ${UIPC_PYUIPC_NATIVE_DIR}"
            VERBATIM
        )
    endif()
endfunction()


# ---------------------------------------------------------------------------
# Add backends
# ---------------------------------------------------------------------------
add_subdirectory(none)

if(UIPC_WITH_CUDA_BACKEND)
    add_subdirectory(cuda)
endif()


