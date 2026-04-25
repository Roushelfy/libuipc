file(GLOB SOURCE "*.cpp" "*.h")
uipc_add_test(sim_case ${SOURCE})
target_link_libraries(sim_case PRIVATE uipc::backends muda::muda)

uipc_register_catch_test(sim_case_cuda_mixed_contract sim_case "[cuda_mixed]")

if(TARGET uipc::backend::cuda_mixed)
    uipc_add_test(sim_case_cuda_mixed_only 86_cuda_mixed_precision_contracts.cpp)
    target_link_libraries(sim_case_cuda_mixed_only PRIVATE
        uipc::backend::cuda_mixed
        muda::muda)
    target_link_directories(sim_case_cuda_mixed_only PRIVATE
        "${CUDAToolkit_ROOT}/lib64")
    target_include_directories(sim_case_cuda_mixed_only PRIVATE ${PROJECT_SOURCE_DIR}/src)
    uipc_register_catch_test(
        sim_case_cuda_mixed_only_linear_solver
        sim_case_cuda_mixed_only
        "86_cuda_mixed_linear_solver_selection_smoke")
endif()

# Serialize PDB writes on MSVC to avoid C1041 when many .cpp files compile in parallel
if(MSVC)
    target_compile_options(sim_case PRIVATE /FS)
    if(TARGET sim_case_cuda_mixed_only)
        target_compile_options(sim_case_cuda_mixed_only PRIVATE /FS)
    endif()
endif()
