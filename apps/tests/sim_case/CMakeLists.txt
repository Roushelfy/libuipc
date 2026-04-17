file(GLOB SOURCE "*.cpp" "*.h")
uipc_add_test(sim_case ${SOURCE})
target_link_libraries(sim_case PRIVATE uipc::backends muda::muda)

uipc_register_catch_test(sim_case_cuda_mixed_contract sim_case "[cuda_mixed]")

# Serialize PDB writes on MSVC to avoid C1041 when many .cpp files compile in parallel
if(MSVC)
    target_compile_options(sim_case PRIVATE /FS)
endif()
