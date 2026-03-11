file(GLOB SOURCES "*.cpp" "*.cu" "*.h" "details/*.inl")
target_sources(cuda_mixed PRIVATE ${SOURCES})
