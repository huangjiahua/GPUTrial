add_executable(memAlloc "Allocation.cuh" AllocationTest.cu)

if (WIN32)
    target_compile_definitions(memAlloc PRIVATE WIN32=1)
endif (WIN32)
