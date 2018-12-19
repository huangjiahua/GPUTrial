add_executable(atomicAdd "atomic_add.cuh" "atm.cu")

if (WIN32)
    target_compile_definitions(atomicAdd PRIVATE WIN32=1)
endif (WIN32)
