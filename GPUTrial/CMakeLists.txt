# CMakeList.txt : CMake project for GPUTrial, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# C4819 will happen on windows due to language encoding issue, it won't affect anything.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/wd 4819\" " )


# It needs to add this flag, because this project should be compiled seperately
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true " )

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 " )

# Add source to this project's executable.

add_subdirectory("BandwidthTrail")
add_subdirectory("WorkloadTrail")
add_subdirectory("AtomicOperations")
add_subdirectory("MemAollocation")
add_subdirectory("DynamicMemoryAllocation")

# TODO: Add tests and install targets if needed.
