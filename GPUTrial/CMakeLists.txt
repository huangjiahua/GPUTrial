# CMakeList.txt : CMake project for GPUTrial, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.

add_subdirectory("BandwidthTrail")
add_subdirectory("WorkloadTrail")
add_subdirectory("AtomicOperations")

# TODO: Add tests and install targets if needed.
