add_executable(gpuWorkloadTrail "WorkloadTest.cuh" "gpu.cu")
add_executable(cpuWorkloadTrail "WorkloadTest.cuh" "cpu.cu")
add_executable(gpuScheduleTrail "schedulegpu.cu")

if (WIN32) 
	target_compile_definitions(gpuWorkloadTrail PRIVATE WIN32=1)
	target_compile_definitions(cpuWorkloadTrail PRIVATE WIN32=1)
	target_compile_definitions(gpuScheduleTrail PRIVATE WIN32=1)
endif (WIN32)
