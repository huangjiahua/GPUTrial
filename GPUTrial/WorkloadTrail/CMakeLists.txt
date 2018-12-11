add_executable(gpuWorkloadTrail "workloadTest.cuh" "gpu.cu")
add_executable(cpuWorkloadTrail "workloadTest.cuh" "cpu.cu")
add_executable(gpuScheduleTrail "SchedulerTest.cuh" "schedulegpu.cu")

if (WIN32) 
	target_compile_definitions(gpuWorkloadTrail PRIVATE WIN32=1)
	target_compile_definitions(cpuWorkloadTrail PRIVATE WIN32=1)
	target_compile_definitions(gpuScheduleTrail PRIVATE WIN32=1)
endif (WIN32)
