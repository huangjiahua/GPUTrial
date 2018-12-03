add_executable(bandwidthTrail "bandwidthTrail.cu")
if (WIN32)
	target_compile_definitions(bandwidthTrail PRIVATE WIN32=1)
endif (WIN32)	