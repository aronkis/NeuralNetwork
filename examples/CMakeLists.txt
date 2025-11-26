cmake_minimum_required(VERSION 3.16)

project(NeuralNetworkExamples LANGUAGES CXX)

# Pull in the top-level project so configuring from this folder builds everything.
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/NeuralNetwork)

# Wire example sources/headers into the main executable so they are available when building from here.
if(TARGET neural_network)
	set(EXAMPLES_ROOT ${CMAKE_CURRENT_LIST_DIR})

	file(GLOB_RECURSE EXAMPLE_SOURCES CONFIGURE_DEPENDS
		${EXAMPLES_ROOT}/*.cc)
	if(EXAMPLE_SOURCES)
		target_sources(neural_network PRIVATE ${EXAMPLE_SOURCES})
	endif()

	file(GLOB_RECURSE EXAMPLE_HEADERS CONFIGURE_DEPENDS
		${EXAMPLES_ROOT}/*.h)
	if(EXAMPLE_HEADERS)
		set(EXAMPLE_INCLUDE_DIRS)
		foreach(header ${EXAMPLE_HEADERS})
			get_filename_component(header_dir ${header} DIRECTORY)
			list(APPEND EXAMPLE_INCLUDE_DIRS ${header_dir})
		endforeach()
		list(REMOVE_DUPLICATES EXAMPLE_INCLUDE_DIRS)
		target_include_directories(neural_network PRIVATE ${EXAMPLE_INCLUDE_DIRS})
	endif()
endif()
