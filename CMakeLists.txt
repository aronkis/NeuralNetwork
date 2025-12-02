cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(POLICY CMP0169)
	cmake_policy(SET CMP0169 OLD)
endif()

if(POLICY CMP0167)
	cmake_policy(SET CMP0167 OLD)
endif()

project(NeuralNetwork 
	VERSION 1.0
	LANGUAGES CXX
	DESCRIPTION "Neural Network from Scratch in C++")

option(NN_DISABLE_EIGEN_PARALLELISM "Disable Eigen parallelism" ON)
option(NN_BUILD_TESTS "Build unit tests" ON)
option(NN_ENABLE_FAST_MATH "Enable fast math optimizations" ON)
option(NN_USE_BLAS "Use BLAS for matrix operations" ON)
option(NN_ENABLE_OPENMP "Enable OpenMP support" ON)

# FetchContent dependency options
option(NN_FETCH_EIGEN "Fetch Eigen library via FetchContent" ON)
option(NN_FETCH_MATPLOTLIB "Fetch matplotlib-cpp via FetchContent" OFF)
option(NN_FETCH_CPR "Fetch CPR (HTTP client) via FetchContent" ON)
option(NN_FETCH_MINIZIP "Fetch minizip-ng via FetchContent" ON)
option(NN_FETCH_STB "Fetch stb (image loading) via FetchContent" ON)
option(NN_FETCH_CEREAL "Fetch Cereal (serialization) via FetchContent" ON)
option(NN_FETCH_ZMQ "Fetch libzmq via FetchContent" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	add_compile_options(-O3 -march=native)
endif()

include(FetchContent)

# Eigen
if(NN_FETCH_EIGEN)
	FetchContent_Declare(
		Eigen3
		GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
		GIT_TAG 3.4.0
		GIT_SHALLOW TRUE
	)
	set(EIGEN_BUILD_DOC OFF CACHE BOOL "Don't build Eigen docs")
	set(BUILD_TESTING OFF CACHE BOOL "Don't build Eigen tests")
	set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Don't build Eigen pkg-config")
	FetchContent_MakeAvailable(Eigen3)
else()
	find_package(Eigen3 REQUIRED)
endif()

find_package(CURL REQUIRED)

if(NN_USE_BLAS)
	find_package(BLAS QUIET)

	if(BLAS_FOUND)
		message(STATUS "Found system BLAS: ${BLAS_LIBRARIES}")
		set(BLAS_TARGET_NAME "BLAS::BLAS")
	else()
		message(STATUS "System BLAS not found, please install it to your system.")
	endif()
endif()

if(NN_ENABLE_OPENMP)
	find_package(OpenMP)
	if(OpenMP_CXX_FOUND)
		message(STATUS "OpenMP support enabled: ${OpenMP_CXX_FLAGS}")
	else()
		message(WARNING "NN_ENABLE_OPENMP is ON but OpenMP was not found on this system.")
	endif()
endif()

# matplotlib-cpp
if(NN_FETCH_MATPLOTLIB)
	find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Embed)
	
	FetchContent_Declare(
		matplotlib_cpp
		GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
		GIT_TAG master
	)
	FetchContent_Populate(matplotlib_cpp)

	add_library(matplotlib_cpp INTERFACE)
	target_include_directories(matplotlib_cpp SYSTEM INTERFACE ${matplotlib_cpp_SOURCE_DIR})
	target_link_libraries(matplotlib_cpp INTERFACE Python3::Python)
else()
	message(STATUS "matplotlib-cpp disabled. Enable NN_FETCH_MATPLOTLIB to use plotting features.")
endif()

# CPR (HTTP client)
if(NN_FETCH_CPR)
	FetchContent_Declare(
		cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
		GIT_TAG da40186618909b1a7363d4e4495aa899c6e0eb75
	)
	set(CPR_USE_SYSTEM_CURL ON CACHE BOOL "" FORCE)
	set(CPR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(cpr)
else()
	message(STATUS "CPR disabled. Enable NN_FETCH_CPR to use HTTP download features.")
endif()

# minizip-ng
if(NN_FETCH_MINIZIP)
	FetchContent_Declare(
		minizip-ng
		GIT_REPOSITORY https://github.com/zlib-ng/minizip-ng.git
		GIT_TAG 4.0.10
	)

	set(MZ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(MZ_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
	set(MZ_BUILD_FUZZ_TESTS OFF CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(minizip-ng)

	FetchContent_GetProperties(minizip-ng)
	if(NOT minizip-ng_POPULATED)
		FetchContent_Populate(minizip-ng)
	endif()
else()
	message(STATUS "minizip-ng disabled. Enable NN_FETCH_MINIZIP to use ZIP extraction features.")
endif()

# stb (image loading)
if(NN_FETCH_STB)
	FetchContent_Declare(
		stb
		GIT_REPOSITORY https://github.com/nothings/stb.git
		GIT_TAG master
	)
	FetchContent_MakeAvailable(stb)
else()
	message(STATUS "stb disabled. Enable NN_FETCH_STB to use image loading features.")
endif()

# Cereal (serialization)
if(NN_FETCH_CEREAL)
	FetchContent_Declare(
		cereal
		GIT_REPOSITORY https://github.com/USCiLab/cereal.git
		GIT_TAG v1.3.2
	)
	FetchContent_GetProperties(cereal)
	if(NOT cereal_POPULATED)
		FetchContent_Populate(cereal)
	endif()
else()
	message(STATUS "Cereal disabled. Enable NN_FETCH_CEREAL to use serialization features.")
endif()

# libzmq
if(NN_FETCH_ZMQ)
	set(WITH_PERF_TOOL OFF CACHE BOOL "" FORCE)
	set(ZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
	set(ENABLE_CPACK OFF CACHE BOOL "" FORCE)
	set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE) # Build as static library

	FetchContent_Declare(
		libzmq
		GIT_REPOSITORY https://github.com/zeromq/libzmq.git
		GIT_TAG        b91a6201307b72beb522300366aad763d19b1456 # Post-v4.3.5 commit that fixes cmake_minimum_required for CMake 3.31+
	)
	FetchContent_MakeAvailable(libzmq)
else()
	message(STATUS "libzmq disabled. Enable NN_FETCH_ZMQ to use ZeroMQ streaming features.")
endif()

file(GLOB_RECURSE SOURCES "src/*.cc")
file(GLOB_RECURSE EXAMPLE_SOURCES "examples/*.cc")

# Create the executable FIRST
add_executable(neural_network main.cc ${SOURCES} ${EXAMPLE_SOURCES})

# Now set compile definitions AFTER the target exists
target_compile_definitions(neural_network PRIVATE
	WITHOUT_NUMPY
)

# Add BLAS compile definitions if enabled
if(NN_USE_BLAS)
	target_compile_definitions(neural_network PRIVATE EIGEN_USE_BLAS)
	message(STATUS "EIGEN_USE_BLAS enabled")
endif()

target_include_directories(neural_network PRIVATE 
	include
	include/Accuracy
	include/Activation
	include/Layer
	include/Loss
	include/Model
	include/Optimizer
	include/Utils
	examples
)

# Conditional include directories based on enabled features
if(NN_FETCH_MINIZIP)
	target_include_directories(neural_network PRIVATE ${minizip-ng_SOURCE_DIR})
endif()
if(NN_FETCH_CEREAL)
	target_include_directories(neural_network PRIVATE ${cereal_SOURCE_DIR}/include)
endif()
if(NN_FETCH_STB)
	target_include_directories(neural_network PRIVATE ${stb_SOURCE_DIR})
endif()
if(NN_FETCH_MATPLOTLIB)
	target_include_directories(neural_network PRIVATE ${Python3_INCLUDE_DIRS})
endif()

file(GLOB EXAMPLE_SUBDIRS RELATIVE ${CMAKE_SOURCE_DIR} "examples/*")
foreach(_ex_dir ${EXAMPLE_SUBDIRS})
	if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/${_ex_dir})
		target_include_directories(neural_network PRIVATE ${CMAKE_SOURCE_DIR}/${_ex_dir})
	endif()
endforeach()

# Base libraries
target_link_libraries(neural_network PRIVATE Eigen3::Eigen)

# Conditional libraries based on enabled features
if(NN_FETCH_MATPLOTLIB)
	target_link_libraries(neural_network PRIVATE matplotlib_cpp Python3::Python)
endif()
if(NN_FETCH_CPR)
	target_link_libraries(neural_network PRIVATE cpr::cpr)
endif()
if(NN_FETCH_MINIZIP)
	target_link_libraries(neural_network PRIVATE minizip)
endif()
if(NN_FETCH_ZMQ)
	target_link_libraries(neural_network PRIVATE libzmq)
endif()

if(NN_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
	target_link_libraries(neural_network PRIVATE OpenMP::OpenMP_CXX)
	target_compile_definitions(neural_network PRIVATE USE_OPENMP)
	message(STATUS "Linked neural_network with OpenMP")
endif()

if(NN_USE_BLAS)
	if(BLAS_FOUND)
		target_link_libraries(neural_network PRIVATE ${BLAS_LIBRARIES})
	else()
		target_link_libraries(neural_network PRIVATE ${BLAS_TARGET_NAME})
	endif()
endif()

if(MSVC)
	target_compile_options(neural_network PRIVATE 
		/W4
		/WX-     # Don't treat warnings as errors
		/wd4996  # Disable deprecated declarations warning on MSVC
		/wd4127  # Disable "conditional expression is constant" warning (common in Eigen)
		/wd4244  # Disable conversion warnings (type conversions with possible data loss)
		/wd4267  # Disable size_t conversion warnings
		/wd4100  # Disable unreferenced parameter warnings
		/wd4456  # Disable declaration hides previous local declaration
	)
	if(NN_ENABLE_FAST_MATH)
		target_compile_options(neural_network PRIVATE /fp:fast)
	endif()
else()
	target_compile_options(neural_network PRIVATE 
		-Wall 
		-Wextra 
		-pedantic
		-Wno-deprecated-declarations  # Disable deprecated declarations on GCC/Clang
		-Wno-ignored-qualifiers       # Suppress const on primitive return types
		-Wno-unused-parameter         # Suppress unused parameter warnings
		-Wno-unused-variable          # Suppress unused variable warnings
	)
	if(NN_ENABLE_FAST_MATH)
		target_compile_options(neural_network PRIVATE -ffast-math)
	endif()
endif()

if(NN_DISABLE_EIGEN_PARALLELISM)
	message(STATUS "Disabling Eigen internal parallelism (EIGEN_DONT_PARALLELIZE)")
	target_compile_definitions(neural_network PRIVATE EIGEN_DONT_PARALLELIZE)
endif()

if(NN_BUILD_TESTS)
	add_subdirectory(test)
endif()

# Suppress Eigen infinity warnings
if(NOT MSVC)
    target_compile_options(neural_network PRIVATE -Wno-nan-infinity-disabled)
endif()