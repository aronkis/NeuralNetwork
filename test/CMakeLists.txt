cmake_minimum_required(VERSION 3.16)

include(FetchContent)
FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG        v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

file(GLOB_RECURSE NEURAL_NETWORK_SOURCES "../src/*.cc")

set(TEST_SOURCES
	TestMain.cc
	TestLayerDense.cc
	TestActivations.cc
	TestLayerDropout.cc
	TestLosses.cc
	TestOptimizers.cc
	TestAccuracy.cc
	TestModel.cc
	TestModelSaveLoadIntegration.cc
	TestHelpers.cc
	TestConvolution.cc
	TestPooling.cc
	TestConvolution1D.cc
	TestConvolution1DGradients.cc
	TestMaxPooling1D.cc
	TestZMQ.cc
)

add_executable(neural_network_tests ${TEST_SOURCES} ${NEURAL_NETWORK_SOURCES})

target_include_directories(neural_network_tests PRIVATE
	../include
	../include/Accuracy
	../include/Activation
	../include/Layer
	../include/Loss
	../include/Model
	../include/Optimizer
	../include/Utils
)

# Conditional include directories based on enabled features
if(NN_FETCH_MINIZIP)
	target_include_directories(neural_network_tests PRIVATE ${minizip-ng_SOURCE_DIR})
endif()
if(NN_FETCH_CEREAL)
	target_include_directories(neural_network_tests PRIVATE ${cereal_SOURCE_DIR}/include)
endif()
if(NN_FETCH_STB)
	target_include_directories(neural_network_tests PRIVATE ${stb_SOURCE_DIR})
endif()
if(NN_FETCH_MATPLOTLIB)
	target_include_directories(neural_network_tests PRIVATE ${Python3_INCLUDE_DIRS})
endif()

# Base libraries
target_link_libraries(neural_network_tests PRIVATE
	gtest_main
	Eigen3::Eigen
)

# Conditional libraries based on enabled features
if(NN_FETCH_MATPLOTLIB)
	target_link_libraries(neural_network_tests PRIVATE matplotlib_cpp Python3::Python)
endif()
if(NN_FETCH_CPR)
	target_link_libraries(neural_network_tests PRIVATE cpr::cpr)
endif()
if(NN_FETCH_MINIZIP)
	target_link_libraries(neural_network_tests PRIVATE minizip)
endif()
if(NN_FETCH_ZMQ)
	target_link_libraries(neural_network_tests PRIVATE libzmq)
endif()

target_compile_definitions(neural_network_tests PRIVATE WITHOUT_NUMPY)

if(MSVC)
	target_compile_options(neural_network_tests PRIVATE
		/W4
		/WX-
		/wd4996
		/wd4127
		/wd4244
		/wd4267
		/wd4100
		/wd4456
	)
else()
	target_compile_options(neural_network_tests PRIVATE
		-Wall
		-Wextra
		-pedantic
		-Wno-deprecated-declarations
		-Wno-ignored-qualifiers
		-Wno-unused-parameter
		-Wno-unused-variable
	)
endif()

gtest_discover_tests(neural_network_tests)