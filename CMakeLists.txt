cmake_minimum_required(VERSION 3.16)

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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native)
endif()

find_package(Eigen3 3.3 REQUIRED)
find_package(CURL REQUIRED)

include(FetchContent)
FetchContent_Declare(
    matplotlib_cpp
    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
    GIT_TAG master
)
FetchContent_Populate(matplotlib_cpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Embed)
add_library(matplotlib_cpp INTERFACE)
target_include_directories(matplotlib_cpp SYSTEM INTERFACE ${matplotlib_cpp_SOURCE_DIR})
target_link_libraries(matplotlib_cpp INTERFACE Python3::Python)

FetchContent_Declare(
    cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
    GIT_TAG da40186618909b1a7363d4e4495aa899c6e0eb75
)
set(CPR_USE_SYSTEM_CURL ON CACHE BOOL "" FORCE)
set(CPR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cpr)

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

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(stb)

FetchContent_Declare(
    cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG v1.3.2
)
FetchContent_GetProperties(cereal)
if(NOT cereal_POPULATED)
    FetchContent_Populate(cereal)
endif()

file(GLOB_RECURSE SOURCES "src/*.cc")

add_executable(neural_network main.cc ${SOURCES})
target_include_directories(neural_network PRIVATE 
    include
    include/Accuracy
    include/Activation
    include/Layer
    include/Loss
    include/Model
    include/Optimizer
    include/Utils
    ${minizip-ng_SOURCE_DIR}
    ${cereal_SOURCE_DIR}/include
    ${stb_SOURCE_DIR}  # Add stb include directory
)
target_link_libraries(neural_network PRIVATE Eigen3::Eigen matplotlib_cpp cpr::cpr minizip)
target_compile_definitions(neural_network PRIVATE WITHOUT_NUMPY)
target_compile_options(neural_network PRIVATE -Wno-deprecated-declarations)

set(NN_EPOCHS 10 CACHE STRING "Number of training epochs")
set(NN_PRINT_EVERY 1 CACHE STRING "Print interval")
set(BATCH_SIZE 1 CACHE STRING "Batch size")
target_compile_definitions(neural_network PRIVATE NN_EPOCHS=${NN_EPOCHS} NN_PRINT_EVERY=${NN_PRINT_EVERY} BATCH_SIZE=${BATCH_SIZE})

option(NN_ENABLE_FAST_MATH "Enable -ffast-math (may change numerics)" OFF)
if(NN_ENABLE_FAST_MATH)
    message(STATUS "FastMath enabled.")
    target_compile_options(neural_network PRIVATE -ffast-math)
endif()

option(NN_DISABLE_EIGEN_PARALLELISM "Define EIGEN_DONT_PARALLELIZE to opt out of Eigen's internal parallelism" OFF)
if(NN_DISABLE_EIGEN_PARALLELISM)
    message(STATUS "Disabling Eigen internal parallelism (EIGEN_DONT_PARALLELIZE)")
    target_compile_definitions(neural_network PRIVATE EIGEN_DONT_PARALLELIZE)
endif()