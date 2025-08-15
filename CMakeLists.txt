cmake_minimum_required(VERSION 3.16)
project(NeuralNetwork 
    VERSION 1.0
    LANGUAGES CXX
    DESCRIPTION "Neural Network from Scratch in C++")

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure optimized builds by default and enable vectorization-friendly flags
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native)
endif()

find_package(Eigen3 3.3 REQUIRED)

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

file(GLOB_RECURSE SOURCES "src/*.cc")

add_executable(neural_network main.cc ${SOURCES})
target_include_directories(neural_network PRIVATE 
    include
    include/Activation
    include/Layer
    include/Loss
    include/Optimizer
)
target_link_libraries(neural_network PRIVATE Eigen3::Eigen matplotlib_cpp)
target_compile_definitions(neural_network PRIVATE WITHOUT_NUMPY)
target_compile_options(neural_network PRIVATE -Wno-deprecated-declarations)

# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
#   target_compile_options(neural_network PRIVATE -ffast-math)
# endif()
