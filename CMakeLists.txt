cmake_minimum_required(VERSION 3.16)
project(NeuralNetwork 
    VERSION 1.0
    LANGUAGES CXX OBJCXX
    DESCRIPTION "Neural Network from Scratch in C++")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

find_package(Eigen3 3.3 REQUIRED)
find_package(Matplot++ QUIET)

# Check if we're on macOS and add Metal support
if(APPLE)
    find_library(METAL_LIBRARY Metal)
    find_library(METALKIT_LIBRARY MetalKit)
    find_library(METALPERFORMANCESHADERS_LIBRARY MetalPerformanceShaders)
    find_library(FOUNDATION_LIBRARY Foundation)
    
    if(METAL_LIBRARY AND METALKIT_LIBRARY AND METALPERFORMANCESHADERS_LIBRARY)
        set(METAL_FOUND TRUE)
        message(STATUS "Metal frameworks found - GPU acceleration enabled")
    else()
        message(WARNING "Metal frameworks not found - CPU only")
    endif()
endif()

file(GLOB_RECURSE SOURCES "src/*.cc")
file(GLOB_RECURSE HEADERS "include/*.h")

# Add Metal sources if available
if(METAL_FOUND)
    file(GLOB_RECURSE METAL_SOURCES "src/Metal/*.mm")
    list(APPEND SOURCES ${METAL_SOURCES})
endif()

add_executable(neural_network main.cc ${SOURCES})
target_include_directories(neural_network PRIVATE 
    include
    include/Activation
    include/Loss
    include/Optimizer
    include/Metal
)
target_link_libraries(neural_network PRIVATE Eigen3::Eigen)

# Link Metal frameworks if available
if(METAL_FOUND)
    target_link_libraries(neural_network PRIVATE 
        ${METAL_LIBRARY}
        ${METALKIT_LIBRARY}
        ${METALPERFORMANCESHADERS_LIBRARY}
        ${FOUNDATION_LIBRARY}
    )
    target_compile_definitions(neural_network PRIVATE METAL_ENABLED)
endif()

if(Matplot++_FOUND)
    target_link_libraries(neural_network PRIVATE Matplot++::matplot)
    target_compile_definitions(neural_network PRIVATE HAS_MATPLOT)
endif()