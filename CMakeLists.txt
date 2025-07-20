cmake_minimum_required(VERSION 3.16)
project(NeuralNetwork 
    VERSION 1.0
    LANGUAGES CXX
    DESCRIPTION "Neural Network from Scratch in C++")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

find_package(Eigen3 3.3 REQUIRED)
find_package(Matplot++ QUIET)

file(GLOB_RECURSE SOURCES "src/*.cc")
file(GLOB_RECURSE HEADERS "include/*.h")

add_executable(neural_network main.cc ${SOURCES})
target_include_directories(neural_network PRIVATE 
    include
    include/Activation
    include/Loss
    include/Optimizer
)
target_link_libraries(neural_network PRIVATE Eigen3::Eigen)

if(Matplot++_FOUND)
    target_link_libraries(neural_network PRIVATE Matplot++::matplot)
    target_compile_definitions(neural_network PRIVATE HAS_MATPLOT)
endif()