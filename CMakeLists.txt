cmake_minimum_required(VERSION 3.14)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Matplot++ REQUIRED)
find_package (Eigen3 3.3 REQUIRED)


add_executable(main main.cc src/LayerDense.cc 
                            src/Helpers.cc
                            src/Activation/ActivationReLU.cc
                            src/Activation/ActivationSoftmax.cc
                            src/Loss/Loss.cc
                            src/Loss/LossCategoricalCrossentropy.cc)

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/
                                        ${CMAKE_CURRENT_SOURCE_DIR}/include/Activation
                                        ${CMAKE_CURRENT_SOURCE_DIR}/include/Loss)

target_link_libraries(main PRIVATE Matplot++::matplot Eigen3::Eigen)
