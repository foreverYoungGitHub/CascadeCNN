cmake_minimum_required(VERSION 3.6)
project(MegaFace_evaluation)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(MegaFace_evaluation ${SOURCE_FILES})