cmake_minimum_required(VERSION 3.6)
project(face_detection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp ../../../detection/CascadeCNN.cpp ../../../detection/CascadeCNN.h fddb_detection.cpp fddb_detection.h)
add_executable(face_detection ${SOURCE_FILES})

target_link_libraries(face_detection ${Caffe_LIBRARIES})