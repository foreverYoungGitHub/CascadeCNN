cmake_minimum_required(VERSION 3.5)
project(CascadeCNN)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set(OpenCV_DIR "/usr/local/lib")
#find_package(OpenCV)

#set(Caffe_DIR "/home/xileli/Documents/library/caffe/build")
#set(Caffe_DEFINITIONS "-DCMAKE_BUILD_TYPE=Release")
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})

#message(STATUS ${Caffe_INCLUDE_DIRS})
#message(STATUS ${Caffe_LIBRARIES})

set(SOURCE_FILES main.cpp CascadeCNN.cpp CascadeCNN.h)
add_executable(CascadeCNN ${SOURCE_FILES})

#target_link_libraries(CascadeCNN ${OpenCV_LIBS} )
target_link_libraries(CascadeCNN ${Caffe_LIBRARIES})