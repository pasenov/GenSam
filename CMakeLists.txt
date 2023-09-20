cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(main)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(nflowwrapper SHARED nflowwrapper.cpp)
target_link_libraries(nflowwrapper ${TORCH_LIBRARIES})
set_property(TARGET nflowwrapper PROPERTY CXX_STANDARD 17)

add_executable(GenSam main.cpp)
target_link_libraries(GenSam nflowwrapper ${TORCH_LIBRARIES})

set_property(TARGET GenSam PROPERTY CXX_STANDARD 17)
