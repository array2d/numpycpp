cmake_minimum_required(VERSION 3.16)
project(numpycpp_example LANGUAGES CXX)

find_package(numpycpp REQUIRED)

add_executable(example main.cpp)
target_link_libraries(example PRIVATE numpycpp::numpycpp)
