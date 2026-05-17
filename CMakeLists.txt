cmake_minimum_required(VERSION 3.16)
project(numpycpp VERSION 0.1.0 LANGUAGES CXX)

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Collect sources
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/numpy/*.cpp)

# Headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Python module
pybind11_add_module(numpycpp ${SOURCES})

target_include_directories(numpycpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(numpycpp PRIVATE Eigen3::Eigen)

set_target_properties(numpycpp PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Install
install(TARGETS numpcpp
    LIBRARY DESTINATION ${Python3_SITEARCH}
)

install(DIRECTORY numpy/
    DESTINATION include/numpycpp
    FILES_MATCHING PATTERN "*.h"
)

message(STATUS "numpycpp v${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python:       ${Python3_VERSION}")
message(STATUS "  Eigen3:       ${EIGEN3_INCLUDE_DIR}")
