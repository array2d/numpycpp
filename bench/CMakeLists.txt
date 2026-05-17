cmake_minimum_required(VERSION 3.16)
project(numpycpp-bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# FetchContent google-benchmark
include(FetchContent)
FetchContent_Declare(googlebench
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebench)

add_executable(bench_core bench_core.cpp)
target_include_directories(bench_core PRIVATE ..)
target_link_libraries(bench_core PRIVATE benchmark::benchmark)
