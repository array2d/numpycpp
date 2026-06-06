# Build numpycpp Python test/benchmark module.
# Alignment: BIT-LEVEL — all tests verify bit-exact C++ vs numpy results.
#
# Build:
#   cmake -S tests -B tests/build [-DCMAKE_BUILD_TYPE=Release]
#   cmake --build tests/build
#
# Or from the repo root:
#   cmake --build build --target numpycpp_pymod   (after root cmake -B build)
#
# Run tests:
#   cd tests && python3 -m pytest test_all.py -q

cmake_minimum_required(VERSION 3.16)
project(numpycpp_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Find dependencies -------------------------------------------------------
# Python3 (for pybind11 extension)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# pybind11 — prefer the copy installed alongside Python
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmake
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_cmake_result
)
if(_pybind11_cmake_result EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED
        PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)
else()
    find_package(pybind11 CONFIG REQUIRED)
endif()

# Eigen3 (optional — used for norm/std helpers in module.cpp)
find_package(Eigen3 CONFIG QUIET)

# OpenMP (optional — einsum uses #pragma omp only when _OPENMP is defined)
find_package(OpenMP)

# ---- Python extension module -------------------------------------------------
pybind11_add_module(numpycpp MODULE module.cpp)

# Include the numpycpp header-only library (parent dir)
target_include_directories(numpycpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../pycpp
)
if(Eigen3_FOUND)
    target_include_directories(numpycpp PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# Compiler flags — matches the bit-exact requirements
target_compile_options(numpycpp PRIVATE
    -O2
    -ffp-contract=off          # no implicit FMA for a+b*c (keeps Cody-Waite exact)
    -msse4.1 -mavx512f -mfma  # enable AVX-512 code paths
    # disable builtin replacements so our calls go through SVML/npy_math paths
    -fno-builtin-exp   -fno-builtin-log   -fno-builtin-sin
    -fno-builtin-cos   -fno-builtin-tan   -fno-builtin-pow
    -fno-builtin-sqrt  -fno-builtin-atan2 -fno-builtin-log2
    -fno-builtin-log10 -fno-builtin-asin  -fno-builtin-acos
    -fno-builtin-atan  -fno-builtin-exp2
    -fno-builtin-cbrt  -fno-builtin-expm1 -fno-builtin-log1p
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(numpycpp PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(numpycpp PRIVATE dl)

# Place .so next to the test scripts for easy import
set_target_properties(numpycpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX ""
    SUFFIX ".so"
)

# ---- Test target -------------------------------------------------------------
add_custom_target(test
    COMMAND "${Python3_EXECUTABLE}" -m pytest
            "${CMAKE_CURRENT_SOURCE_DIR}/test_all.py" -q --tb=short --no-header
    DEPENDS numpycpp
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running bit-exact alignment tests"
)

message(STATUS "numpycpp test module")
message(STATUS "  Python: ${Python3_VERSION}  (${Python3_EXECUTABLE})")
message(STATUS "  pybind11: ${pybind11_VERSION}")
message(STATUS "  Build: cmake --build <build_dir>")
message(STATUS "  Test:  cmake --build <build_dir> --target test")
