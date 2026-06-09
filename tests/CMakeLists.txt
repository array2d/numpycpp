# Build numpycpp Python test/benchmark module.
#
# Two build modes (select with -DNUMPYCPP_STD_ONLY):
#
#   OFF (default) — bit-exact mode:
#     All 900 tests verify IEEE 754 bit-identical results vs numpy.
#     Requires: dlsym, numpy .so loaded, AVX-512 capable machine (or fallback).
#     cmake -S tests -B tests/build
#
#   ON — std / performance-first mode:
#     Transcendental + linalg tests use atol/rtol tolerance (0–2 ULP allowed).
#     No external dependencies — pure C++17 std only.
#     cmake -S tests -B tests/build -DNUMPYCPP_STD_ONLY=ON
#
# Build:   cmake --build tests/build
# Test:    cd tests && python3 -m pytest test_all.py -q

cmake_minimum_required(VERSION 3.16)
project(numpycpp_tests CXX)

option(NUMPYCPP_STD_ONLY
    "Use only C++ standard library (no SVML/BLAS/dlopen). Performance-first, not bit-exact."
    OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Find dependencies -------------------------------------------------------
# Python3 (for pybind11 extension)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# pybind11 — prefer the copy installed alongside Python
# pybind11 v2.x uses --cmake; v3.x uses --cmakedir — try both
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_cmake_result
    ERROR_QUIET
)
if(NOT (_pybind11_cmake_result EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}"))
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmake
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_cmake_result
        ERROR_QUIET
    )
endif()
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

# AVX-512 runtime probe — compile AND run a tiny binary that executes a ZMM
# instruction.  check_cxx_source_runs returns TRUE only if the binary exits 0,
# so it correctly distinguishes machines where CPUID says avx512f but the
# hypervisor traps ZMM execution (e.g. some Azure VM SKUs used by GitHub Actions).
include(CheckCXXSourceRuns)
set(CMAKE_REQUIRED_FLAGS "-mavx512f -mfma")
check_cxx_source_runs("
    #include <immintrin.h>
    int main() {
        __m512d a = _mm512_set1_pd(2.0);
        __m512d b = _mm512_sqrt_pd(a);
        (void)b;
        return 0;
    }
" NUMPYCPP_AVX512_WORKS)
unset(CMAKE_REQUIRED_FLAGS)

# ---- Python extension module -------------------------------------------------
pybind11_add_module(numpycpp MODULE module.cpp)

# Include the numpycpp header-only library (parent dir)
target_include_directories(numpycpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
if(Eigen3_FOUND)
    target_include_directories(numpycpp PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

if(NUMPYCPP_STD_ONLY)
    # ── std / performance-first build ─────────────────────────────────────────
    # No bit-exact requirement: let the compiler optimise freely.
    # FMA contraction, full auto-vectorisation, and native SIMD are all allowed.
    target_compile_definitions(numpycpp PRIVATE NUMPYCPP_STD_ONLY)
    target_compile_options(numpycpp PRIVATE
        -O3
        -march=native            # auto-vectorise with available SIMD (AVX-512 etc.)
        # No -ffp-contract=off: FMA fusion allowed for performance
        # No -fno-builtin-*: std::exp etc. can use GCC builtins (fast path)
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(numpycpp PRIVATE OpenMP::OpenMP_CXX)
    endif()
    # No -ldl: no dlsym in std backend
    message(STATUS "Test build: std mode  (-O3 -march=native, no dlsym, no SVML)")
else()
    # ── bit-exact build (default) ─────────────────────────────────────────────
    # Flags determined empirically: each flag's removal was tested against all
    # 900 tests. Only flags whose removal caused failures are marked REQUIRED.
    target_compile_options(numpycpp PRIVATE
        -O2
        -ffp-contract=off          # REQUIRED: no implicit FMA (keeps Cody-Waite exact)
        -msse4.1 -mfma             # baseline SSE4.1 + FMA
        # disable builtin replacements so calls go through SVML/npy_math paths
        -fno-builtin-exp   -fno-builtin-log   -fno-builtin-sin
        -fno-builtin-cos   -fno-builtin-tan   -fno-builtin-pow
        -fno-builtin-sqrt  -fno-builtin-atan2 -fno-builtin-log2
        -fno-builtin-log10 -fno-builtin-asin  -fno-builtin-acos
        -fno-builtin-atan  -fno-builtin-exp2
        -fno-builtin-cbrt  -fno-builtin-expm1 -fno-builtin-log1p
    )
    if(NUMPYCPP_AVX512_WORKS)
        target_compile_options(numpycpp PRIVATE
            -mavx512f -mprefer-vector-width=256)
        message(STATUS "AVX-512: usable — SVML + wide-loop path enabled")
    else()
        message(STATUS "AVX-512: not usable (missing or hypervisor-trapped) — scalar npy_* path (still bit-exact)")
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(numpycpp PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(numpycpp PRIVATE dl)
    message(STATUS "Test build: bit-exact mode  (-O2 -ffp-contract=off, dlsym, SVML)")
endif()

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
