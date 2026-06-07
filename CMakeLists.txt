cmake_minimum_required(VERSION 3.16)
project(numpycpp VERSION 1.21.2 LANGUAGES CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Math backend selection --------------------------------------------------
# NUMPYCPP_STD_ONLY=OFF (default): bit-exact backend
#   • dlsym resolves npy_exp/npy_log/... and __svml_exp8/... from numpy's .so
#   • OpenBLAS ILP64 for dot/matmul — same kernels as numpy
#   • Requires: -ldl at link time
#   • Results: IEEE 754 bit-identical to numpy on all architectures
#
# NUMPYCPP_STD_ONLY=ON: pure C++ standard-library backend
#   • std::exp, std::log, std::sin, ... from <cmath>
#   • Naive C++ loops for dot/gemm — compiler auto-vectorises with -O3
#   • No external dependencies (no dlopen, no numpy .so)
#   • Results: 0–2 ULP from numpy, NOT bit-exact
option(NUMPYCPP_STD_ONLY
    "Use only C++ standard library (no SVML/BLAS/dlopen). Performance-first, not bit-exact."
    OFF)

# ---- INTERFACE target for find_package consumers ----------------------------
add_library(numpycpp INTERFACE)
target_include_directories(numpycpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/numpycpp>
)
target_compile_features(numpycpp INTERFACE cxx_std_17)

if(NUMPYCPP_STD_ONLY)
    # Propagate the flag so every consumer gets the right backend automatically
    target_compile_definitions(numpycpp INTERFACE NUMPYCPP_STD_ONLY)
    message(STATUS "numpycpp backend: std  (pure <cmath> + C++ loops — no bit-exact guarantee)")
else()
    # dlsym required for SVML bridge + BLAS bridge
    target_link_libraries(numpycpp INTERFACE dl)
    message(STATUS "numpycpp backend: bitexact  (SVML + OpenBLAS ILP64 via dlsym)")
endif()

# ---- Install — header-only C++ library --------------------------------------
install(DIRECTORY numpy
    DESTINATION include/numpycpp
    FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY pycpp
    DESTINATION include/numpycpp
    FILES_MATCHING PATTERN "*.h"
    PATTERN ".pytest_cache" EXCLUDE
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    numpycpp-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/numpycpp-config.cmake
    INSTALL_DESTINATION lib/cmake/numpycpp
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/numpycpp-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/numpycpp-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/numpycpp-config-version.cmake
    DESTINATION lib/cmake/numpycpp
)

# ---- Docs & examples -------------------------------------------------------
install(FILES README.md
    DESTINATION share/numpycpp
)
install(DIRECTORY example/
    DESTINATION share/numpycpp/example
)

# ---- CPack DEB packaging ----------------------------------------------------
# Single DEB — header-only library, backend is a consumer compile-time choice.
# File: numpycpp-dev-<version>-Linux.deb
set(CPACK_PACKAGE_NAME                     "numpycpp-dev")
set(CPACK_PACKAGE_VERSION                  ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR                   "miaobyte")
set(CPACK_PACKAGE_CONTACT                  "miaobyte")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "C++ numpy-API header-only library (bitexact + std backends, C++17)")
set(CPACK_GENERATOR                        "DEB")
set(CPACK_DEB_COMPONENT_INSTALL            OFF)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER        "miaobyte")
set(CPACK_DEBIAN_PACKAGE_SECTION           "libdevel")
set(CPACK_DEBIAN_PACKAGE_PRIORITY          "optional")
set(CPACK_DEBIAN_FILE_NAME
    "numpycpp-dev-${CPACK_PACKAGE_VERSION}-Linux.deb")
set(CPACK_PACKAGING_INSTALL_PREFIX         "/usr")
# Wipe the old include tree on install/upgrade so stale headers cannot accumulate.
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/preinst")

include(CPack)

# Convenience target
add_custom_target(deb
    COMMAND cpack -G DEB
    COMMENT "Packaging numpycpp-dev-${CPACK_PACKAGE_VERSION}-Linux.deb"
)

message(STATUS "numpycpp v${PROJECT_VERSION}  (header-only C++ library)")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  DEB:  cmake --build <build_dir> --target deb")
message(STATUS "  Test: cmake -S tests -B tests/build && cmake --build tests/build --target test")
