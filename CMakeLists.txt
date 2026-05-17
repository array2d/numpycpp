cmake_minimum_required(VERSION 3.16)
project(numpycpp VERSION 1.21.3 LANGUAGES CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- INTERFACE target for find_package consumers ----------------------------
add_library(numpycpp INTERFACE)
target_include_directories(numpycpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/numpycpp>
)
target_compile_features(numpycpp INTERFACE cxx_std_17)

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

install(TARGETS numpycpp
    EXPORT numpycpp-targets
    DESTINATION lib/cmake/numpycpp
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

install(EXPORT numpycpp-targets
    NAMESPACE numpycpp::
    DESTINATION lib/cmake/numpycpp
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/numpycpp-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/numpycpp-config-version.cmake
    DESTINATION lib/cmake/numpycpp
)

# ---- CPack DEB packaging ----------------------------------------------------
set(CPACK_PACKAGE_NAME                     "numpycpp-dev")
set(CPACK_PACKAGE_VERSION                  ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR                   "miaobyte")
set(CPACK_PACKAGE_CONTACT                  "miaobyte")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY      "C++ pixel-level alignment of Python numpy — header-only library")
set(CPACK_GENERATOR                        "DEB")
set(CPACK_DEB_COMPONENT_INSTALL            OFF)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER        "miaobyte")
set(CPACK_DEBIAN_PACKAGE_SECTION           "libdevel")
set(CPACK_DEBIAN_PACKAGE_PRIORITY          "optional")
set(CPACK_DEBIAN_FILE_NAME                 "numpycpp-dev-${CPACK_PACKAGE_VERSION}-Linux.deb")
set(CPACK_PACKAGING_INSTALL_PREFIX         "/usr")

include(CPack)

# Convenience target
add_custom_target(deb
    COMMAND cpack -G DEB
    COMMENT "Packaging numpycpp-dev-${CPACK_PACKAGE_VERSION}-Linux.deb"
)

message(STATUS "numpycpp v${PROJECT_VERSION}  (header-only C++ library)")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  DEB:  make deb  →  numpycpp-dev-${CPACK_PACKAGE_VERSION}-Linux.deb")
message(STATUS "  Test: cd tests/ && make  →  build + run 336 tests")
