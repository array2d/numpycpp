cmake_minimum_required(VERSION 3.16)
project(numpycpp VERSION 1.21.2 LANGUAGES CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Install — header-only C++ library --------------------------------------
install(DIRECTORY numpy/
    DESTINATION include/numpycpp
    FILES_MATCHING PATTERN "*.h"
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
    COMMENT "Packaging numpcpp-dev-${CPACK_PACKAGE_VERSION}-Linux.deb"
)

message(STATUS "numpycpp v${PROJECT_VERSION}  (header-only C++ library)")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  DEB:  make deb  →  numpcpp-dev-${CPACK_PACKAGE_VERSION}-Linux.deb")
message(STATUS "  Test: cd tests/ && make  →  build + run 336 tests")
