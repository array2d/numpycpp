# numpcpp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-%3E%3D3.16-green.svg)](https://cmake.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Background

NumPy is fast — but its ceiling is locked by Python.

We created `numpycpp` to keep NumPy's familiar usage patterns while letting C++ break through Python's performance ceiling and accelerate your code further.

## Overview

`numpycpp` is a **header-only C++ library** implementing numpy's core API (`numpy.*`, `numpy.linalg.*`, `numpy.einsum`) with pixel-level precision alignment. Raw pointer + size interface. Zero external dependencies — pure C++17 standard library.

## Quick Start

### Dependencies

- **C++17** compiler (GCC >= 9, Clang >= 7, MSVC >= 2019)

### Usage

```cpp
#include "numpy/core.h"

std::vector<double> data = {1.0, 4.0, 9.0};
std::vector<double> result(data.size());

numpy::sqrt(data.data(), result.data(), data.size());
// result → {1.0, 2.0, 3.0}

double s = numpy::sum(data.data(), data.size());
// s → 14.0
```

### Install

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make deb
sudo dpkg -i numpcpp-dev-*.deb
# headers installed to /usr/include/numpycpp/
```

### Testing

The test suite verifies pixel-level precision alignment between every C++ function and Python numpy.

```bash
cd tests
make                    # compile test module
make test               # run all 336 tests
```

## Project Structure

```
numpycpp/
├── numpy/              # native C++ headers (zero dependency)
│   ├── core.h          # numpy.* equivalents (~80 functions)
│   ├── linalg.h        # numpy.linalg.* equivalents
│   └── einsum.h        # numpy.einsum
├── pycpp/              # pybind11 wrappers (optional)
│   ├── core_py.h
│   ├── linalg_py.h
│   └── einsum_py.h
├── tests/              # precision tests + test module
│   ├── module.cpp      # pybind11 module for testing
│   ├── Makefile
│   └── test_*.py
├── CMakeLists.txt      # build & .deb packaging
└── README.md
```

## License

[MIT](LICENSE)
