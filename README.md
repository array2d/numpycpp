# numpycpp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-%3E%3D3.16-green.svg)](https://cmake.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Background

NumPy is fast — but its ceiling is locked by Python.

We created `numpycpp` to keep NumPy's familiar usage patterns while letting C++ break through Python's performance ceiling and accelerate your code further.

## Overview

`numpycpp` is a **header-only C++ library** implementing numpy's core API (`numpy.*`, `numpy.linalg.*`, `numpy.einsum`) with **bit-level precision alignment**. Raw pointer + size interface. Zero external dependencies — pure C++17 standard library.

All APIs are tested against Python numpy under strict bit-level comparison: every IEEE 754 float bit must match exactly. Where bit-exact parity is unattainable due to differing math library implementations (1‑ULP), it is documented explicitly.

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

**Ubuntu (DEB)**

Download the [latest `.deb` release](https://github.com/array2d/numpycpp/releases) or build from source:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make deb
sudo dpkg -i numpycpp-dev-*.deb
```

Headers are installed to `/usr/include/numpycpp/` along with CMake config. Consuming projects use:

```cmake
find_package(numpycpp REQUIRED)
target_link_libraries(myapp PRIVATE numpycpp::numpycpp)
```

**pybind11_add_module users**

With certain CMake / pybind11 version combinations, `pybind11_add_module` may lose IMPORTED targets
during generation (target exists at configure time but disappears at generate time).
If you hit this, use the variables-based fallback:

```cmake
find_package(numpycpp REQUIRED)
pybind11_add_module(mymodule module.cpp)
target_include_directories(mymodule PRIVATE ${numpycpp_INCLUDE_DIRS})
target_compile_features(mymodule PRIVATE cxx_std_17)
```

**Manual (header-only)**

Add `-Ipath/to/numpycpp` to your compiler flags and include the headers directly. No build step, no copy required.

### Testing

The test suite verifies **bit-level precision alignment** between every C++ function and Python numpy.
No tolerance, no `atol`/`rtol` — raw IEEE 754 bits must match exactly.

```bash
cd tests
make                    # compile C++ test module
make test               # run all tests (default: 337 tests, float64 + float32)
```

**API category filter** — limit tests to specific API groups via env var:

```bash
# Run only creation + reduction APIs (zeros_like, sum, mean, etc.)
NUMPYCPP_TEST_APIS=creation,reduction make test

# Run only elementary math (sqrt, exp, sin, pow, etc.)
NUMPYCPP_TEST_APIS=math make test

# Run only einsum patterns
NUMPYCPP_TEST_APIS=einsum make test

# Run all (default)
NUMPYCPP_TEST_APIS=all make test
```

Available categories:

| Category      | APIs covered |
|---------------|-------------|
| `creation`    | zeros_like, ones_like, full_like, empty_like, zeros, ones |
| `astype`      | astype int/bool, truncate_to_float32 |
| `math`        | sqrt, abs, exp, log, sin, cos, tan, power, clip, log10, log2, arcsin, arccos, arctan, round, floor, ceil, degrees, radians, sign |
| `reduction`   | sum, mean, max, min, any, all |
| `comparison`  | greater, less, equal, greater_equal, less_equal, not_equal |
| `logical`     | logical_and, logical_or, logical_not, logical_xor |
| `special`     | isnan, isinf, isfinite |
| `binary`      | arctan2, maximum, minimum |
| `manipulation`| diff, stack, concatenate, vstack, hstack, where, transpose, flatten, mean_axis, slice, take_cols, slice_assign, roll, flip, repeat, tile |
| `statistical` | std, var |
| `sorting`     | argsort, argmax, argmin |
| `setops`      | isin, intersect1d, interp, safe_divide |
| `access`      | array_get, asarray, to_vector |
| `linalg`      | norm, norm_axis, dot |
| `einsum`      | all einsum patterns (explicit + implicit mode) |

### Alignment status

The table below reflects the current bit-level parity between `numpycpp` C++ and Python numpy.
Tests marked ✅ are bit-exact (all IEEE 754 bits match). Tests marked ⚠️ differ by ≤ 1 ULP.

| API group         | float64 | float32 | Notes |
|-------------------|:-------:|:-------:|-------|
| Creation          | ✅ | ✅ | All creation APIs bit-exact |
| Astype            | ✅ | ✅ | All conversions bit-exact |
| Comparison        | ✅ | ✅ | All comparisons bit-exact |
| Logical           | ✅ | ✅ | bool-only, always exact |
| Special values    | ✅ | ✅ | isnan / isinf / isfinite bit-exact |
| Manipulation      | ✅ | ✅ | diff, stack, transpose, slice etc. bit-exact |
| Sorting           | ✅ | ✅ | argsort, argmax, argmin bit-exact |
| Setops / interp   | ✅ | ✅ | isin, intersect1d, interp bit-exact |
| Access / convert  | ✅ | ✅ | array_get, asarray, to_vector bit-exact |
| **Math — sqrt, abs, clip, round, floor, ceil, degrees, radians, sign** | ✅ | ✅ | These are bit-exact |
| **Math — transcendental** (exp, log, sin, cos, tan, log10, log2, arcsin, arccos, arctan) | ⚠️ | ⚠️ | 1-ULP: `std::` vs numpy libm |
| **Math — power**   | ✅ | ⚠️ | float32: 1-ULP from libm |
| **Reduction** (sum 2d, mean float32) | ⚠️ | ⚠️ | Accumulation-order differences |
| Statistical (std, var) | ⚠️ | ⚠️ | Accumulation-order differences |
| Binary (arctan2 scalar float32) | ✅ | ⚠️ | 1-ULP from libm |
| slice_assign float32 | ✅ | ⚠️ | pybind11 overload dispatch issue |
| **Dot product**    | ⚠️ | ✅ | float64: accumulation-order |
| **Norm**           | ✅ | ⚠️ | float32: sqrt + accumulation |
| **Einsum** (most patterns) | ✅ | ✅ | Simple patterns bit-exact |
| **Einsum** (large accumulations) | ⚠️ | ⚠️ | Multi-element accumulation drift |

> **Why 1‑ULP?** The C++ standard library (`std::exp`, `std::log`, etc.) and numpy's underlying libm may use different polynomial approximations or rounding strategies, leading to a single-bit difference in the last place. This is inherent to the math library, not a bug in `numpycpp`.

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
├── tests/              # bit-level precision tests + test module
│   ├── module.cpp      # pybind11 module for testing
│   ├── test_all.py     # single entry — all APIs, float64+float32
│   ├── conftest.py     # fixtures + NUMPYCPP_TEST_APIS filter
│   ├── utils.py        # bit-level comparison engine
│   └── Makefile
├── CMakeLists.txt      # build & .deb packaging
└── README.md
```

## License

[MIT](LICENSE)
