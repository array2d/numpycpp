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

All APIs are tested against Python numpy under strict bit-level comparison: every IEEE 754 float bit must match exactly (336 tests, float64 + float32).

**Bit-exact math** is achieved via an SVML bridge that resolves numpy's own transcendental functions (`__svml_exp8`, `__svml_sin8`, etc.) from the loaded `_multiarray_umath.so` at runtime. This guarantees that `exp`, `log`, `sin`, `cos`, `tan`, and all other transcendental functions produce the exact same bits as numpy. On platforms without AVX-512, the bridge falls back to `std::` (1‑ULP).

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
No tolerance, no `atol`/`rtol` — raw IEEE 754 bits must match exactly. 336 tests, float64 + float32.

```bash
cd tests
make                    # compile C++ test module
make test               # run all 336 tests (silent mode: only failures print)
```

To run with verbose output:

```bash
PYTHONPATH=tests:$PYTHONPATH python3 -m pytest tests/test_all.py -v
```

### Alignment status

The table below reflects the current bit-level parity between `numpycpp` C++ and Python numpy.
All 336 tests pass under strict IEEE 754 bit comparison (float64 + float32).

✅ = bit-exact on AVX-512 (SVML bridge active).  
🔶 = 1-ULP on non-AVX-512 (falls back to `std::` math).

| API group         | float64 | float32 | Notes |
|-------------------|:-------:|:-------:|-------|
| Creation          | ✅ | ✅ | zeros_like, ones_like, full_like, zeros, ones |
| Astype            | ✅ | ✅ | astype int/bool, truncate float32 |
| Comparison        | ✅ | ✅ | greater, less, equal, not_equal, etc. |
| Logical           | ✅ | ✅ | bool-only (and/or/not/xor) |
| Special values    | ✅ | ✅ | isnan, isinf, isfinite |
| Manipulation      | ✅ | ✅ | diff, stack, concatenate, transpose, slice, roll, flip, repeat, tile, where |
| Sorting           | ✅ | ✅ | argsort, argmax, argmin |
| Setops / interp   | ✅ | ✅ | isin, intersect1d, interp, safe_divide |
| Access / convert  | ✅ | ✅ | array_get, asarray, to_vector |
| **Math — element-wise** (sqrt, abs, sign, clip, round, floor, ceil, degrees, radians) | ✅ | ✅ | Pure C++, no libm dependency |
| **Math — transcendental** (exp, log, sin, cos, tan, asin, acos, atan, log10, log2, exp2) | ✅ | 🔶 | SVML bridge → bit-exact; fallback → std:: (1-ULP) |
| **Math — power**   | ✅ | 🔶 | SVML bridge for f64; f32: std::pow |
| **Math — atan2**   | ✅ | 🔶 | npy_atan2 via SVML bridge |
| **Reduction** (sum, mean, max, min, any, all) | ✅ | ✅ | pairwise_sum matches numpy exactly |
| Statistical (std, var) | ✅ | ✅ | pairwise_sum + sqrt |
| Binary (maximum, minimum) | ✅ | ✅ | std::max/min, deterministic |
| **Dot product**    | ✅ | ✅ | pairwise_sum(a*b) — matches np.sum(a*b) |
| **Norm**           | ✅ | ✅ | pairwise_sum of squares + sqrt |
| **Norm (axis)**    | ✅ | ✅ | Fiber-wise pairwise_sum + sqrt |
| **Einsum**         | ✅ | ✅ | All patterns (ij,ij→i, ij,jk→ik, bij,bjk→bik, etc.) |

> **SVML bridge**: On AVX-512 platforms, `numpycpp` resolves numpy's own SVML vector functions (`__svml_exp8`, `__svml_sin8`, etc.) from the loaded `_multiarray_umath.so` via `dlsym`. This guarantees bit-identical transcendental results. On non-AVX-512, `std::` fallbacks produce ≤ 1 ULP difference.
>
> **Reductions**: All reductions use numpy's pairwise summation algorithm (recursive split, 8-accumulator unrolled). This matches `np.sum` exactly. Dot products and norms build on pairwise_sum, not BLAS — matching `np.sum(a*b)` and `np.sqrt(np.sum(a*a))` respectively.

## Project Structure

```
numpycpp/
├── numpy/              # native C++ headers (zero dependency)
│   ├── core.h          # numpy.* equivalents (pairwise_sum, element-wise, reductions, etc.)
│   ├── linalg.h        # numpy.linalg.* (norm, norm_axis)
│   ├── einsum.h        # numpy.einsum (SSE SIMD xmm, OpenMP, stride-based fast path)
│   ├── svml_bridge.h  # runtime dlsym resolver for numpy's SVML vector functions
│   └── npy_math_float.h # numpy's own float32 polynomial approximations
├── pycpp/              # pybind11 wrappers (optional)
│   ├── core_py.h
│   ├── linalg_py.h
│   └── einsum_py.h
├── tests/              # bit-level precision tests + test module
│   ├── module.cpp      # pybind11 module for testing
│   ├── test_all.py     # single entry — all APIs, 336 tests, float64+float32
│   ├── conftest.py     # silent-mode output suppression
│   └── Makefile
├── CMakeLists.txt      # build & .deb packaging
└── README.md
```

## License

[MIT](LICENSE)
