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

All APIs are tested against Python numpy under strict bit-level comparison: every IEEE 754 float bit must match exactly (500 tests, float64 + float32).

**Bit-exact math** is achieved by resolving numpy's own math functions from `_multiarray_umath.so` at runtime. The SVML bridge auto-detects your CPU and selects the same path numpy uses: AVX‑512 SVML (`__svml_exp8`) when available, or scalar `npy_exp`/`npy_log`/etc. otherwise. AVX‑512 intrinsics are isolated behind `__attribute__((target))` — the binary is safe on any x86_64 CPU (no SIGILL). Every transcendental function produces the exact same IEEE 754 bits as numpy on **all architectures**.

## Quick Start

### Dependencies

- **C++17** compiler (GCC >= 9, Clang >= 7, MSVC >= 2019)

### Usage

**Public headers** — these are the only files you should `#include`:

```cpp
#include "numpy/core.h"     // numpy.* equivalents
#include "numpy/linalg.h"   // numpy.linalg.*
#include "numpy/einsum.h"   // numpy.einsum
```

> `numpy/svml_bridge.h` and `numpy/npy_math_float.h` are **internal** — they are
> automatically pulled in by `core.h`. Do not include them directly.

```cpp
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
No tolerance, no `atol`/`rtol` — raw IEEE 754 bits must match exactly. 500 tests, float64 + float32.

```bash
cd tests
make                    # compile C++ test module
make test               # run all 500 tests (silent mode: only failures print)
```

To run with verbose output:

```bash
PYTHONPATH=tests:$PYTHONPATH python3 -m pytest tests/test_all.py -v
```

### Compiler flags for bit-exact alignment

Achieving bit-identical results with numpy requires strict control over floating-point code generation.
The Makefile applies the following flags:

```makefile
CXXFLAGS ?= -std=c++17 -O2 -fPIC -fopenmp                \
            -ffp-contract=off -ffloat-store -msse4.1      \
            -mavx512f -mfma                                \
            -fno-builtin-exp    -fno-builtin-log           \
            -fno-builtin-sin    -fno-builtin-cos           \
            -fno-builtin-tan    -fno-builtin-pow           \
            -fno-builtin-sqrt   -fno-builtin-atan2         \
            -fno-builtin-log2   -fno-builtin-log10         \
            -fno-builtin-asin   -fno-builtin-acos          \
            -fno-builtin-atan   -fno-builtin-exp2         \
            -fno-builtin-cbrt   -fno-builtin-expm1      \
            -fno-builtin-log1p
LDFLAGS   = -shared -ldl
```

| Flag | Purpose |
|------|---------|
| `-ffp-contract=off` | Disable FMA contraction — numpy does not contract |
| `-ffloat-store` | Prevent excess x87 precision in registers |
| `-msse4.1` | Required for einsum SSE intrinsics (`_mm_hadd_pd`, `_mm_insert_epi32`) |
| `-mavx512f -mfma` | Enable AVX‑512 compilation for SVML bridge. Intrinsics are runtime‑guarded via `__attribute__((target))` — safe on any x86_64 CPU (no SIGILL) |
| `-fno-builtin-<func>` | Prevent GCC from replacing math calls with built‑ins, ensuring the SVML bridge intercepts every call |
| `-ldl` | Required for `dlsym` at runtime to resolve numpy's math functions from `_multiarray_umath.so` |

> **Runtime CPU dispatch**: The SVML bridge auto‑detects AVX‑512 at runtime
> (`__builtin_cpu_supports`). On AVX‑512 hardware it calls numpy's SVML vector functions
> (`__svml_exp8`, etc.); otherwise it falls back to numpy's scalar math functions
> (`npy_exp`, `npy_log`, etc.). Both paths are resolved from the loaded
> `_multiarray_umath.so` via `dlsym`. AVX‑512 intrinsics are isolated behind
> `__attribute__((target("avx512f")))` so the binary runs safely on ANY
> x86_64 CPU — no SIGILL.

### Alignment status

The table below reflects the current bit-level parity between `numpycpp` C++ and Python numpy.
All 500 tests pass under strict IEEE 754 bit comparison (float64 + float32).

✅ = bit-exact on ALL architectures (SVML bridge with runtime CPU dispatch).

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
| **Math — transcendental** (exp, log, sin, cos, tan, asin, acos, atan, log10, log2, exp2, cbrt, expm1, log1p) | ✅ | ✅ | dlsym npy_* or SVML via bridge, bit-exact on all archs |
| **Math — power**   | ✅ | ✅ | npy_pow / npy_powf via SVML bridge |
| **Math — hypot**   | ✅ | ✅ | std::hypot — bit-exact (numpy matches libm) |
| **Math — atan2**   | ✅ | ✅ | npy_atan2 / npy_atan2f via SVML bridge |
| **Reduction** (sum, mean, max, min, any, all) | ✅ | ✅ | pairwise_sum matches numpy exactly |
| Statistical (std, var) | ✅ | ✅ | pairwise_sum + sqrt |
| Binary (maximum, minimum) | ✅ | ✅ | std::max/min, deterministic |
| **Dot product**    | ✅ | ✅ | pairwise_sum(a*b) — matches np.sum(a*b) |
| **Norm**           | ✅ | ✅ | pairwise_sum of squares + sqrt |
| **Norm (axis)**    | ✅ | ✅ | Fiber-wise pairwise_sum + sqrt |
| **Einsum**         | ✅ | ✅ | All patterns (ij,ij→i, ij,jk→ik, bij,bjk→bik, etc.) |

> **SVML bridge**: At runtime, `numpycpp` detects CPU features (`__builtin_cpu_supports("avx512f")`) and selects the same math path numpy uses — AVX‑512 SVML vector functions (`__svml_exp8`, etc.) on supported hardware, or scalar `npy_exp`/`npy_log`/etc. otherwise. Both are resolved from the loaded `_multiarray_umath.so` via `dlsym`. AVX‑512 intrinsics are isolated behind `__attribute__((target("avx512f")))` — the binary compiles and runs safely on ANY x86_64 CPU without SIGILL.
>
> **Reductions**: All reductions use numpy's pairwise summation algorithm (recursive split, 8-accumulator unrolled). This matches `np.sum` exactly. Dot products and norms build on pairwise_sum, not BLAS — matching `np.sum(a*b)` and `np.sqrt(np.sum(a*a))` respectively.

## Project Structure

```
numpycpp/
├── numpy/              # native C++ headers
│   ├── core.h          # [PUBLIC] numpy.* equivalents
│   ├── linalg.h        # [PUBLIC] numpy.linalg.*
│   ├── einsum.h        # [PUBLIC] numpy.einsum
│   ├── svml_bridge.h   # [INTERNAL] do not include directly
│   └── npy_math_float.h # [INTERNAL] do not include directly
├── pycpp/              # pybind11 wrappers (optional)
│   ├── core_py.h
│   ├── linalg_py.h
│   └── einsum_py.h
├── tests/              # bit-level precision tests + test module
│   ├── module.cpp      # pybind11 module for testing
│   ├── test_all.py     # single entry — all APIs, 500 tests, float64+float32
│   ├── conftest.py     # silent-mode output suppression
│   └── Makefile
├── CMakeLists.txt      # build & .deb packaging
└── README.md
```

## License

[MIT](LICENSE)
