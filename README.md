# numpycpp

[![CI](https://github.com/array2d/numpycpp/actions/workflows/ci.yml/badge.svg)](https://github.com/array2d/numpycpp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-%3E%3D3.16-green.svg)](https://cmake.org/)
[![Tests](https://img.shields.io/badge/tests-981%20bit--exact-brightgreen.svg)](tests/test_all.py)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Background

NumPy is fast — but its ceiling is locked by Python.

We created `numpycpp` to keep NumPy's familiar usage patterns while letting C++ break through Python's performance ceiling and accelerate your code further.

## Overview

`numpycpp` is a **header-only C++ library** implementing numpy's core API (`numpy.*`, `numpy.linalg.*`, `numpy.einsum`) with **bit-level precision alignment**. Raw pointer + size interface. Zero external dependencies — pure C++17 standard library.

All APIs are tested against Python numpy under strict bit-level comparison: every IEEE 754 float bit must match exactly (981 tests, float64 + float32, including NaN passthrough, signed-zero, ±∞, domain-error cases, and advanced indexing).

**Bit-exact math** is achieved by resolving numpy's own math functions from `_multiarray_umath.so` at runtime. The SVML bridge auto-detects your CPU and selects the same path numpy uses: AVX‑512 SVML (`__svml_exp8`) when available, or scalar `npy_exp`/`npy_log`/etc. otherwise. AVX‑512 intrinsics are isolated behind `__attribute__((target))` — the binary is safe on any x86_64 CPU (no SIGILL). Every transcendental function produces the exact same IEEE 754 bits as numpy on **all architectures**.

## Quick Start

### Dependencies

- **C++17** compiler (GCC >= 9, Clang >= 7, MSVC >= 2019)

### Usage

**Public headers** — include the umbrella or individual modules:

```cpp
#include <numpycpp/numpy.h>          // ← single entry point (recommended)

// or include only what you need:
#include <numpycpp/init.h>           // zeros_like, ones_like, full, arange, …
#include <numpycpp/elementwise.h>    // sqrt, exp, sin, astype, …
#include <numpycpp/reduce.h>         // sum, mean, std, var, cumsum, …
#include <numpycpp/manipulation.h>   // transpose, take, slice, putmask, …
#include <numpycpp/io.h>             // isin, interp, unwrap, …
#include <numpycpp/linalg.h>         // dot, norm, matmul, einsum
```

> `numpycpp/detail/` headers are **internal** — automatically pulled in by the
> public headers. Do not include them directly.
>
> **pybind11 users** — include `<numpycpp/numpy_py.h>` instead to get the full
> set of pybind11 wrapper functions (`numpy::sum(py::array_t<T>)` etc.).

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
cmake ..
make deb
sudo dpkg -i numpycpp-dev-*.deb
```

Headers are installed to `/usr/include/numpycpp/` along with a CMake config that supports both backends.

**CMake — bitexact backend (default)**

```cmake
find_package(numpycpp REQUIRED)
target_link_libraries(myapp PRIVATE numpycpp::numpycpp)
# cmake propagates -ldl automatically — no extra flags needed
```

**CMake — std backend**

```cmake
set(NUMPYCPP_STD_ONLY ON)           # set BEFORE find_package
find_package(numpycpp REQUIRED)
target_link_libraries(myapp PRIVATE numpycpp::numpycpp)
# cmake propagates -DNUMPYCPP_STD_ONLY automatically — no extra flags needed
```

**pybind11_add_module users**

With certain CMake / pybind11 version combinations, `pybind11_add_module` may lose IMPORTED targets
during generation. If you hit this, use the variables-based fallback:

```cmake
set(NUMPYCPP_STD_ONLY OFF)          # or ON for std backend
find_package(numpycpp REQUIRED)
pybind11_add_module(mymodule module.cpp)
target_include_directories(mymodule PRIVATE ${numpycpp_INCLUDE_DIRS})
target_compile_features(mymodule PRIVATE cxx_std_17)
# bitexact: add manually → target_link_libraries(mymodule PRIVATE dl)
# std:      add manually → target_compile_definitions(mymodule PRIVATE NUMPYCPP_STD_ONLY)
```

**Manual (header-only)**

Add `-Ipath/to/numpycpp` to your compiler flags and include the headers directly. No build step, no copy required.
- Bitexact backend: add `-ldl` at link time. See compiler flags table below for required flags (`-ffp-contract=off`, `-msse4.1`, etc.).
- Std backend: add `-DNUMPYCPP_STD_ONLY` (no `-ldl` needed)

### Testing

The test suite verifies **bit-level precision alignment** between every C++ function and Python numpy.
No tolerance, no `atol`/`rtol` — raw IEEE 754 bits must match exactly.
981 tests: float64 + float32, including NaN passthrough, signed-zero, ±∞, domain errors, advanced indexing, and AVX-512 boundary sizes.

```bash
# build
cmake -S tests -B tests/build
cmake --build tests/build -j$(nproc)

# run (silent on pass — failures print hex diff)
cd tests && python3 -m pytest test_all.py -q --tb=short --no-header
```

### Two backends — choose at cmake time

numpycpp ships two interchangeable math backends selected via a single cmake flag.
All public APIs (`numpy::exp`, `numpy::dot`, `numpy::einsum`, …) are identical;
only the internal implementation and precision guarantee differ.

The library is **header-only** — both backends live in the same installed headers.
The backend is a **consumer compile-time choice**, not an install-time choice.
One DEB installs everything; `NUMPYCPP_STD_ONLY` selects the backend.

```cmake
cmake -DNUMPYCPP_STD_ONLY=OFF ..   # default — bit-exact backend
cmake -DNUMPYCPP_STD_ONLY=ON  ..   # std / performance-first backend
```

| Property | `NUMPYCPP_STD_ONLY=OFF` (bitexact) | `NUMPYCPP_STD_ONLY=ON` (std) |
|---|---|---|
| **Transcendental math** | `dlsym` → `npy_exp` / `__svml_exp8` | `std::exp`, `std::log`, … |
| **Dot / matmul** | OpenBLAS ILP64 via `dlsym` | Pure C++ loops (auto-vectorised) |
| **Precision vs numpy** | **IEEE 754 bit-identical** | 0–2 ULP (not bit-exact) |
| **External deps** | libdl + numpy `.so` loaded | **None** — pure C++17 |
| **DEB package** | same `numpycpp-dev-<ver>-Linux.deb` | same `numpycpp-dev-<ver>-Linux.deb` |
| **cmake propagation** | `target_link_libraries(… dl)` | `target_compile_definitions(… NUMPYCPP_STD_ONLY)` |

#### Compiler flags — bitexact backend (`NUMPYCPP_STD_ONLY=OFF`)

The minimum set was determined empirically: each flag was removed in isolation
and the full 981-test suite was re-run. Only flags whose removal caused at
least one test failure are marked **required**.

The SVML bridge compiles cleanly with or without `-mavx512f` — all AVX-512 code
is guarded by `#ifdef __AVX512F__`. Without `-mavx512f`, the scalar `npy_*`
path is used (resolved via `dlsym` from numpy's `.so`), which is still bit-exact.
We recommend using cmake's `check_cxx_source_runs` to probe AVX-512 at configure
time — see [`tests/CMakeLists.txt`](tests/CMakeLists.txt) for a complete example.

```cmake
target_compile_options(<target> PRIVATE
    -O2
    -ffp-contract=off          # REQUIRED — see below
    -msse4.1                   # REQUIRED — see below
    -mfma                      # REQUIRED with -mavx512f — see below
    -mavx512f                  # recommended — see below (use cmake probe)
    -mprefer-vector-width=256  # REQUIRED with -mavx512f — see below
    # Defensive: prevent GCC from substituting npy_* call sites with builtins.
    # No test currently depends on these — kept as future-proofing.
    -fno-builtin-exp   -fno-builtin-log   -fno-builtin-sin
    -fno-builtin-cos   -fno-builtin-tan   -fno-builtin-pow
    -fno-builtin-sqrt  -fno-builtin-atan2 -fno-builtin-log2
    -fno-builtin-log10 -fno-builtin-asin  -fno-builtin-acos
    -fno-builtin-atan  -fno-builtin-exp2
    -fno-builtin-cbrt  -fno-builtin-expm1 -fno-builtin-log1p
)
target_link_libraries(<target> PRIVATE dl)   # REQUIRED — dlsym
```

| Flag | Status | Why required | Consequence of removal |
|------|:------:|-------------|------------------------|
| `-O2` | recommended | Standard optimization level. Without optimization, bit-exact results are preserved but performance degrades significantly. | Slow execution (correctness unaffected). |
| `-ffp-contract=off` | **required** | Prevents silent FMA fusion of `a*b+c`. einsum loops must match numpy's BLAS multiply-then-add order. | 36 einsum tests fail with ±1 ULP (verified). |
| `-msse4.1` | **required** | `linalg.h` uses `_mm_insert_epi32` (SSE4.1 instruction) unconditionally. | Hard compile error: `'__builtin_ia32_pinsrd' requires SSE4.1`. |
| `-mfma` | **required** | `avx512_loops.h` uses `_mm512_fmadd_ps/pd` inside `#ifdef __AVX512F__`. Only needed together with `-mavx512f`. | Hard compile error if `-mavx512f` is enabled. |
| `-mavx512f` | recommended | Enables the AVX-512 SVML vector path (`__svml_exp8`, etc.) and wide-loop specializations in `avx512_loops.h`. Without it, the scalar `npy_*` path is used — still bit-exact, but 4–8× slower on large arrays. **Safe on non-AVX-512 CPUs:** all AVX-512 code is isolated behind `__attribute__((target("avx512f")))` + runtime `cpu_has_avx512f()` guard. | Fallback to scalar `npy_*` path (still bit-exact, slower). |
| `-mprefer-vector-width=256` | **required** | Prevents GCC from emitting ZMM (512-bit) instructions in auto-vectorized code when `-mavx512f` is enabled. Some cloud VMs expose `avx512f` in CPUID but trap ZMM via hypervisor XSAVE. Explicit AVX-512 intrinsics are safe (runtime-guarded), but unguarded auto-vectorized ZMM causes SIGILL. No effect without `-mavx512f`. | SIGILL at startup on some cloud VMs (GitHub Actions azure runners). |
| `-ldl` | **required** | `dlsym`/`dlopen` locate numpy's `_multiarray_umath.so` at runtime. | Link error: `undefined reference to 'dlsym'`. |
| `-fno-builtin-*` (full list) | recommended | Prevents GCC from substituting npy_* call sites with builtin implementations. numpycpp resolves math functions via dlsym at runtime, never calling `exp()` from `<cmath>` directly — so no current effect. Kept as defensive guard against future GCC versions. | No test failure when removed today (verified on GCC 9–14). |

#### Compiler flags — std backend (`NUMPYCPP_STD_ONLY=ON`)

```cmake
target_compile_definitions(<target> PRIVATE NUMPYCPP_STD_ONLY)
target_compile_options(<target> PRIVATE
    -O3
    -march=native              # auto-vectorise with all available SIMD
)
# No -ldl needed — no dlsym in std backend
```

| Flag | Status | Why |
|------|:------:|-----|
| `NUMPYCPP_STD_ONLY` | **required** | Selects `std_math_backend.h` + `std_linalg_backend.h` instead of SVML/BLAS bridges. Set via `cmake -DNUMPYCPP_STD_ONLY=ON` or `set(NUMPYCPP_STD_ONLY ON)` before `find_package`. |
| `-O3 -march=native` | recommended | Enables full auto-vectorisation of the C++ loops (exp/dot/gemm). Without optimisation, std backend is slow. |
| `-ffp-contract=off` | **not needed** | FMA contraction is welcome in std mode — improves precision and performance of gemm/dot. |
| `-mavx512f -mprefer-vector-width=256` | **not needed** | SVML bridge not compiled in; no ZMM-trap risk. `-march=native` selects appropriate SIMD automatically. |
| `-ldl` | **not needed** | No dlopen/dlsym in std backend. |

> **Runtime CPU dispatch (bitexact only)**: The SVML bridge auto‑detects AVX‑512
> at runtime (`__builtin_cpu_supports("avx512f")`). On AVX‑512 hardware it calls
> numpy's SVML vector functions (`__svml_exp8`, etc.); otherwise it falls back to
> numpy's scalar `npy_exp`/`npy_log`/etc. Both paths are resolved from the
> loaded `_multiarray_umath.so` via `dlsym`. AVX‑512 intrinsics are isolated
> behind `__attribute__((target("avx512f")))` — safe on any x86_64 CPU.

### Alignment status

Two backends, same API — choose with `cmake -DNUMPYCPP_STD_ONLY=ON/OFF`.

| Legend | Meaning |
|--------|---------|
| ✅ | IEEE 754 bit-identical to numpy (float64 + float32) |
| 〜 | Correct result, 0–2 ULP from numpy (not bit-exact) |

| Category | Functions | `bitexact` (`STD_ONLY=OFF`) | `std` (`STD_ONLY=ON`) |
|----------|-----------|:---------------------------:|:---------------------:|
| **Creation** | `zeros_like` `ones_like` `full_like` `empty_like` `zeros` `ones` `full` | ✅ | ✅ |
| **Type conversion** | `astype` (int/float/bool/int64) `truncate_to_float32` | ✅ | ✅ |
| **Comparison** | `greater` `less` `equal` `not_equal` `greater_equal` `less_equal` | ✅ | ✅ |
| **Logical** | `logical_and` `logical_or` `logical_not` `logical_xor` | ✅ | ✅ |
| **Special values** | `isnan` `isinf` `isfinite` | ✅ | ✅ |
| **Manipulation** | `diff` `stack` `vstack` `hstack` `concatenate` `transpose` `flatten` `squeeze` `roll` `flip` `repeat` `tile` `where` | ✅ | ✅ |
| **Advanced indexing** | `take` `compress` `slice` (N-D + step) `put` `putmask` `slice_assign` | ✅ | ✅ |
| **Sorting** | `argsort` `argmax` `argmin` | ✅ | ✅ |
| **Set / interp** | `isin` `intersect1d` `interp` `unwrap` `flatnonzero` `safe_divide` | ✅ | ✅ |
| **Reduction** | `sum` `mean` `max` `min` `any` `all` `std` `var` `cumsum` `mean` (axis) | ✅ | ✅ |
| **Math — pure C++** | `sqrt` `abs` `sign` `clip` `round` `floor` `ceil` `degrees` `radians` `maximum` `minimum` | ✅ | ✅ |
| **Math — transcendental** | `exp` `log` `sin` `cos` `tan` `arcsin` `arccos` `arctan` `log10` `log2` `exp2` `cbrt` `expm1` `log1p` | ✅ | 〜 0–1 ULP |
| **Math — power / atan2** | `power` `arctan2` | ✅ | 〜 0–1 ULP |
| **Math — hypot** | `hypot` | ✅ | ✅ |
| **Dot product** | `numpy.dot` (1-D) | ✅ | 〜 0–1 ULP |
| **Norm** | `numpy.linalg.norm` (scalar + axis) | ✅ | 〜 0–1 ULP |
| **Matmul** | `numpy.matmul` (2-D, 1-D×2-D, 2-D×1-D, batched 3-D) | ✅ | 〜 0–2 ULP |
| **Einsum** | `ij,ij→i` `ij,jk→ik` `bij,bjk→bik` and all 2-operand patterns | ✅ | 〜 0–2 ULP |
| **Matrix inverse** | `numpy.linalg.inv` (N×N) | ✅ | 〜 0–2 ULP |

> **bitexact backend**: transcendentals resolved via `dlsym` from numpy's
> `_multiarray_umath.so` — same `npy_exp`/`npy_log` kernels numpy uses, with
> AVX‑512 SVML vector path (`__svml_exp8` etc.) when available.
> Dot/matmul/einsum use OpenBLAS ILP64 (`cblas_sgemm64_`) — the same BLAS
> numpy delegates to. Results are IEEE 754 bit-identical on **all architectures**.
>
> **std backend**: transcendentals use `std::exp`/`std::sin`/… from `<cmath>`
> (glibc, typically 0–1 ULP). Dot/matmul/einsum use plain C++ loops
> (compiler auto-vectorises with `-O3 -march=native`). No external dependencies.
>
> **Reductions** (both backends): pairwise summation algorithm (recursive split,
> 8-accumulator unrolled) — matches `np.sum` exactly.
> **hypot** (both backends): `std::hypot` — numpy delegates to the same libm call.

## Project Structure

```
numpycpp/
├── numpycpp/                   # header-only library (all public + internal headers)
│   ├── numpy.h                 # [PUBLIC]   umbrella — includes all core modules below
│   ├── numpy_py.h              # [PUBLIC]   umbrella — includes all pybind11 wrappers below
│   ├── init.h                  # [PUBLIC]   zeros_like, ones_like, full, arange, linspace, …
│   ├── init_py.h               # [PUBLIC]   pybind11 wrappers for init.h
│   ├── elementwise.h           # [PUBLIC]   sqrt/exp/sin/…, comparison, logical, astype
│   ├── elementwise_py.h        # [PUBLIC]   pybind11 wrappers for elementwise.h
│   ├── reduce.h                # [PUBLIC]   sum/mean/std/var/cumsum, axis reductions
│   ├── reduce_py.h             # [PUBLIC]   pybind11 wrappers for reduce.h
│   ├── manipulation.h          # [PUBLIC]   transpose/take/slice/put/putmask/argsort/…
│   ├── manipulation_py.h       # [PUBLIC]   pybind11 wrappers for manipulation.h
│   ├── io.h                    # [PUBLIC]   isin, interp, unwrap, safe_divide, …
│   ├── io_py.h                 # [PUBLIC]   pybind11 wrappers for io.h
│   ├── linalg.h                # [PUBLIC]   dot, norm, matmul, einsum
│   ├── linalg_py.h             # [PUBLIC]   pybind11 wrappers for linalg.h
│   └── detail/                 # [INTERNAL] do not include directly
│       ├── macros.h            #   NUMPY_UNROLL4, NUMPY_SMALL_STACK
│       ├── svml_bridge.h       #   bitexact: SVML / npy_* scalar math (dlsym)
│       ├── std_math_backend.h  #   std: pure <cmath> std::exp/log/sin/… (no deps)
│       ├── blas_bridge.h       #   bitexact: OpenBLAS ILP64 cblas wrappers (dlsym)
│       ├── std_linalg_backend.h#   std: pure C++ loop dot/gemm (no deps)
│       ├── avx512_loops.h      #   bitexact: AVX-512 vectorised exp/sin/cos loops
│       └── npy_math_float.h    #   bitexact: npy_* float32 wrappers
├── bench/                      # performance benchmarks
│   ├── CMakeLists.txt
│   ├── bench_core.cpp          # C++ benchmark driver
│   ├── bench.py                # pybind11-based benchmark runner
│   └── bench_numpy.py          # pure-numpy baseline
├── tests/                      # bit-level precision tests + test module
│   ├── module.cpp              # pybind11 module for testing
│   ├── test_all.py             # single entry — all APIs, 981 tests, float64+float32
│   ├── conftest.py             # silent-mode output suppression
│   ├── make_csv.py             # ULP precision CSV generator
│   ├── diagnose_numpy.py       # numpy internal diagnostic tool
│   ├── ulp_precision.csv       # per-function ULP comparison data
│   └── CMakeLists.txt          # test-module build
├── example/                    # minimal usage examples
│   ├── CMakeLists.txt
│   └── main.cpp
├── cmake/
│   └── preinst                 # DEB pre-install script (clean old headers)
├── issue/                      # issue tracking & root-cause analysis
│   └── 001-mean_pairwise_sum_vs_sequential.md
├── CMakeLists.txt              # build & .deb packaging
└── README.md
```

## License

[MIT](LICENSE)
