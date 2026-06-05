// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  INTERNAL HEADER — DIRECT INCLUSION IS A COMPILE ERROR                 ║
// ║                                                                          ║
// ║  This file wraps OpenBLAS ILP64 (Linux x86_64 only) via dlsym/dlopen.  ║
// ║  All symbols live in numpy::detail — an implementation namespace that   ║
// ║  external code must never reference.                                     ║
// ║                                                                          ║
// ║  ✗  #include "numpy/blas_bridge.h"      ← compile error                ║
// ║  ✗  numpy::detail::blas_sdot(...)       ← undefined behaviour          ║
// ║  ✓  #include "numpy/core.h"             ← only correct entry point      ║
// ║  ✓  numpy::dot(a, b, n)                 ← public API                    ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// BLAS bridge — bit-exact dot/norm vs numpy's OpenBLAS-backed np.dot /
// np.linalg.norm (without axis).
//
// numpy routes 1-D dot and Frobenius norm through BLAS (OpenBLAS ILP64):
//   np.dot(a, b)      → sdot_64_ / ddot_64_
//   np.linalg.norm(a) → sqrt(x.dot(x)) → same sdot_64_ / ddot_64_
//
// np.linalg.norm(a, axis=k) uses numpy's own pairwise sum — already
// handled by norm_axis() in core.h, no BLAS needed.
//
// The OpenBLAS library path is auto-discovered from /proc/self/maps
// (numpy loads it when imported), so no compile-time link flag is needed.
//
// ILP64 Fortran calling convention (OpenBLAS built with BLAS_SYMBOL_SUFFIX=64_):
//   sdot_64_(n*, x*, incx*, y*, incy*)  → float   (return in xmm0)
//   ddot_64_(n*, x*, incx*, y*, incy*)  → double  (return in xmm0)
//
// Fallback (if OpenBLAS not discovered): sequential accumulation.

#pragma once

#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  error "blas_bridge.h is an internal header — do not include directly. \
Use #include \"numpy/core.h\" instead."
#endif

#include <cstdint>
#include <cmath>
#include <dlfcn.h>
#include <fstream>
#include <string>

namespace numpy {
namespace detail {

inline void* g_blas_handle = nullptr;

inline const char* find_openblas_path() {
    static std::string path;
    static bool tried = false;
    if (tried) return path.empty() ? nullptr : path.c_str();
    tried = true;

    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find("libopenblas") != std::string::npos &&
            line.find(".so")         != std::string::npos) {
            auto pos   = line.rfind('/');
            auto start = line.rfind(' ', pos);
            if (start != std::string::npos && pos != std::string::npos) {
                path = line.substr(start + 1);
                // trim trailing whitespace / newline
                while (!path.empty() && (path.back() == '\n' || path.back() == '\r'
                                         || path.back() == ' '))
                    path.pop_back();
                break;
            }
        }
    }
    return path.empty() ? nullptr : path.c_str();
}

inline void* resolve_blas(const char* sym) {
    if (!g_blas_handle) {
        const char* path = find_openblas_path();
        if (path) g_blas_handle = dlopen(path, RTLD_NOLOAD | RTLD_LAZY);
    }
    return g_blas_handle ? dlsym(g_blas_handle, sym) : nullptr;
}

// ILP64 Fortran function types (all int args are int64_t by pointer)
using sdot64_fn = float  (const int64_t*, const float*,  const int64_t*,
                           const float*,  const int64_t*);
using ddot64_fn = double (const int64_t*, const double*, const int64_t*,
                           const double*, const int64_t*);

inline float blas_sdot(const float* x, const float* y, size_t n) {
    static auto fn = (sdot64_fn*)resolve_blas("sdot_64_");
    if (__builtin_expect(fn != nullptr, 1)) {
        const int64_t ni = static_cast<int64_t>(n), inc = 1;
        return fn(&ni, x, &inc, y, &inc);
    }
    // Fallback: sequential accumulation
    float r = 0.0f;
    for (size_t i = 0; i < n; ++i) r += x[i] * y[i];
    return r;
}

inline double blas_ddot(const double* x, const double* y, size_t n) {
    static auto fn = (ddot64_fn*)resolve_blas("ddot_64_");
    if (__builtin_expect(fn != nullptr, 1)) {
        const int64_t ni = static_cast<int64_t>(n), inc = 1;
        return fn(&ni, x, &inc, y, &inc);
    }
    double r = 0.0;
    for (size_t i = 0; i < n; ++i) r += x[i] * y[i];
    return r;
}

// Template dispatcher
template<typename T> struct blas_ops;

template<> struct blas_ops<float> {
    static float  dot (const float*  x, const float*  y, size_t n) { return blas_sdot(x, y, n); }
    static float  norm(const float*  x,                  size_t n) { return std::sqrt(blas_sdot(x, x, n)); }
};
template<> struct blas_ops<double> {
    static double dot (const double* x, const double* y, size_t n) { return blas_ddot(x, y, n); }
    static double norm(const double* x,                  size_t n) { return std::sqrt(blas_ddot(x, x, n)); }
};

} // namespace detail
} // namespace numpy
