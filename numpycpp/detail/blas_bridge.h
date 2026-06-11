// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  INTERNAL HEADER — DIRECT INCLUSION IS A COMPILE ERROR                 ║
// ║                                                                          ║
// ║  This file wraps OpenBLAS ILP64 (Linux x86_64 only) via dlsym/dlopen.  ║
// ║  All symbols live in numpy::detail — an implementation namespace that   ║
// ║  external code must never reference.                                     ║
// ║                                                                          ║
// ║  ✗  #include "numpy/detail/blas_bridge.h"      ← compile error                ║
// ║  ✗  numpy::detail::blas_sdot(...)       ← undefined behaviour          ║
// ║  ✓  #include "numpycpp/numpy.h"             ← recommended entry point      ║
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
// handled by norm_axis() in linalg.h, no BLAS needed.
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
Use #include "numpycpp/numpy.h" instead."
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

// cblas_sgemm64_ / cblas_dgemm64_  — C BLAS interface, ILP64 (BLAS_SYMBOL_SUFFIX=64_)
// Signature: (layout, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
//   layout  : 101 = CblasRowMajor
//   transA/B: 111 = CblasNoTrans
using cblas_sgemm64_fn = void(int, int, int,
                               int64_t, int64_t, int64_t,
                               float,  const float*,  int64_t,
                                       const float*,  int64_t,
                               float,        float*,  int64_t);
using cblas_dgemm64_fn = void(int, int, int,
                               int64_t, int64_t, int64_t,
                               double, const double*, int64_t,
                                       const double*, int64_t,
                               double,       double*, int64_t);

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

// cblas_sgemv64_ / cblas_dgemv64_  — matrix-vector, ILP64
// Signature: (layout, trans, M, N, alpha, A, lda, x, incx, beta, y, incy)
using cblas_sgemv64_fn = void(int, int, int64_t, int64_t,
                               float,  const float*,  int64_t,
                                       const float*,  int64_t,
                               float,        float*,  int64_t);
using cblas_dgemv64_fn = void(int, int, int64_t, int64_t,
                               double, const double*, int64_t,
                                       const double*, int64_t,
                               double,       double*, int64_t);

// y[M] = A[M×K] @ x[K]  — 2D × 1D case
inline void blas_sgemv(const float* A, const float* x, float* y, size_t M, size_t K) {
    static auto fn = (cblas_sgemv64_fn*)resolve_blas("cblas_sgemv64_");
    if (__builtin_expect(fn != nullptr, 1)) {
        fn(101, 111, (int64_t)M, (int64_t)K, 1.0f, A, (int64_t)K,
                                              x, 1, 0.0f, y, 1);
        return;
    }
    for (size_t i = 0; i < M; ++i) {
        float s = 0.0f;
        for (size_t k = 0; k < K; ++k) s += A[i*K+k] * x[k];
        y[i] = s;
    }
}
inline void blas_dgemv(const double* A, const double* x, double* y, size_t M, size_t K) {
    static auto fn = (cblas_dgemv64_fn*)resolve_blas("cblas_dgemv64_");
    if (__builtin_expect(fn != nullptr, 1)) {
        fn(101, 111, (int64_t)M, (int64_t)K, 1.0, A, (int64_t)K,
                                              x, 1, 0.0, y, 1);
        return;
    }
    for (size_t i = 0; i < M; ++i) {
        double s = 0.0;
        for (size_t k = 0; k < K; ++k) s += A[i*K+k] * x[k];
        y[i] = s;
    }
}

// y[N] = B^T[K×N] @ a[K]  — 1D × 2D case (Trans=112)
inline void blas_sgemv_t(const float* B, const float* a, float* y, size_t K, size_t N) {
    static auto fn = (cblas_sgemv64_fn*)resolve_blas("cblas_sgemv64_");
    if (__builtin_expect(fn != nullptr, 1)) {
        fn(101, 112, (int64_t)K, (int64_t)N, 1.0f, B, (int64_t)N,
                                              a, 1, 0.0f, y, 1);
        return;
    }
    for (size_t j = 0; j < N; ++j) {
        float s = 0.0f;
        for (size_t k = 0; k < K; ++k) s += B[k*N+j] * a[k];
        y[j] = s;
    }
}
inline void blas_dgemv_t(const double* B, const double* a, double* y, size_t K, size_t N) {
    static auto fn = (cblas_dgemv64_fn*)resolve_blas("cblas_dgemv64_");
    if (__builtin_expect(fn != nullptr, 1)) {
        fn(101, 112, (int64_t)K, (int64_t)N, 1.0, B, (int64_t)N,
                                              a, 1, 0.0, y, 1);
        return;
    }
    for (size_t j = 0; j < N; ++j) {
        double s = 0.0;
        for (size_t k = 0; k < K; ++k) s += B[k*N+j] * a[k];
        y[j] = s;
    }
}

// C = A @ B  (all row-major)  C[M×N] = A[M×K] @ B[K×N]
// Uses cblas_sgemm64_ — same kernel numpy.matmul calls → 0 ULP by construction.
inline void blas_sgemm(const float* A, const float* B, float* C,
                       size_t M, size_t K, size_t N) {
    static auto fn = (cblas_sgemm64_fn*)resolve_blas("cblas_sgemm64_");
    if (__builtin_expect(fn != nullptr, 1)) {
        fn(101, 111, 111,                        // RowMajor, NoTrans, NoTrans
           (int64_t)M, (int64_t)N, (int64_t)K,
           1.0f, A, (int64_t)K, B, (int64_t)N,
           0.0f, C, (int64_t)N);
        return;
    }
    // Fallback (no OpenBLAS): naive triple loop — not bit-exact but always correct
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (size_t k = 0; k < K; ++k) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

inline void blas_dgemm(const double* A, const double* B, double* C,
                       size_t M, size_t K, size_t N) {
    static auto fn = (cblas_dgemm64_fn*)resolve_blas("cblas_dgemm64_");
    if (__builtin_expect(fn != nullptr, 1)) {
        fn(101, 111, 111,
           (int64_t)M, (int64_t)N, (int64_t)K,
           1.0, A, (int64_t)K, B, (int64_t)N,
           0.0, C, (int64_t)N);
        return;
    }
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < K; ++k) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

// ============================================================================
// LAPACK — LU factorisation + matrix inverse (numpy.linalg.inv)
// ============================================================================
// numpy.linalg.inv uses LAPACKE (C interface) routing through OpenBLAS ILP64.
// LAPACKE internally handles row→column conversion and workspace allocation,
// producing the exact same floating-point rounding path as numpy.
//
// LAPACKE function signatures (ILP64, return info as int64_t):
//   LAPACKE_sgetrf64_(layout, m, n, a, lda, ipiv)
//   LAPACKE_sgetri64_(layout, n, a, lda, ipiv)
// layout = 101 (LAPACK_ROW_MAJOR)

using LAPACKE_sgetrf64_fn = int64_t(int, int64_t, int64_t, float*,  int64_t, int64_t*);
using LAPACKE_dgetrf64_fn = int64_t(int, int64_t, int64_t, double*, int64_t, int64_t*);
using LAPACKE_sgetri64_fn = int64_t(int, int64_t, float*,  int64_t, const int64_t*);
using LAPACKE_dgetri64_fn = int64_t(int, int64_t, double*, int64_t, const int64_t*);

/// LAPACKE-based matrix inverse (C interface, RowMajor).
/// Uses ?getrf (LU factorisation) + ?getri (inverse from LU).
/// Matches numpy.linalg.inv exactly — same LAPACKE path, same ILP64 ABI.
inline bool blas_sinv(float* A, size_t N) {
    static auto getrf = (LAPACKE_sgetrf64_fn*)resolve_blas("LAPACKE_sgetrf64_");
    static auto getri = (LAPACKE_sgetri64_fn*)resolve_blas("LAPACKE_sgetri64_");
    if (__builtin_expect(getrf == nullptr || getri == nullptr, 0)) return false;
    int64_t n = static_cast<int64_t>(N);
    auto ipiv = std::make_unique<int64_t[]>(N);
    int64_t info = getrf(101, n, n, A, n, ipiv.get());
    if (info != 0) return false;
    info = getri(101, n, A, n, ipiv.get());
    return info == 0;
}
inline bool blas_dinv(double* A, size_t N) {
    static auto getrf = (LAPACKE_dgetrf64_fn*)resolve_blas("LAPACKE_dgetrf64_");
    static auto getri = (LAPACKE_dgetri64_fn*)resolve_blas("LAPACKE_dgetri64_");
    if (__builtin_expect(getrf == nullptr || getri == nullptr, 0)) return false;
    int64_t n = static_cast<int64_t>(N);
    auto ipiv = std::make_unique<int64_t[]>(N);
    int64_t info = getrf(101, n, n, A, n, ipiv.get());
    if (info != 0) return false;
    info = getri(101, n, A, n, ipiv.get());
    return info == 0;
}

// Template dispatcher
template<typename T> struct blas_ops;

template<> struct blas_ops<float> {
    static float dot  (const float*  x, const float*  y, size_t n) { return blas_sdot(x, y, n); }
    static float norm (const float*  x,                  size_t n) { return std::sqrt(blas_sdot(x, x, n)); }
    static void  gemm (const float*  A, const float*  B, float*  C,
                       size_t M, size_t K, size_t N) { blas_sgemm(A, B, C, M, K, N); }
    // y[M] = A[M×K] @ x[K]
    static void  gemv (const float*  A, const float*  x, float*  y,
                       size_t M, size_t K) { blas_sgemv(A, x, y, M, K); }
    // y[N] = B^T @ a[K]   (1D × 2D case)
    static void  gemvt(const float*  B, const float*  a, float*  y,
                       size_t K, size_t N) { blas_sgemv_t(B, a, y, K, N); }
    // A_inv[N×N] = inv(A[N×N]) — in-place, returns true on success
    static bool  inv  (float* A, size_t N) { return blas_sinv(A, N); }
};
template<> struct blas_ops<double> {
    static double dot  (const double* x, const double* y, size_t n) { return blas_ddot(x, y, n); }
    static double norm (const double* x,                  size_t n) { return std::sqrt(blas_ddot(x, x, n)); }
    static void   gemm (const double* A, const double* B, double* C,
                        size_t M, size_t K, size_t N) { blas_dgemm(A, B, C, M, K, N); }
    static void   gemv (const double* A, const double* x, double* y,
                        size_t M, size_t K) { blas_dgemv(A, x, y, M, K); }
    static void   gemvt(const double* B, const double* a, double* y,
                        size_t K, size_t N) { blas_dgemv_t(B, a, y, K, N); }
    static bool   inv  (double* A, size_t N) { return blas_dinv(A, N); }
};

} // namespace detail
} // namespace numpy
