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
// No fallback: if OpenBLAS ILP64 symbols are not found, throws
// std::runtime_error immediately.  Silent failures are forbidden.

#pragma once

#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  error "blas_bridge.h is an internal header — do not include directly. \
Use #include "numpycpp/numpy.h" instead."
#endif

#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <dlfcn.h>
#include <fstream>
#include <string>

namespace numpy {
namespace detail {

inline void* g_blas_handle = nullptr;

inline const std::string& find_openblas_path() {
    static std::string path;
    static bool tried = false;
    if (tried) {
        if (path.empty())
            throw std::runtime_error(
                "OpenBLAS ILP64 library not found in /proc/self/maps. "
                "Import numpy (or another package that loads libopenblas64) "
                "before calling numpycpp BLAS/LAPACK functions.");
        return path;
    }
    tried = true;

    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        // Match only ILP64 OpenBLAS (e.g. libopenblas64_p-*.so).
        // scipy's LP64 libopenblas**p**-*.so does NOT have _64_ symbols
        // (dgesv_64_, sdot_64_, etc.), so picking it causes silent failures.
        if (line.find("libopenblas64") != std::string::npos &&
            line.find(".so")           != std::string::npos) {
            auto pos   = line.rfind('/');
            auto start = line.rfind(' ', pos);
            if (start != std::string::npos && pos != std::string::npos) {
                path = line.substr(start + 1);
                while (!path.empty() && (path.back() == '\n' || path.back() == '\r'
                                         || path.back() == ' '))
                    path.pop_back();
                return path;
            }
        }
    }
    // Not found — throw immediately.  No retry, no fallback.
    throw std::runtime_error(
        "OpenBLAS ILP64 library (libopenblas64) not found in /proc/self/maps. "
        "Import numpy before calling numpycpp BLAS/LAPACK functions.");
}

inline void* resolve_blas(const char* sym) {
    if (!g_blas_handle) {
        const std::string& path = find_openblas_path();  // throws if not found
        void* h = dlopen(path.c_str(), RTLD_NOLOAD | RTLD_LAZY);
        if (!h)
            throw std::runtime_error(std::string("dlopen failed for ") + path +
                                     ": " + (dlerror() ? dlerror() : "unknown"));
        // Validate the symbol exists before caching the handle
        if (!dlsym(h, sym))
            throw std::runtime_error(std::string("BLAS symbol '") + sym +
                                     "' not found in " + path +
                                     " (ILP64 ABI mismatch? Check libopenblas64_p-*.so)");
        g_blas_handle = h;
    }
    void* fn = dlsym(g_blas_handle, sym);
    if (!fn)
        throw std::runtime_error(std::string("BLAS symbol '") + sym +
                                 "' disappeared from cached handle");
    return fn;
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
    static sdot64_fn* fn = (sdot64_fn*)resolve_blas("sdot_64_");
    const int64_t ni = static_cast<int64_t>(n), inc = 1;
    return fn(&ni, x, &inc, y, &inc);
}

inline double blas_ddot(const double* x, const double* y, size_t n) {
    static ddot64_fn* fn = (ddot64_fn*)resolve_blas("ddot_64_");
    const int64_t ni = static_cast<int64_t>(n), inc = 1;
    return fn(&ni, x, &inc, y, &inc);
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
    static cblas_sgemv64_fn* fn = (cblas_sgemv64_fn*)resolve_blas("cblas_sgemv64_");
    fn(101, 111, (int64_t)M, (int64_t)K, 1.0f, A, (int64_t)K,
                                          x, 1, 0.0f, y, 1);
}
inline void blas_dgemv(const double* A, const double* x, double* y, size_t M, size_t K) {
    static cblas_dgemv64_fn* fn = (cblas_dgemv64_fn*)resolve_blas("cblas_dgemv64_");
    fn(101, 111, (int64_t)M, (int64_t)K, 1.0, A, (int64_t)K,
                                          x, 1, 0.0, y, 1);
}

// y[N] = B^T[K×N] @ a[K]  — 1D × 2D case (Trans=112)
inline void blas_sgemv_t(const float* B, const float* a, float* y, size_t K, size_t N) {
    static cblas_sgemv64_fn* fn = (cblas_sgemv64_fn*)resolve_blas("cblas_sgemv64_");
    fn(101, 112, (int64_t)K, (int64_t)N, 1.0f, B, (int64_t)N,
                                          a, 1, 0.0f, y, 1);
}
inline void blas_dgemv_t(const double* B, const double* a, double* y, size_t K, size_t N) {
    static cblas_dgemv64_fn* fn = (cblas_dgemv64_fn*)resolve_blas("cblas_dgemv64_");
    fn(101, 112, (int64_t)K, (int64_t)N, 1.0, B, (int64_t)N,
                                          a, 1, 0.0, y, 1);
}

// C = A @ B  (all row-major)  C[M×N] = A[M×K] @ B[K×N]
// Uses cblas_sgemm64_ — same kernel numpy.matmul calls → 0 ULP by construction.
inline void blas_sgemm(const float* A, const float* B, float* C,
                       size_t M, size_t K, size_t N) {
    static cblas_sgemm64_fn* fn = (cblas_sgemm64_fn*)resolve_blas("cblas_sgemm64_");
    fn(101, 111, 111,                        // RowMajor, NoTrans, NoTrans
       (int64_t)M, (int64_t)N, (int64_t)K,
       1.0f, A, (int64_t)K, B, (int64_t)N,
       0.0f, C, (int64_t)N);
}

inline void blas_dgemm(const double* A, const double* B, double* C,
                       size_t M, size_t K, size_t N) {
    static cblas_dgemm64_fn* fn = (cblas_dgemm64_fn*)resolve_blas("cblas_dgemm64_");
    fn(101, 111, 111,
       (int64_t)M, (int64_t)N, (int64_t)K,
       1.0, A, (int64_t)K, B, (int64_t)N,
       0.0, C, (int64_t)N);
}

// ============================================================================
// LAPACK — matrix inverse (numpy.linalg.inv) via Fortran DGESV
// ============================================================================
// numpy.linalg.inv calls LAPACK ?gesv (solve A·X = I).  DGESV fuses LU
// factorisation + forward/back substitution in a single kernel.
//
// On this OpenBLAS build, sgesv_64_ produces 1‑ULP differences vs numpy for
// float32 inputs.  NumPy's float32 inv is bit‑equivalent to: promote to
// float64 → dgesv → demote to float32.  We follow that same path so both
// dtypes are IEEE‑754 bit‑identical to numpy.
//
// Fortran DGESV signature (ILP64, _64_ suffix):
//   dgesv_64_(int64_t *N, int64_t *NRHS, double *A, int64_t *LDA,
//             int64_t *IPIV, double *B, int64_t *LDB, int64_t *INFO);

using dgesv64_fn = void(int64_t*, int64_t*, double*, int64_t*,
                         int64_t*, double*, int64_t*, int64_t*);

/// Fortran DGESV-based matrix inverse.  Matches numpy.linalg.inv exactly
/// — same Fortran symbol, same ILP64 ABI, same memory layout.
/// Throws std::runtime_error on singular matrix (LAPACK info != 0).
template<typename T> inline void blas_gesv_inv(T* A, size_t N);

template<> inline void blas_gesv_inv<float>(float* A, size_t N) {
    // numpy.linalg.inv for float32 produces the same bits as:
    //   float32 → float64 → dgesv → float32
    // (OpenBLAS sgesv_64_ gives 1-ULP-off results vs numpy on this build;
    //  the float64 path is bit-identical for both types.)
    static dgesv64_fn* gesv = (dgesv64_fn*)resolve_blas("dgesv_64_");
    int64_t n = static_cast<int64_t>(N);
    auto ipiv = std::make_unique<int64_t[]>(N);
    // Double-precision work buffers (column-major)
    auto A_col = std::make_unique<double[]>(N * N);
    auto B_col = std::make_unique<double[]>(N * N);
    // Promote A row-major → A_col column-major (float→double)
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A_col[j*N + i] = static_cast<double>(A[i*N + j]);
    // B = identity (column-major, double)
    std::memset(B_col.get(), 0, N * N * sizeof(double));
    for (size_t i = 0; i < N; ++i)
        B_col[i + i*N] = 1.0;
    int64_t nrhs = n, lda = n, ldb = n, info = 0;
    gesv(&n, &nrhs, A_col.get(), &lda, ipiv.get(), B_col.get(), &ldb, &info);
    if (info != 0)
        throw std::runtime_error("linalg.inv: singular matrix (DGESV info=" +
                                 std::to_string(info) + ")");
    // Demote solution back to float32 row-major
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i*N + j] = static_cast<float>(B_col[j*N + i]);
}

template<> inline void blas_gesv_inv<double>(double* A, size_t N) {
    static dgesv64_fn* gesv = (dgesv64_fn*)resolve_blas("dgesv_64_");
    int64_t n = static_cast<int64_t>(N);
    auto ipiv = std::make_unique<int64_t[]>(N);
    auto A_col = std::make_unique<double[]>(N * N);
    auto B_col = std::make_unique<double[]>(N * N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A_col[j*N + i] = A[i*N + j];
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            B_col[i + j*N] = (i == j) ? 1.0 : 0.0;
    int64_t nrhs = n, lda = n, ldb = n, info = 0;
    gesv(&n, &nrhs, A_col.get(), &lda, ipiv.get(), B_col.get(), &ldb, &info);
    if (info != 0)
        throw std::runtime_error("linalg.inv: singular matrix (DGESV info=" +
                                 std::to_string(info) + ")");
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i*N + j] = B_col[j*N + i];
}

// Template dispatcher
template<typename T> struct blas_ops;

template<> struct blas_ops<float> {
    static float dot  (const float*  x, const float*  y, size_t n) { return blas_sdot(x, y, n); }
    static float norm (const float*  x,                  size_t n) { return std::sqrt(blas_sdot(x, x, n)); }
    static void  gemm (const float*  A, const float*  B, float*  C,
                       size_t M, size_t K, size_t N) { blas_sgemm(A, B, C, M, K, N); }
    static void  gemv (const float*  A, const float*  x, float*  y,
                       size_t M, size_t K) { blas_sgemv(A, x, y, M, K); }
    static void  gemvt(const float*  B, const float*  a, float*  y,
                       size_t K, size_t N) { blas_sgemv_t(B, a, y, K, N); }
    static void  inv  (float* A, size_t N) { blas_gesv_inv<float>(A, N); }
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
    static void   inv  (double* A, size_t N) { blas_gesv_inv<double>(A, N); }
};

} // namespace detail
} // namespace numpy
