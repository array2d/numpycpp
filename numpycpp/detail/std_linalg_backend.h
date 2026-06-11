// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  INTERNAL HEADER — DIRECT INCLUSION IS A COMPILE ERROR                 ║
// ║                                                                          ║
// ║  Pure C++ standard-library linear-algebra backend.                      ║
// ║  Provides the same numpy::detail::blas_ops<T> API as blas_bridge.h     ║
// ║  but uses only C++ loops — no dlopen, no OpenBLAS, no Fortran ABI.     ║
// ║                                                                          ║
// ║  ✗  #include "numpy/detail/std_linalg_backend.h"  ← compile error      ║
// ║  ✓  #include "numpycpp/numpy.h"                      ← entry point         ║
// ║  ✓  cmake … -DNUMPYCPP_STD_ONLY=ON               ← selects this backend║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// Performance notes (NUMPYCPP_STD_ONLY=ON):
//   • dot/norm: simple sequential accumulation — with -O3 -march=native the
//     compiler auto-vectorises to SIMD (SSE/AVX/AVX-512) transparently.
//   • gemm: row-major triple loop; GCC/Clang with -O3 will auto-vectorise
//     the inner k-loop.  For production-grade performance (LAPACK parity)
//     consider linking OpenBLAS explicitly instead of NUMPYCPP_STD_ONLY.
//   • Precision: NOT bit-exact with numpy.matmul (which uses OpenBLAS);
//     differences of 0–2 ULP expected for large matrices.

#pragma once

#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  error "std_linalg_backend.h is an internal header — do not include directly. \
Use #include \"numpycpp/numpy.h\" instead."
#endif

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace numpy {
namespace detail {

// dot — sequential accumulation; compiler can auto-vectorise with -O3
inline float  std_sdot(const float*  x, const float*  y, size_t n) {
    float r = 0.0f;
    for (size_t i = 0; i < n; ++i) r += x[i] * y[i];
    return r;
}
inline double std_ddot(const double* x, const double* y, size_t n) {
    double r = 0.0;
    for (size_t i = 0; i < n; ++i) r += x[i] * y[i];
    return r;
}

// gemv: y[M] = A[M×K] @ x[K]  (row-major, NoTrans)
inline void std_sgemv(const float* A, const float* x, float* y, size_t M, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        float s = 0.0f;
        for (size_t k = 0; k < K; ++k) s += A[i*K+k] * x[k];
        y[i] = s;
    }
}
inline void std_dgemv(const double* A, const double* x, double* y, size_t M, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        double s = 0.0;
        for (size_t k = 0; k < K; ++k) s += A[i*K+k] * x[k];
        y[i] = s;
    }
}

// gemv_t: y[N] = B^T[K×N] @ a[K]  (1D × 2D case, Trans)
inline void std_sgemv_t(const float* B, const float* a, float* y, size_t K, size_t N) {
    for (size_t j = 0; j < N; ++j) {
        float s = 0.0f;
        for (size_t k = 0; k < K; ++k) s += B[k*N+j] * a[k];
        y[j] = s;
    }
}
inline void std_dgemv_t(const double* B, const double* a, double* y, size_t K, size_t N) {
    for (size_t j = 0; j < N; ++j) {
        double s = 0.0;
        for (size_t k = 0; k < K; ++k) s += B[k*N+j] * a[k];
        y[j] = s;
    }
}

// gemm: C[M×N] = A[M×K] @ B[K×N]  (all row-major)
inline void std_sgemm(const float* A, const float* B, float* C,
                      size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (size_t k = 0; k < K; ++k) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}
inline void std_dgemm(const double* A, const double* B, double* C,
                      size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < K; ++k) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

// ============================================================================
// Matrix inverse — Gauss-Jordan elimination with partial pivoting
// (std backend when LAPACK not available)
// ============================================================================
// Augments [A | I], eliminates to [I | A⁻¹], then extracts RHS.
// Throws std::runtime_error if matrix is singular (pivot too small).

template<typename T>
inline void std_inv(T* A, size_t N) {
    // Augmented matrix [A | I] stored row-major: rows of 2N elements
    auto aug = std::make_unique<T[]>(N * 2 * N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) aug[i*2*N + j] = A[i*N + j];
        for (size_t j = 0; j < N; ++j) aug[i*2*N + N + j] = T(i == j);
    }

    for (size_t col = 0; col < N; ++col) {
        // Partial pivoting: find row with max |value| in this column
        size_t pivot_row = col;
        T max_val = std::abs(aug[col*2*N + col]);
        for (size_t row = col + 1; row < N; ++row) {
            T v = std::abs(aug[row*2*N + col]);
            if (v > max_val) { max_val = v; pivot_row = row; }
        }
        if (max_val < T(1e-30))
            throw std::runtime_error("linalg.inv: singular matrix (Gauss-Jordan pivot too small)");

        // Swap rows if needed
        if (pivot_row != col) {
            for (size_t j = 0; j < 2 * N; ++j)
                std::swap(aug[col*2*N + j], aug[pivot_row*2*N + j]);
        }

        // Normalise pivot row
        T pivot = aug[col*2*N + col];
        for (size_t j = 0; j < 2 * N; ++j)
            aug[col*2*N + j] /= pivot;

        // Eliminate all other rows
        for (size_t row = 0; row < N; ++row) {
            if (row == col) continue;
            T factor = aug[row*2*N + col];
            for (size_t j = 0; j < 2 * N; ++j)
                aug[row*2*N + j] -= factor * aug[col*2*N + j];
        }
    }

    // Extract inverse from augmented RHS
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i*N + j] = aug[i*2*N + N + j];
}

// ============================================================================
// blas_ops<T> — same template interface as blas_bridge.h.
// linalg.h calls numpy::detail::blas_ops<T>::dot/norm/gemm/gemv/gemvt.
// ============================================================================

template<typename T> struct blas_ops;

template<> struct blas_ops<float> {
    static float dot  (const float* x, const float* y, size_t n) {
        return std_sdot(x, y, n);
    }
    static float norm (const float* x, size_t n) {
        return std::sqrt(std_sdot(x, x, n));
    }
    static void gemm (const float* A, const float* B, float* C,
                      size_t M, size_t K, size_t N) {
        std_sgemm(A, B, C, M, K, N);
    }
    static void gemv (const float* A, const float* x, float* y,
                      size_t M, size_t K) {
        std_sgemv(A, x, y, M, K);
    }
    static void gemvt(const float* B, const float* a, float* y,
                      size_t K, size_t N) {
        std_sgemv_t(B, a, y, K, N);
    }
    static void inv  (float* A, size_t N) { std_inv(A, N); }
};

template<> struct blas_ops<double> {
    static double dot  (const double* x, const double* y, size_t n) {
        return std_ddot(x, y, n);
    }
    static double norm (const double* x, size_t n) {
        return std::sqrt(std_ddot(x, x, n));
    }
    static void gemm (const double* A, const double* B, double* C,
                      size_t M, size_t K, size_t N) {
        std_dgemm(A, B, C, M, K, N);
    }
    static void gemv (const double* A, const double* x, double* y,
                      size_t M, size_t K) {
        std_dgemv(A, x, y, M, K);
    }
    static void gemvt(const double* B, const double* a, double* y,
                      size_t K, size_t N) {
        std_dgemv_t(B, a, y, K, N);
    }
    static void inv  (double* A, size_t N) { std_inv(A, N); }
};

} // namespace detail
} // namespace numpy
