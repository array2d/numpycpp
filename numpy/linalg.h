// Native C++ linalg functions — zero pybind11 dependency.
// Uses numpy:: helpers from core.h for shared computation.

#pragma once

#include "core.h"
#include <cmath>

namespace numpy {
namespace linalg {

/// numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) — vector / Frobenius
//  np.linalg.norm(a) internally computes sqrt(a.dot(a)) via BLAS sdot/ddot.
//  We call the same OpenBLAS routine (auto-discovered) for bit-exact match.
template<typename T>
inline T norm(const T* data, size_t n) {
    return numpy::detail::blas_ops<T>::norm(data, n);
}

/// numpy.linalg.norm(x, ord=None, axis=N, keepdims=False) — N-D
template<typename T>
inline void norm_axis(const T* src, T* dst, const ptrdiff_t* shape, int ndim, int axis) {
    numpy::norm_axis(src, dst, shape, ndim, axis);
}

/// numpy.matmul — single 2D slice: mirrors numpy's cblas_matrixproduct dispatch.
/// numpy selects sdot / sgemv / dgemv / sgemm based on output dimensions:
///   M==1 && N==1 → sdot  (scalar inner product, highest precision path)
///   M==1          → sgemv(Trans)   — row-vector × matrix
///   N==1          → sgemv(NoTrans) — matrix × col-vector
///   otherwise     → sgemm
template<typename T>
inline void matmul_slice(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {
    if (M == 1 && N == 1) {
        C[0] = numpy::detail::blas_ops<T>::dot(A, B, K);   // A[0..K-1] · B[0..K-1]
    } else if (M == 1) {
        numpy::detail::blas_ops<T>::gemvt(B, A, C, K, N);  // y[N] = B^T @ A[0]
    } else if (N == 1) {
        numpy::detail::blas_ops<T>::gemv(A, B, C, M, K);   // y[M] = A @ B[:,0]
    } else {
        numpy::detail::blas_ops<T>::gemm(A, B, C, M, K, N);
    }
}

/// numpy.matmul — 2D: C[M,N] = A[M,K] @ B[K,N]  (row-major)
template<typename T>
inline void matmul(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {
    matmul_slice<T>(A, B, C, M, K, N);
}

/// numpy.matmul — 2D×1D: y[M] = A[M,K] @ x[K]
template<typename T>
inline void matmul_mv(const T* A, const T* x, T* y, size_t M, size_t K) {
    numpy::detail::blas_ops<T>::gemv(A, x, y, M, K);
}

/// numpy.matmul — 1D×2D: y[N] = a[K] @ B[K,N]  (= B^T @ a)
/// When N==1, numpy uses sdot (dot product path), not sgemv.
template<typename T>
inline void matmul_vm(const T* a, const T* B, T* y, size_t K, size_t N) {
    if (N == 1)
        y[0] = numpy::detail::blas_ops<T>::dot(a, B, K);  // a · B[:,0]
    else
        numpy::detail::blas_ops<T>::gemvt(B, a, y, K, N);
}

/// numpy.matmul — batched 3D: C[batch,M,N] = A[batch,M,K] @ B[batch,K,N]
/// Each slice uses the same sdot/gemv/gemm dispatch as numpy.
template<typename T>
inline void matmul(const T* A, const T* B, T* C,
                   size_t batch, size_t M, size_t K, size_t N) {
    for (size_t b = 0; b < batch; ++b)
        matmul_slice<T>(
            A + b * M * K,
            B + b * K * N,
            C + b * M * N,
            M, K, N);
}

} // namespace linalg
} // namespace numpy
