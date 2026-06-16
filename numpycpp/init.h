// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — numpy/init.h                              [PUBLIC HEADER]
//
//  Array initialisation / creation routines.
//
//      numpy.zeros_like   numpy.ones_like   numpy.full
//      numpy.arange       numpy.linspace    numpy.logspace   numpy.geomspace
//      numpy.eye          numpy.identity
//      numpy.diag         (build from 1-D / extract from 2-D)
//
//  Recommended entry point: #include "numpy/numpy.h"
//  Direct include is also valid for standalone use.
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstddef>
#include <cmath>
#include <algorithm>

namespace numpy {

// ============================================================================
// Fill helpers (used by zeros_like / ones_like / full)
// ============================================================================

/// numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
template<typename T>
inline void zeros_like(T* dst, size_t n) {
    std::fill_n(dst, n, T(0));
}

/// numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)
template<typename T>
inline void ones_like(T* dst, size_t n) {
    std::fill_n(dst, n, T(1));
}

/// numpy.full(shape, fill_value, dtype=None, order='C')
template<typename T>
inline void full(T* dst, size_t n, T fill_value) {
    std::fill_n(dst, n, fill_value);
}

// ============================================================================
// numpy.arange
// ============================================================================

/// Number of elements produced by numpy.arange(start, stop, step).
/// Uses double precision for the ceiling so float32 step errors don't
/// cause an off-by-one (matches numpy's internal length computation).
template<typename T>
inline size_t arange_size(T start, T stop, T step) {
    if (step == T(0)) return 0;
    double n = std::ceil(
        (static_cast<double>(stop) - static_cast<double>(start))
        / static_cast<double>(step));
    return (n > 0.0) ? static_cast<size_t>(n) : 0;
}

/// numpy.arange([start,] stop[, step,], dtype=None)
/// dst must have at least arange_size(start, stop, step) elements.
/// Values: dst[i] = start + T(i)*step  (index-based, not accumulated).
template<typename T>
inline void arange(T* dst, T start, T stop, T step) {
    size_t n = arange_size(start, stop, step);
    for (size_t i = 0; i < n; ++i)
        dst[i] = start + static_cast<T>(i) * step;
}

// ============================================================================
// numpy.linspace
// ============================================================================

/// numpy.linspace(start, stop, num=50, endpoint=True, ...)
/// Values: dst[i] = start + i * step  where step = (stop-start)/(num-1 or num).
/// When endpoint=true the last element is set to stop exactly.
template<typename T>
inline void linspace(T* dst, T start, T stop, size_t num,
                     bool endpoint = true) {
    if (num == 0) return;
    if (num == 1) { dst[0] = start; return; }
    T div  = static_cast<T>(endpoint ? num - 1 : num);
    T step = (stop - start) / div;
    for (size_t i = 0; i < num; ++i)
        dst[i] = start + static_cast<T>(i) * step;
    if (endpoint) dst[num - 1] = stop;   // exact endpoint (matches numpy)
}

// ============================================================================
// numpy.logspace
// ============================================================================

/// numpy.logspace(start, stop, num=50, endpoint=True, base=10.0)
/// dst[i] = base ^ linspace(start, stop, num, endpoint)[i]
template<typename T>
inline void logspace(T* dst, T start, T stop, size_t num,
                     bool endpoint = true, T base = T(10)) {
    linspace(dst, start, stop, num, endpoint);
    for (size_t i = 0; i < num; ++i)
        dst[i] = detail::pow(base, dst[i]);
}

// ============================================================================
// numpy.geomspace
// ============================================================================

/// numpy.geomspace(start, stop, num=50, endpoint=True)
/// Geometric sequence: num points evenly spaced on a log scale.
/// Matches numpy's algorithm: log10(start..stop) → logspace(..., base=10).
/// Both start and stop must be non-zero and have the same sign.
template<typename T>
inline void geomspace(T* dst, T start, T stop, size_t num,
                      bool endpoint = true) {
    if (num == 0) return;
    T sign = (start < T(0)) ? T(-1) : T(1);
    T abs_start = sign * start;
    T abs_stop  = sign * stop;
    // Delegate to logspace(log10(start), log10(stop), base=10) — same as numpy
    logspace(dst, std::log10(abs_start), std::log10(abs_stop),
             num, endpoint, T(10));
    // Apply sign for negative-to-negative ranges
    if (sign < T(0))
        for (size_t i = 0; i < num; ++i) dst[i] = -dst[i];
    // Fix endpoints exactly (numpy guarantees start/stop are exact)
    dst[0] = start;
    if (endpoint && num > 1) dst[num - 1] = stop;
}

// ============================================================================
// numpy.eye / numpy.identity
// ============================================================================

/// numpy.eye(N, M=N, k=0, dtype=float)
/// N-row × M-col matrix; 1 on the k-th diagonal, 0 elsewhere.
/// dst must hold N*M elements (row-major).
template<typename T>
inline void eye(T* dst, size_t N, size_t M, int k = 0) {
    std::fill_n(dst, N * M, T(0));
    size_t row0 = (k < 0) ? static_cast<size_t>(-k) : 0;
    size_t col0 = (k >= 0) ? static_cast<size_t>(k)  : 0;
    if (row0 >= N || col0 >= M) return;
    size_t len = std::min(N - row0, M - col0);
    for (size_t i = 0; i < len; ++i)
        dst[(row0 + i) * M + (col0 + i)] = T(1);
}

/// numpy.identity(n, dtype=float)  ≡  eye(n, n, 0)
template<typename T>
inline void identity(T* dst, size_t n) {
    eye(dst, n, n, 0);
}

// ============================================================================
// numpy.diag
// ============================================================================

/// Diagonal length when extracting the k-th diagonal from an N×M matrix.
inline size_t diag_size(size_t N, size_t M, int k) {
    size_t row0 = (k < 0) ? static_cast<size_t>(-k) : 0;
    size_t col0 = (k >= 0) ? static_cast<size_t>(k)  : 0;
    if (row0 >= N || col0 >= M) return 0;
    return std::min(N - row0, M - col0);
}

/// numpy.diag(v, k=0) — 1-D input: build (n+|k|)×(n+|k|) diagonal matrix.
/// dst must hold (n+|k|)^2 elements.
template<typename T>
inline void diag_from_vec(T* dst, const T* src, size_t n, int k = 0) {
    size_t N = n + static_cast<size_t>(std::abs(k));
    std::fill_n(dst, N * N, T(0));
    size_t row0 = (k < 0) ? static_cast<size_t>(-k) : 0;
    size_t col0 = (k >= 0) ? static_cast<size_t>(k)  : 0;
    for (size_t i = 0; i < n; ++i)
        dst[(row0 + i) * N + (col0 + i)] = src[i];
}

/// numpy.diag(v, k=0) — 2-D input: extract k-th diagonal.
/// dst must hold diag_size(N, M, k) elements.
template<typename T>
inline void diag_from_mat(T* dst, const T* src, size_t N, size_t M, int k = 0) {
    size_t row0 = (k < 0) ? static_cast<size_t>(-k) : 0;
    size_t col0 = (k >= 0) ? static_cast<size_t>(k)  : 0;
    size_t len = diag_size(N, M, k);
    for (size_t i = 0; i < len; ++i)
        dst[i] = src[(row0 + i) * M + (col0 + i)];
}

} // namespace numpy
