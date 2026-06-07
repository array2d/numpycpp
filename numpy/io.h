// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — numpy/io.h                                [PUBLIC HEADER]
//
//  Set operations, interpolation, and numeric utilities.
//
//      numpy.isin          numpy.intersect1d   numpy.flatnonzero
//      numpy.interp        numpy.unwrap        numpy.cumsum
//      safe_divide  (numpycpp utility — not a numpy API)
//
//  Recommended entry point: #include "numpy/numpy.h"
//  Direct include is also valid for standalone use.
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstddef>
#include <cmath>
#include <vector>
#include <unordered_set>

namespace numpy {

// ============================================================================
// Set operations
// ============================================================================

/// numpy.isin(element, test_elements, assume_unique=False, invert=False)
template<typename T>
inline void isin(const T* arr, bool* dst, size_t n,
                 const T* values, size_t nv) {
    std::unordered_set<T> vs(values, values + nv);
    for (size_t i = 0; i < n; ++i) dst[i] = vs.count(arr[i]) > 0;
}

/// numpy.flatnonzero(a)
template<typename T>
inline std::vector<size_t> flatnonzero(const T* arr, size_t n) {
    std::vector<size_t> idx;
    for (size_t i = 0; i < n; ++i)
        if (arr[i] != T(0)) idx.push_back(i);
    return idx;
}

/// numpy.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)
template<typename T>
inline std::vector<T> intersect1d(const T* a, size_t na,
                                   const T* b, size_t nb) {
    std::unordered_set<T> sa(a, a + na), sb(b, b + nb);
    std::vector<T> inter;
    for (T v : sa)
        if (sb.count(v)) inter.push_back(v);
    return inter;
}

// ============================================================================
// Interpolation
// ============================================================================

/// numpy.interp(x, xp, fp, left=None, right=None, period=None)
inline void interp(const double* x, double* dst, size_t nx,
                   const double* xp, const double* fp, size_t np_sz) {
    for (size_t i = 0; i < nx; ++i) {
        double xi = x[i];
        if (xi <= xp[0]) { dst[i] = fp[0]; }
        else if (xi >= xp[np_sz - 1]) { dst[i] = fp[np_sz - 1]; }
        else {
            size_t j = 1;
            while (j < np_sz && xp[j] < xi) ++j;
            double t = (xi - xp[j - 1]) / (xp[j] - xp[j - 1]);
            dst[i] = fp[j - 1] + t * (fp[j] - fp[j - 1]);
        }
    }
}

// ============================================================================
// Utility
// ============================================================================

/// safe_divide(a, b, default_val=0.0): returns a/b, or default_val if b == 0
inline double safe_divide(double a, double b, double default_val = 0.0) {
    return b == 0.0 ? default_val : a / b;
}

/// numpy.unwrap(p, discont=None, axis=-1)
/// 1D only: unwrap phase angles by correcting jumps >= discont.
/// Uses numpy's exact computation path: (dd+period/2) % period - period/2
/// to guarantee bit-identical float32 results.
template<typename T>
inline void unwrap(const T* src, T* dst, size_t n, T discont = T(M_PI)) {
    if (n == 0) return;
    T period = T(2) * discont;
    T p2 = period / T(2);
    dst[0] = src[0];
    T cum_correct = T(0);
    for (size_t i = 1; i < n; ++i) {
        T dd = src[i] - src[i - 1];
        T ph_correct = T(0);
        if (std::isnan(dd)) {
            cum_correct = dd;
        } else if (std::abs(dd) >= discont) {
            T val = dd + p2;
            T val_mod_signed = std::fmod(val, period);
            T val_mod = (val_mod_signed < T(0)) ? val_mod_signed + period : val_mod_signed;
            T ddmod = val_mod - p2;
            if (dd > T(0) && ddmod == -p2) ddmod = p2;
            ph_correct = ddmod - dd;
            cum_correct += ph_correct;
        } else {
            cum_correct += ph_correct;
        }
        dst[i] = src[i] + cum_correct;
    }
}

} // namespace numpy
