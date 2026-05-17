// Native C++ implementations — zero pybind11 dependency.
// All functions operate on raw pointers + sizes.
// Namespace: numpy::impl
//
// Usable by any C++ project via #include "numpy/core.h"

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <cstring>
#include <cstddef>
#include <stdexcept>

namespace numpy {

// ============================================================================
// Array creation
// ============================================================================
template<typename T>
inline void zeros_like(T* dst, size_t n) {
    std::fill_n(dst, n, T(0));
}

template<typename T>
inline void ones_like(T* dst, size_t n) {
    std::fill_n(dst, n, T(1));
}

template<typename T>
inline void full_like(T* dst, size_t n, T fill_value) {
    std::fill_n(dst, n, fill_value);
}

// ============================================================================
// Element-wise math — T in → T out
// ============================================================================
template<typename T>
inline void sqrt(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::sqrt(src[i]);
}

template<typename T>
inline void abs(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::abs(src[i]);
}

template<typename T>
inline void exp(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::exp(src[i]);
}

template<typename T>
inline void log(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::log(src[i]);
}

template<typename T>
inline void sin(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::sin(src[i]);
}

template<typename T>
inline void cos(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::cos(src[i]);
}

template<typename T>
inline void tan(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::tan(src[i]);
}

template<typename T>
inline void power(const T* src, T* dst, size_t n, T exponent) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::pow(src[i], exponent);
}

template<typename T>
inline void clip(const T* src, T* dst, size_t n, T min_val, T max_val) {
    for (size_t i = 0; i < n; ++i)
        dst[i] = std::max(min_val, std::min(max_val, src[i]));
}

template<typename T>
inline void log10(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::log10(src[i]);
}

template<typename T>
inline void log2(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::log2(src[i]);
}

template<typename T>
inline void arcsin(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::asin(src[i]);
}

template<typename T>
inline void arccos(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::acos(src[i]);
}

template<typename T>
inline void arctan(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::atan(src[i]);
}

template<typename T>
inline void round(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::round(src[i]);
}

template<typename T>
inline void floor(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::floor(src[i]);
}

template<typename T>
inline void ceil(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::ceil(src[i]);
}

template<typename T>
inline void degrees(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = src[i] * T(180.0 / M_PI);
}

template<typename T>
inline void radians(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = src[i] * T(M_PI / 180.0);
}

template<typename T>
inline void sign(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = T((src[i] > T(0)) - (src[i] < T(0)));
}

// ============================================================================
// Reduction — T in → T out
// ============================================================================
template<typename T>
inline T sum(const T* data, size_t n) {
    T total = T(0);
    for (size_t i = 0; i < n; ++i) total += data[i];
    return total;
}

template<typename T>
inline T mean(const T* data, size_t n) {
    if (n == 0) return T(0);
    return sum(data, n) / static_cast<T>(n);
}

template<typename T>
inline T max(const T* data, size_t n) {
    if (n == 0) return T(0);
    T m = data[0];
    for (size_t i = 1; i < n; ++i)
        if (data[i] > m) m = data[i];
    return m;
}

template<typename T>
inline T min(const T* data, size_t n) {
    if (n == 0) return T(0);
    T m = data[0];
    for (size_t i = 1; i < n; ++i)
        if (data[i] < m) m = data[i];
    return m;
}

inline bool any(const bool* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (data[i]) return true;
    return false;
}

inline bool all(const bool* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!data[i]) return false;
    return true;
}

template<typename T>
inline T stddev(const T* data, size_t n) {
    if (n == 0) return T(0);
    T m = mean(data, n);
    T sum_sq = T(0);
    for (size_t i = 0; i < n; ++i) {
        T diff = data[i] - m;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / static_cast<T>(n));
}

template<typename T>
inline T var(const T* data, size_t n) {
    if (n == 0) return T(0);
    T m = mean(data, n);
    T sum_sq = T(0);
    for (size_t i = 0; i < n; ++i) {
        T diff = data[i] - m;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<T>(n);
}

// ============================================================================
// Comparison — T in → bool out
// ============================================================================
template<typename T>
inline void greater(const T* src, bool* dst, size_t n, T threshold) {
    for (size_t i = 0; i < n; ++i) dst[i] = (src[i] > threshold);
}

template<typename T>
inline void less(const T* src, bool* dst, size_t n, T threshold) {
    for (size_t i = 0; i < n; ++i) dst[i] = (src[i] < threshold);
}

template<typename T>
inline void equal(const T* src, bool* dst, size_t n, T val) {
    for (size_t i = 0; i < n; ++i) dst[i] = (src[i] == val);
}

template<typename T>
inline void greater_equal(const T* src, bool* dst, size_t n, T threshold) {
    for (size_t i = 0; i < n; ++i) dst[i] = (src[i] >= threshold);
}

template<typename T>
inline void less_equal(const T* src, bool* dst, size_t n, T threshold) {
    for (size_t i = 0; i < n; ++i) dst[i] = (src[i] <= threshold);
}

template<typename T>
inline void not_equal_scalar(const T* src, bool* dst, size_t n, T val) {
    for (size_t i = 0; i < n; ++i) dst[i] = (src[i] != val);
}

template<typename T>
inline void not_equal_array(const T* a, const T* b, bool* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = (a[i] != b[i]);
}

// ============================================================================
// Logical — bool in → bool out
// ============================================================================
inline void logical_and(const bool* a, const bool* b, bool* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] && b[i];
}

inline void logical_or(const bool* a, const bool* b, bool* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] || b[i];
}

inline void logical_not(const bool* src, bool* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = !src[i];
}

inline void logical_xor(const bool* a, const bool* b, bool* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] ^ b[i];
}

// ============================================================================
// Special value helpers — T in → bool out
// ============================================================================
template<typename T>
inline void isnan(const T* src, bool* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::isnan(src[i]);
}

template<typename T>
inline void isinf(const T* src, bool* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::isinf(src[i]);
}

template<typename T>
inline void isfinite(const T* src, bool* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::isfinite(src[i]);
}

// ============================================================================
// Binary element-wise — 2 arrays T in → T out
// ============================================================================
template<typename T>
inline void arctan2_array(const T* a, const T* b, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::atan2(a[i], b[i]);
}

template<typename T>
inline void arctan2_scalar(const T* src, T* dst, size_t n, T b) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::atan2(src[i], b);
}

template<typename T>
inline void maximum_array(const T* a, const T* b, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::max(a[i], b[i]);
}

template<typename T>
inline void maximum_scalar(const T* src, T* dst, size_t n, T b) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::max(src[i], b);
}

template<typename T>
inline void minimum_array(const T* a, const T* b, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::min(a[i], b[i]);
}

template<typename T>
inline void minimum_scalar(const T* src, T* dst, size_t n, T b) {
    for (size_t i = 0; i < n; ++i) dst[i] = std::min(src[i], b);
}

// ============================================================================
// Array manipulation
// ============================================================================

// diff_1d: src has n elements, dst has n-1 elements
template<typename T>
inline void diff_1d(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n - 1; ++i) dst[i] = src[i + 1] - src[i];
}

// diff_2d_axis0: result has (rows-1) * cols
template<typename T>
inline void diff_2d_axis0(const T* src, T* dst, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows - 1; ++i)
        for (size_t j = 0; j < cols; ++j)
            dst[i * cols + j] = src[(i + 1) * cols + j] - src[i * cols + j];
}

// diff_2d_axis1: result has rows * (cols-1)
template<typename T>
inline void diff_2d_axis1(const T* src, T* dst, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols - 1; ++j)
            dst[i * (cols - 1) + j] = src[i * cols + j + 1] - src[i * cols + j];
}

// stack: n_arrays each with elem_size, dst is flat (n_arrays * elem_size)
template<typename T>
inline void stack(const T* const* arrays, T* dst, size_t n_arrays, size_t elem_size) {
    for (size_t i = 0; i < n_arrays; ++i)
        std::memcpy(dst + i * elem_size, arrays[i], elem_size * sizeof(T));
}

// concatenate: arrays with individual elem_sizes, dst is flat
template<typename T>
inline void concatenate(const T* const* arrays, T* dst, const size_t* sizes, size_t n_arrays) {
    size_t off = 0;
    for (size_t i = 0; i < n_arrays; ++i) {
        std::memcpy(dst + off, arrays[i], sizes[i] * sizeof(T));
        off += sizes[i];
    }
}

// where with scalar x, y
template<typename T>
inline void where_scalar(const bool* cond, T* dst, size_t n, T x, T y) {
    for (size_t i = 0; i < n; ++i) dst[i] = cond[i] ? x : y;
}

// where with array x, y
template<typename T>
inline void where_array(const bool* cond, T* dst, size_t n, const T* x, const T* y) {
    for (size_t i = 0; i < n; ++i) dst[i] = cond[i] ? x[i] : y[i];
}

// transpose_2d: src is rows×cols, dst is cols×rows
template<typename T>
inline void transpose_2d(const T* src, T* dst, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            dst[j * rows + i] = src[i * cols + j];
}

// roll: cyclic shift (shift >= 0)
template<typename T>
inline void roll(const T* src, T* dst, size_t n, ptrdiff_t shift) {
    shift = shift % static_cast<ptrdiff_t>(n);
    if (shift < 0) shift += static_cast<ptrdiff_t>(n);
    size_t s = static_cast<size_t>(shift);
    for (size_t i = 0; i < n; ++i)
        dst[(i + s) % n] = src[i];
}

template<typename T>
inline void flip(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i)
        dst[i] = src[n - 1 - i];
}

// repeat: each element repeated 'reps' times
template<typename T>
inline void repeat(const T* src, T* dst, size_t n, size_t reps) {
    for (size_t i = 0; i < n; ++i)
        for (size_t r = 0; r < reps; ++r)
            dst[i * reps + r] = src[i];
}

// tile: entire array repeated 'reps' times as blocks
template<typename T>
inline void tile(const T* src, T* dst, size_t n, size_t reps) {
    for (size_t r = 0; r < reps; ++r)
        std::memcpy(dst + r * n, src, n * sizeof(T));
}

// take_cols: extract first n_cols from each row
template<typename T>
inline void take_cols(const T* src, T* dst, size_t rows, size_t src_cols, size_t n_cols) {
    for (size_t i = 0; i < rows; ++i)
        std::memcpy(dst + i * n_cols, src + i * src_cols, n_cols * sizeof(T));
}

// slice_assign: fill data[start..n) with value
template<typename T>
inline void slice_assign(T* data, size_t n, size_t start, T value) {
    if (start >= n) return;
    std::fill(data + start, data + n, value);
}

// ============================================================================
// Sorting and indexing
// ============================================================================
template<typename T>
inline void argsort(const T* data, ptrdiff_t* indices, size_t n) {
    for (size_t i = 0; i < n; ++i) indices[i] = static_cast<ptrdiff_t>(i);
    std::stable_sort(indices, indices + n,
        [data](ptrdiff_t a, ptrdiff_t b) { return data[a] < data[b]; });
}

template<typename T>
inline ptrdiff_t argmax(const T* data, size_t n) {
    if (n == 0) return -1;
    ptrdiff_t mi = 0;
    for (size_t i = 1; i < n; ++i)
        if (data[i] > data[static_cast<size_t>(mi)]) mi = static_cast<ptrdiff_t>(i);
    return mi;
}

template<typename T>
inline ptrdiff_t argmin(const T* data, size_t n) {
    if (n == 0) return -1;
    ptrdiff_t mi = 0;
    for (size_t i = 1; i < n; ++i)
        if (data[i] < data[static_cast<size_t>(mi)]) mi = static_cast<ptrdiff_t>(i);
    return mi;
}

// ============================================================================
// Set operations
// ============================================================================
inline void isin(const double* arr, bool* dst, size_t n,
                  const double* values, size_t nv) {
    std::unordered_set<double> vs(values, values + nv);
    for (size_t i = 0; i < n; ++i) dst[i] = vs.count(arr[i]) > 0;
}

inline std::vector<double> intersect1d(const double* a, size_t na,
                                        const double* b, size_t nb) {
    std::unordered_set<double> sa(a, a + na), sb(b, b + nb);
    std::vector<double> inter;
    for (double v : sa)
        if (sb.count(v)) inter.push_back(v);
    return inter;
}

// ============================================================================
// Interpolation
// ============================================================================
inline void interp(const double* x, double* dst, size_t nx,
                    const double* xp, const double* fp, size_t np) {
    for (size_t i = 0; i < nx; ++i) {
        double xi = x[i];
        if (xi <= xp[0]) { dst[i] = fp[0]; }
        else if (xi >= xp[np - 1]) { dst[i] = fp[np - 1]; }
        else {
            size_t j = 1;
            while (j < np && xp[j] < xi) ++j;
            double t = (xi - xp[j - 1]) / (xp[j] - xp[j - 1]);
            dst[i] = fp[j - 1] + t * (fp[j] - fp[j - 1]);
        }
    }
}

// ============================================================================
// Safe division
// ============================================================================
inline double safe_divide(double a, double b, double default_val = 0.0) {
    return b == 0.0 ? default_val : a / b;
}

// ============================================================================
// astype conversions
// ============================================================================
inline void astype_int(const double* src, int* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<int>(src[i]);
}

inline void astype_bool(const double* src, bool* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<bool>(src[i]);
}

inline void astype_bool_from_int(const int* src, bool* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<bool>(src[i]);
}

inline void truncate_to_float32(const double* src, double* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float tmp = static_cast<float>(src[i]);
        dst[i] = static_cast<double>(tmp);
    }
}

// ============================================================================
// mean_axis: T in → double out (matching numpy dtype promotion)
// ============================================================================

template<typename T>
inline void mean_axis0_2d(const T* src, double* dst, size_t rows, size_t cols) {
    for (size_t j = 0; j < cols; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < rows; ++i)
            sum += static_cast<double>(src[i * cols + j]);
        dst[j] = sum / static_cast<double>(rows);
    }
}

template<typename T>
inline void mean_axis1_2d(const T* src, double* dst, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j)
            sum += static_cast<double>(src[i * cols + j]);
        dst[i] = sum / static_cast<double>(cols);
    }
}

template<typename T>
inline void mean_axis0_3d(const T* src, double* dst, size_t d0, size_t d1, size_t d2) {
    size_t d1d2 = d1 * d2;
    for (size_t i = 0; i < d1d2; ++i) {
        double sum = 0.0;
        for (size_t k = 0; k < d0; ++k)
            sum += static_cast<double>(src[k * d1d2 + i]);
        dst[i] = sum / static_cast<double>(d0);
    }
}

template<typename T>
inline void mean_axis1_3d(const T* src, double* dst, size_t d0, size_t d1, size_t d2) {
    for (size_t i = 0; i < d0; ++i) {
        for (size_t k = 0; k < d2; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < d1; ++j)
                sum += static_cast<double>(src[i * d1 * d2 + j * d2 + k]);
            dst[i * d2 + k] = sum / static_cast<double>(d1);
        }
    }
}

template<typename T>
inline void mean_axis2_3d(const T* src, double* dst, size_t d0, size_t d1, size_t d2) {
    for (size_t i = 0; i < d0; ++i) {
        for (size_t j = 0; j < d1; ++j) {
            double sum = 0.0;
            const T* row = src + i * d1 * d2 + j * d2;
            for (size_t k = 0; k < d2; ++k)
                sum += static_cast<double>(row[k]);
            dst[i * d1 + j] = sum / static_cast<double>(d2);
        }
    }
}

// ============================================================================
// norm, dot — used by linalg
// ============================================================================
template<typename T>
inline T norm_sq(const T* data, size_t n) {
    T s = T(0);
    for (size_t i = 0; i < n; ++i) s += data[i] * data[i];
    return s;
}

template<typename T>
inline T dot(const T* a, const T* b, size_t n) {
    T s = T(0);
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

template<typename T>
inline void norm_axis1(const T* src, double* dst, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            double v = static_cast<double>(src[i * cols + j]);
            sum += v * v;
        }
        dst[i] = std::sqrt(sum);
    }
}

} // namespace numpy
