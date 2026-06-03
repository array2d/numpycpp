// Native C++ implementations — zero pybind11 dependency.
// All functions operate on raw pointers + sizes.
//
// Usable by any C++ project via #include "numpy/core.h"
//
// Convention: each function is annotated with its Python numpy equivalent,
// e.g. /// numpy.sqrt(x, /, out=None, *, where=True, ...)
//
// Acceleration (安全优化，保持 bit-exact 对齐):
//   - Loop unrolling (4x) for element-wise functions
//   - Stack allocation for small buffers (n ≤ 128)
//   - Reusable fiber buffer in axis reductions
//   - Fused multiply-accumulate in norm_sq/dot

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <cstring>
#include <cstddef>
#include <stdexcept>

#include "svml_bridge.h"

namespace numpy {

// Stack-allocation threshold for small-array optimizations
#define NUMPY_SMALL_STACK 128

// ============================================================================
// Array creation
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
// Element-wise math — T in → T out
//
// NUMPY_UNROLL4 macro: 4x loop unrolling to reduce branch overhead.
// All calls are inlined → identical codegen to hand-written unrolled loops.
// ============================================================================

#define NUMPY_UNROLL4(dst_i, body)       \
    do { size_t _i = 0;                  \
        for (; _i + 3 < n; _i += 4) {    \
            size_t dst_i = _i + 0; body; \
            dst_i = _i + 1; body;        \
            dst_i = _i + 2; body;        \
            dst_i = _i + 3; body;        \
        }                                \
        for (; _i < n; ++_i) {           \
            size_t dst_i = _i; body;     \
        }                                \
    } while(0)

/// numpy.sqrt(x, /, out=None, *, where=True, ...)
template<typename T>
inline void sqrt(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::sqrt(src[i]));
}

/// numpy.abs(x, /, out=None, *, where=True, ...)
template<typename T>
inline void abs(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::abs(src[i]));
}

/// numpy.exp(x, /, out=None, *, where=True, ...)
template<typename T>
inline void exp(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::exp(src[i]));
}

/// numpy.log(x, /, out=None, *, where=True, ...)
template<typename T>
inline void log(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::log(src[i]));
}

/// numpy.sin(x, /, out=None, *, where=True, ...)
template<typename T>
inline void sin(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::sin(src[i]));
}

/// numpy.cos(x, /, out=None, *, where=True, ...)
template<typename T>
inline void cos(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::cos(src[i]));
}

/// numpy.tan(x, /, out=None, *, where=True, ...)
template<typename T>
inline void tan(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::tan(src[i]));
}

/// numpy.cbrt(x, /, out=None, *, where=True, ...)
template<typename T>
inline void cbrt(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::cbrt(src[i]));
}

/// numpy.expm1(x, /, out=None, *, where=True, ...)
template<typename T>
inline void expm1(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::expm1(src[i]));
}

/// numpy.log1p(x, /, out=None, *, where=True, ...)
template<typename T>
inline void log1p(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::log1p(src[i]));
}

/// numpy.power(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void power(const T* src, T* dst, size_t n, T exponent) {
    NUMPY_UNROLL4(i, dst[i] = detail::pow(src[i], exponent));
}

/// numpy.clip(a, a_min, a_max, out=None, **kwargs)
template<typename T>
inline void clip(const T* src, T* dst, size_t n, T min_val, T max_val) {
    NUMPY_UNROLL4(i, dst[i] = std::max(min_val, std::min(max_val, src[i])));
}

/// numpy.log10(x, /, out=None, *, where=True, ...)
template<typename T>
inline void log10(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::log10(src[i]));
}

/// numpy.log2(x, /, out=None, *, where=True, ...)
template<typename T>
inline void log2(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::log2(src[i]));
}

/// numpy.arcsin(x, /, out=None, *, where=True, ...)
template<typename T>
inline void arcsin(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::asin(src[i]));
}

/// numpy.arccos(x, /, out=None, *, where=True, ...)
template<typename T>
inline void arccos(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::acos(src[i]));
}

/// numpy.arctan(x, /, out=None, *, where=True, ...)
template<typename T>
inline void arctan(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::atan(src[i]));
}

/// numpy.round(a, decimals=0, out=None)
template<typename T>
inline void round(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::nearbyint(src[i]));
}

/// numpy.floor(x, /, out=None, *, where=True, ...)
template<typename T>
inline void floor(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::floor(src[i]));
}

/// numpy.ceil(x, /, out=None, *, where=True, ...)
template<typename T>
inline void ceil(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::ceil(src[i]));
}

/// numpy.degrees(x, /, out=None, *, where=True, ...)
template<typename T>
inline void degrees(const T* src, T* dst, size_t n) {
    T factor = T(180.0) / T(M_PI);
    NUMPY_UNROLL4(i, dst[i] = src[i] * factor);
}

/// numpy.radians(x, /, out=None, *, where=True, ...)
template<typename T>
inline void radians(const T* src, T* dst, size_t n) {
    T factor = T(M_PI) / T(180.0);
    NUMPY_UNROLL4(i, dst[i] = src[i] * factor);
}

/// numpy.sign(x, /, out=None, *, where=True, ...)
template<typename T>
inline void sign(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = T((src[i] > T(0)) - (src[i] < T(0))));
}

// ============================================================================
// Pairwise summation — matches numpy's accumulation order exactly
// ============================================================================

/// Pairwise summation of type T values (numpy's reduction algorithm).
/// Recursively splits, 8-accumulator unrolled for medium sizes,
/// simple sequential for base case (n < 8).
/// Start with -0.0 to preserve negative zero (matching numpy).
/// Stack-allocated accumulator for n ≤ 128 (avoids heap allocation).
template<typename T>
inline T pairwise_sum(const T* data, size_t n) {
    if (n == 0) return T(0);
    if (n < 8) {
        T res = T(-0.0);
        for (size_t i = 0; i < n; ++i) res += data[i];
        return res;
    }
    if (n <= 128) {
        T r[8];
        size_t i = 0;
        for (; i < 8 && i < n; ++i) r[i] = data[i];
        for (; i + 7 < n; i += 8) {
            r[0] += data[i + 0];
            r[1] += data[i + 1];
            r[2] += data[i + 2];
            r[3] += data[i + 3];
            r[4] += data[i + 4];
            r[5] += data[i + 5];
            r[6] += data[i + 6];
            r[7] += data[i + 7];
        }
        // numpy's exact combining order: ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))
        T res = ((r[0] + r[1]) + (r[2] + r[3])) +
                ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += data[i];
        return res;
    }
    // recursive split — ensure multiple of 8 (matching numpy)
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(data, n2) +
           pairwise_sum(data + n2, n - n2);
}

// ============================================================================
// Reduction — T in → T out
// ============================================================================

/// numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False, ...)
//  Accumulation matches input dtype (matching numpy: float32 stays float32).
//  Uses numpy's pairwise summation algorithm for exact bit-level alignment.
template<typename T>
inline T sum(const T* data, size_t n) {
    return pairwise_sum(data, n);
}

/// numpy.mean(a, axis=None, dtype=None, out=None, keepdims=False, *)
template<typename T>
inline T mean(const T* data, size_t n) {
    if (n == 0) return T(0);
    T s = pairwise_sum(data, n);
    return s / static_cast<T>(n);
}

/// numpy.max(a, axis=None, out=None, keepdims=False, initial=..., where=...)
template<typename T>
inline T max(const T* data, size_t n) {
    if (n == 0) return T(0);
    T m = data[0];
    for (size_t i = 1; i < n; ++i)
        if (data[i] > m) m = data[i];
    return m;
}

/// numpy.min(a, axis=None, out=None, keepdims=False, initial=..., where=...)
template<typename T>
inline T min(const T* data, size_t n) {
    if (n == 0) return T(0);
    T m = data[0];
    for (size_t i = 1; i < n; ++i)
        if (data[i] < m) m = data[i];
    return m;
}

/// numpy.any(a, axis=None, out=None, keepdims=False, *)
inline bool any(const bool* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (data[i]) return true;
    return false;
}

/// numpy.all(a, axis=None, out=None, keepdims=False, *)
inline bool all(const bool* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!data[i]) return false;
    return true;
}

/// numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *)
//  NOTE: named 'stddev' because 'std' conflicts with C++ namespace std.
template<typename T>
inline T stddev(const T* data, size_t n) {
    if (n == 0) return T(0);
    T m = mean(data, n);
    std::vector<T> diffs(n);
    for (size_t i = 0; i < n; ++i) {
        T diff = data[i] - m;
        diffs[i] = diff * diff;
    }
    T sum_sq = pairwise_sum(diffs.data(), n);
    return std::sqrt(sum_sq / static_cast<T>(n));
}

/// numpy.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *)
template<typename T>
inline T var(const T* data, size_t n) {
    if (n == 0) return T(0);
    T m = mean(data, n);
    std::vector<T> diffs(n);
    for (size_t i = 0; i < n; ++i) {
        T diff = data[i] - m;
        diffs[i] = diff * diff;
    }
    T sum_sq = pairwise_sum(diffs.data(), n);
    return sum_sq / static_cast<T>(n);
}

// ============================================================================
// Comparison — T in → bool out
//
// NOTE: _scalar / _array suffix convention for internal native helpers that
// share the same numpy API name. C++ disallows overloads with identical
// signatures, so scalar-vs-array variants are disambiguated in the native
// layer. Wrappers (pycpp/) expose the unified numpy name via their own
// overloads; these suffixed helpers are NOT bound directly to Python.
// ============================================================================

/// numpy.greater(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void greater(const T* src, bool* dst, size_t n, T threshold) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] > threshold));
}
/// numpy.less(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void less(const T* src, bool* dst, size_t n, T threshold) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] < threshold));
}
/// numpy.equal(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void equal(const T* src, bool* dst, size_t n, T val) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] == val));
}
/// numpy.greater_equal(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void greater_equal(const T* src, bool* dst, size_t n, T threshold) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] >= threshold));
}
/// numpy.less_equal(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void less_equal(const T* src, bool* dst, size_t n, T threshold) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] <= threshold));
}
/// numpy.not_equal(x1, x2, /, out=None, *, where=True, ...) — scalar variant
template<typename T>
inline void not_equal_scalar(const T* src, bool* dst, size_t n, T val) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] != val));
}
/// numpy.not_equal(x1, x2, /, out=None, *, where=True, ...) — array variant
template<typename T>
inline void not_equal_array(const T* a, const T* b, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = (a[i] != b[i]));
}

// ============================================================================
// Logical — bool in → bool out
// ============================================================================

/// numpy.logical_and(x1, x2, /, out=None, *, where=True, ...)
inline void logical_and(const bool* a, const bool* b, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = a[i] && b[i]);
}
/// numpy.logical_or(x1, x2, /, out=None, *, where=True, ...)
inline void logical_or(const bool* a, const bool* b, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = a[i] || b[i]);
}
/// numpy.logical_not(x, /, out=None, *, where=True, ...)
inline void logical_not(const bool* src, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = !src[i]);
}
/// numpy.logical_xor(x1, x2, /, out=None, *, where=True, ...)
inline void logical_xor(const bool* a, const bool* b, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = a[i] ^ b[i]);
}

// ============================================================================
// Special value helpers — T in → bool out
// ============================================================================

/// numpy.isnan(x, /, out=None, *, where=True, ...)
template<typename T>
inline void isnan(const T* src, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::isnan(src[i]));
}
/// numpy.isinf(x, /, out=None, *, where=True, ...)
template<typename T>
inline void isinf(const T* src, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::isinf(src[i]));
}
/// numpy.isfinite(x, /, out=None, *, where=True, ...)
template<typename T>
inline void isfinite(const T* src, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::isfinite(src[i]));
}

// ============================================================================
// Binary element-wise — 2 arrays T in → T out
// ============================================================================

/// numpy.hypot(x1, x2, /, out=None, *, where=True, ...) — array-array
template<typename T>
inline void hypot_array(const T* a, const T* b, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::hypot(a[i], b[i]));
}

/// numpy.arctan2(x1, x2, /, out=None, *, where=True, ...) — array-array
template<typename T>
inline void arctan2_array(const T* a, const T* b, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::atan2(a[i], b[i]));
}
/// numpy.arctan2(x1, x2, /, out=None, *, where=True, ...) — array-scalar
template<typename T>
inline void arctan2_scalar(const T* src, T* dst, size_t n, T b) {
    NUMPY_UNROLL4(i, dst[i] = detail::atan2(src[i], b));
}
/// numpy.maximum(x1, x2, /, out=None, *, where=True, ...) — array-array
template<typename T>
inline void maximum_array(const T* a, const T* b, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::max(a[i], b[i]));
}
/// numpy.maximum(x1, x2, /, out=None, *, where=True, ...) — scalar variant
template<typename T>
inline void maximum_scalar(const T* src, T* dst, size_t n, T b) {
    NUMPY_UNROLL4(i, dst[i] = std::max(src[i], b));
}
/// numpy.minimum(x1, x2, /, out=None, *, where=True, ...) — array-array
template<typename T>
inline void minimum_array(const T* a, const T* b, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::min(a[i], b[i]));
}
/// numpy.minimum(x1, x2, /, out=None, *, where=True, ...) — scalar variant
template<typename T>
inline void minimum_scalar(const T* src, T* dst, size_t n, T b) {
    NUMPY_UNROLL4(i, dst[i] = std::min(src[i], b));
}

// ============================================================================
// Array manipulation
// ============================================================================

/// numpy.diff(a, n=1, axis=-1, prepend=..., append=...) — N-D, n=1
template<typename T>
inline void diff(const T* src, T* dst, const ptrdiff_t* shape, int ndim, int axis) {
    if (ndim == 0) return;

    // Normalize axis
    if (axis < 0) axis += ndim;
    ptrdiff_t axis_size = shape[axis];
    if (axis_size < 2) return;

    // Strides for C-contiguous layout
    std::vector<ptrdiff_t> in_stride(ndim);
    in_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        in_stride[d] = in_stride[d + 1] * shape[d + 1];

    // Output shape: same as input but axis_size - 1 along 'axis'
    std::vector<ptrdiff_t> out_shape(shape, shape + ndim);
    out_shape[axis] = axis_size - 1;

    std::vector<ptrdiff_t> out_stride(ndim);
    out_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        out_stride[d] = out_stride[d + 1] * out_shape[d + 1];

    ptrdiff_t in_axis_stride = in_stride[axis];
    ptrdiff_t out_axis_stride = out_stride[axis];

    // Number of fibers = product of all non-axis dimensions
    ptrdiff_t n_fibers = 1;
    for (int d = 0; d < ndim; ++d)
        if (d != axis) n_fibers *= shape[d];

    // For each fiber, do 1D diff along axis
    for (ptrdiff_t f = 0; f < n_fibers; ++f) {
        // Decompose fiber index f → multi-index (non-axis dims)
        ptrdiff_t rem = f;
        ptrdiff_t in_base = 0, out_base = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            if (d == axis) continue;
            ptrdiff_t idx = rem % shape[d];
            rem /= shape[d];
            in_base += idx * in_stride[d];
            out_base += idx * out_stride[d];
        }

        for (ptrdiff_t i = 0; i < axis_size - 1; ++i) {
            dst[out_base + i * out_axis_stride] =
                src[in_base + (i + 1) * in_axis_stride] -
                src[in_base + i * in_axis_stride];
        }
    }
}

/// numpy.stack(arrays, axis=0, out=None, *, dtype=None, casting=...)
template<typename T>
inline void stack(const T* const* arrays, T* dst, size_t n_arrays, size_t elem_size) {
    for (size_t i = 0; i < n_arrays; ++i)
        std::memcpy(dst + i * elem_size, arrays[i], elem_size * sizeof(T));
}

/// numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting=...)
template<typename T>
inline void concatenate(const T* const* arrays, T* dst, const size_t* sizes, size_t n_arrays) {
    size_t off = 0;
    for (size_t i = 0; i < n_arrays; ++i) {
        std::memcpy(dst + off, arrays[i], sizes[i] * sizeof(T));
        off += sizes[i];
    }
}

/// numpy.where(condition, x, y) — scalar x, y
template<typename T>
inline void where_scalar(const bool* cond, T* dst, size_t n, T x, T y) {
    NUMPY_UNROLL4(i, dst[i] = cond[i] ? x : y);
}
/// numpy.where(condition, x, y) — array x, y
template<typename T>
inline void where_array(const bool* cond, T* dst, size_t n, const T* x, const T* y) {
    NUMPY_UNROLL4(i, dst[i] = cond[i] ? x[i] : y[i]);
}

/// numpy.transpose(a, axes=None) — 2D only
/// numpy.transpose(a, axes=None) — N-D with arbitrary axis permutation
template<typename T>
inline void transpose(const T* src, T* dst, const ptrdiff_t* shape, int ndim, const int* axes) {
    if (ndim <= 1) {
        if (ndim == 1) std::memcpy(dst, src, shape[0] * sizeof(T));
        return;
    }

    // Build permutation: use axes if given, else reverse
    std::vector<int> perm(ndim);
    if (axes) {
        for (int i = 0; i < ndim; ++i) perm[i] = axes[i];
    } else {
        for (int i = 0; i < ndim; ++i) perm[i] = ndim - 1 - i;
    }

    // Input strides (C-contiguous)
    std::vector<ptrdiff_t> in_stride(ndim);
    in_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        in_stride[i] = in_stride[i + 1] * shape[i + 1];

    // Output shape
    std::vector<ptrdiff_t> out_shape(ndim);
    for (int i = 0; i < ndim; ++i) out_shape[i] = shape[perm[i]];

    // Output strides (C-contiguous)
    std::vector<ptrdiff_t> out_stride(ndim);
    out_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        out_stride[i] = out_stride[i + 1] * out_shape[i + 1];

    // Inverse permutation: output dim j → input dim perm[j]
    // For output coord out_coord, input coord in_coord[perm[j]] = out_coord[j]
    // → in_coord[perm[j]] = out_coord[j] → in_flat = sum_j out_coord[j] * in_stride[perm[j]]

    ptrdiff_t total = 1;
    for (int i = 0; i < ndim; ++i) total *= out_shape[i];

    // Iterate output in C order
    for (ptrdiff_t out_idx = 0; out_idx < total; ++out_idx) {
        ptrdiff_t rem = out_idx;
        ptrdiff_t in_idx = 0;
        for (int j = ndim - 1; j >= 0; --j) {
            ptrdiff_t c = rem % out_shape[j];
            rem /= out_shape[j];
            in_idx += c * in_stride[perm[j]];
        }
        dst[out_idx] = src[in_idx];
    }
}

/// numpy.roll(a, shift, axis=None)
template<typename T>
inline void roll(const T* src, T* dst, size_t n, ptrdiff_t shift) {
    shift = shift % static_cast<ptrdiff_t>(n);
    if (shift < 0) shift += static_cast<ptrdiff_t>(n);
    size_t s = static_cast<size_t>(shift);
    NUMPY_UNROLL4(i, dst[(i + s) % n] = src[i]);
}

/// numpy.flip(m, axis=None)
template<typename T>
inline void flip(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = src[n - 1 - i]);
}

/// numpy.repeat(a, repeats, axis=None)
template<typename T>
inline void repeat(const T* src, T* dst, size_t n, size_t reps) {
    for (size_t i = 0; i < n; ++i)
        for (size_t r = 0; r < reps; ++r)
            dst[i * reps + r] = src[i];
}

/// numpy.tile(A, reps)
template<typename T>
inline void tile(const T* src, T* dst, size_t n, size_t reps) {
    for (size_t r = 0; r < reps; ++r)
        std::memcpy(dst + r * n, src, n * sizeof(T));
}

/// numpy.take(a, indices, axis=1) — extract first n_cols from each row
template<typename T>
inline void take_cols(const T* src, T* dst, size_t rows, size_t src_cols, size_t n_cols) {
    for (size_t i = 0; i < rows; ++i)
        std::memcpy(dst + i * n_cols, src + i * src_cols, n_cols * sizeof(T));
}

/// slice assignment: a[start:] = value
template<typename T>
inline void slice_assign(T* data, size_t n, size_t start, T value) {
    if (start >= n) return;
    std::fill(data + start, data + n, value);
}

// ============================================================================
// Sorting and indexing
// ============================================================================

/// numpy.argsort(a, axis=-1, kind=None, order=None)
template<typename T>
inline void argsort(const T* data, ptrdiff_t* indices, size_t n) {
    for (size_t i = 0; i < n; ++i) indices[i] = static_cast<ptrdiff_t>(i);
    std::stable_sort(indices, indices + n,
        [data](ptrdiff_t a, ptrdiff_t b) { return data[a] < data[b]; });
}

/// numpy.argmax(a, axis=None, out=None, *, keepdims=...)
template<typename T>
inline ptrdiff_t argmax(const T* data, size_t n) {
    if (n == 0) return -1;
    ptrdiff_t mi = 0;
    for (size_t i = 1; i < n; ++i)
        if (data[i] > data[static_cast<size_t>(mi)]) mi = static_cast<ptrdiff_t>(i);
    return mi;
}

/// numpy.argmin(a, axis=None, out=None, *, keepdims=...)
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
// Safe division — utility, not a numpy API
// ============================================================================

/// safe_divide(a, b, default_val=0.0): returns a/b, or default_val if b == 0
inline double safe_divide(double a, double b, double default_val = 0.0) {
    return b == 0.0 ? default_val : a / b;
}

/// numpy.unwrap(p, discont=None, axis=-1)
/// 1D only: unwrap phase angles by correcting jumps > discont.
/// Uses numpy's exact computation path: (dd+period/2)%period - period/2
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
        if (std::abs(dd) >= discont) {
            // numpy: ddmod = (dd + period/2) % period - period/2
            // Python-style mod using floor division (numpy's mod):
            T val = dd + p2;
            T val_mod = val - std::floor(val / period) * period;
            T ddmod = val_mod - p2;
            // boundary_ambiguous: when dd > 0 and ddmod == -period/2, use +period/2
            if (dd > T(0) && ddmod == -p2) ddmod = p2;
            ph_correct = ddmod - dd;
        }
        cum_correct += ph_correct;
        dst[i] = src[i] + cum_correct;
    }
}

/// numpy.cumsum(a, axis=None, dtype=None, out=None)
/// 1D cumulative sum: dst[i] = sum_{j=0}^{i} src[j]
template<typename T>
inline void cumsum(const T* src, T* dst, size_t n) {
    if (n == 0) return;
    dst[0] = src[0];
    for (size_t i = 1; i < n; ++i) {
        dst[i] = dst[i-1] + src[i];
    }
}

// ============================================================================
// astype conversions
// ============================================================================

/// ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
template<typename Tout, typename Tin>
inline void astype(const Tin* src, Tout* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = static_cast<Tout>(src[i]));
}

/// float64 → float32 → float64 roundtrip (for precision testing)
inline void truncate_to_float32(const double* src, double* dst, size_t n) {
    NUMPY_UNROLL4(i, { float tmp = static_cast<float>(src[i]);
                       dst[i] = static_cast<double>(tmp); });
}

// ============================================================================
// Axis reductions — shared fiber-skeleton for mean_axis, norm_axis, etc.
// ============================================================================

/// Shared axis-reduction skeleton: stride setup, fiber loop, coordinate decomp.
/// Calls op(in_base, out_base, axis_stride, axis_size, buf) per fiber.
template<typename T, typename F>
inline void axis_reduce_impl(const T* src, T* dst, const ptrdiff_t* shape, int ndim,
                              int axis, F&& op) {
    if (ndim == 0) return;
    if (axis < 0) axis += ndim;
    ptrdiff_t axis_size = shape[axis];
    if (axis_size == 0) return;

    std::vector<ptrdiff_t> stride(ndim);
    stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        stride[d] = stride[d + 1] * shape[d + 1];

    ptrdiff_t axis_stride = stride[axis];

    ptrdiff_t n_fibers = 1;
    for (int d = 0; d < ndim; ++d)
        if (d != axis) n_fibers *= shape[d];

    std::vector<ptrdiff_t> out_shape(shape, shape + ndim);
    out_shape[axis] = 1;
    std::vector<ptrdiff_t> out_stride(ndim);
    out_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        out_stride[d] = out_stride[d + 1] * out_shape[d + 1];

    std::vector<T> buf(static_cast<size_t>(axis_size));

    for (ptrdiff_t f = 0; f < n_fibers; ++f) {
        ptrdiff_t rem = f, in_base = 0, out_base = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            if (d == axis) continue;
            ptrdiff_t idx = rem % shape[d];
            rem /= shape[d];
            in_base += idx * stride[d];
            out_base += idx * out_stride[d];
        }
        op(in_base, out_base, axis_stride, axis_size, buf.data());
    }
}

/// ndarray.mean(axis=N) — N-D, T in → T out
template<typename T>
inline void mean_axis(const T* src, T* dst, const ptrdiff_t* shape, int ndim, int axis) {
    axis_reduce_impl<T>(src, dst, shape, ndim, axis,
        [&](ptrdiff_t ib, ptrdiff_t ob, ptrdiff_t as, ptrdiff_t n, T* buf) {
            for (ptrdiff_t i = 0; i < n; ++i)
                buf[static_cast<size_t>(i)] = src[ib + i * as];
            T sum = pairwise_sum(buf, static_cast<size_t>(n));
            dst[ob] = sum / static_cast<T>(n);
        });
}

// ============================================================================
// norm, dot — used by linalg
// ============================================================================

/// squared L2 norm / dot — stack allocation for n ≤ 128
template<typename T>
inline T norm_sq(const T* data, size_t n) {
    T buf[NUMPY_SMALL_STACK];
    T* squares = (n <= NUMPY_SMALL_STACK) ? buf : new T[n];
    for (size_t i = 0; i < n; ++i) squares[i] = data[i] * data[i];
    T result = pairwise_sum(squares, n);
    if (n > NUMPY_SMALL_STACK) delete[] squares;
    return result;
}

/// numpy.dot(a, b, out=None) — pairwise sum, matches np.sum(a*b)
template<typename T>
inline T dot(const T* a, const T* b, size_t n) {
    T buf[NUMPY_SMALL_STACK];
    T* prods = (n <= NUMPY_SMALL_STACK) ? buf : new T[n];
    for (size_t i = 0; i < n; ++i) prods[i] = a[i] * b[i];
    T result = pairwise_sum(prods, n);
    if (n > NUMPY_SMALL_STACK) delete[] prods;
    return result;
}

/// numpy.linalg.norm(x, ord=None, axis=N, keepdims=False) — N-D
template<typename T>
inline void norm_axis(const T* src, T* dst, const ptrdiff_t* shape, int ndim, int axis) {
    axis_reduce_impl<T>(src, dst, shape, ndim, axis,
        [&](ptrdiff_t ib, ptrdiff_t ob, ptrdiff_t as, ptrdiff_t n, T* buf) {
            for (ptrdiff_t i = 0; i < n; ++i) {
                T v = src[ib + i * as];
                buf[static_cast<size_t>(i)] = v * v;
            }
            T sum = pairwise_sum(buf, static_cast<size_t>(n));
            dst[ob] = std::sqrt(sum);
        });
}

} // namespace numpy
