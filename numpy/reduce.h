// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — numpy/reduce.h                            [PUBLIC HEADER]
//
//  Reduction routines (whole-array and axis-wise).
//
//      numpy.sum        numpy.mean       numpy.max        numpy.min
//      numpy.any        numpy.all        numpy.std        numpy.var
//      numpy.cumsum
//      numpy.mean (axis)   numpy.linalg.norm (axis)   [axis-wise skeletons]
//
//  Recommended entry point: #include "numpy/numpy.h"
//  Direct include is also valid for standalone use.
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstddef>
#include <cmath>
#include <vector>

namespace numpy {

// ============================================================================
// Summation helpers
// ============================================================================

/// Sequential (left-fold) summation from -0.0 — matches numpy's axis-reduction
/// algorithm for multi-dimensional arrays (np.sum / np.mean with an axis= arg).
///
/// numpy's np.add.reduce on axis k of an N-D array processes the reduction
/// dimension sequentially (element by element), regardless of array size.
/// This is empirically verified to match for all n ∈ [1, 1000+].
///
/// Start from -0.0 to preserve negative-zero output when all inputs are -0.0
/// (matching numpy's signed-zero semantics).
template<typename T>
inline T sequential_sum(const T* data, size_t n) {
    if (n == 0) return T(0);
    T res = T(-0.0);
    for (size_t i = 0; i < n; ++i) res += data[i];
    return res;
}

/// Pairwise summation of type T values — matches numpy's np.sum / np.mean
/// accumulation order for CONTIGUOUS 1-D reductions (axis=None).
///
/// Three tiers matching numpy's np.add.reduce on a flat 1-D contiguous array:
///
///   n < 8          Sequential loop from -0.0 (numpy's base case).
///
///   8 ≤ n ≤ 128   8-accumulator interleaved loop; remaining elements are
///                  appended to the running total AFTER the 8-way combine —
///                  matching numpy's empirically verified accumulation order.
///
///   n > 128        Recursive split aligned to multiples of 8, matching
///                  numpy's PW_BLOCKSIZE boundary.
///
/// NOTE: this function is used only for whole-array sum/mean (no axis arg).
/// For axis-wise reductions (mean_axis, norm_axis), numpy uses sequential_sum.
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
            r[0] += data[i + 0]; r[1] += data[i + 1];
            r[2] += data[i + 2]; r[3] += data[i + 3];
            r[4] += data[i + 4]; r[5] += data[i + 5];
            r[6] += data[i + 6]; r[7] += data[i + 7];
        }
        // numpy's exact combining order: ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))
        T res = ((r[0] + r[1]) + (r[2] + r[3])) +
                ((r[4] + r[5]) + (r[6] + r[7]));
        // Remaining elements appended after the 8-way combine.
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
// Whole-array reductions
// ============================================================================

/// numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False, ...)
/// Uses pairwise summation for exact bit-level alignment with numpy.
template<typename T>
inline T sum(const T* data, size_t n) {
    return pairwise_sum(data, n);
}

/// numpy.mean(a, axis=None, dtype=None, out=None, keepdims=False, *)
template<typename T>
inline T mean(const T* data, size_t n) {
    if (n == 0) return T(0);
    return pairwise_sum(data, n) / static_cast<T>(n);
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
/// NOTE: named 'stddev' because 'std' conflicts with C++ namespace std.
template<typename T>
inline T stddev(const T* data, size_t n) {
    if (n == 0) return T(0);
    T m = mean(data, n);
    std::vector<T> diffs(n);
    for (size_t i = 0; i < n; ++i) {
        T diff = data[i] - m;
        diffs[i] = diff * diff;
    }
    return std::sqrt(pairwise_sum(diffs.data(), n) / static_cast<T>(n));
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
    return pairwise_sum(diffs.data(), n) / static_cast<T>(n);
}

/// numpy.cumsum(a, axis=None, dtype=None, out=None)
/// 1D cumulative sum: dst[i] = sum_{j=0}^{i} src[j]
template<typename T>
inline void cumsum(const T* src, T* dst, size_t n) {
    if (n == 0) return;
    dst[0] = src[0];
    for (size_t i = 1; i < n; ++i)
        dst[i] = dst[i - 1] + src[i];
}

// ============================================================================
// Axis-reduction skeleton — shared by mean_axis, norm_axis, etc.
// ============================================================================

/// Shared axis-reduction skeleton: stride setup, fiber loop, coordinate decomp.
/// Calls op(in_base, out_base, axis_stride, axis_size, buf) per fiber.
template<typename T, typename F>
inline void axis_reduce_impl(const T* src, T* dst,
                              const ptrdiff_t* shape, int ndim,
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
            in_base  += idx * stride[d];
            out_base += idx * out_stride[d];
        }
        op(in_base, out_base, axis_stride, axis_size, buf.data());
    }
}

/// ndarray.mean(axis=N) — N-D, T in → T out
///
/// issue #001 fix: numpy's accumulation order for axis reductions depends on
/// whether the reduced axis is memory-contiguous (axis_stride == 1) or not:
///
///   axis_stride == 1  →  pairwise_sum  (same as numpy's contiguous 1-D path)
///   axis_stride >  1  →  sequential_sum (numpy's row-by-row strided path)
///
/// The distinction is empirically verified: np.mean([[2^24,1,1,1,1,1,1,1]],
/// axis=1) [stride=1] → pairwise result; np.mean(same_values.reshape(8,1).T,
/// axis=0) [stride>1] → sequential result.  For n < 8 both paths are
/// identical (both use sequential internally), so only n ≥ 8 exposes the
/// difference.
template<typename T>
inline void mean_axis(const T* src, T* dst,
                      const ptrdiff_t* shape, int ndim, int axis) {
    axis_reduce_impl<T>(src, dst, shape, ndim, axis,
        [&](ptrdiff_t ib, ptrdiff_t ob, ptrdiff_t as, ptrdiff_t n, T* buf) {
            for (ptrdiff_t i = 0; i < n; ++i)
                buf[static_cast<size_t>(i)] = src[ib + i * as];
            T s = (as == 1)
                  ? pairwise_sum (buf, static_cast<size_t>(n))
                  : sequential_sum(buf, static_cast<size_t>(n));
            dst[ob] = s / static_cast<T>(n);
        });
}

/// numpy.linalg.norm(x, ord=None, axis=N) — N-D vector norm along one axis
///
/// Same stride-dependent algorithm selection as mean_axis:
///   axis_stride == 1  →  pairwise_sum  for the squared elements
///   axis_stride >  1  →  sequential_sum
template<typename T>
inline void norm_axis(const T* src, T* dst,
                      const ptrdiff_t* shape, int ndim, int axis) {
    axis_reduce_impl<T>(src, dst, shape, ndim, axis,
        [&](ptrdiff_t ib, ptrdiff_t ob, ptrdiff_t as, ptrdiff_t n, T* buf) {
            for (ptrdiff_t i = 0; i < n; ++i) {
                T v = src[ib + i * as];
                buf[static_cast<size_t>(i)] = v * v;
            }
            T s = (as == 1)
                  ? pairwise_sum (buf, static_cast<size_t>(n))
                  : sequential_sum(buf, static_cast<size_t>(n));
            dst[ob] = std::sqrt(s);
        });
}

} // namespace numpy
