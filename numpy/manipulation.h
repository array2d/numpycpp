// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — numpy/manipulation.h                      [PUBLIC HEADER]
//
//  Array manipulation, shape operations, sorting, and fancy indexing.
//
//  Shape / structure:
//      numpy.diff         numpy.stack        numpy.concatenate
//      numpy.transpose    numpy.roll         numpy.flip
//      numpy.repeat       numpy.tile         numpy.where
//
//  Sorting / searching:
//      numpy.argsort      numpy.argmax       numpy.argmin
//
//  Fancy indexing:
//      numpy.take         numpy.compress
//      numpy.put          numpy.putmask
//      numpy.slice  (nd-slice)  numpy.slice_assign
//
//  Recommended entry point: #include "numpy/numpy.h"
//  Direct include is also valid for standalone use.
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstddef>
#include <cstring>
#include <vector>
#include <algorithm>
#include "detail/macros.h"   // NUMPY_UNROLL4

namespace numpy {

// ============================================================================
// Shape / structure
// ============================================================================

/// numpy.diff(a, n=1, axis=-1, prepend=..., append=...) — N-D, n=1
template<typename T>
inline void diff(const T* src, T* dst,
                 const ptrdiff_t* shape, int ndim, int axis) {
    if (ndim == 0) return;
    if (axis < 0) axis += ndim;
    ptrdiff_t axis_size = shape[axis];
    if (axis_size < 2) return;

    std::vector<ptrdiff_t> in_stride(ndim);
    in_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        in_stride[d] = in_stride[d + 1] * shape[d + 1];

    std::vector<ptrdiff_t> out_shape(shape, shape + ndim);
    out_shape[axis] = axis_size - 1;
    std::vector<ptrdiff_t> out_stride(ndim);
    out_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        out_stride[d] = out_stride[d + 1] * out_shape[d + 1];

    ptrdiff_t in_axis_stride  = in_stride[axis];
    ptrdiff_t out_axis_stride = out_stride[axis];

    ptrdiff_t n_fibers = 1;
    for (int d = 0; d < ndim; ++d)
        if (d != axis) n_fibers *= shape[d];

    for (ptrdiff_t f = 0; f < n_fibers; ++f) {
        ptrdiff_t rem = f, in_base = 0, out_base = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            if (d == axis) continue;
            ptrdiff_t idx = rem % shape[d];
            rem /= shape[d];
            in_base  += idx * in_stride[d];
            out_base += idx * out_stride[d];
        }
        for (ptrdiff_t i = 0; i < axis_size - 1; ++i)
            dst[out_base + i * out_axis_stride] =
                src[in_base + (i + 1) * in_axis_stride] -
                src[in_base +  i      * in_axis_stride];
    }
}

/// numpy.stack(arrays, axis=0, out=None, *, dtype=None, casting=...)
template<typename T>
inline void stack(const T* const* arrays, T* dst,
                  size_t n_arrays, size_t elem_size) {
    for (size_t i = 0; i < n_arrays; ++i)
        std::memcpy(dst + i * elem_size, arrays[i], elem_size * sizeof(T));
}

/// numpy.concatenate — 1D flat overload
template<typename T>
inline void concatenate(const T* const* arrays, T* dst,
                         const size_t* sizes, size_t n_arrays) {
    size_t off = 0;
    for (size_t i = 0; i < n_arrays; ++i) {
        std::memcpy(dst + off, arrays[i], sizes[i] * sizeof(T));
        off += sizes[i];
    }
}

/// numpy.concatenate — N-D with axis support
template<typename T>
inline void concatenate(const T* const* arrays, T* dst,
                         const ptrdiff_t* shape, int ndim, int axis,
                         const size_t* axis_sizes, size_t n_arrays) {
    if (n_arrays == 0 || ndim == 0) return;
    if (axis < 0) axis += ndim;

    ptrdiff_t trailing = 1;
    for (int d = axis + 1; d < ndim; ++d) trailing *= shape[d];

    ptrdiff_t out_axis = 0;
    for (size_t i = 0; i < n_arrays; ++i)
        out_axis += static_cast<ptrdiff_t>(axis_sizes[i]);

    std::vector<std::vector<ptrdiff_t>> in_stride(n_arrays);
    for (size_t k = 0; k < n_arrays; ++k) {
        in_stride[k].resize(ndim);
        in_stride[k][ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d) {
            ptrdiff_t s = (d + 1 == axis)
                ? static_cast<ptrdiff_t>(axis_sizes[k])
                : shape[d + 1];
            in_stride[k][d] = in_stride[k][d + 1] * s;
        }
    }

    std::vector<ptrdiff_t> out_shape(shape, shape + ndim);
    out_shape[axis] = out_axis;
    std::vector<ptrdiff_t> out_stride(ndim);
    out_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        out_stride[d] = out_stride[d + 1] * out_shape[d + 1];

    std::vector<size_t> slice_n(n_arrays);
    for (size_t i = 0; i < n_arrays; ++i)
        slice_n[i] = static_cast<size_t>(axis_sizes[i]) * static_cast<size_t>(trailing);

    ptrdiff_t n_slices = 1;
    for (int d = 0; d < axis; ++d) n_slices *= shape[d];

    ptrdiff_t out_slice_bytes = static_cast<ptrdiff_t>(
        static_cast<size_t>(out_axis) * static_cast<size_t>(trailing) * sizeof(T));

    for (ptrdiff_t s = 0; s < n_slices; ++s) {
        ptrdiff_t rem = s;
        std::vector<size_t> in_off(n_arrays, 0);
        for (int d = axis - 1; d >= 0; --d) {
            ptrdiff_t idx = rem % shape[d];
            rem /= shape[d];
            for (size_t k = 0; k < n_arrays; ++k)
                in_off[k] += static_cast<size_t>(idx) *
                              static_cast<size_t>(in_stride[k][d]);
        }
        char* out_start = reinterpret_cast<char*>(dst) + s * out_slice_bytes;
        size_t out_byte_off = 0;
        for (size_t i = 0; i < n_arrays; ++i) {
            size_t bytes = slice_n[i] * sizeof(T);
            std::memcpy(out_start + out_byte_off, arrays[i] + in_off[i], bytes);
            out_byte_off += bytes;
        }
    }
}

/// numpy.transpose(a, axes=None) — N-D with arbitrary axis permutation
template<typename T>
inline void transpose(const T* src, T* dst,
                      const ptrdiff_t* shape, int ndim, const int* axes) {
    if (ndim <= 1) {
        if (ndim == 1) std::memcpy(dst, src, shape[0] * sizeof(T));
        return;
    }
    std::vector<int> perm(ndim);
    if (axes) {
        for (int i = 0; i < ndim; ++i) perm[i] = axes[i];
    } else {
        for (int i = 0; i < ndim; ++i) perm[i] = ndim - 1 - i;
    }
    std::vector<ptrdiff_t> in_stride(ndim);
    in_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        in_stride[i] = in_stride[i + 1] * shape[i + 1];

    std::vector<ptrdiff_t> out_shape(ndim);
    for (int i = 0; i < ndim; ++i) out_shape[i] = shape[perm[i]];

    std::vector<ptrdiff_t> out_stride(ndim);
    out_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        out_stride[i] = out_stride[i + 1] * out_shape[i + 1];

    ptrdiff_t total = 1;
    for (int i = 0; i < ndim; ++i) total *= out_shape[i];

    for (ptrdiff_t out_idx = 0; out_idx < total; ++out_idx) {
        ptrdiff_t rem = out_idx, in_idx = 0;
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

/// numpy.where(condition, x, y) — scalar x, y
template<typename T>
inline void where_scalar(const bool* cond, T* dst, size_t n, T x, T y) {
    NUMPY_UNROLL4(i, dst[i] = cond[i] ? x : y);
}

/// numpy.where(condition, x, y) — array x, y
template<typename T>
inline void where_array(const bool* cond, T* dst, size_t n,
                         const T* x, const T* y) {
    NUMPY_UNROLL4(i, dst[i] = cond[i] ? x[i] : y[i]);
}

/// numpy.take(a, indices, axis=1) — extract first n_cols from each row
/// (legacy 2-D overload; use take() for the full N-D version below)
template<typename T>
inline void take_cols(const T* src, T* dst,
                      size_t rows, size_t src_cols, size_t n_cols) {
    for (size_t i = 0; i < rows; ++i)
        std::memcpy(dst + i * n_cols, src + i * src_cols, n_cols * sizeof(T));
}

// ============================================================================
// Sorting and searching
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
// Fancy indexing helpers
// ============================================================================

/// Compute the number of elements produced by slice(start, stop, step).
/// Pre-condition: step != 0, start/stop already normalised
/// (stop = -1 is the "before index 0" sentinel for negative-step slices).
///
/// Matches Python's len(range(start, stop, step)) exactly:
///   step > 0:  max(0, ceil((stop-start)/step))
///   step < 0:  max(0, ceil((start-stop)/-step))
inline ptrdiff_t slice_len(ptrdiff_t start, ptrdiff_t stop,
                            ptrdiff_t step) noexcept {
    if (step > 0)
        return (stop > start) ? (stop - start + step - 1) / step : 0;
    else
        return (start > stop) ? (start - stop - step - 1) / (-step) : 0;
}

/// numpy.take(a, indices, axis=None)
/// axis = -1 means axis=None (gather from flattened array).
/// Negative indices are wrapped modulo the axis dimension.
template<typename T>
inline void take(const T* src, T* dst,
                 const ptrdiff_t* shape, int ndim, int axis,
                 const ptrdiff_t* indices, size_t ni) {
    if (ni == 0) return;

    if (axis < 0) {
        ptrdiff_t total = 1;
        for (int d = 0; d < ndim; ++d) total *= shape[d];
        for (size_t k = 0; k < ni; ++k) {
            ptrdiff_t idx = indices[k];
            if (idx < 0) idx += total;
            dst[k] = src[idx];
        }
        return;
    }

    ptrdiff_t leading = 1;
    for (int d = 0; d < axis; ++d) leading *= shape[d];
    ptrdiff_t axis_size = shape[axis];
    ptrdiff_t trailing  = 1;
    for (int d = axis + 1; d < ndim; ++d) trailing *= shape[d];

    for (ptrdiff_t l = 0; l < leading; ++l) {
        const T* src_l = src + l * axis_size * trailing;
        T*       dst_l = dst + l * static_cast<ptrdiff_t>(ni) * trailing;
        for (size_t k = 0; k < ni; ++k) {
            ptrdiff_t idx = indices[k];
            if (idx < 0) idx += axis_size;
            std::memcpy(dst_l + static_cast<ptrdiff_t>(k) * trailing,
                        src_l + idx * trailing,
                        static_cast<size_t>(trailing) * sizeof(T));
        }
    }
}

/// numpy.compress(condition, a, axis=None)
/// Gathers elements from src (treated as flat) where mask[i] == true.
/// Returns the count of elements written to dst.
template<typename T>
inline size_t compress(const T* src, T* dst, const bool* mask, size_t n) {
    size_t cnt = 0;
    for (size_t i = 0; i < n; ++i)
        if (mask[i]) dst[cnt++] = src[i];
    return cnt;
}

/// N-D slice: dst = src[s0:e0:k0, s1:e1:k1, ...]
/// starts/stops/steps are pre-normalised by caller
/// (stop = -1 is the valid "before index 0" sentinel for negative-step slices).
template<typename T>
inline void slice(const T* src, T* dst,
                  const ptrdiff_t* shape, int ndim,
                  const ptrdiff_t* starts, const ptrdiff_t* stops,
                  const ptrdiff_t* steps) {
    if (ndim == 0) return;

    std::vector<ptrdiff_t> out_shape(ndim);
    ptrdiff_t total = 1;
    for (int d = 0; d < ndim; ++d) {
        out_shape[d] = slice_len(starts[d], stops[d], steps[d]);
        if (out_shape[d] <= 0) return;
        total *= out_shape[d];
    }

    std::vector<ptrdiff_t> in_stride(ndim);
    in_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        in_stride[d] = in_stride[d + 1] * shape[d + 1];

    for (ptrdiff_t out_idx = 0; out_idx < total; ++out_idx) {
        ptrdiff_t rem = out_idx, in_idx = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            ptrdiff_t od = rem % out_shape[d];
            rem /= out_shape[d];
            in_idx += (starts[d] + od * steps[d]) * in_stride[d];
        }
        dst[out_idx] = src[in_idx];
    }
}

/// N-D slice assignment — scalar: dst[slice] = value
template<typename T>
inline void slice_assign(T* dst,
                          const ptrdiff_t* shape, int ndim,
                          const ptrdiff_t* starts, const ptrdiff_t* stops,
                          const ptrdiff_t* steps, T value) {
    if (ndim == 0) return;

    std::vector<ptrdiff_t> out_shape(ndim);
    ptrdiff_t total = 1;
    for (int d = 0; d < ndim; ++d) {
        out_shape[d] = slice_len(starts[d], stops[d], steps[d]);
        if (out_shape[d] <= 0) return;
        total *= out_shape[d];
    }

    std::vector<ptrdiff_t> in_stride(ndim);
    in_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        in_stride[d] = in_stride[d + 1] * shape[d + 1];

    for (ptrdiff_t out_idx = 0; out_idx < total; ++out_idx) {
        ptrdiff_t rem = out_idx, in_idx = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            ptrdiff_t od = rem % out_shape[d];
            rem /= out_shape[d];
            in_idx += (starts[d] + od * steps[d]) * in_stride[d];
        }
        dst[in_idx] = value;
    }
}

/// N-D slice assignment — array: dst[slice] = values
template<typename T>
inline void slice_assign(T* dst,
                          const ptrdiff_t* shape, int ndim,
                          const ptrdiff_t* starts, const ptrdiff_t* stops,
                          const ptrdiff_t* steps, const T* values) {
    if (ndim == 0) return;

    std::vector<ptrdiff_t> out_shape(ndim);
    ptrdiff_t total = 1;
    for (int d = 0; d < ndim; ++d) {
        out_shape[d] = slice_len(starts[d], stops[d], steps[d]);
        if (out_shape[d] <= 0) return;
        total *= out_shape[d];
    }

    std::vector<ptrdiff_t> in_stride(ndim);
    in_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        in_stride[d] = in_stride[d + 1] * shape[d + 1];

    for (ptrdiff_t out_idx = 0; out_idx < total; ++out_idx) {
        ptrdiff_t rem = out_idx, in_idx = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            ptrdiff_t od = rem % out_shape[d];
            rem /= out_shape[d];
            in_idx += (starts[d] + od * steps[d]) * in_stride[d];
        }
        dst[in_idx] = values[out_idx];
    }
}

/// numpy.put(a, indices, values, mode='raise')
/// Scatters values into dst (flat) at the given indices.
template<typename T>
inline void put(T* dst, size_t n,
                const ptrdiff_t* indices, const T* values, size_t ni) {
    for (size_t k = 0; k < ni; ++k) {
        ptrdiff_t idx = indices[k];
        if (idx < 0) idx += static_cast<ptrdiff_t>(n);
        if (idx >= 0 && idx < static_cast<ptrdiff_t>(n))
            dst[static_cast<size_t>(idx)] = values[k];
    }
}

/// numpy.putmask(a, mask, values) — scalar variant
/// Sets dst[i] = value for every i where mask[i] is true.
template<typename T>
inline void putmask(T* dst, const bool* mask, size_t n, T value) {
    for (size_t i = 0; i < n; ++i)
        if (mask[i]) dst[i] = value;
}

/// numpy.putmask(a, mask, values) — array variant
/// Sets dst[i] = values[j++] for every i where mask[i] is true (sequential).
template<typename T>
inline void putmask(T* dst, const bool* mask, size_t n, const T* values) {
    size_t j = 0;
    for (size_t i = 0; i < n; ++i)
        if (mask[i]) dst[i] = values[j++];
}

// ============================================================================
// Convenience slice-assign overload: a[start:] = value  (1-D, scalar)
// ============================================================================

/// slice assignment: a[start:] = value (1-D, scalar, legacy convenience overload)
template<typename T>
inline void slice_assign(T* data, size_t n, size_t start, T value) {
    if (start >= n) return;
    std::fill(data + start, data + n, value);
}

} // namespace numpy
