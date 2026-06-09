// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — pycpp/manipulation_py.h                   [PUBLIC HEADER]
//  Pybind11 wrappers: array manipulation, indexing, and fancy indexing.
//
//  Shape / structure:
//      diff  stack  concatenate  vstack  hstack  transpose  flatten  squeeze
//      where  roll  flip  repeat  tile
//
//  Sorting / searching:
//      argsort  argmax  argmin
//
//  Fancy indexing:
//      take  compress  slice (1-D and N-D)  slice_assign
//      put  putmask  take_cols
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../numpy/numpy.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace numpy {

// ============================================================================
// Shape / structure
// ============================================================================

/// numpy.diff(a, n=1, axis=-1)
template<typename T>
py::array_t<T> diff(const py::array_t<T>& arr, int n = 1, int axis = -1) {
    auto buf = arr.request();
    if (buf.ndim == 0 || buf.size < 2) return py::array_t<T>{};
    int ax = (axis == -1) ? static_cast<int>(buf.ndim - 1) : axis;
    if (ax < 0 || ax >= static_cast<int>(buf.ndim)) return py::array_t<T>{};
    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());
    if (shape[ax] < 2) return py::array_t<T>{};
    std::vector<ptrdiff_t> out_shape = shape;
    out_shape[ax]--;
    std::vector<py::ssize_t> py_out(out_shape.begin(), out_shape.end());
    py::array_t<T> result(py_out);
    numpy::diff(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr),
                shape.data(), static_cast<int>(buf.ndim), ax);
    return result;
}

/// numpy.stack(arrays, axis=0)
template<typename T>
py::array_t<T> stack(const std::vector<py::array_t<T>>& arrays) {
    if (arrays.empty()) return py::array_t<T>{};
    auto buf0 = arrays[0].request();
    py::array_t<T> result({static_cast<py::ssize_t>(arrays.size()), buf0.size});
    T* dst = static_cast<T*>(result.request().ptr);
    for (size_t i = 0; i < arrays.size(); ++i) {
        auto buf = arrays[i].request();
        std::memcpy(dst + i * buf0.size, static_cast<const T*>(buf.ptr),
                    buf.size * sizeof(T));
    }
    return result;
}

/// numpy.concatenate((a1,a2,...), axis=0)
template<typename T>
py::array_t<T> concatenate(const std::vector<py::array_t<T>>& arrays,
                            int axis = 0) {
    if (arrays.empty()) return py::array_t<T>{};
    auto buf0 = arrays[0].request();
    int ndim = static_cast<int>(buf0.ndim);
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim)
        throw std::invalid_argument("concatenate: axis out of range");
    for (const auto& arr : arrays)
        if (arr.request().ndim != ndim)
            throw std::invalid_argument(
                "concatenate: all arrays must have same ndim");

    std::vector<ptrdiff_t> shape(ndim);
    for (int d = 0; d < ndim; ++d) shape[d] = buf0.shape[d];

    std::vector<size_t> axis_sizes(arrays.size());
    for (size_t i = 0; i < arrays.size(); ++i)
        axis_sizes[i] = static_cast<size_t>(arrays[i].request().shape[axis]);

    for (size_t i = 0; i < arrays.size(); ++i) {
        auto buf = arrays[i].request();
        for (int d = 0; d < ndim; ++d) {
            if (d == axis) continue;
            if (buf.shape[d] != shape[d])
                throw std::invalid_argument(
                    "concatenate: shape mismatch on non-concat axis");
        }
    }

    std::vector<ptrdiff_t> out_shape = shape;
    ptrdiff_t total_axis = 0;
    for (auto s : axis_sizes) total_axis += static_cast<ptrdiff_t>(s);
    out_shape[axis] = total_axis;

    std::vector<py::ssize_t> py_out(out_shape.begin(), out_shape.end());
    py::array_t<T> result(py_out);
    std::vector<const T*> ptrs(arrays.size());
    for (size_t i = 0; i < arrays.size(); ++i)
        ptrs[i] = static_cast<const T*>(arrays[i].request().ptr);

    numpy::concatenate(ptrs.data(), static_cast<T*>(result.request().ptr),
                       shape.data(), ndim, axis, axis_sizes.data(), arrays.size());
    return result;
}

/// numpy.vstack(tup)
template<typename T>
py::array_t<T> vstack(const std::vector<py::array_t<T>>& arrays) {
    if (arrays.empty()) return py::array_t<T>{};
    if (arrays[0].request().ndim == 1) {
        auto buf0 = arrays[0].request();
        py::array_t<T> result({static_cast<py::ssize_t>(arrays.size()),
                                static_cast<py::ssize_t>(buf0.size)});
        T* dst = static_cast<T*>(result.request().ptr);
        for (size_t i = 0; i < arrays.size(); ++i) {
            auto buf = arrays[i].request();
            std::memcpy(dst + i * buf0.size, static_cast<const T*>(buf.ptr),
                        buf.size * sizeof(T));
        }
        return result;
    }
    return concatenate(arrays, 0);
}

/// numpy.hstack(tup)
template<typename T>
py::array_t<T> hstack(const std::vector<py::array_t<T>>& arrays) {
    if (arrays.empty()) return py::array_t<T>{};
    return concatenate(arrays, (arrays[0].request().ndim == 1) ? 0 : 1);
}

/// numpy.transpose(a)
template<typename T>
py::array_t<T> transpose(const py::array_t<T>& arr) {
    auto buf = arr.request();
    int ndim = static_cast<int>(buf.ndim);
    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());
    std::vector<int> axes(ndim);
    for (int i = 0; i < ndim; ++i) axes[i] = ndim - 1 - i;
    std::vector<py::ssize_t> out_shape(ndim);
    for (int i = 0; i < ndim; ++i) out_shape[i] = buf.shape[axes[i]];
    py::array_t<T> result(out_shape);
    if (ndim > 0)
        numpy::transpose(static_cast<const T*>(buf.ptr),
                         static_cast<T*>(result.request().ptr),
                         shape.data(), ndim, axes.data());
    return result;
}

/// ndarray.flatten()
template<typename T>
py::array_t<T> flatten(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result({buf.size});
    std::memcpy(static_cast<T*>(result.request().ptr),
                static_cast<const T*>(buf.ptr), buf.size * sizeof(T));
    return result;
}

/// numpy.squeeze(a)
template<typename T>
py::array_t<T> squeeze(const py::array_t<T>& arr) {
    auto buf = arr.request();
    std::vector<py::ssize_t> new_shape;
    for (auto s : buf.shape) if (s != 1) new_shape.push_back(s);
    if (new_shape.empty()) new_shape.push_back(1);
    py::array_t<T> result(new_shape);
    std::memcpy(result.request().ptr, buf.ptr, buf.size * sizeof(T));
    return result;
}

/// numpy.where(condition, x, y) — scalar
template<typename T>
py::array_t<T> where(const py::array_t<bool>& cond, T x, T y) {
    auto buf = cond.request();
    py::array_t<T> result(buf.shape);
    numpy::where_scalar(static_cast<const bool*>(buf.ptr),
                        static_cast<T*>(result.request().ptr), buf.size, x, y);
    return result;
}

/// numpy.where(condition, x, y) — array
template<typename T>
py::array_t<T> where(const py::array_t<bool>& cond,
                     const py::array_t<T>& x, const py::array_t<T>& y) {
    auto bc = cond.request(), bx = x.request(), by = y.request();
    py::array_t<T> result(bc.shape);
    numpy::where_array(static_cast<const bool*>(bc.ptr),
                       static_cast<T*>(result.request().ptr),
                       std::min({bc.size, bx.size, by.size}),
                       static_cast<const T*>(bx.ptr),
                       static_cast<const T*>(by.ptr));
    return result;
}

/// numpy.roll(a, shift)
template<typename T>
py::array_t<T> roll(const py::array_t<T>& arr, py::ssize_t shift) {
    auto buf = arr.request();
    if (buf.size == 0) return py::array_t<T>{};
    py::array_t<T> result(buf.shape);
    numpy::roll(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size, shift);
    return result;
}

/// numpy.flip(m)
template<typename T>
py::array_t<T> flip(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    numpy::flip(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

/// numpy.repeat(a, repeats)
template<typename T>
py::array_t<T> repeat(const py::array_t<T>& arr, py::ssize_t repeats) {
    auto buf = arr.request();
    py::array_t<T> result({buf.size * repeats});
    numpy::repeat(static_cast<const T*>(buf.ptr),
                  static_cast<T*>(result.request().ptr), buf.size, repeats);
    return result;
}

/// numpy.tile(A, reps)
template<typename T>
py::array_t<T> tile(const py::array_t<T>& arr, py::ssize_t reps) {
    auto buf = arr.request();
    py::array_t<T> result({buf.size * reps});
    numpy::tile(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size, reps);
    return result;
}

// ============================================================================
// Sorting / searching
// ============================================================================

/// numpy.argsort(a)
template<typename T>
py::array_t<py::ssize_t> argsort(const py::array_t<T>& arr) {
    auto buf = arr.request();
    std::vector<ptrdiff_t> idx(buf.size);
    numpy::argsort(static_cast<const T*>(buf.ptr), idx.data(), buf.size);
    return py::array_t<py::ssize_t>(buf.size, idx.data());
}

/// numpy.argmax(a)
template<typename T>
py::ssize_t argmax(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return static_cast<py::ssize_t>(
        numpy::argmax(static_cast<const T*>(buf.ptr), buf.size));
}

/// numpy.argmin(a)
template<typename T>
py::ssize_t argmin(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return static_cast<py::ssize_t>(
        numpy::argmin(static_cast<const T*>(buf.ptr), buf.size));
}

// ============================================================================
// Slice helpers — 1-D convenience overloads
// ============================================================================

/// ndarray[start:stop] — typed
template<typename T>
py::array_t<T> slice(const py::array_t<T>& arr,
                     py::ssize_t start, py::ssize_t stop) {
    auto buf = arr.request();
    if (buf.ndim < 1) return py::array_t<T>{};
    py::ssize_t dim0 = buf.shape[0];
    if (start < 0) start = 0;
    if (stop > dim0) stop = dim0;
    if (start >= stop) return py::array_t<T>{};
    py::ssize_t n = stop - start, trailing = 1;
    for (int i = 1; i < buf.ndim; ++i) trailing *= buf.shape[i];
    std::vector<py::ssize_t> new_shape = {n};
    for (int i = 1; i < buf.ndim; ++i) new_shape.push_back(buf.shape[i]);
    py::array_t<T> result(new_shape);
    std::memcpy(static_cast<T*>(result.request().ptr),
                static_cast<const T*>(buf.ptr) + start * trailing,
                n * trailing * sizeof(T));
    return result;
}

/// ndarray[start:stop] — generic (preserves dtype)
inline py::array slice(const py::array& arr,
                       py::ssize_t start, py::ssize_t stop) {
    auto buf = arr.request();
    if (buf.ndim < 1) return py::array{};
    py::ssize_t dim0 = buf.shape[0];
    if (start < 0) start = 0;
    if (stop > dim0) stop = dim0;
    if (start >= stop) return py::array{};
    py::ssize_t n = stop - start, itemsize = buf.itemsize, trailing = 1;
    for (int i = 1; i < buf.ndim; ++i) trailing *= buf.shape[i];
    std::vector<py::ssize_t> new_shape = {n};
    for (int i = 1; i < buf.ndim; ++i) new_shape.push_back(buf.shape[i]);
    py::array result(arr.dtype(), new_shape);
    std::memcpy(static_cast<char*>(result.request().ptr),
                static_cast<const char*>(buf.ptr) + start * trailing * itemsize,
                n * trailing * itemsize);
    return result;
}

// ── slice() N-D overload ──────────────────────────────────────────────────────

/// slice(a, starts, stops, steps) — N-D with per-dimension step
template<typename T>
py::array_t<T> slice(const py::array_t<T>& arr,
                     const std::vector<py::ssize_t>& starts,
                     const std::vector<py::ssize_t>& stops,
                     const std::vector<py::ssize_t>& steps) {
    auto buf = arr.request();
    int ndim = static_cast<int>(buf.ndim);
    if (static_cast<int>(starts.size()) != ndim ||
        static_cast<int>(stops.size())  != ndim ||
        static_cast<int>(steps.size())  != ndim)
        throw std::invalid_argument("slice: starts/stops/steps must match ndim");

    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());
    std::vector<ptrdiff_t> st(starts.begin(), starts.end());
    std::vector<ptrdiff_t> sp(stops.begin(),  stops.end());
    std::vector<ptrdiff_t> sv(steps.begin(),  steps.end());

    std::vector<py::ssize_t> out_shape(ndim);
    for (int d = 0; d < ndim; ++d) {
        ptrdiff_t len = numpy::slice_len(st[d], sp[d], sv[d]);
        if (len <= 0)
            return py::array_t<T>(std::vector<py::ssize_t>(ndim, 0));
        out_shape[d] = static_cast<py::ssize_t>(len);
    }

    py::array_t<T> result(out_shape);
    numpy::slice(static_cast<const T*>(buf.ptr),
                 static_cast<T*>(result.request().ptr),
                 shape.data(), ndim, st.data(), sp.data(), sv.data());
    return result;
}

// ── slice_assign convenience overload (1-D, scalar) ──────────────────────────

/// a[start:] = value (in-place)
template<typename T>
void slice_assign(py::array_t<T> arr, py::ssize_t start, T value) {
    auto buf = arr.request();
    if (buf.ndim < 1 || start >= buf.shape[0]) return;
    if (start < 0) start = 0;
    numpy::slice_assign(static_cast<T*>(buf.ptr), buf.size,
                        static_cast<size_t>(start), value);
}

inline void slice_assign(py::array_t<int>  arr, py::ssize_t start, int  value) {
    auto buf = arr.request();
    if (buf.ndim < 1 || start >= buf.shape[0]) return;
    if (start < 0) start = 0;
    std::fill(static_cast<int*>(buf.ptr) + start,
              static_cast<int*>(buf.ptr) + buf.size, value);
}

inline void slice_assign(py::array_t<bool> arr, py::ssize_t start, bool value) {
    auto buf = arr.request();
    if (buf.ndim < 1 || start >= buf.shape[0]) return;
    if (start < 0) start = 0;
    std::fill(static_cast<bool*>(buf.ptr) + start,
              static_cast<bool*>(buf.ptr) + buf.size, value);
}

// ── slice_assign N-D overloads ────────────────────────────────────────────────

/// a[s0:e0:k0, ...] = scalar  (in-place)
template<typename T>
void slice_assign(py::array_t<T> arr,
                  const std::vector<py::ssize_t>& starts,
                  const std::vector<py::ssize_t>& stops,
                  const std::vector<py::ssize_t>& steps,
                  T value) {
    auto buf = arr.request();
    int ndim = static_cast<int>(buf.ndim);
    if (static_cast<int>(starts.size()) != ndim ||
        static_cast<int>(stops.size())  != ndim ||
        static_cast<int>(steps.size())  != ndim)
        throw std::invalid_argument("slice_assign: starts/stops/steps must match ndim");
    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());
    std::vector<ptrdiff_t> st(starts.begin(), starts.end());
    std::vector<ptrdiff_t> sp(stops.begin(),  stops.end());
    std::vector<ptrdiff_t> sv(steps.begin(),  steps.end());
    numpy::slice_assign(static_cast<T*>(buf.ptr),
                        shape.data(), ndim, st.data(), sp.data(), sv.data(), value);
}

/// a[s0:e0:k0, ...] = values_array  (in-place)
template<typename T>
void slice_assign(py::array_t<T> arr,
                  const std::vector<py::ssize_t>& starts,
                  const std::vector<py::ssize_t>& stops,
                  const std::vector<py::ssize_t>& steps,
                  const py::array_t<T>& values) {
    auto buf = arr.request();
    int ndim = static_cast<int>(buf.ndim);
    if (static_cast<int>(starts.size()) != ndim ||
        static_cast<int>(stops.size())  != ndim ||
        static_cast<int>(steps.size())  != ndim)
        throw std::invalid_argument("slice_assign: starts/stops/steps must match ndim");
    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());
    std::vector<ptrdiff_t> st(starts.begin(), starts.end());
    std::vector<ptrdiff_t> sp(stops.begin(),  stops.end());
    std::vector<ptrdiff_t> sv(steps.begin(),  steps.end());
    auto vbuf = values.request();
    numpy::slice_assign(static_cast<T*>(buf.ptr),
                        shape.data(), ndim, st.data(), sp.data(), sv.data(),
                        static_cast<const T*>(vbuf.ptr));
}

// ── take_cols ─────────────────────────────────────────────────────────────────

/// numpy.take — first N columns of a 2-D array (legacy helper)
template<typename T>
py::array_t<T> take_cols(const py::array_t<T>& arr, py::ssize_t n) {
    auto buf = arr.request();
    if (buf.ndim != 2 || n > buf.shape[1])
        return arr.attr("copy")().template cast<py::array_t<T>>();
    py::ssize_t rows = buf.shape[0], src_cols = buf.shape[1];
    py::array_t<T> result({rows, n});
    numpy::take_cols(static_cast<const T*>(buf.ptr),
                     static_cast<T*>(result.request().ptr), rows, src_cols, n);
    return result;
}

// ── numpy.take (full) ────────────────────────────────────────────────────────

/// numpy.take(a, indices, axis=None)
template<typename T>
py::array_t<T> take(const py::array_t<T>& arr,
                    const py::array_t<py::ssize_t>& indices,
                    int axis = -1) {
    auto buf  = arr.request();
    auto ibuf = indices.request();
    int ndim = static_cast<int>(buf.ndim);
    if (ndim == 0) return py::array_t<T>{};
    if (axis < -1 || axis >= ndim)
        throw std::invalid_argument("take: axis out of range");

    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());
    size_t ni = static_cast<size_t>(ibuf.size);

    std::vector<py::ssize_t> out_shape;
    if (axis == -1) {
        out_shape.push_back(static_cast<py::ssize_t>(ni));
    } else {
        for (int d = 0; d < ndim; ++d)
            out_shape.push_back(d == axis ? static_cast<py::ssize_t>(ni) : buf.shape[d]);
    }

    py::array_t<T> result(out_shape);
    numpy::take(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr),
                shape.data(), ndim, axis,
                static_cast<const ptrdiff_t*>(ibuf.ptr), ni);
    return result;
}

// ── numpy.compress ────────────────────────────────────────────────────────────

/// numpy.compress(condition, a)
template<typename T>
py::array_t<T> compress(const py::array_t<T>& arr,
                         const py::array_t<bool>& mask) {
    auto buf  = arr.request();
    auto mbuf = mask.request();
    size_t use = static_cast<size_t>(std::min(buf.size, mbuf.size));
    const bool* m = static_cast<const bool*>(mbuf.ptr);
    size_t cnt = 0;
    for (size_t i = 0; i < use; ++i) if (m[i]) ++cnt;
    py::array_t<T> result({static_cast<py::ssize_t>(cnt)});
    if (cnt > 0)
        numpy::compress(static_cast<const T*>(buf.ptr),
                        static_cast<T*>(result.request().ptr), m, use);
    return result;
}

// ── numpy.put ─────────────────────────────────────────────────────────────────

/// numpy.put(a, indices, values)
template<typename T>
void put(py::array_t<T> arr,
         const py::array_t<py::ssize_t>& indices,
         const py::array_t<T>& values) {
    auto buf  = arr.request();
    auto ibuf = indices.request();
    auto vbuf = values.request();
    numpy::put(static_cast<T*>(buf.ptr),
               static_cast<size_t>(buf.size),
               static_cast<const ptrdiff_t*>(ibuf.ptr),
               static_cast<const T*>(vbuf.ptr),
               static_cast<size_t>(std::min(ibuf.size, vbuf.size)));
}

// ── numpy.putmask ─────────────────────────────────────────────────────────────

/// numpy.putmask(a, mask, scalar)
template<typename T>
void putmask(py::array_t<T> arr,
             const py::array_t<bool>& mask, T value) {
    auto buf  = arr.request();
    auto mbuf = mask.request();
    numpy::putmask(static_cast<T*>(buf.ptr),
                   static_cast<const bool*>(mbuf.ptr),
                   static_cast<size_t>(std::min(buf.size, mbuf.size)),
                   value);
}

/// numpy.putmask(a, mask, values_array)
template<typename T>
void putmask(py::array_t<T> arr,
             const py::array_t<bool>& mask,
             const py::array_t<T>& values) {
    auto buf  = arr.request();
    auto mbuf = mask.request();
    auto vbuf = values.request();
    numpy::putmask(static_cast<T*>(buf.ptr),
                   static_cast<const bool*>(mbuf.ptr),
                   static_cast<size_t>(std::min(buf.size, mbuf.size)),
                   static_cast<const T*>(vbuf.ptr));
}

} // namespace numpy
