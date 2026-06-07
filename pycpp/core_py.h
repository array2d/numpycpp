// Pybind11 wrappers for core.h native functions.
// Thin layer: extract pointers from py::array_t, call *, wrap results.
//
// Each wrapper matches a Python numpy API; native dispatch is handled
// by the functions in numpy/core.h (q.v. for individual numpy annotations).

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../numpy/numpy.h"
#include <vector>
#include <cstring>
#include <cstdint>

namespace py = pybind11;

namespace numpy {

// ============================================================================
// Array creation — numpy.zeros_like, numpy.ones_like, numpy.full_like, etc.
// ============================================================================

/// numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
template<typename T>
py::array_t<T> zeros_like(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    zeros_like(static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

/// numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)
template<typename T>
py::array_t<T> ones_like(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    ones_like(static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

/// numpy.full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None)
template<typename T>
py::array_t<T> full_like(const py::array_t<T>& arr, T fill_value) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    numpy::full(static_cast<T*>(result.request().ptr), buf.size, fill_value);
    return result;
}

/// numpy.empty_like(prototype, dtype=None, order='K', subok=True, shape=None)
template<typename T>
py::array_t<T> empty_like(const py::array_t<T>& arr) {
    return py::array_t<T>(arr.request().shape);
}

/// numpy.zeros(shape, dtype=float, order='C', *, like=None)
inline py::array_t<double> zeros(const std::vector<py::ssize_t>& shape) {
    py::array_t<double> result(shape);
    zeros_like(static_cast<double*>(result.request().ptr), result.request().size);
    return result;
}

/// numpy.ones(shape, dtype=float, order='C', *, like=None)
inline py::array_t<double> ones(const std::vector<py::ssize_t>& shape) {
    py::array_t<double> result(shape);
    ones_like(static_cast<double*>(result.request().ptr), result.request().size);
    return result;
}

/// numpy.full(shape, fill_value, dtype=float, order='C')
inline py::array_t<double> full(const std::vector<py::ssize_t>& shape, double fill_value) {
    py::array_t<double> result(shape);
    numpy::full(static_cast<double*>(result.request().ptr), result.request().size, fill_value);
    return result;
}

// Bool specializations — return-type disambiguation via dtype parameter
// where pybind11 cannot distinguish overloads by return type alone.
//
// full_like(arr, bool_val): bool fill_value naturally disambiguates from
// the template full_like(arr, T val) where T=double/float.
// zeros_like(arr, dtype_str) / ones_like(arr, dtype_str): string dtype
// parameter mirrors numpy's dtype= kwarg for disambiguation.
inline py::array_t<bool> full_like(const py::array_t<double>& arr, bool fill_value) {
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    std::fill_n(static_cast<bool*>(result.request().ptr), buf.size, fill_value);
    return result;
}

inline py::array zeros_like(const py::array& arr, const std::string& dtype) {
    auto buf = arr.request();
    if (dtype == "bool") {
        py::array_t<bool> result(buf.shape);
        std::fill_n(static_cast<bool*>(result.request().ptr), buf.size, false);
        return result;
    }
    throw std::runtime_error("unsupported dtype: " + dtype);
}

inline py::array ones_like(const py::array& arr, const std::string& dtype) {
    auto buf = arr.request();
    if (dtype == "bool") {
        py::array_t<bool> result(buf.shape);
        std::fill_n(static_cast<bool*>(result.request().ptr), buf.size, true);
        return result;
    }
    throw std::runtime_error("unsupported dtype: " + dtype);
}

// ============================================================================
// astype — ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
// ============================================================================

/// ndarray.astype(dtype) — unified dtype dispatch
inline py::array astype(const py::array& arr, const std::string& dtype) {
    auto buf = arr.request();
    auto dt = arr.dtype();

    // float64 input
    if (dt.is(py::dtype::of<double>())) {
        auto* src = static_cast<const double*>(buf.ptr);
        if (dtype == "float32" || dtype == "float") {
            py::array_t<float> r(buf.shape);
            astype<float, double>(src, static_cast<float*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "int" || dtype == "int32") {
            py::array_t<int> r(buf.shape);
            astype<int, double>(src, static_cast<int*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "int64") {
            py::array_t<int64_t> r(buf.shape);
            astype<int64_t, double>(src, static_cast<int64_t*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "bool") {
            py::array_t<bool> r(buf.shape);
            astype<bool, double>(src, static_cast<bool*>(r.request().ptr), buf.size);
            return r;
        }
    }

    // float32 input
    if (dt.is(py::dtype::of<float>())) {
        auto* src = static_cast<const float*>(buf.ptr);
        if (dtype == "float64" || dtype == "double") {
            py::array_t<double> r(buf.shape);
            astype<double, float>(src, static_cast<double*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "int" || dtype == "int32") {
            py::array_t<int> r(buf.shape);
            astype<int, float>(src, static_cast<int*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "int64") {
            py::array_t<int64_t> r(buf.shape);
            astype<int64_t, float>(src, static_cast<int64_t*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "bool") {
            py::array_t<bool> r(buf.shape);
            astype<bool, float>(src, static_cast<bool*>(r.request().ptr), buf.size);
            return r;
        }
    }

    // int32 input
    if (dt.is(py::dtype::of<int>())) {
        auto* src = static_cast<const int*>(buf.ptr);
        if (dtype == "float64" || dtype == "double") {
            py::array_t<double> r(buf.shape);
            astype<double, int>(src, static_cast<double*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "float32" || dtype == "float") {
            py::array_t<float> r(buf.shape);
            astype<float, int>(src, static_cast<float*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "int64") {
            py::array_t<int64_t> r(buf.shape);
            astype<int64_t, int>(src, static_cast<int64_t*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "bool") {
            py::array_t<bool> r(buf.shape);
            astype<bool, int>(src, static_cast<bool*>(r.request().ptr), buf.size);
            return r;
        }
    }

    // int64 input
    if (dt.is(py::dtype::of<int64_t>())) {
        auto* src = static_cast<const int64_t*>(buf.ptr);
        if (dtype == "float64" || dtype == "double") {
            py::array_t<double> r(buf.shape);
            astype<double, int64_t>(src, static_cast<double*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "float32" || dtype == "float") {
            py::array_t<float> r(buf.shape);
            astype<float, int64_t>(src, static_cast<float*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "int" || dtype == "int32") {
            py::array_t<int> r(buf.shape);
            astype<int, int64_t>(src, static_cast<int*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "bool") {
            py::array_t<bool> r(buf.shape);
            astype<bool, int64_t>(src, static_cast<bool*>(r.request().ptr), buf.size);
            return r;
        }
    }

    // bool input
    if (dt.is(py::dtype::of<bool>())) {
        auto* src = static_cast<const bool*>(buf.ptr);
        if (dtype == "float64" || dtype == "double") {
            py::array_t<double> r(buf.shape);
            astype<double, bool>(src, static_cast<double*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "float32" || dtype == "float") {
            py::array_t<float> r(buf.shape);
            astype<float, bool>(src, static_cast<float*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "int" || dtype == "int32") {
            py::array_t<int> r(buf.shape);
            astype<int, bool>(src, static_cast<int*>(r.request().ptr), buf.size);
            return r;
        }
        if (dtype == "int64") {
            py::array_t<int64_t> r(buf.shape);
            astype<int64_t, bool>(src, static_cast<int64_t*>(r.request().ptr), buf.size);
            return r;
        }
    }

    throw std::runtime_error("astype: unsupported conversion " + std::string(py::str(dt)) + " -> " + dtype);
}

/// float64 → float32 → float64 roundtrip (precision testing helper)
inline py::array_t<double> truncate_to_float32(const py::array_t<double>& arr) {
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    truncate_to_float32(static_cast<const double*>(buf.ptr),
                               static_cast<double*>(result.mutable_data()), buf.size);
    return result;
}

// ============================================================================
// Element-wise math — numpy.sqrt, abs, exp, log, sin, cos, tan, ...
// ============================================================================
#define DEF_ELEMWISE(name) \
template<typename T> \
py::array_t<T> name(const py::array_t<T>& arr) { \
    auto buf = arr.request(); \
    py::array_t<T> result(buf.shape); \
    name(static_cast<const T*>(buf.ptr), \
               static_cast<T*>(result.request().ptr), buf.size); \
    return result; \
}

/// numpy.sqrt(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(sqrt)
/// numpy.abs(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(abs)
/// numpy.exp(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(exp)
/// numpy.log(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(log)
/// numpy.sin(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(sin)
/// numpy.cos(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(cos)
/// numpy.tan(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(tan)
/// numpy.cbrt(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(cbrt)
/// numpy.expm1(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(expm1)
/// numpy.log1p(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(log1p)
/// numpy.log10(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(log10)
/// numpy.log2(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(log2)
/// numpy.arcsin(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(arcsin)
/// numpy.arccos(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(arccos)
/// numpy.arctan(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(arctan)
/// numpy.round(a, decimals=0, out=None)
DEF_ELEMWISE(round)
/// numpy.floor(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(floor)
/// numpy.ceil(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(ceil)
/// numpy.degrees(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(degrees)
/// numpy.radians(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(radians)
/// numpy.sign(x, /, out=None, *, where=True, ...)
DEF_ELEMWISE(sign)

#undef DEF_ELEMWISE

/// numpy.power(x1, x2, /, out=None, *, where=True, ...) — scalar exponent only
template<typename T>
py::array_t<T> power(const py::array_t<T>& arr, T exponent) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    power(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size, exponent);
    return result;
}

/// numpy.clip(a, a_min, a_max, out=None, **kwargs)
template<typename T>
py::array_t<T> clip(const py::array_t<T>& arr, T min_val, T max_val) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    clip(static_cast<const T*>(buf.ptr),
               static_cast<T*>(result.request().ptr), buf.size, min_val, max_val);
    return result;
}

// ============================================================================
// Reduction — numpy.sum, mean, max, min, any, all, std, var
// ============================================================================

/// numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False, ...)
template<typename T>
T sum(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return sum(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.mean(a, axis=None, dtype=None, out=None, keepdims=False, *)
template<typename T>
T mean(const py::array_t<T>& arr) {
    auto buf = arr.request();
    if (buf.size == 0) return T(0);
    return mean(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.max(a, axis=None, out=None, keepdims=False, initial=..., where=...)
template<typename T>
T max(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return max(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.min(a, axis=None, out=None, keepdims=False, initial=..., where=...)
template<typename T>
T min(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return min(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.any(a, axis=None, out=None, keepdims=False, *)
inline bool any(const py::array_t<bool>& arr) {
    auto buf = arr.request();
    return any(static_cast<const bool*>(buf.ptr), buf.size);
}

/// numpy.all(a, axis=None, out=None, keepdims=False, *)
inline bool all(const py::array_t<bool>& arr) {
    auto buf = arr.request();
    return all(static_cast<const bool*>(buf.ptr), buf.size);
}

/// numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *)
template<typename T>
T std(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return stddev(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *)
template<typename T>
T var(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return var(static_cast<const T*>(buf.ptr), buf.size);
}

// ============================================================================
// Comparison — numpy.greater, less, equal, greater_equal, less_equal, not_equal
// ============================================================================
#define DEF_COMPARE(name) \
template<typename T> \
py::array_t<bool> name(const py::array_t<T>& a, T b) { \
    auto buf = a.request(); \
    py::array_t<bool> result(buf.shape); \
    name(static_cast<const T*>(buf.ptr), \
               static_cast<bool*>(result.request().ptr), buf.size, b); \
    return result; \
}

/// numpy.greater(x1, x2, /, out=None, *, where=True, ...)
DEF_COMPARE(greater)
/// numpy.less(x1, x2, /, out=None, *, where=True, ...)
DEF_COMPARE(less)
/// numpy.equal(x1, x2, /, out=None, *, where=True, ...)
DEF_COMPARE(equal)
/// numpy.greater_equal(x1, x2, /, out=None, *, where=True, ...)
DEF_COMPARE(greater_equal)
/// numpy.less_equal(x1, x2, /, out=None, *, where=True, ...)
DEF_COMPARE(less_equal)

#undef DEF_COMPARE

/// numpy.not_equal(x1, x2, /, out=None, *, where=True, ...) — scalar
template<typename T>
py::array_t<bool> not_equal(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    not_equal_scalar(static_cast<const T*>(buf.ptr),
                            static_cast<bool*>(result.request().ptr), buf.size, b);
    return result;
}

/// numpy.not_equal(x1, x2, /, out=None, *, where=True, ...) — array
template<typename T>
py::array_t<bool> not_equal(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<bool> result(ba.shape);
    not_equal_array(static_cast<const T*>(ba.ptr),
                           static_cast<const T*>(bb.ptr),
                           static_cast<bool*>(result.request().ptr),
                           std::min(ba.size, bb.size));
    return result;
}

// ============================================================================
// Logical — numpy.logical_and, logical_or, logical_not, logical_xor
// ============================================================================

/// numpy.logical_and(x1, x2, /, out=None, *, where=True, ...)
inline py::array_t<bool> logical_and(const py::array_t<bool>& a, const py::array_t<bool>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<bool> result(ba.shape);
    logical_and(static_cast<const bool*>(ba.ptr),
                       static_cast<const bool*>(bb.ptr),
                       static_cast<bool*>(result.request().ptr),
                       std::min(ba.size, bb.size));
    return result;
}

/// numpy.logical_or(x1, x2, /, out=None, *, where=True, ...)
inline py::array_t<bool> logical_or(const py::array_t<bool>& a, const py::array_t<bool>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<bool> result(ba.shape);
    logical_or(static_cast<const bool*>(ba.ptr),
                      static_cast<const bool*>(bb.ptr),
                      static_cast<bool*>(result.request().ptr),
                      std::min(ba.size, bb.size));
    return result;
}

/// numpy.logical_not(x, /, out=None, *, where=True, ...)
inline py::array_t<bool> logical_not(const py::array_t<bool>& a) {
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    logical_not(static_cast<const bool*>(buf.ptr),
                       static_cast<bool*>(result.request().ptr), buf.size);
    return result;
}

/// numpy.logical_xor(x1, x2, /, out=None, *, where=True, ...)
inline py::array_t<bool> logical_xor(const py::array_t<bool>& a, const py::array_t<bool>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<bool> result(ba.shape);
    logical_xor(static_cast<const bool*>(ba.ptr),
                       static_cast<const bool*>(bb.ptr),
                       static_cast<bool*>(result.request().ptr),
                       std::min(ba.size, bb.size));
    return result;
}

// ============================================================================
// Special value helpers — numpy.isnan, isinf, isfinite
// ============================================================================
#define DEF_SPECIAL(name) \
template<typename T> \
py::array_t<bool> name(const py::array_t<T>& arr) { \
    auto buf = arr.request(); \
    py::array_t<bool> result(buf.shape); \
    name(static_cast<const T*>(buf.ptr), \
               static_cast<bool*>(result.request().ptr), buf.size); \
    return result; \
}

/// numpy.isnan(x, /, out=None, *, where=True, ...)
DEF_SPECIAL(isnan)
/// numpy.isinf(x, /, out=None, *, where=True, ...)
DEF_SPECIAL(isinf)
/// numpy.isfinite(x, /, out=None, *, where=True, ...)
DEF_SPECIAL(isfinite)

#undef DEF_SPECIAL

// ============================================================================
// Array access — ndarray indexing
// ============================================================================

/// ndarray[idx] — 1D get
inline double array_get(const py::array_t<double>& arr, py::ssize_t idx) {
    auto buf = arr.request();
    if (idx < 0) idx = buf.size + idx;
    return (idx >= 0 && idx < buf.size)
        ? static_cast<const double*>(buf.ptr)[idx] : 0.0;
}

/// ndarray[i, j] — 2D get
inline double array_get(const py::array_t<double>& arr, py::ssize_t i, py::ssize_t j) {
    auto buf = arr.request();
    if (buf.ndim != 2) return 0.0;
    py::ssize_t cols = buf.shape[1];
    if (i < 0) i = buf.shape[0] + i;
    if (j < 0) j = cols + j;
    return (i >= 0 && i < buf.shape[0] && j >= 0 && j < cols)
        ? static_cast<const double*>(buf.ptr)[i * cols + j] : 0.0;
}

inline bool array_get(const py::array_t<bool>& arr, py::ssize_t idx) {
    auto buf = arr.request();
    if (idx < 0) idx = buf.size + idx;
    return (idx >= 0 && idx < buf.size)
        ? static_cast<const bool*>(buf.ptr)[idx] : false;
}

inline double array_get(const py::array& arr, py::ssize_t idx) {
    return arr[py::int_(idx)].cast<double>();
}

// ============================================================================
// Dict helpers
// ============================================================================
inline py::array_t<double> get_array(const py::dict& d, const char* key) {
    return d[key].cast<py::array_t<double>>();
}
inline void set_array(py::dict& d, const char* key, const py::array_t<double>& arr) {
    d[key] = arr;
}

// ============================================================================
// Conversion — numpy.asarray, numpy.array
// ============================================================================

/// numpy.asarray(a, dtype=None, order=None, *, like=None)
inline py::array_t<double> asarray(const std::vector<double>& vec) {
    return py::array_t<double>(vec.size(), vec.data());
}
inline py::array_t<double> asarray(const py::array_t<double>& arr) { return arr; }

/// numpy.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)
inline py::array_t<double> array(const std::vector<double>& vec) {
    return py::array_t<double>(vec.size(), vec.data());
}
inline py::array_t<double> array(const py::array_t<double>& arr) { return arr; }

// ============================================================================
// numpy.transpose / ndarray.flatten
// ============================================================================

/// numpy.transpose(a, axes=None) — N-D, defaults to reversing all axes
template<typename T>
py::array_t<T> transpose(const py::array_t<T>& arr) {
    auto buf = arr.request();
    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());
    int ndim = static_cast<int>(buf.ndim);

    // Build default permutation: reverse all axes
    std::vector<int> axes(ndim);
    for (int i = 0; i < ndim; ++i) axes[i] = ndim - 1 - i;

    // Output shape
    std::vector<py::ssize_t> out_shape(ndim);
    for (int i = 0; i < ndim; ++i) out_shape[i] = buf.shape[axes[i]];

    py::array_t<T> result(out_shape);
    if (ndim > 0) {
        transpose(static_cast<const T*>(buf.ptr),
                         static_cast<T*>(result.request().ptr),
                         shape.data(), ndim, axes.data());
    }
    return result;
}

/// ndarray.flatten(order='C')
template<typename T>
py::array_t<T> flatten(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result({buf.size});
    std::memcpy(static_cast<T*>(result.request().ptr),
                static_cast<const T*>(buf.ptr), buf.size * sizeof(T));
    return result;
}

// ============================================================================
// numpy.mean / ndarray.mean with axis
// ============================================================================

/// numpy.mean(a, axis=..., dtype=None, out=None, keepdims=False, *) — N-D
template<typename T>
py::array_t<T> mean_axis(const py::array_t<T>& arr, int axis) {
    auto buf = arr.request();
    if (buf.ndim < 1) throw std::invalid_argument("Array must have at least 1 dimension");

    int ax = (axis == -1) ? static_cast<int>(buf.ndim - 1) : axis;

    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());

    // Output shape: drop the reduced axis
    std::vector<ptrdiff_t> out_shape;
    for (int d = 0; d < buf.ndim; ++d)
        if (d != ax) out_shape.push_back(shape[d]);
    if (out_shape.empty()) out_shape.push_back(1);

    std::vector<py::ssize_t> py_out_shape(out_shape.begin(), out_shape.end());
    py::array_t<T> result(py_out_shape);
    mean_axis(static_cast<const T*>(buf.ptr),
                     static_cast<T*>(result.request().ptr),
                     shape.data(), static_cast<int>(buf.ndim), ax);
    return result;
}

/// numpy.mean(a, axis=..., ...) — float64
inline py::array_t<double> mean(const py::array_t<double>& arr, int axis) { return mean_axis(arr, axis); }
/// numpy.mean(a, axis=..., ...) — float32
inline py::array_t<float> mean(const py::array_t<float>& arr, int axis)  { return mean_axis(arr, axis); }

// ============================================================================
// to_vector — utility
// ============================================================================
inline std::vector<bool> to_vector(const py::array_t<bool>& arr) {
    auto buf = arr.request();
    const bool* p = static_cast<const bool*>(buf.ptr);
    return std::vector<bool>(p, p + buf.size);
}

template<typename T>
std::vector<T> to_vector(const py::array_t<T>& arr) {
    auto buf = arr.request();
    const T* p = static_cast<const T*>(buf.ptr);
    return std::vector<T>(p, p + buf.size);
}

// ============================================================================
// Slice helpers — ndarray[start:stop]
// ============================================================================

/// ndarray[start:stop] — typed
template<typename T>
py::array_t<T> slice(const py::array_t<T>& arr, py::ssize_t start, py::ssize_t stop) {
    auto buf = arr.request();
    if (buf.ndim < 1) return py::array_t<T>{};

    py::ssize_t dim0 = buf.shape[0];
    if (start < 0) start = 0;
    if (stop > dim0) stop = dim0;
    if (start >= stop) return py::array_t<T>{};

    py::ssize_t n = stop - start;
    py::ssize_t trailing = 1;
    for (int i = 1; i < buf.ndim; ++i) trailing *= buf.shape[i];

    std::vector<py::ssize_t> new_shape = {n};
    for (int i = 1; i < buf.ndim; ++i) new_shape.push_back(buf.shape[i]);

    py::array_t<T> result(new_shape);
    const T* src = static_cast<const T*>(buf.ptr);
    T* dst = static_cast<T*>(result.request().ptr);
    std::memcpy(dst, src + start * trailing, n * trailing * sizeof(T));
    return result;
}

/// ndarray[start:stop] — generic (preserves dtype)
inline py::array slice(const py::array& arr, py::ssize_t start, py::ssize_t stop) {
    auto buf = arr.request();
    if (buf.ndim < 1) return py::array{};
    py::ssize_t dim0 = buf.shape[0];
    if (start < 0) start = 0;
    if (stop > dim0) stop = dim0;
    if (start >= stop) return py::array{};

    py::ssize_t n = stop - start, itemsize = buf.itemsize;
    py::ssize_t trailing = 1;
    for (int i = 1; i < buf.ndim; ++i) trailing *= buf.shape[i];

    std::vector<py::ssize_t> new_shape = {n};
    for (int i = 1; i < buf.ndim; ++i) new_shape.push_back(buf.shape[i]);

    py::array result(arr.dtype(), new_shape);
    const char* src = static_cast<const char*>(buf.ptr);
    char* dst = static_cast<char*>(result.request().ptr);
    std::memcpy(dst, src + start * trailing * itemsize, n * trailing * itemsize);
    return result;
}

// ============================================================================
// numpy.take(a, indices, axis=1) — extract first N columns
// ============================================================================

/// numpy.take(a, indices, axis=1) — specialized subset: first N columns, 2D only.
//  NOTE: named take_cols, not take — numpy.take supports arbitrary indices and
//  arbitrary axis; this is a narrow helper with a simpler (arr, n) signature.
template<typename T>
py::array_t<T> take_cols(const py::array_t<T>& arr, py::ssize_t n) {
    auto buf = arr.request();
    if (buf.ndim != 2 || n > buf.shape[1])
        return arr.attr("copy")().template cast<py::array_t<T>>();

    py::ssize_t rows = buf.shape[0], src_cols = buf.shape[1];
    py::array_t<T> result({rows, n});
    take_cols(static_cast<const T*>(buf.ptr),
                     static_cast<T*>(result.request().ptr),
                     rows, src_cols, n);
    return result;
}

// ============================================================================
// Slice assignment — a[start:] = value
// ============================================================================

/// a[start:] = value (in-place)
template<typename T>
void slice_assign(py::array_t<T> arr, py::ssize_t start, T value) {
    auto buf = arr.request();
    if (buf.ndim < 1 || start >= buf.shape[0]) return;
    if (start < 0) start = 0;
    T* ptr = static_cast<T*>(buf.ptr);
    slice_assign(ptr, buf.size, static_cast<size_t>(start), value);
}

inline void slice_assign(py::array_t<int> arr, py::ssize_t start, int value) {
    auto buf = arr.request();
    if (buf.ndim < 1 || start >= buf.shape[0]) return;
    if (start < 0) start = 0;
    int* ptr = static_cast<int*>(buf.ptr);
    std::fill(ptr + start, ptr + buf.size, value);
}

inline void slice_assign(py::array_t<bool> arr, py::ssize_t start, bool value) {
    auto buf = arr.request();
    if (buf.ndim < 1 || start >= buf.shape[0]) return;
    if (start < 0) start = 0;
    bool* ptr = static_cast<bool*>(buf.ptr);
    std::fill(ptr + start, ptr + buf.size, value);
}

// ============================================================================
// Binary element-wise — numpy.arctan2, maximum, minimum
// ============================================================================

/// numpy.hypot(x1, x2, /, out=None, *, where=True, ...) — array-array
template<typename T>
py::array_t<T> hypot(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<T> result(ba.shape);
    hypot(static_cast<const T*>(ba.ptr),
                        static_cast<const T*>(bb.ptr),
                        static_cast<T*>(result.request().ptr),
                        std::min(ba.size, bb.size));
    return result;
}

/// numpy.arctan2(x1, x2, /, out=None, *, where=True, ...) — array-array
template<typename T>
py::array_t<T> arctan2(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<T> result(ba.shape);
    arctan2(static_cast<const T*>(ba.ptr),
                          static_cast<const T*>(bb.ptr),
                          static_cast<T*>(result.request().ptr),
                          std::min(ba.size, bb.size));
    return result;
}

/// numpy.arctan2(x1, x2, /, out=None, *, where=True, ...) — array-scalar
template<typename T>
py::array_t<T> arctan2(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<T> result(buf.shape);
    arctan2(static_cast<const T*>(buf.ptr),
                           static_cast<T*>(result.request().ptr), buf.size, b);
    return result;
}

/// numpy.maximum(x1, x2, /, out=None, *, where=True, ...) — array-array
template<typename T>
py::array_t<T> maximum(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<T> result(ba.shape);
    maximum(static_cast<const T*>(ba.ptr),
                          static_cast<const T*>(bb.ptr),
                          static_cast<T*>(result.request().ptr),
                          std::min(ba.size, bb.size));
    return result;
}

/// numpy.maximum(x1, x2, /, out=None, *, where=True, ...) — scalar
template<typename T>
py::array_t<T> maximum(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<T> result(buf.shape);
    maximum(static_cast<const T*>(buf.ptr),
                           static_cast<T*>(result.request().ptr), buf.size, b);
    return result;
}

/// numpy.minimum(x1, x2, /, out=None, *, where=True, ...) — array-array
template<typename T>
py::array_t<T> minimum(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<T> result(ba.shape);
    minimum(static_cast<const T*>(ba.ptr),
                          static_cast<const T*>(bb.ptr),
                          static_cast<T*>(result.request().ptr),
                          std::min(ba.size, bb.size));
    return result;
}

/// numpy.minimum(x1, x2, /, out=None, *, where=True, ...) — scalar
template<typename T>
py::array_t<T> minimum(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<T> result(buf.shape);
    minimum(static_cast<const T*>(buf.ptr),
                           static_cast<T*>(result.request().ptr), buf.size, b);
    return result;
}

// ============================================================================
// Array manipulation — numpy.diff, stack, concatenate, where, roll, flip, ...
// ============================================================================

/// numpy.diff(a, n=1, axis=-1, prepend=..., append=...) — N-D
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
    std::vector<py::ssize_t> py_out_shape(out_shape.begin(), out_shape.end());

    py::array_t<T> result(py_out_shape);
    diff(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr),
                shape.data(), static_cast<int>(buf.ndim), ax);
    return result;
}

/// numpy.stack(arrays, axis=0, out=None, *, dtype=None, casting=...)
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

/// numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting=...)
template<typename T>
py::array_t<T> concatenate(const std::vector<py::array_t<T>>& arrays, int axis = 0) {
    if (arrays.empty()) return py::array_t<T>{};

    auto buf0 = arrays[0].request();
    int ndim = static_cast<int>(buf0.ndim);

    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim)
        throw std::invalid_argument("concatenate: axis out of range");

    // Validate that all arrays have same number of dimensions
    for (const auto& arr : arrays) {
        if (arr.request().ndim != ndim)
            throw std::invalid_argument("concatenate: all arrays must have same number of dimensions");
    }

    // Collect shape (from first array) and per-array axis sizes
    std::vector<ptrdiff_t> shape(ndim);
    for (int d = 0; d < ndim; ++d) shape[d] = buf0.shape[d];

    std::vector<size_t> axis_sizes(arrays.size());
    for (size_t i = 0; i < arrays.size(); ++i) {
        auto buf = arrays[i].request();
        axis_sizes[i] = static_cast<size_t>(buf.shape[axis]);
    }

    // Validate non-axis dimensions match
    for (size_t i = 0; i < arrays.size(); ++i) {
        auto buf = arrays[i].request();
        for (int d = 0; d < ndim; ++d) {
            if (d == axis) continue;
            if (buf.shape[d] != shape[d])
                throw std::invalid_argument(
                    "concatenate: all arrays must have same shape except along axis");
        }
    }

    // Compute output shape
    std::vector<ptrdiff_t> out_shape = shape;
    ptrdiff_t total_axis = 0;
    for (size_t i = 0; i < arrays.size(); ++i)
        total_axis += static_cast<ptrdiff_t>(axis_sizes[i]);
    out_shape[axis] = total_axis;

    std::vector<py::ssize_t> py_out_shape(out_shape.begin(), out_shape.end());
    py::array_t<T> result(py_out_shape);
    T* dst = static_cast<T*>(result.request().ptr);

    // Build pointer array
    std::vector<const T*> ptrs(arrays.size());
    for (size_t i = 0; i < arrays.size(); ++i)
        ptrs[i] = static_cast<const T*>(arrays[i].request().ptr);

    numpy::concatenate(ptrs.data(), dst, shape.data(), ndim, axis,
                       axis_sizes.data(), arrays.size());
    return result;
}

/// numpy.vstack(tup, *, dtype=None, casting=...)
template<typename T>
py::array_t<T> vstack(const std::vector<py::array_t<T>>& arrays) {
    if (arrays.empty()) return py::array_t<T>{};
    int ndim = static_cast<int>(arrays[0].request().ndim);
    if (ndim == 1) {
        // numpy.vstack: 1D arrays are reshaped to (1, N) before stacking
        auto buf0 = arrays[0].request();
        py::array_t<T> result({static_cast<py::ssize_t>(arrays.size()), static_cast<py::ssize_t>(buf0.size)});
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

/// numpy.hstack(tup, *, dtype=None, casting=...)
template<typename T>
py::array_t<T> hstack(const std::vector<py::array_t<T>>& arrays) {
    if (arrays.empty()) return py::array_t<T>{};
    int ndim = static_cast<int>(arrays[0].request().ndim);
    // 1D arrays: hstack is identical to concatenate along axis=0
    // 2D+ arrays: hstack concatenates along axis=1
    return concatenate(arrays, (ndim == 1) ? 0 : 1);
}

/// numpy.where(condition, x, y) — scalar x, y
template<typename T>
py::array_t<T> where(const py::array_t<bool>& cond, T x, T y) {
    auto buf = cond.request();
    py::array_t<T> result(buf.shape);
    where_scalar(static_cast<const bool*>(buf.ptr),
                        static_cast<T*>(result.request().ptr), buf.size, x, y);
    return result;
}

/// numpy.where(condition, x, y) — array x, y
template<typename T>
py::array_t<T> where(const py::array_t<bool>& cond, const py::array_t<T>& x, const py::array_t<T>& y) {
    auto bc = cond.request(), bx = x.request(), by = y.request();
    py::array_t<T> result(bc.shape);
    where_array(static_cast<const bool*>(bc.ptr),
                       static_cast<T*>(result.request().ptr),
                       std::min({bc.size, bx.size, by.size}),
                       static_cast<const T*>(bx.ptr),
                       static_cast<const T*>(by.ptr));
    return result;
}

/// numpy.roll(a, shift, axis=None)
template<typename T>
py::array_t<T> roll(const py::array_t<T>& arr, py::ssize_t shift) {
    auto buf = arr.request();
    if (buf.size == 0) return py::array_t<T>{};
    py::array_t<T> result(buf.shape);
    roll(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size, shift);
    return result;
}

/// numpy.flip(m, axis=None)
template<typename T>
py::array_t<T> flip(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    flip(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

/// numpy.repeat(a, repeats, axis=None)
template<typename T>
py::array_t<T> repeat(const py::array_t<T>& arr, py::ssize_t repeats) {
    auto buf = arr.request();
    py::array_t<T> result({buf.size * repeats});
    repeat(static_cast<const T*>(buf.ptr),
                  static_cast<T*>(result.request().ptr), buf.size, repeats);
    return result;
}

/// numpy.tile(A, reps)
template<typename T>
py::array_t<T> tile(const py::array_t<T>& arr, py::ssize_t reps) {
    auto buf = arr.request();
    py::array_t<T> result({buf.size * reps});
    tile(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size, reps);
    return result;
}

// ============================================================================
// Sorting and indexing — numpy.argsort, argmax, argmin
// ============================================================================

/// numpy.argsort(a, axis=-1, kind=None, order=None)
template<typename T>
py::array_t<py::ssize_t> argsort(const py::array_t<T>& arr) {
    auto buf = arr.request();
    std::vector<ptrdiff_t> idx(buf.size);
    argsort(static_cast<const T*>(buf.ptr), idx.data(), buf.size);
    return py::array_t<py::ssize_t>(buf.size, idx.data());
}

/// numpy.argmax(a, axis=None, out=None, *, keepdims=...)
template<typename T>
py::ssize_t argmax(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return static_cast<py::ssize_t>(
        argmax(static_cast<const T*>(buf.ptr), buf.size));
}

/// numpy.argmin(a, axis=None, out=None, *, keepdims=...)
template<typename T>
py::ssize_t argmin(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return static_cast<py::ssize_t>(
        argmin(static_cast<const T*>(buf.ptr), buf.size));
}

// ============================================================================
// Set operations — numpy.isin, numpy.intersect1d
// ============================================================================

/// numpy.isin(element, test_elements, assume_unique=False, invert=False)
inline py::array_t<bool> isin(const py::array_t<double>& arr, const std::vector<double>& values) {
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    numpy::isin(static_cast<const double*>(buf.ptr),
                static_cast<bool*>(result.request().ptr), buf.size,
                values.data(), values.size());
    return result;
}

/// isin with int test_elements (e.g. np.isin(arr, [1, 3, 5]))
inline py::array_t<bool> isin(const py::array_t<double>& arr, const std::vector<int>& values) {
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    std::vector<double> dv(values.begin(), values.end());
    numpy::isin(static_cast<const double*>(buf.ptr),
                static_cast<bool*>(result.request().ptr), buf.size,
                dv.data(), dv.size());
    return result;
}

/// numpy.flatnonzero(a)
inline py::array_t<py::ssize_t> flatnonzero(const py::array_t<double>& arr) {
    auto buf = arr.request();
    auto idx = numpy::flatnonzero(static_cast<const double*>(buf.ptr), buf.size);
    py::array_t<py::ssize_t> result(idx.size());
    std::copy(idx.begin(), idx.end(),
              static_cast<py::ssize_t*>(result.request().ptr));
    return result;
}

/// numpy.unwrap(p, discont=None, axis=-1) — 1D only
template<typename T>
py::array_t<T> unwrap(const py::array_t<T>& arr, T discont = T(M_PI)) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    numpy::unwrap(static_cast<const T*>(buf.ptr),
                  static_cast<T*>(result.request().ptr), buf.size, discont);
    return result;
}

/// numpy.cumsum(a, axis=None) — 1D cumulative sum
template<typename T>
py::array_t<T> cumsum(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    numpy::cumsum(static_cast<const T*>(buf.ptr),
                  static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

/// numpy.squeeze(a, axis=None) — remove axes of length 1
template<typename T>
py::array_t<T> squeeze(const py::array_t<T>& arr) {
    auto buf = arr.request();
    std::vector<py::ssize_t> new_shape;
    for (auto s : buf.shape)
        if (s != 1) new_shape.push_back(s);
    if (new_shape.empty()) new_shape.push_back(1);
    py::array_t<T> result(new_shape);
    std::memcpy(result.request().ptr, buf.ptr, buf.size * sizeof(T));
    return result;
}

/// numpy.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)
template<typename T>
py::array_t<T> intersect1d(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    auto inter = intersect1d(
        static_cast<const T*>(ba.ptr), ba.size,
        static_cast<const T*>(bb.ptr), bb.size);
    return py::array_t<T>(inter.size(), inter.data());
}

// ============================================================================
// Interpolation — numpy.interp
// ============================================================================

/// numpy.interp(x, xp, fp, left=None, right=None, period=None)
inline py::array_t<double> interp(const py::array_t<double>& x,
                                   const py::array_t<double>& xp,
                                   const py::array_t<double>& fp) {
    auto bx = x.request(), bxp = xp.request(), bfp = fp.request();
    py::array_t<double> result(bx.shape);
    interp(
        static_cast<const double*>(bx.ptr),
        static_cast<double*>(result.request().ptr), bx.size,
        static_cast<const double*>(bxp.ptr),
        static_cast<const double*>(bfp.ptr), bxp.size);
    return result;
}

// ============================================================================
// Advanced / Fancy indexing
// ============================================================================
//
// These wrappers implement the full set of numpy advanced-indexing operations.
// All names align with the numpy function API:
//
//   take(a, indices, axis)       — numpy.take  : gather along an axis
//   compress(a, mask)            — numpy.compress : boolean mask gather
//   slice(a, starts,stops,steps) — overload of slice() with per-dim step
//   put(a, indices, values)      — numpy.put   : in-place scatter by flat index
//   putmask(a, mask, scalar)     — numpy.putmask (scalar):  a[mask] = scalar
//   putmask(a, mask, values)     — numpy.putmask (array):   a[mask] = array
//   slice_assign(a,starts,stops,steps,scalar) — overload: a[s:e:k]=scalar
//   slice_assign(a,starts,stops,steps,values) — overload: a[s:e:k]=array
//
// All slice starts/stops/steps must be pre-normalized by the caller using
// Python's slice.indices(n) convention:
//   • step != 0
//   • stop = -1 is the valid "before index 0" sentinel for negative-step slices
//     (e.g. slice(None,None,-1).indices(n) → (n-1, -1, -1))
// ============================================================================

// ── Helper: normalize Python slice parameters for a single dimension ─────────
// Mirrors the logic of PySlice_GetIndicesEx / slice.indices(length).
// None is represented as py::ssize_t min (LLONG_MIN / 2) by convention from
// callers; this helper is also exposed so callers can avoid that dependency.
inline void _slice_indices(py::ssize_t length,
                            py::ssize_t start_in,  bool start_none,
                            py::ssize_t stop_in,   bool stop_none,
                            py::ssize_t step,
                            py::ssize_t& start_out, py::ssize_t& stop_out) {
    if (step == 0) throw std::invalid_argument("slice step cannot be zero");

    py::ssize_t start, stop;

    if (step > 0) {
        start = start_none ? 0      : start_in;
        stop  = stop_none  ? length : stop_in;
        // wrap negatives
        if (start < 0) start += length;
        if (stop  < 0) stop  += length;
        // clamp
        start = std::max(py::ssize_t(0), std::min(length,   start));
        stop  = std::max(py::ssize_t(0), std::min(length,   stop));
    } else {
        start = start_none ? length - 1 : start_in;
        stop  = stop_none  ? -(length + 1) : stop_in;   // sentinel: before idx 0
        // wrap negatives (but NOT for stop when already a "before 0" sentinel)
        if (start < 0) start += length;
        if (stop  < 0 && !stop_none) stop += length;
        // clamp
        start = std::max(py::ssize_t(-1), std::min(length - 1, start));
        stop  = std::max(py::ssize_t(-1), std::min(length - 1, stop));
    }
    start_out = start;
    stop_out  = stop;
}

// ── numpy.take ───────────────────────────────────────────────────────────────

/// numpy.take(a, indices, axis=None)
///   axis = -1 means axis=None (gather from flattened array).
///   indices must be int64 (np.int64 / np.intp array).
///   Negative indices are wrapped mod axis_size.
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
    const ptrdiff_t* idx_ptr = static_cast<const ptrdiff_t*>(ibuf.ptr);

    // Build output shape
    std::vector<py::ssize_t> out_shape;
    if (axis == -1) {
        // axis=None → flat gather
        out_shape.push_back(static_cast<py::ssize_t>(ni));
    } else {
        for (int d = 0; d < ndim; ++d) {
            if (d == axis) out_shape.push_back(static_cast<py::ssize_t>(ni));
            else           out_shape.push_back(buf.shape[d]);
        }
    }

    py::array_t<T> result(out_shape);
    numpy::take(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr),
                shape.data(), ndim, axis, idx_ptr, ni);
    return result;
}

// ── numpy.compress ───────────────────────────────────────────────────────────

/// numpy.compress(condition, a, axis=None)
/// Boolean mask gather.  The array is treated as flat (axis=None).
/// Returns a 1-D array of elements where mask[i] is true.
template<typename T>
py::array_t<T> compress(const py::array_t<T>& arr,
                         const py::array_t<bool>& mask) {
    auto buf  = arr.request();
    auto mbuf = mask.request();
    size_t n   = static_cast<size_t>(buf.size);
    size_t nm  = static_cast<size_t>(mbuf.size);
    size_t use = std::min(n, nm);
    const bool* m = static_cast<const bool*>(mbuf.ptr);

    // Count trues first so we can allocate exact output
    size_t cnt = 0;
    for (size_t i = 0; i < use; ++i) if (m[i]) ++cnt;

    py::array_t<T> result({static_cast<py::ssize_t>(cnt)});
    if (cnt > 0)
        numpy::compress(static_cast<const T*>(buf.ptr),
                        static_cast<T*>(result.request().ptr),
                        m, use);
    return result;
}

// ── slice() overload: N-D slice with per-dimension step ──────────────────────

/// slice(a, starts, stops, steps) — N-D generalization of slice(a, start, stop).
///
/// Overload of the existing slice() function; this variant accepts three
/// equal-length vectors (one entry per array dimension) instead of scalar
/// start/stop, enabling arbitrary step and multi-dimensional slicing.
///
/// starts/stops/steps must be pre-normalized (use Python's
///   slice(s,e,k).indices(dim_size) to obtain canonical values).
///
/// Examples:
///   slice(a, {0}, {10}, {2})        → a[::2]
///   slice(a, {9}, {-1}, {-1})       → a[::-1]   (stop=-1 = "before index 0")
///   slice(a, {1,2}, {3,5}, {1,1})   → a[1:3, 2:5]
///   slice(a, {n-1,0}, {-1,m}, {-1,2}) → a[::-1, ::2]
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
        throw std::invalid_argument("slice: starts/stops/steps must match array ndim");

    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());
    std::vector<ptrdiff_t> st(starts.begin(), starts.end());
    std::vector<ptrdiff_t> sp(stops.begin(),  stops.end());
    std::vector<ptrdiff_t> sv(steps.begin(),  steps.end());

    // Compute output shape
    std::vector<py::ssize_t> out_shape(ndim);
    for (int d = 0; d < ndim; ++d) {
        ptrdiff_t len = numpy::slice_len(st[d], sp[d], sv[d]);
        if (len <= 0) {
            // Return empty array with zeros in all dims (matches numpy)
            std::vector<py::ssize_t> empty_shape(ndim, 0);
            return py::array_t<T>(empty_shape);
        }
        out_shape[d] = static_cast<py::ssize_t>(len);
    }

    py::array_t<T> result(out_shape);
    numpy::slice(static_cast<const T*>(buf.ptr),
                 static_cast<T*>(result.request().ptr),
                 shape.data(), ndim,
                 st.data(), sp.data(), sv.data());
    return result;
}

// ── numpy.put ────────────────────────────────────────────────────────────────

/// numpy.put(a, indices, values, mode='raise')
/// Scatters values into arr at flat indices (in-place).
/// Negative indices: idx < 0 → idx += n.
/// Out-of-range indices are silently ignored (safe / clip behaviour).
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

// ── putmask: numpy.putmask ────────────────────────────────────────────────────

/// numpy.putmask(a, mask, values) — scalar variant.
/// In-place: sets a[i] = value for every i where mask[i] is True.
/// Equivalent to Python:  a[mask] = scalar
template<typename T>
void putmask(py::array_t<T> arr,
             const py::array_t<bool>& mask, T value) {
    auto buf  = arr.request();
    auto mbuf = mask.request();
    numpy::putmask(
        static_cast<T*>(buf.ptr),
        static_cast<const bool*>(mbuf.ptr),
        static_cast<size_t>(std::min(buf.size, mbuf.size)),
        value);
}

/// numpy.putmask(a, mask, values) — array variant.
/// In-place: sets a[i] = values[j++] for the j-th True position i.
/// Equivalent to Python:  a[mask] = array_of_values
template<typename T>
void putmask(py::array_t<T> arr,
             const py::array_t<bool>& mask,
             const py::array_t<T>& values) {
    auto buf  = arr.request();
    auto mbuf = mask.request();
    auto vbuf = values.request();
    numpy::putmask(
        static_cast<T*>(buf.ptr),
        static_cast<const bool*>(mbuf.ptr),
        static_cast<size_t>(std::min(buf.size, mbuf.size)),
        static_cast<const T*>(vbuf.ptr));
}

// ── slice_assign() overload: N-D in-place slice assignment ───────────────────

/// slice_assign(a, starts, stops, steps, scalar) — N-D in-place slice assign.
/// Overload of the existing slice_assign(); this variant accepts step vectors.
/// In-place: a[s0:e0:k0, s1:e1:k1, ...] = scalar
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
        throw std::invalid_argument(
            "slice_assign: starts/stops/steps must match array ndim");

    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());
    std::vector<ptrdiff_t> st(starts.begin(), starts.end());
    std::vector<ptrdiff_t> sp(stops.begin(),  stops.end());
    std::vector<ptrdiff_t> sv(steps.begin(),  steps.end());

    numpy::slice_assign(
        static_cast<T*>(buf.ptr),
        shape.data(), ndim,
        st.data(), sp.data(), sv.data(), value);
}

/// slice_assign(a, starts, stops, steps, values) — N-D in-place slice assign.
/// Overload of slice_assign(); this variant accepts an array of values.
/// values must contain exactly product(slice_shape) elements in C-order.
/// In-place: a[s0:e0:k0, s1:e1:k1, ...] = values_array
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
        throw std::invalid_argument(
            "slice_assign: starts/stops/steps must match array ndim");

    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());
    std::vector<ptrdiff_t> st(starts.begin(), starts.end());
    std::vector<ptrdiff_t> sp(stops.begin(),  stops.end());
    std::vector<ptrdiff_t> sv(steps.begin(),  steps.end());

    auto vbuf = values.request();
    numpy::slice_assign(
        static_cast<T*>(buf.ptr),
        shape.data(), ndim,
        st.data(), sp.data(), sv.data(),
        static_cast<const T*>(vbuf.ptr));
}

} // namespace numpy
