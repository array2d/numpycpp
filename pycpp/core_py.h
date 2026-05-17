// Pybind11 wrappers for core.h native functions.
// Thin layer: extract pointers from py::array_t, call *, wrap results.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../numpy/core.h"
#include <vector>
#include <cstring>

namespace py = pybind11;

namespace numpy {

// ============================================================================
// Array creation
// ============================================================================
template<typename T>
py::array_t<T> zeros_like(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    zeros_like(static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

template<typename T>
py::array_t<T> ones_like(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    ones_like(static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

template<typename T>
py::array_t<T> full_like(const py::array_t<T>& arr, T fill_value) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    full_like(static_cast<T*>(result.request().ptr), buf.size, fill_value);
    return result;
}

template<typename T>
py::array_t<T> empty_like(const py::array_t<T>& arr) {
    return py::array_t<T>(arr.request().shape);
}

inline py::array_t<double> zeros(const std::vector<py::ssize_t>& shape) {
    py::array_t<double> result(shape);
    zeros_like(static_cast<double*>(result.request().ptr), result.request().size);
    return result;
}

inline py::array_t<double> ones(const std::vector<py::ssize_t>& shape) {
    py::array_t<double> result(shape);
    ones_like(static_cast<double*>(result.request().ptr), result.request().size);
    return result;
}

// Bool specializations
inline py::array_t<bool> full_like_bool(const py::array_t<double>& arr, bool fill_value) {
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    std::fill_n(static_cast<bool*>(result.request().ptr), buf.size, fill_value);
    return result;
}

inline py::array_t<bool> zeros_like_bool(const py::array_t<double>& arr) {
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    std::fill_n(static_cast<bool*>(result.request().ptr), buf.size, false);
    return result;
}

inline py::array_t<bool> ones_like_bool(const py::array_t<double>& arr) {
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    std::fill_n(static_cast<bool*>(result.request().ptr), buf.size, true);
    return result;
}

// ============================================================================
// astype
// ============================================================================
inline py::array_t<int> astype_int(const py::array_t<double>& arr) {
    auto buf = arr.request();
    py::array_t<int> result(buf.shape);
    astype_int(static_cast<const double*>(buf.ptr),
                     static_cast<int*>(result.request().ptr), buf.size);
    return result;
}

inline py::array_t<bool> astype_bool(const py::array_t<double>& arr) {
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    astype_bool(static_cast<const double*>(buf.ptr),
                      static_cast<bool*>(result.request().ptr), buf.size);
    return result;
}

inline py::array_t<bool> astype_bool_from_int(const py::array_t<int>& arr) {
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    astype_bool_from_int(static_cast<const int*>(buf.ptr),
                                static_cast<bool*>(result.request().ptr), buf.size);
    return result;
}

inline py::array_t<double> truncate_to_float32(const py::array_t<double>& arr) {
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    truncate_to_float32(static_cast<const double*>(buf.ptr),
                               static_cast<double*>(result.mutable_data()), buf.size);
    return result;
}

// ============================================================================
// Element-wise math
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

DEF_ELEMWISE(sqrt)
DEF_ELEMWISE(abs)
DEF_ELEMWISE(exp)
DEF_ELEMWISE(log)
DEF_ELEMWISE(sin)
DEF_ELEMWISE(cos)
DEF_ELEMWISE(tan)
DEF_ELEMWISE(log10)
DEF_ELEMWISE(log2)
DEF_ELEMWISE(arcsin)
DEF_ELEMWISE(arccos)
DEF_ELEMWISE(arctan)
DEF_ELEMWISE(round)
DEF_ELEMWISE(floor)
DEF_ELEMWISE(ceil)
DEF_ELEMWISE(degrees)
DEF_ELEMWISE(radians)
DEF_ELEMWISE(sign)

#undef DEF_ELEMWISE

template<typename T>
py::array_t<T> power(const py::array_t<T>& arr, T exponent) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    power(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size, exponent);
    return result;
}

template<typename T>
py::array_t<T> clip(const py::array_t<T>& arr, T min_val, T max_val) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    clip(static_cast<const T*>(buf.ptr),
               static_cast<T*>(result.request().ptr), buf.size, min_val, max_val);
    return result;
}

// ============================================================================
// Reduction
// ============================================================================
template<typename T>
T sum(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return sum(static_cast<const T*>(buf.ptr), buf.size);
}

template<typename T>
T mean(const py::array_t<T>& arr) {
    auto buf = arr.request();
    if (buf.size == 0) return T(0);
    return mean(static_cast<const T*>(buf.ptr), buf.size);
}

template<typename T>
T max(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return max(static_cast<const T*>(buf.ptr), buf.size);
}

template<typename T>
T min(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return min(static_cast<const T*>(buf.ptr), buf.size);
}

inline bool any(const py::array_t<bool>& arr) {
    auto buf = arr.request();
    return any(static_cast<const bool*>(buf.ptr), buf.size);
}

inline bool all(const py::array_t<bool>& arr) {
    auto buf = arr.request();
    return all(static_cast<const bool*>(buf.ptr), buf.size);
}

template<typename T>
T std(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return stddev(static_cast<const T*>(buf.ptr), buf.size);
}

template<typename T>
T var(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return var(static_cast<const T*>(buf.ptr), buf.size);
}

// ============================================================================
// Comparison
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

DEF_COMPARE(greater)
DEF_COMPARE(less)
DEF_COMPARE(equal)
DEF_COMPARE(greater_equal)
DEF_COMPARE(less_equal)

#undef DEF_COMPARE

template<typename T>
py::array_t<bool> not_equal(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    not_equal_scalar(static_cast<const T*>(buf.ptr),
                            static_cast<bool*>(result.request().ptr), buf.size, b);
    return result;
}

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
// Logical
// ============================================================================
inline py::array_t<bool> logical_and(const py::array_t<bool>& a, const py::array_t<bool>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<bool> result(ba.shape);
    logical_and(static_cast<const bool*>(ba.ptr),
                       static_cast<const bool*>(bb.ptr),
                       static_cast<bool*>(result.request().ptr),
                       std::min(ba.size, bb.size));
    return result;
}

inline py::array_t<bool> logical_or(const py::array_t<bool>& a, const py::array_t<bool>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<bool> result(ba.shape);
    logical_or(static_cast<const bool*>(ba.ptr),
                      static_cast<const bool*>(bb.ptr),
                      static_cast<bool*>(result.request().ptr),
                      std::min(ba.size, bb.size));
    return result;
}

inline py::array_t<bool> logical_not(const py::array_t<bool>& a) {
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    logical_not(static_cast<const bool*>(buf.ptr),
                       static_cast<bool*>(result.request().ptr), buf.size);
    return result;
}

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
// Special value helpers
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

DEF_SPECIAL(isnan)
DEF_SPECIAL(isinf)
DEF_SPECIAL(isfinite)

#undef DEF_SPECIAL

// ============================================================================
// Array access
// ============================================================================
inline double array_get(const py::array_t<double>& arr, py::ssize_t idx) {
    auto buf = arr.request();
    if (idx < 0) idx = buf.size + idx;
    return (idx >= 0 && idx < buf.size)
        ? static_cast<const double*>(buf.ptr)[idx] : 0.0;
}

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
// Conversion
// ============================================================================
inline py::array_t<double> asarray(const std::vector<double>& vec) {
    return py::array_t<double>(vec.size(), vec.data());
}
inline py::array_t<double> asarray(const py::array_t<double>& arr) { return arr; }
inline py::array_t<double> array(const std::vector<double>& vec) {
    return py::array_t<double>(vec.size(), vec.data());
}
inline py::array_t<double> array(const py::array_t<double>& arr) { return arr; }

// ============================================================================
// transpose / flatten
// ============================================================================
template<typename T>
py::array_t<T> transpose(const py::array_t<T>& arr) {
    auto buf = arr.request();
    if (buf.ndim == 1) {
        py::array_t<T> result(buf.shape);
        std::memcpy(static_cast<T*>(result.request().ptr),
                    static_cast<const T*>(buf.ptr), buf.size * sizeof(T));
        return result;
    }
    if (buf.ndim == 2) {
        py::array_t<T> result({buf.shape[1], buf.shape[0]});
        transpose_2d(static_cast<const T*>(buf.ptr),
                            static_cast<T*>(result.request().ptr),
                            buf.shape[0], buf.shape[1]);
        return result;
    }
    py::array_t<T> result({buf.size});
    std::memcpy(static_cast<T*>(result.request().ptr),
                static_cast<const T*>(buf.ptr), buf.size * sizeof(T));
    return result;
}

template<typename T>
py::array_t<T> flatten(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result({buf.size});
    std::memcpy(static_cast<T*>(result.request().ptr),
                static_cast<const T*>(buf.ptr), buf.size * sizeof(T));
    return result;
}

// ============================================================================
// mean with axis
// ============================================================================
template<typename T>
py::array_t<double> mean_axis(const py::array_t<T>& arr, int axis) {
    auto buf = arr.request();
    if (buf.ndim < 1 || buf.ndim > 3)
        throw std::invalid_argument("Array must be 1D, 2D, or 3D");

    int ax = (axis == -1) ? static_cast<int>(buf.ndim - 1) : axis;

    if (buf.ndim == 1) {
        if (ax != 0) throw std::invalid_argument("For 1D array, axis must be 0 or -1");
        py::array_t<double> result({1});
        *static_cast<double*>(result.request().ptr) =
            static_cast<double>(mean(arr));
        return result;
    }

    if (buf.ndim == 2) {
        py::ssize_t rows = buf.shape[0], cols = buf.shape[1];
        const T* src = static_cast<const T*>(buf.ptr);
        if (ax == 0) {
            py::array_t<double> result({cols});
            mean_axis0_2d(src, static_cast<double*>(result.request().ptr),
                                 rows, cols);
            return result;
        }
        if (ax == 1) {
            py::array_t<double> result({rows});
            mean_axis1_2d(src, static_cast<double*>(result.request().ptr),
                                 rows, cols);
            return result;
        }
        throw std::invalid_argument("For 2D array, axis must be 0, 1, or -1");
    }

    // ndim == 3
    py::ssize_t d0 = buf.shape[0], d1 = buf.shape[1], d2 = buf.shape[2];
    const T* src = static_cast<const T*>(buf.ptr);
    if (ax == 0) {
        py::array_t<double> result({d1, d2});
        mean_axis0_3d(src, static_cast<double*>(result.request().ptr),
                              d0, d1, d2);
        return result;
    }
    if (ax == 1) {
        py::array_t<double> result({d0, d2});
        mean_axis1_3d(src, static_cast<double*>(result.request().ptr),
                              d0, d1, d2);
        return result;
    }
    if (ax == 2) {
        py::array_t<double> result({d0, d1});
        mean_axis2_3d(src, static_cast<double*>(result.request().ptr),
                              d0, d1, d2);
        return result;
    }
    throw std::invalid_argument("For 3D array, axis must be 0, 1, 2, or -1");
}

inline py::array_t<double> mean(const py::array_t<double>& arr, int axis) { return mean_axis(arr, axis); }
inline py::array_t<double> mean(const py::array_t<float>& arr, int axis)  { return mean_axis(arr, axis); }

// ============================================================================
// to_vector
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
// Slice helpers
// ============================================================================
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
// Column slice
// ============================================================================
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
// Slice assignment
// ============================================================================
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
// Binary element-wise: array-array and array-scalar
// ============================================================================
template<typename T>
py::array_t<T> arctan2(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<T> result(ba.shape);
    arctan2_array(static_cast<const T*>(ba.ptr),
                          static_cast<const T*>(bb.ptr),
                          static_cast<T*>(result.request().ptr),
                          std::min(ba.size, bb.size));
    return result;
}

template<typename T>
py::array_t<T> arctan2(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<T> result(buf.shape);
    arctan2_scalar(static_cast<const T*>(buf.ptr),
                           static_cast<T*>(result.request().ptr), buf.size, b);
    return result;
}

template<typename T>
py::array_t<T> maximum(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<T> result(ba.shape);
    maximum_array(static_cast<const T*>(ba.ptr),
                          static_cast<const T*>(bb.ptr),
                          static_cast<T*>(result.request().ptr),
                          std::min(ba.size, bb.size));
    return result;
}

template<typename T>
py::array_t<T> maximum(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<T> result(buf.shape);
    maximum_scalar(static_cast<const T*>(buf.ptr),
                           static_cast<T*>(result.request().ptr), buf.size, b);
    return result;
}

template<typename T>
py::array_t<T> minimum(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<T> result(ba.shape);
    minimum_array(static_cast<const T*>(ba.ptr),
                          static_cast<const T*>(bb.ptr),
                          static_cast<T*>(result.request().ptr),
                          std::min(ba.size, bb.size));
    return result;
}

template<typename T>
py::array_t<T> minimum(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<T> result(buf.shape);
    minimum_scalar(static_cast<const T*>(buf.ptr),
                           static_cast<T*>(result.request().ptr), buf.size, b);
    return result;
}

// ============================================================================
// Array manipulation: diff, stack, concatenate, etc.
// ============================================================================
template<typename T>
py::array_t<T> diff(const py::array_t<T>& arr, int n = 1, int axis = -1) {
    auto buf = arr.request();
    if (buf.ndim == 0 || buf.size < 2) return py::array_t<T>{};

    if (buf.ndim == 1) {
        py::array_t<T> result({buf.size - 1});
        diff_1d(static_cast<const T*>(buf.ptr),
                       static_cast<T*>(result.request().ptr), buf.size);
        return result;
    }

    if (buf.ndim == 2) {
        py::ssize_t rows = buf.shape[0], cols = buf.shape[1];
        const T* src = static_cast<const T*>(buf.ptr);
        int ax = (axis == -1) ? 1 : axis;
        if (ax == 0) {
            py::array_t<T> result({rows - 1, cols});
            diff_2d_axis0(src, static_cast<T*>(result.request().ptr), rows, cols);
            return result;
        } else {
            py::array_t<T> result({rows, cols - 1});
            diff_2d_axis1(src, static_cast<T*>(result.request().ptr), rows, cols);
            return result;
        }
    }

    // fallback: flatten
    py::array_t<T> result({buf.size - 1});
    diff_1d(static_cast<const T*>(buf.ptr),
                   static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

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

template<typename T>
py::array_t<T> concatenate(const std::vector<py::array_t<T>>& arrays) {
    if (arrays.empty()) return py::array_t<T>{};
    py::ssize_t total = 0;
    for (const auto& arr : arrays) total += arr.request().size;
    py::array_t<T> result({total});
    T* dst = static_cast<T*>(result.request().ptr);
    py::ssize_t off = 0;
    for (const auto& arr : arrays) {
        auto buf = arr.request();
        std::memcpy(dst + off, static_cast<const T*>(buf.ptr), buf.size * sizeof(T));
        off += buf.size;
    }
    return result;
}

template<typename T>
py::array_t<T> vstack(const std::vector<py::array_t<T>>& arrays) { return stack(arrays); }

template<typename T>
py::array_t<T> hstack(const std::vector<py::array_t<T>>& arrays) { return concatenate(arrays); }

template<typename T>
py::array_t<T> where(const py::array_t<bool>& cond, T x, T y) {
    auto buf = cond.request();
    py::array_t<T> result(buf.shape);
    where_scalar(static_cast<const bool*>(buf.ptr),
                        static_cast<T*>(result.request().ptr), buf.size, x, y);
    return result;
}

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

template<typename T>
py::array_t<T> roll(const py::array_t<T>& arr, py::ssize_t shift) {
    auto buf = arr.request();
    if (buf.size == 0) return py::array_t<T>{};
    py::array_t<T> result(buf.shape);
    roll(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size, shift);
    return result;
}

template<typename T>
py::array_t<T> flip(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    flip(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

template<typename T>
py::array_t<T> repeat(const py::array_t<T>& arr, py::ssize_t repeats) {
    auto buf = arr.request();
    py::array_t<T> result({buf.size * repeats});
    repeat(static_cast<const T*>(buf.ptr),
                  static_cast<T*>(result.request().ptr), buf.size, repeats);
    return result;
}

template<typename T>
py::array_t<T> tile(const py::array_t<T>& arr, py::ssize_t reps) {
    auto buf = arr.request();
    py::array_t<T> result({buf.size * reps});
    tile(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size, reps);
    return result;
}

// ============================================================================
// Sorting and indexing
// ============================================================================
template<typename T>
py::array_t<py::ssize_t> argsort(const py::array_t<T>& arr) {
    auto buf = arr.request();
    std::vector<ptrdiff_t> idx(buf.size);
    argsort(static_cast<const T*>(buf.ptr), idx.data(), buf.size);
    return py::array_t<py::ssize_t>(buf.size, idx.data());
}

template<typename T>
py::ssize_t argmax(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return static_cast<py::ssize_t>(
        argmax(static_cast<const T*>(buf.ptr), buf.size));
}

template<typename T>
py::ssize_t argmin(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return static_cast<py::ssize_t>(
        argmin(static_cast<const T*>(buf.ptr), buf.size));
}

// ============================================================================
// Set operations
// ============================================================================
inline py::array_t<bool> isin(const py::array_t<double>& arr, const std::vector<double>& values) {
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    isin(static_cast<const double*>(buf.ptr),
                static_cast<bool*>(result.request().ptr), buf.size,
                values.data(), values.size());
    return result;
}

inline py::array_t<double> intersect1d(const py::array_t<double>& a, const py::array_t<double>& b) {
    auto ba = a.request(), bb = b.request();
    auto inter = intersect1d(
        static_cast<const double*>(ba.ptr), ba.size,
        static_cast<const double*>(bb.ptr), bb.size);
    return py::array_t<double>(inter.size(), inter.data());
}

// ============================================================================
// Interpolation
// ============================================================================
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

} // namespace numpy
