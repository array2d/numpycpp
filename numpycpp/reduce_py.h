// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — pycpp/reduce_py.h                         [PUBLIC HEADER]
//  Pybind11 wrappers: reduction operations.
//      numpy.sum  numpy.mean  numpy.max  numpy.min
//      numpy.any  numpy.all   numpy.std  numpy.var
//      numpy.mean(axis)       numpy.cumsum
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "numpy.h"
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace numpy {

/// numpy.sum(a)
template<typename T>
T sum(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return numpy::sum(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.mean(a)
template<typename T>
T mean(const py::array_t<T>& arr) {
    auto buf = arr.request();
    if (buf.size == 0) return T(0);
    return numpy::mean(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.max(a)
template<typename T>
T max(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return numpy::max(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.min(a)
template<typename T>
T min(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return numpy::min(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.any(a)
inline bool any(const py::array_t<bool>& arr) {
    auto buf = arr.request();
    return numpy::any(static_cast<const bool*>(buf.ptr), buf.size);
}

/// numpy.all(a)
inline bool all(const py::array_t<bool>& arr) {
    auto buf = arr.request();
    return numpy::all(static_cast<const bool*>(buf.ptr), buf.size);
}

/// numpy.std(a)
template<typename T>
T std(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return numpy::stddev(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.var(a)
template<typename T>
T var(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return numpy::var(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.mean(a, axis) — N-D
template<typename T>
py::array_t<T> mean_axis(const py::array_t<T>& arr, int axis) {
    auto buf = arr.request();
    if (buf.ndim < 1)
        throw std::invalid_argument("Array must have at least 1 dimension");

    int ax = (axis == -1) ? static_cast<int>(buf.ndim - 1) : axis;
    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());

    std::vector<ptrdiff_t> out_shape;
    for (int d = 0; d < static_cast<int>(buf.ndim); ++d)
        if (d != ax) out_shape.push_back(shape[d]);
    if (out_shape.empty()) out_shape.push_back(1);

    std::vector<py::ssize_t> py_out_shape(out_shape.begin(), out_shape.end());
    py::array_t<T> result(py_out_shape);
    numpy::mean_axis(static_cast<const T*>(buf.ptr),
                     static_cast<T*>(result.request().ptr),
                     shape.data(), static_cast<int>(buf.ndim), ax);
    return result;
}

/// numpy.mean(a, axis) — float64 overload
inline py::array_t<double> mean(const py::array_t<double>& arr, int axis) {
    return mean_axis(arr, axis);
}
/// numpy.mean(a, axis) — float32 overload
inline py::array_t<float> mean(const py::array_t<float>& arr, int axis) {
    return mean_axis(arr, axis);
}

/// numpy.cumsum(a)
template<typename T>
py::array_t<T> cumsum(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    numpy::cumsum(static_cast<const T*>(buf.ptr),
                  static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

} // namespace numpy
