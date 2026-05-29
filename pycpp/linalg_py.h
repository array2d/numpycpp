// Pybind11 wrappers for linalg native functions.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../numpy/linalg.h"

namespace py = pybind11;

namespace numpy {
namespace linalg {

/// numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)
template<typename T>
T norm(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return norm(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.linalg.norm(x, ord=None, axis=N, keepdims=False) — N-D with axis
template<typename T>
py::array_t<T> norm(const py::array_t<T>& arr, int axis = -1) {
    auto buf = arr.request();
    int ndim = static_cast<int>(buf.ndim);
    int ax = (axis == -1) ? ndim - 1 : axis;

    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());

    // Output shape: drop the reduced axis
    std::vector<ptrdiff_t> out_shape;
    for (int d = 0; d < ndim; ++d)
        if (d != ax) out_shape.push_back(shape[d]);
    if (out_shape.empty()) out_shape.push_back(1);

    std::vector<py::ssize_t> py_out_shape(out_shape.begin(), out_shape.end());
    py::array_t<T> result(py_out_shape);
    numpy::norm_axis(static_cast<const T*>(buf.ptr),
                            static_cast<T*>(result.request().ptr),
                            shape.data(), ndim, ax);
    return result;
}

} // namespace linalg

/// numpy.dot(a, b, out=None)
template<typename T>
T dot(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    return numpy::dot(static_cast<const T*>(ba.ptr),
                             static_cast<const T*>(bb.ptr),
                             std::min(ba.size, bb.size));
}

} // namespace numpy
