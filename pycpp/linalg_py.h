// Pybind11 wrappers for linalg native functions.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../numpy/linalg.h"

namespace py = pybind11;

namespace numpy {
namespace linalg {

template<typename T>
T norm(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return norm(static_cast<const T*>(buf.ptr), buf.size);
}

template<typename T>
py::array_t<double> norm_axis1(const py::array_t<T>& arr) {
    auto buf = arr.request();
    if (buf.ndim != 2) {
        py::array_t<double> result({1});
        *static_cast<double*>(result.request().ptr) = static_cast<double>(norm(arr));
        return result;
    }
    py::ssize_t rows = buf.shape[0], cols = buf.shape[1];
    py::array_t<double> result({rows});
    norm_axis1(static_cast<const T*>(buf.ptr),
                      static_cast<double*>(result.request().ptr), rows, cols);
    return result;
}

} // namespace linalg

template<typename T>
T dot(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    return numpy::dot(static_cast<const T*>(ba.ptr),
                             static_cast<const T*>(bb.ptr),
                             std::min(ba.size, bb.size));
}

} // namespace numpy
