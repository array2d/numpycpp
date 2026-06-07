// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — pycpp/init_py.h                           [PUBLIC HEADER]
//  Pybind11 wrappers: array creation / initialisation.
//      numpy.zeros_like  numpy.ones_like  numpy.full_like  numpy.empty_like
//      numpy.zeros       numpy.ones       numpy.full
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../numpy/numpy.h"
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

namespace numpy {

// ── Template variants (float64 / float32) ────────────────────────────────────

/// numpy.zeros_like(a, ...)
template<typename T>
py::array_t<T> zeros_like(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    numpy::zeros_like(static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

/// numpy.ones_like(a, ...)
template<typename T>
py::array_t<T> ones_like(const py::array_t<T>& arr) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    numpy::ones_like(static_cast<T*>(result.request().ptr), buf.size);
    return result;
}

/// numpy.full_like(a, fill_value, ...)
template<typename T>
py::array_t<T> full_like(const py::array_t<T>& arr, T fill_value) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    numpy::full(static_cast<T*>(result.request().ptr), buf.size, fill_value);
    return result;
}

/// numpy.empty_like(prototype, ...)
template<typename T>
py::array_t<T> empty_like(const py::array_t<T>& arr) {
    return py::array_t<T>(arr.request().shape);
}

// ── Shape-based creation (float64) ───────────────────────────────────────────

/// numpy.zeros(shape, dtype=float, order='C', *, like=None)
inline py::array_t<double> zeros(const std::vector<py::ssize_t>& shape) {
    py::array_t<double> result(shape);
    numpy::zeros_like(static_cast<double*>(result.request().ptr),
                      result.request().size);
    return result;
}

/// numpy.ones(shape, dtype=float, order='C', *, like=None)
inline py::array_t<double> ones(const std::vector<py::ssize_t>& shape) {
    py::array_t<double> result(shape);
    numpy::ones_like(static_cast<double*>(result.request().ptr),
                     result.request().size);
    return result;
}

/// numpy.full(shape, fill_value, dtype=float, order='C')
inline py::array_t<double> full(const std::vector<py::ssize_t>& shape,
                                 double fill_value) {
    py::array_t<double> result(shape);
    numpy::full(static_cast<double*>(result.request().ptr),
                result.request().size, fill_value);
    return result;
}

// ── Bool specialisations ──────────────────────────────────────────────────────

/// numpy.full_like(arr, bool_val) — bool fill
inline py::array_t<bool> full_like(const py::array_t<double>& arr,
                                    bool fill_value) {
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    std::fill_n(static_cast<bool*>(result.request().ptr), buf.size, fill_value);
    return result;
}

/// numpy.zeros_like(arr, dtype='bool')
inline py::array zeros_like(const py::array& arr, const std::string& dtype) {
    auto buf = arr.request();
    if (dtype == "bool") {
        py::array_t<bool> result(buf.shape);
        std::fill_n(static_cast<bool*>(result.request().ptr), buf.size, false);
        return result;
    }
    throw std::runtime_error("zeros_like: unsupported dtype: " + dtype);
}

/// numpy.ones_like(arr, dtype='bool')
inline py::array ones_like(const py::array& arr, const std::string& dtype) {
    auto buf = arr.request();
    if (dtype == "bool") {
        py::array_t<bool> result(buf.shape);
        std::fill_n(static_cast<bool*>(result.request().ptr), buf.size, true);
        return result;
    }
    throw std::runtime_error("ones_like: unsupported dtype: " + dtype);
}

} // namespace numpy
