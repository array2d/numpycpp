// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — pycpp/init_py.h                           [PUBLIC HEADER]
//  Pybind11 wrappers: array creation / initialisation.
//      numpy.zeros_like  numpy.ones_like  numpy.full_like  numpy.empty_like
//      numpy.zeros       numpy.ones       numpy.full       numpy.empty
//      numpy.arange      numpy.linspace   numpy.logspace   numpy.geomspace
//      numpy.eye         numpy.identity   numpy.diag
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "numpy.h"
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

// ── numpy.empty ──────────────────────────────────────────────────────────────

/// numpy.empty(shape, dtype=float64)
inline py::array_t<double> empty(const std::vector<py::ssize_t>& shape) {
    return py::array_t<double>(shape);
}

// ── numpy.arange ─────────────────────────────────────────────────────────────

/// arange<T> — internal typed helper
template<typename T>
py::array_t<T> arange(T start, T stop, T step) {
    size_t n = numpy::arange_size(start, stop, step);
    py::array_t<T> result(static_cast<py::ssize_t>(n));
    numpy::arange(static_cast<T*>(result.request().ptr), start, stop, step);
    return result;
}

/// numpy.arange([start,] stop[, step])
/// Returns float64 array (numpy default).  Integer stop/step are cast to double.
inline py::array arange(py::object a, py::object b = py::none(),
                         py::object c = py::none()) {
    double start, stop, step;
    if (b.is_none()) {
        start = 0.0;  stop = a.cast<double>();  step = 1.0;
    } else if (c.is_none()) {
        start = a.cast<double>();  stop = b.cast<double>();  step = 1.0;
    } else {
        start = a.cast<double>();  stop = b.cast<double>();  step = c.cast<double>();
    }
    return arange<double>(start, stop, step);
}

// ── numpy.linspace ───────────────────────────────────────────────────────────

/// linspace<T> — internal typed helper
template<typename T>
py::array_t<T> linspace(T start, T stop, py::ssize_t num,
                         bool endpoint = true) {
    if (num < 0) throw std::invalid_argument("linspace: num must be >= 0");
    py::array_t<T> result(num);
    numpy::linspace(static_cast<T*>(result.request().ptr),
                    start, stop, static_cast<size_t>(num), endpoint);
    return result;
}

/// numpy.linspace(start, stop, num=50, endpoint=True)
/// Returns float64 array (numpy default for scalar inputs).
inline py::array linspace(py::object start_o, py::object stop_o,
                           py::ssize_t num = 50, bool endpoint = true) {
    double start = start_o.cast<double>(), stop = stop_o.cast<double>();
    return linspace<double>(start, stop, num, endpoint);
}

// ── numpy.logspace ───────────────────────────────────────────────────────────

/// logspace<T> — internal typed helper
template<typename T>
py::array_t<T> logspace(T start, T stop, py::ssize_t num,
                         bool endpoint = true, T base = T(10)) {
    if (num < 0) throw std::invalid_argument("logspace: num must be >= 0");
    py::array_t<T> result(num);
    numpy::logspace(static_cast<T*>(result.request().ptr),
                    start, stop, static_cast<size_t>(num), endpoint, base);
    return result;
}

/// numpy.logspace(start, stop, num=50, endpoint=True, base=10.0)
/// Returns float64 array (numpy default for scalar inputs).
inline py::array logspace(py::object start_o, py::object stop_o,
                           py::ssize_t num = 50, bool endpoint = true,
                           double base = 10.0) {
    double start = start_o.cast<double>(), stop = stop_o.cast<double>();
    return logspace<double>(start, stop, num, endpoint, base);
}

// ── numpy.geomspace ──────────────────────────────────────────────────────────

/// geomspace<T> — internal typed helper
template<typename T>
py::array_t<T> geomspace(T start, T stop, py::ssize_t num,
                          bool endpoint = true) {
    if (num < 0) throw std::invalid_argument("geomspace: num must be >= 0");
    if (start == T(0) || stop == T(0))
        throw std::invalid_argument("geomspace: start and stop must be non-zero");
    py::array_t<T> result(num);
    numpy::geomspace(static_cast<T*>(result.request().ptr),
                     start, stop, static_cast<size_t>(num), endpoint);
    return result;
}

/// numpy.geomspace(start, stop, num=50, endpoint=True)
/// Returns float64 array (numpy default for scalar inputs).
inline py::array geomspace(py::object start_o, py::object stop_o,
                            py::ssize_t num = 50, bool endpoint = true) {
    double start = start_o.cast<double>(), stop = stop_o.cast<double>();
    return geomspace<double>(start, stop, num, endpoint);
}

// ── numpy.eye ────────────────────────────────────────────────────────────────

/// numpy.eye(N, M=None, k=0, dtype=float64) — M=None 表示方阵，严格对齐 numpy API
template<typename T>
py::array_t<T> eye(py::ssize_t N, py::object M_obj = py::none(), int k = 0) {
    if (N < 0) throw std::invalid_argument("eye: N must be >= 0");
    size_t Ns = static_cast<size_t>(N);
    py::ssize_t M_val = M_obj.is_none() ? N : M_obj.cast<py::ssize_t>();
    if (M_val < 0) throw std::invalid_argument("eye: M must be >= 0");
    size_t Ms = static_cast<size_t>(M_val);
    py::array_t<T> result({N, M_val});
    numpy::eye(static_cast<T*>(result.request().ptr), Ns, Ms, k);
    return result;
}

inline py::array_t<double> eye(py::ssize_t N, py::object M_obj = py::none(),
                                int k = 0) {
    return eye<double>(N, M_obj, k);
}

// ── numpy.identity ───────────────────────────────────────────────────────────

/// numpy.identity(n, dtype=float64)
template<typename T>
py::array_t<T> identity(py::ssize_t n) {
    if (n < 0) throw std::invalid_argument("identity: n must be >= 0");
    size_t ns = static_cast<size_t>(n);
    py::array_t<T> result({n, n});
    numpy::identity(static_cast<T*>(result.request().ptr), ns);
    return result;
}

inline py::array_t<double> identity(py::ssize_t n) {
    return identity<double>(n);
}

// ── numpy.diag ───────────────────────────────────────────────────────────────

/// numpy.diag(v, k=0)
///   1-D input → 2-D diagonal matrix
///   2-D input → 1-D diagonal extraction
template<typename T>
py::array_t<T> diag(const py::array_t<T>& v, int k = 0) {
    auto buf = v.request();
    if (buf.ndim == 1) {
        size_t n = static_cast<size_t>(buf.shape[0]);
        size_t N = n + static_cast<size_t>(std::abs(k));
        py::array_t<T> result({static_cast<py::ssize_t>(N),
                                static_cast<py::ssize_t>(N)});
        numpy::diag_from_vec(static_cast<T*>(result.request().ptr),
                             static_cast<const T*>(buf.ptr), n, k);
        return result;
    }
    if (buf.ndim == 2) {
        size_t rows = static_cast<size_t>(buf.shape[0]);
        size_t cols = static_cast<size_t>(buf.shape[1]);
        size_t len  = numpy::diag_size(rows, cols, k);
        py::array_t<T> result(static_cast<py::ssize_t>(len));
        numpy::diag_from_mat(static_cast<T*>(result.request().ptr),
                             static_cast<const T*>(buf.ptr), rows, cols, k);
        return result;
    }
    throw std::invalid_argument("diag: input must be 1-D or 2-D");
}

} // namespace numpy
