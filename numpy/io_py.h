// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — pycpp/io_py.h                             [PUBLIC HEADER]
//  Pybind11 wrappers: set operations, interpolation, conversion utilities.
//      numpy.isin         numpy.flatnonzero  numpy.intersect1d
//      numpy.interp       numpy.unwrap
//      numpy.asarray      numpy.array
//      array_get          to_vector
//      get_array / set_array  (dict helpers)
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../numpy/numpy.h"
#include <algorithm>
#include <vector>

namespace py = pybind11;

namespace numpy {

// ── Set operations ────────────────────────────────────────────────────────────

/// numpy.isin(element, test_elements) — double values
inline py::array_t<bool> isin(const py::array_t<double>& arr,
                               const std::vector<double>& values) {
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    numpy::isin(static_cast<const double*>(buf.ptr),
                static_cast<bool*>(result.request().ptr), buf.size,
                values.data(), values.size());
    return result;
}

/// numpy.isin(element, test_elements) — int values (auto-converted)
inline py::array_t<bool> isin(const py::array_t<double>& arr,
                               const std::vector<int>& values) {
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

/// numpy.intersect1d(ar1, ar2)
template<typename T>
py::array_t<T> intersect1d(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    auto inter = numpy::intersect1d(static_cast<const T*>(ba.ptr), ba.size,
                                    static_cast<const T*>(bb.ptr), bb.size);
    return py::array_t<T>(inter.size(), inter.data());
}

// ── Interpolation ─────────────────────────────────────────────────────────────

/// numpy.interp(x, xp, fp)
inline py::array_t<double> interp(const py::array_t<double>& x,
                                   const py::array_t<double>& xp,
                                   const py::array_t<double>& fp) {
    auto bx = x.request(), bxp = xp.request(), bfp = fp.request();
    py::array_t<double> result(bx.shape);
    numpy::interp(static_cast<const double*>(bx.ptr),
                  static_cast<double*>(result.request().ptr), bx.size,
                  static_cast<const double*>(bxp.ptr),
                  static_cast<const double*>(bfp.ptr), bxp.size);
    return result;
}

// ── Signal processing ─────────────────────────────────────────────────────────

/// numpy.unwrap(p, discont=pi) — 1-D
template<typename T>
py::array_t<T> unwrap(const py::array_t<T>& arr, T discont = T(M_PI)) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    numpy::unwrap(static_cast<const T*>(buf.ptr),
                  static_cast<T*>(result.request().ptr), buf.size, discont);
    return result;
}

// ── Conversion helpers ────────────────────────────────────────────────────────

/// numpy.asarray(a)
inline py::array_t<double> asarray(const std::vector<double>& vec) {
    return py::array_t<double>(vec.size(), vec.data());
}
inline py::array_t<double> asarray(const py::array_t<double>& arr) { return arr; }

/// numpy.array(a)
inline py::array_t<double> array(const std::vector<double>& vec) {
    return py::array_t<double>(vec.size(), vec.data());
}
inline py::array_t<double> array(const py::array_t<double>& arr) { return arr; }

// ── Array element access ──────────────────────────────────────────────────────

/// ndarray[idx] — 1-D float64 get
inline double array_get(const py::array_t<double>& arr, py::ssize_t idx) {
    auto buf = arr.request();
    if (idx < 0) idx = buf.size + idx;
    return (idx >= 0 && idx < buf.size)
        ? static_cast<const double*>(buf.ptr)[idx] : 0.0;
}

/// ndarray[i, j] — 2-D float64 get
inline double array_get(const py::array_t<double>& arr,
                         py::ssize_t i, py::ssize_t j) {
    auto buf = arr.request();
    if (buf.ndim != 2) return 0.0;
    py::ssize_t cols = buf.shape[1];
    if (i < 0) i += buf.shape[0];
    if (j < 0) j += cols;
    return (i >= 0 && i < buf.shape[0] && j >= 0 && j < cols)
        ? static_cast<const double*>(buf.ptr)[i * cols + j] : 0.0;
}

/// ndarray[idx] — 1-D bool get
inline bool array_get(const py::array_t<bool>& arr, py::ssize_t idx) {
    auto buf = arr.request();
    if (idx < 0) idx = buf.size + idx;
    return (idx >= 0 && idx < buf.size)
        ? static_cast<const bool*>(buf.ptr)[idx] : false;
}

/// ndarray[idx] — generic (returns double)
inline double array_get(const py::array& arr, py::ssize_t idx) {
    return arr[py::int_(idx)].cast<double>();
}

// ── to_vector ─────────────────────────────────────────────────────────────────

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

// ── Dict helpers ──────────────────────────────────────────────────────────────

inline py::array_t<double> get_array(const py::dict& d, const char* key) {
    return d[key].cast<py::array_t<double>>();
}

inline void set_array(py::dict& d, const char* key,
                      const py::array_t<double>& arr) {
    d[key] = arr;
}

} // namespace numpy
