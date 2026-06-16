// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — pycpp/elementwise_py.h                    [PUBLIC HEADER]
//  Pybind11 wrappers: element-wise operations and type conversion.
//      Unary: sqrt abs exp log sin cos tan cbrt expm1 log1p log10 log2
//             arcsin arccos arctan round floor ceil degrees radians sign
//             reciprocal power clip
//      Binary: hypot arctan2 maximum minimum
//      Comparison: greater less equal greater_equal less_equal not_equal
//      Logical: logical_and logical_or logical_not logical_xor
//      Special: isnan isinf isfinite
//      Type conversion: astype  truncate_to_float32
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "numpy.h"
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;

namespace numpy {

// ============================================================================
// Unary element-wise — template macro
// ============================================================================
#define DEF_ELEMWISE(name) \
template<typename T> \
py::array_t<T> name(const py::array_t<T>& arr) { \
    auto buf = arr.request(); \
    py::array_t<T> result(buf.shape); \
    numpy::name(static_cast<const T*>(buf.ptr), \
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
DEF_ELEMWISE(cbrt)
DEF_ELEMWISE(expm1)
DEF_ELEMWISE(log1p)
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
DEF_ELEMWISE(reciprocal)
#undef DEF_ELEMWISE

/// numpy.power(x1, x2) — scalar exponent
template<typename T>
py::array_t<T> power(const py::array_t<T>& arr, T exponent) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    numpy::power(static_cast<const T*>(buf.ptr),
                 static_cast<T*>(result.request().ptr), buf.size, exponent);
    return result;
}

/// numpy.clip(a, a_min, a_max)
template<typename T>
py::array_t<T> clip(const py::array_t<T>& arr, T min_val, T max_val) {
    auto buf = arr.request();
    py::array_t<T> result(buf.shape);
    numpy::clip(static_cast<const T*>(buf.ptr),
                static_cast<T*>(result.request().ptr), buf.size, min_val, max_val);
    return result;
}

// ============================================================================
// Binary element-wise
// ============================================================================

/// numpy.hypot(x1, x2)
template<typename T>
py::array_t<T> hypot(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<T> result(ba.shape);
    numpy::hypot(static_cast<const T*>(ba.ptr), static_cast<const T*>(bb.ptr),
                 static_cast<T*>(result.request().ptr), std::min(ba.size, bb.size));
    return result;
}

/// numpy.arctan2(x1, x2) — array-array
template<typename T>
py::array_t<T> arctan2(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<T> result(ba.shape);
    numpy::arctan2(static_cast<const T*>(ba.ptr), static_cast<const T*>(bb.ptr),
                   static_cast<T*>(result.request().ptr), std::min(ba.size, bb.size));
    return result;
}

/// numpy.arctan2(x1, x2) — array-scalar
template<typename T>
py::array_t<T> arctan2(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<T> result(buf.shape);
    numpy::arctan2(static_cast<const T*>(buf.ptr),
                   static_cast<T*>(result.request().ptr), buf.size, b);
    return result;
}

/// numpy.maximum(x1, x2) — array-array
template<typename T>
py::array_t<T> maximum(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<T> result(ba.shape);
    numpy::maximum(static_cast<const T*>(ba.ptr), static_cast<const T*>(bb.ptr),
                   static_cast<T*>(result.request().ptr), std::min(ba.size, bb.size));
    return result;
}

/// numpy.maximum(x1, x2) — array-scalar
template<typename T>
py::array_t<T> maximum(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<T> result(buf.shape);
    numpy::maximum(static_cast<const T*>(buf.ptr),
                   static_cast<T*>(result.request().ptr), buf.size, b);
    return result;
}

/// numpy.minimum(x1, x2) — array-array
template<typename T>
py::array_t<T> minimum(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<T> result(ba.shape);
    numpy::minimum(static_cast<const T*>(ba.ptr), static_cast<const T*>(bb.ptr),
                   static_cast<T*>(result.request().ptr), std::min(ba.size, bb.size));
    return result;
}

/// numpy.minimum(x1, x2) — array-scalar
template<typename T>
py::array_t<T> minimum(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<T> result(buf.shape);
    numpy::minimum(static_cast<const T*>(buf.ptr),
                   static_cast<T*>(result.request().ptr), buf.size, b);
    return result;
}

// ============================================================================
// Comparison — T → bool
// ============================================================================
#define DEF_COMPARE(name) \
template<typename T> \
py::array_t<bool> name(const py::array_t<T>& a, T b) { \
    auto buf = a.request(); \
    py::array_t<bool> result(buf.shape); \
    numpy::name(static_cast<const T*>(buf.ptr), \
                static_cast<bool*>(result.request().ptr), buf.size, b); \
    return result; \
}

DEF_COMPARE(greater)
DEF_COMPARE(less)
DEF_COMPARE(equal)
DEF_COMPARE(greater_equal)
DEF_COMPARE(less_equal)
#undef DEF_COMPARE

/// numpy.not_equal(x1, x2) — scalar
template<typename T>
py::array_t<bool> not_equal(const py::array_t<T>& a, T b) {
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    numpy::not_equal_scalar(static_cast<const T*>(buf.ptr),
                            static_cast<bool*>(result.request().ptr), buf.size, b);
    return result;
}

/// numpy.not_equal(x1, x2) — array
template<typename T>
py::array_t<bool> not_equal(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<bool> result(ba.shape);
    numpy::not_equal_array(static_cast<const T*>(ba.ptr),
                           static_cast<const T*>(bb.ptr),
                           static_cast<bool*>(result.request().ptr),
                           std::min(ba.size, bb.size));
    return result;
}

// ============================================================================
// Logical — bool → bool
// ============================================================================

inline py::array_t<bool> logical_and(const py::array_t<bool>& a,
                                      const py::array_t<bool>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<bool> result(ba.shape);
    numpy::logical_and(static_cast<const bool*>(ba.ptr),
                       static_cast<const bool*>(bb.ptr),
                       static_cast<bool*>(result.request().ptr),
                       std::min(ba.size, bb.size));
    return result;
}

inline py::array_t<bool> logical_or(const py::array_t<bool>& a,
                                     const py::array_t<bool>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<bool> result(ba.shape);
    numpy::logical_or(static_cast<const bool*>(ba.ptr),
                      static_cast<const bool*>(bb.ptr),
                      static_cast<bool*>(result.request().ptr),
                      std::min(ba.size, bb.size));
    return result;
}

inline py::array_t<bool> logical_not(const py::array_t<bool>& a) {
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    numpy::logical_not(static_cast<const bool*>(buf.ptr),
                       static_cast<bool*>(result.request().ptr), buf.size);
    return result;
}

inline py::array_t<bool> logical_xor(const py::array_t<bool>& a,
                                      const py::array_t<bool>& b) {
    auto ba = a.request(), bb = b.request();
    py::array_t<bool> result(ba.shape);
    numpy::logical_xor(static_cast<const bool*>(ba.ptr),
                       static_cast<const bool*>(bb.ptr),
                       static_cast<bool*>(result.request().ptr),
                       std::min(ba.size, bb.size));
    return result;
}

// ============================================================================
// Special value tests — T → bool
// ============================================================================
#define DEF_SPECIAL(name) \
template<typename T> \
py::array_t<bool> name(const py::array_t<T>& arr) { \
    auto buf = arr.request(); \
    py::array_t<bool> result(buf.shape); \
    numpy::name(static_cast<const T*>(buf.ptr), \
                static_cast<bool*>(result.request().ptr), buf.size); \
    return result; \
}
DEF_SPECIAL(isnan)
DEF_SPECIAL(isinf)
DEF_SPECIAL(isfinite)
#undef DEF_SPECIAL

// ============================================================================
// Type conversion
// ============================================================================

/// ndarray.astype(dtype) — unified dtype dispatch
inline py::array astype(const py::array& arr, const std::string& dtype) {
    auto buf = arr.request();
    auto dt  = arr.dtype();

#define _ASTYPE_CASE(SrcT, dst_str, DstT) \
    if (dt.is(py::dtype::of<SrcT>()) && (dtype == dst_str)) { \
        py::array_t<DstT> r(buf.shape); \
        numpy::astype<DstT, SrcT>(static_cast<const SrcT*>(buf.ptr), \
                                   static_cast<DstT*>(r.request().ptr), buf.size); \
        return r; \
    }

    // float64
    _ASTYPE_CASE(double, "float64", double)  // 自转换
    _ASTYPE_CASE(double, "double",  double)
    _ASTYPE_CASE(double, "float32", float)
    _ASTYPE_CASE(double, "float",   float)
    _ASTYPE_CASE(double, "int",     int)
    _ASTYPE_CASE(double, "int32",   int)
    _ASTYPE_CASE(double, "int64",   int64_t)
    _ASTYPE_CASE(double, "bool",    bool)
    // float32
    _ASTYPE_CASE(float, "float64", double)
    _ASTYPE_CASE(float, "double",  double)
    _ASTYPE_CASE(float, "float",   double)   // numpy 约定: np.float32(1).astype(float) → float64
    _ASTYPE_CASE(float, "float32", float)    // 自转换: 无操作
    _ASTYPE_CASE(float, "int",     int)
    _ASTYPE_CASE(float, "int32",   int)
    _ASTYPE_CASE(float, "int64",   int64_t)
    _ASTYPE_CASE(float, "bool",    bool)
    // int32
    _ASTYPE_CASE(int, "int32",   int)     // 自转换
    _ASTYPE_CASE(int, "int",     int)
    _ASTYPE_CASE(int, "float64", double)
    _ASTYPE_CASE(int, "double",  double)
    _ASTYPE_CASE(int, "float32", float)
    _ASTYPE_CASE(int, "float",   float)
    _ASTYPE_CASE(int, "int64",   int64_t)
    _ASTYPE_CASE(int, "bool",    bool)
    // int64
    _ASTYPE_CASE(int64_t, "int64",   int64_t)  // 自转换
    _ASTYPE_CASE(int64_t, "float64", double)
    _ASTYPE_CASE(int64_t, "double",  double)
    _ASTYPE_CASE(int64_t, "float32", float)
    _ASTYPE_CASE(int64_t, "float",   float)
    _ASTYPE_CASE(int64_t, "int",     int)
    _ASTYPE_CASE(int64_t, "int32",   int)
    _ASTYPE_CASE(int64_t, "bool",    bool)
    // bool
    _ASTYPE_CASE(bool, "bool",    bool)    // 自转换
    _ASTYPE_CASE(bool, "float64", double)
    _ASTYPE_CASE(bool, "double",  double)
    _ASTYPE_CASE(bool, "float32", float)
    _ASTYPE_CASE(bool, "float",   float)
    _ASTYPE_CASE(bool, "int",     int)
    _ASTYPE_CASE(bool, "int32",   int)
    _ASTYPE_CASE(bool, "int64",   int64_t)
#undef _ASTYPE_CASE

    throw std::runtime_error(
        "astype: unsupported conversion " + std::string(py::str(dt)) +
        " -> " + dtype + ".  Available targets: float64/double, float32/float, "
        "int/int32, int64, bool.  Also accepts self-conversion (e.g. float32->float32).");
}

/// float64 → float32 → float64 roundtrip
inline py::array_t<double> truncate_to_float32(const py::array_t<double>& arr) {
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    numpy::truncate_to_float32(static_cast<const double*>(buf.ptr),
                               static_cast<double*>(result.mutable_data()),
                               buf.size);
    return result;
}

} // namespace numpy
