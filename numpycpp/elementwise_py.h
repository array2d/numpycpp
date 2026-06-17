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
#include <type_traits>
#include "detail/buffer_dtype.h"

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

/// ndarray.astype(dtype) — unified dtype dispatch.
/// 零 Python 调用: 仅通过 buf.format + itemsize (C API) 检测源类型；
/// 不使用 py::dtype::is() / kind()，避免 Python↔C++ 递归。
/// 目标必须显式指定精度: "float64"/"double", "float32", 不接受模糊的 "float"。
inline py::array astype(const py::array& arr, const std::string& dtype) {
    auto buf = arr.request();                     // C API — 零 Python
    auto src = detail::analyze_buffer(buf);       // 纯 C++ — 零 Python
    auto dst = detail::target_dtype(dtype);

    if (dst.kind == detail::BufKind::UNKNOWN) {
        auto dt = arr.dtype();  // 仅错误路径: 获取 dtype 名称用于报错
        throw std::runtime_error(
            "astype: unsupported target dtype '" + dtype + "'.  "
            "Available: float64/double, float32, int/int32, int64, bool.  "
            "Source: " + std::string(py::str(dt)));
    }

    // dispatch: 源类型 × 目标类型 → 调用对应模板实例
    switch (src.kind) {
    case detail::BufKind::FLOAT64:
        switch (dst.kind) {
        case detail::BufKind::FLOAT64: { py::array_t<double> r(buf.shape); numpy::astype<double,double>(static_cast<const double*>(buf.ptr),static_cast<double*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::FLOAT32: { py::array_t<float>  r(buf.shape); numpy::astype<float, double>(static_cast<const double*>(buf.ptr),static_cast<float*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::INT32:   { py::array_t<int>    r(buf.shape); numpy::astype<int,   double>(static_cast<const double*>(buf.ptr),static_cast<int*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::INT64:   { py::array_t<int64_t>r(buf.shape); numpy::astype<int64_t,double>(static_cast<const double*>(buf.ptr),static_cast<int64_t*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::BOOL:    { py::array_t<bool>   r(buf.shape); numpy::astype<bool,  double>(static_cast<const double*>(buf.ptr),static_cast<bool*>(r.request().ptr),buf.size); return r; }
        default: break;
        }
        break;
    case detail::BufKind::FLOAT32:
        switch (dst.kind) {
        case detail::BufKind::FLOAT64: { py::array_t<double> r(buf.shape); numpy::astype<double,float>(static_cast<const float*>(buf.ptr),static_cast<double*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::FLOAT32: { py::array_t<float>  r(buf.shape); numpy::astype<float, float>(static_cast<const float*>(buf.ptr),static_cast<float*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::INT32:   { py::array_t<int>    r(buf.shape); numpy::astype<int,   float>(static_cast<const float*>(buf.ptr),static_cast<int*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::INT64:   { py::array_t<int64_t>r(buf.shape); numpy::astype<int64_t,float>(static_cast<const float*>(buf.ptr),static_cast<int64_t*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::BOOL:    { py::array_t<bool>   r(buf.shape); numpy::astype<bool,  float>(static_cast<const float*>(buf.ptr),static_cast<bool*>(r.request().ptr),buf.size); return r; }
        default: break;
        }
        break;
    case detail::BufKind::INT32:
        switch (dst.kind) {
        case detail::BufKind::FLOAT64: { py::array_t<double> r(buf.shape); numpy::astype<double,int>(static_cast<const int*>(buf.ptr),static_cast<double*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::FLOAT32: { py::array_t<float>  r(buf.shape); numpy::astype<float, int>(static_cast<const int*>(buf.ptr),static_cast<float*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::INT32:   { py::array_t<int>    r(buf.shape); numpy::astype<int,   int>(static_cast<const int*>(buf.ptr),static_cast<int*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::INT64:   { py::array_t<int64_t>r(buf.shape); numpy::astype<int64_t,int>(static_cast<const int*>(buf.ptr),static_cast<int64_t*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::BOOL:    { py::array_t<bool>   r(buf.shape); numpy::astype<bool,  int>(static_cast<const int*>(buf.ptr),static_cast<bool*>(r.request().ptr),buf.size); return r; }
        default: break;
        }
        break;
    case detail::BufKind::INT64:
        switch (dst.kind) {
        case detail::BufKind::FLOAT64: { py::array_t<double> r(buf.shape); numpy::astype<double,int64_t>(static_cast<const int64_t*>(buf.ptr),static_cast<double*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::FLOAT32: { py::array_t<float>  r(buf.shape); numpy::astype<float, int64_t>(static_cast<const int64_t*>(buf.ptr),static_cast<float*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::INT32:   { py::array_t<int>    r(buf.shape); numpy::astype<int,   int64_t>(static_cast<const int64_t*>(buf.ptr),static_cast<int*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::INT64:   { py::array_t<int64_t>r(buf.shape); numpy::astype<int64_t,int64_t>(static_cast<const int64_t*>(buf.ptr),static_cast<int64_t*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::BOOL:    { py::array_t<bool>   r(buf.shape); numpy::astype<bool,  int64_t>(static_cast<const int64_t*>(buf.ptr),static_cast<bool*>(r.request().ptr),buf.size); return r; }
        default: break;
        }
        break;
    case detail::BufKind::BOOL:
        switch (dst.kind) {
        case detail::BufKind::FLOAT64: { py::array_t<double> r(buf.shape); numpy::astype<double,bool>(static_cast<const bool*>(buf.ptr),static_cast<double*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::FLOAT32: { py::array_t<float>  r(buf.shape); numpy::astype<float, bool>(static_cast<const bool*>(buf.ptr),static_cast<float*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::INT32:   { py::array_t<int>    r(buf.shape); numpy::astype<int,   bool>(static_cast<const bool*>(buf.ptr),static_cast<int*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::INT64:   { py::array_t<int64_t>r(buf.shape); numpy::astype<int64_t,bool>(static_cast<const bool*>(buf.ptr),static_cast<int64_t*>(r.request().ptr),buf.size); return r; }
        case detail::BufKind::BOOL:    { py::array_t<bool>   r(buf.shape); numpy::astype<bool,  bool>(static_cast<const bool*>(buf.ptr),static_cast<bool*>(r.request().ptr),buf.size); return r; }
        default: break;
        }
        break;
    default:
        break;
    }

    auto dt = arr.dtype();  // 仅错误路径
    throw std::runtime_error(
        "astype: unsupported conversion " + std::string(py::str(dt)) +
        " -> " + dtype + ".  Available targets: float64/double, float, "
        "float32, int/int32, int64, bool.");
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
