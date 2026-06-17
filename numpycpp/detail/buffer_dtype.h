// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — detail/buffer_dtype.h                    [INTERNAL HEADER]
//
//  零 Python 调用的 dtype 检测工具。从 py::buffer_info (C API PyObject_GetBuffer
//  一次性获取的 C 结构体) 解析数组类型，不触发任何 Python 属性访问。
//
//  安全用于任何调用上下文——包括 Python→C++→C++ 的嵌套调用链。
//
//  用法:
//      auto buf = arr.request();
//      auto bd  = numpy::detail::analyze_buffer(buf);
//      if (bd.is_float32()) { ... }
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>

namespace numpy {
namespace detail {

enum class BufKind { FLOAT32, FLOAT64, INT32, INT64, BOOL, UNKNOWN };

struct BufDtype {
    BufKind kind;
    int     itemsize;
    char    fmt_char;  // raw format char after stripping byte-order prefix

    bool is_float32() const { return kind == BufKind::FLOAT32; }
    bool is_float64() const { return kind == BufKind::FLOAT64; }
    bool is_int32()   const { return kind == BufKind::INT32; }
    bool is_int64()   const { return kind == BufKind::INT64; }
    bool is_bool()    const { return kind == BufKind::BOOL; }
    bool is_float()   const { return is_float32() || is_float64(); }
};

/// 纯 C++ 解析 buffer_info::format + itemsize，零 Python 调用。
/// PEP 3118 format 字符: f=float, d=double, i/l=int, q=longlong, ?/b=bool.
/// 字节序前缀 @/=/</>/! 被自动剥离。
inline BufDtype analyze_buffer(const py::buffer_info& buf) {
    char c = buf.format.empty() ? '\0' : buf.format[0];
    if (c == '@' || c == '=' || c == '<' || c == '>' || c == '!')
        c = buf.format.size() > 1 ? buf.format[1] : '\0';

    int sz = static_cast<int>(buf.itemsize);
    BufKind kind = BufKind::UNKNOWN;

    switch (c) {
    case 'f':
        kind = (sz == 4) ? BufKind::FLOAT32
             : (sz == 8) ? BufKind::FLOAT64 : BufKind::UNKNOWN;
        break;
    case 'd':
        kind = (sz == 8) ? BufKind::FLOAT64 : BufKind::UNKNOWN;
        break;
    case 'i': case 'l':
        kind = (sz == 4) ? BufKind::INT32
             : (sz == 8) ? BufKind::INT64 : BufKind::UNKNOWN;
        break;
    case 'q':
        kind = (sz == 8) ? BufKind::INT64 : BufKind::UNKNOWN;
        break;
    case '?': case 'b':
        kind = BufKind::BOOL;
        break;
    default:
        break;
    }
    return {kind, sz, c};
}

/// 目标 dtype 字符串 → BufKind + itemsize。
/// 必须显式指定精度: "float64"/"double", "float32", "int"/"int32", "int64", "bool"。
/// 不支持模糊的 "float" — 避免 numpy float=float64 的隐式约定歧义。
inline BufDtype target_dtype(const std::string& dtype) {
    if (dtype == "float64" || dtype == "double")
        return {BufKind::FLOAT64, 8, 'd'};
    if (dtype == "float32")
        return {BufKind::FLOAT32, 4, 'f'};
    if (dtype == "int" || dtype == "int32")
        return {BufKind::INT32, 4, 'i'};
    if (dtype == "int64")
        return {BufKind::INT64, 8, 'q'};
    if (dtype == "bool")
        return {BufKind::BOOL, 1, '?'};
    return {BufKind::UNKNOWN, 0, '\0'};
}

} // namespace detail
} // namespace numpy
