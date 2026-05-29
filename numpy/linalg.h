// Native C++ linalg functions — zero pybind11 dependency.
// Uses numpy:: helpers from core.h for shared computation.

#pragma once

#include "core.h"
#include <cmath>

namespace numpy {
namespace linalg {

/// numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) — frobenius/vector
//  numpy 1.23.5 uses x.dot(x) + sqrt in native type (NO double promotion).
//  For float32, dot() and sqrt() stay in float32.
template<typename T>
inline T norm(const T* data, size_t n) {
    T sqnorm = numpy::dot(data, data, n);  // dot product in native type
    return std::sqrt(sqnorm);
}

/// numpy.linalg.norm(x, ord=None, axis=N, keepdims=False) — N-D
template<typename T>
inline void norm_axis(const T* src, T* dst, const ptrdiff_t* shape, int ndim, int axis) {
    numpy::norm_axis(src, dst, shape, ndim, axis);
}

} // namespace linalg
} // namespace numpy
