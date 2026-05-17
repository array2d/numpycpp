// Native C++ linalg functions — zero pybind11 dependency.
// Uses numpy:: helpers from core.h for shared computation.

#pragma once

#include "core.h"
#include <cmath>

namespace numpy {
namespace linalg {

/// numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) — frobenius/vector
template<typename T>
inline T norm(const T* data, size_t n) {
    return std::sqrt(numpy::norm_sq(data, n));
}

/// numpy.linalg.norm(x, ord=None, axis=N, keepdims=False) — N-D
template<typename T>
inline void norm_axis(const T* src, double* dst, const ptrdiff_t* shape, int ndim, int axis) {
    numpy::norm_axis(src, dst, shape, ndim, axis);
}

} // namespace linalg
} // namespace numpy
