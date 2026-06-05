// Native C++ linalg functions — zero pybind11 dependency.
// Uses numpy:: helpers from core.h for shared computation.

#pragma once

#include "core.h"
#include <cmath>

namespace numpy {
namespace linalg {

/// numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) — vector / Frobenius
//  np.linalg.norm(a) internally computes sqrt(a.dot(a)) via BLAS sdot/ddot.
//  We call the same OpenBLAS routine (auto-discovered) for bit-exact match.
template<typename T>
inline T norm(const T* data, size_t n) {
    return numpy::detail::blas_ops<T>::norm(data, n);
}

/// numpy.linalg.norm(x, ord=None, axis=N, keepdims=False) — N-D
template<typename T>
inline void norm_axis(const T* src, T* dst, const ptrdiff_t* shape, int ndim, int axis) {
    numpy::norm_axis(src, dst, shape, ndim, axis);
}

} // namespace linalg
} // namespace numpy
