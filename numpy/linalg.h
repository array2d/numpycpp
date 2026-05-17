// Native C++ linalg functions — zero pybind11 dependency.
// Uses numpy::impl:: helpers from core.h for shared computation.

#pragma once

#include "core.h"
#include <cmath>

namespace numpy {
namespace linalg {

template<typename T>
inline T norm(const T* data, size_t n) {
    return std::sqrt(numpy::norm_sq(data, n));
}

template<typename T>
inline void norm_axis1(const T* src, double* dst, size_t rows, size_t cols) {
    numpy::norm_axis1(src, dst, rows, cols);
}

} // namespace linalg
} // namespace numpy
