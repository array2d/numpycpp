// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — numpy/init.h                              [PUBLIC HEADER]
//
//  Array initialisation / creation routines.
//
//      numpy.zeros_like   numpy.ones_like   numpy.full
//
//  Recommended entry point: #include "numpy/numpy.h"
//  Direct include is also valid for standalone use.
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstddef>
#include <algorithm>

namespace numpy {

/// numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
template<typename T>
inline void zeros_like(T* dst, size_t n) {
    std::fill_n(dst, n, T(0));
}

/// numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)
template<typename T>
inline void ones_like(T* dst, size_t n) {
    std::fill_n(dst, n, T(1));
}

/// numpy.full(shape, fill_value, dtype=None, order='C')
template<typename T>
inline void full(T* dst, size_t n, T fill_value) {
    std::fill_n(dst, n, fill_value);
}

} // namespace numpy
