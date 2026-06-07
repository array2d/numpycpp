// Pybind11 wrappers for einsum native functions.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../numpy/numpy.h"
#include <vector>

namespace py = pybind11;

namespace numpy {

/// numpy.einsum(subscripts, *operands, out=None, dtype=None, order='K',
///              casting='safe', optimize=False)
//  Currently supports 2-operand patterns only.
template<typename T>
py::array_t<T> einsum(const std::string& subscripts,
                      const py::array_t<T>& a,
                      const py::array_t<T>& b) {
    auto bufa = a.request(), bufb = b.request();

    std::vector<ptrdiff_t> a_shape(bufa.shape.begin(), bufa.shape.end());
    std::vector<ptrdiff_t> b_shape(bufb.shape.begin(), bufb.shape.end());

    auto out_shape = einsum_detail::einsum_output_shape(
        subscripts, a_shape.data(), static_cast<int>(a_shape.size()),
        b_shape.data(), static_cast<int>(b_shape.size()));

    std::vector<py::ssize_t> py_out_shape(out_shape.begin(), out_shape.end());
    py::array_t<T> result(py_out_shape);

    einsum_detail::einsum(
        subscripts,
        static_cast<const T*>(bufa.ptr), a_shape.data(), static_cast<int>(a_shape.size()),
        static_cast<const T*>(bufb.ptr), b_shape.data(), static_cast<int>(b_shape.size()),
        static_cast<T*>(result.request().ptr));

    return result;
}

} // namespace numpy
