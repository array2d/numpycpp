// Python Source: numpy/core/einsumfunc.py (numpy.einsum)
// Line Range: numpy.einsum public API
// Alignment: strict

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

namespace numpy {

// Python: numpy.einsum(subscripts, *operands)
//
// Einstein summation convention. Supports explicit mode (with '->')
// and implicit mode (without '->').
//
// Supported patterns:
//   "ij,ij->i"    - row-wise dot product
//   "ij,jk->ik"   - matrix multiplication
//   "bij,bjk->bik" - batch matrix multiplication
//   "aij,aij->ai"  - batch row-wise dot product
//   "ijk,mkl->mjl" - broadcasting with contraction
//
// Args:
//   subscripts: Einstein summation subscript string
//   a: first operand
//   b: second operand
//
// Returns:
//   Result array with shape determined by output subscripts
py::array_t<double> einsum(const std::string& subscripts,
                           const py::array_t<double>& a,
                           const py::array_t<double>& b);

} // namespace numpy
