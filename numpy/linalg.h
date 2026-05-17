// Python Source: numpy/linalg/ (numpy.linalg module)
// Line Range: numpy.linalg.norm, numpy.dot
// Alignment: strict

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

namespace numpy {
namespace linalg {

// Python: numpy.linalg.norm(arr)
double norm(const py::array_t<double> &arr);

// Python: numpy.linalg.norm(arr) — float32 input.
// Internal computation in float32 to match Python's precision path,
// then promoted to double for return (matching numpy convention).
double norm(const py::array_t<float> &arr);

// Python: numpy.linalg.norm(arr, axis=1)
py::array_t<double> norm_axis1(const py::array_t<double> &arr);

} // namespace linalg

// Python: numpy.dot(a, b)
double dot(const py::array_t<double> &a, const py::array_t<double> &b);

} // namespace numpy
