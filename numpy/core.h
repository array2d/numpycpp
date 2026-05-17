// Python Source: numpy/core/ (top-level numpy functions)
// Line Range: numpy public API
// Alignment: strict

#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <set>

namespace py = pybind11;

namespace numpy {

// ============================================================================
// Array creation
// Python: numpy.zeros, numpy.ones, numpy.full, numpy.empty, etc.
// ============================================================================
py::array_t<double> zeros_like(const py::array_t<double> &arr);
py::array_t<double> ones_like(const py::array_t<double> &arr);
py::array_t<double> full_like(const py::array_t<double> &arr, double fill_value);
py::array_t<bool> full_like_bool(const py::array_t<double> &arr, bool fill_value);
py::array_t<bool> zeros_like_bool(const py::array_t<double> &arr);
py::array_t<bool> ones_like_bool(const py::array_t<double> &arr);
py::array_t<double> zeros(const std::vector<py::ssize_t> &shape);
py::array_t<double> ones(const std::vector<py::ssize_t> &shape);
py::array_t<double> empty_like(const py::array_t<double> &arr);

// ============================================================================
// astype
// Python: ndarray.astype
// ============================================================================
py::array_t<int> astype_int(const py::array_t<double> &arr);
py::array_t<bool> astype_bool(const py::array_t<double> &arr);
py::array_t<bool> astype_bool_from_int(const py::array_t<int> &arr);

// Truncate double array to float32-equivalent values (double→float→double).
py::array_t<double> truncate_to_float32(const py::array_t<double> &arr);

// ============================================================================
// Math (element-wise)
// Python: numpy.sqrt, numpy.abs, numpy.exp, numpy.log, etc.
// ============================================================================
py::array_t<double> sqrt(const py::array_t<double> &arr);
py::array_t<double> abs(const py::array_t<double> &arr);
py::array_t<double> exp(const py::array_t<double> &arr);
py::array_t<double> log(const py::array_t<double> &arr);
py::array_t<double> sin(const py::array_t<double> &arr);
py::array_t<double> cos(const py::array_t<double> &arr);
py::array_t<double> tan(const py::array_t<double> &arr);
py::array_t<double> power(const py::array_t<double> &arr, double exponent);
py::array_t<double> clip(const py::array_t<double> &arr, double min_val, double max_val);

// ============================================================================
// Reduction
// Python: numpy.sum, numpy.mean, numpy.max, numpy.min, numpy.any, numpy.all
// ============================================================================
double sum(const py::array_t<double> &arr);
double mean(const py::array_t<double> &arr);
double max(const py::array_t<double> &arr);
double min(const py::array_t<double> &arr);
bool any(const py::array_t<bool> &arr);
bool all(const py::array_t<bool> &arr);

// ============================================================================
// Comparison (element-wise)
// Python: numpy.greater, numpy.less, numpy.equal, etc.
// ============================================================================
py::array_t<bool> greater(const py::array_t<double> &a, double b);
py::array_t<bool> less(const py::array_t<double> &a, double b);
py::array_t<bool> equal(const py::array_t<double> &a, double b);
py::array_t<bool> greater_equal(const py::array_t<double> &a, double b);
py::array_t<bool> less_equal(const py::array_t<double> &a, double b);

// ============================================================================
// Logical (element-wise)
// Python: numpy.logical_and, numpy.logical_or, numpy.logical_not
// ============================================================================
py::array_t<bool> logical_and(const py::array_t<bool> &a, const py::array_t<bool> &b);
py::array_t<bool> logical_or(const py::array_t<bool> &a, const py::array_t<bool> &b);
py::array_t<bool> logical_not(const py::array_t<bool> &a);

// ============================================================================
// Array access
// ============================================================================
double array_get(const py::array_t<double> &arr, py::ssize_t idx);
double array_get(const py::array_t<double> &arr, py::ssize_t i, py::ssize_t j);
bool array_get(const py::array_t<bool> &arr, py::ssize_t idx);
double array_get(const py::array &arr, py::ssize_t idx);

// ============================================================================
// Dict helpers
// ============================================================================
py::array_t<double> get_array(const py::dict &d, const char *key);
void set_array(py::dict &d, const char *key, const py::array_t<double> &arr);

// ============================================================================
// Conversion
// Python: numpy.asarray, numpy.array
// ============================================================================
py::array_t<double> asarray(const std::vector<double> &vec);
py::array_t<double> array(const std::vector<double> &vec);
py::array_t<double> asarray(const py::array_t<double> &arr);
py::array_t<double> array(const py::array_t<double> &arr);

// ============================================================================
// transpose and flatten
// Python: numpy.transpose (ndarray.T), ndarray.flatten()
// ============================================================================
py::array_t<double> transpose(const py::array_t<double> &arr);
py::array_t<double> flatten(const py::array_t<double> &arr);

// ============================================================================
// mean using Eigen
// Python: numpy.mean
// ============================================================================
py::array_t<double> mean(const py::array_t<double> &arr, int axis);

// ============================================================================
// to_vector helpers
// ============================================================================
std::vector<bool> to_vector(const py::array_t<bool> &arr);
std::vector<double> to_vector(const py::array_t<double> &arr);

// ============================================================================
// Slice helpers (axis=0)
// Python: arr[start:stop]
// ============================================================================
py::array_t<double> slice(const py::array_t<double> &arr, py::ssize_t start, py::ssize_t stop);
py::array slice(const py::array &arr, py::ssize_t start, py::ssize_t stop);

// ============================================================================
// Column slice helpers (axis=1)
// Python: arr[:, :n]
// ============================================================================
py::array_t<double> take_cols(const py::array_t<double> &arr, py::ssize_t n);
py::array_t<float> take_cols(const py::array_t<float> &arr, py::ssize_t n);

// ============================================================================
// Slice assignment helpers (1D)
// Python: arr[start:] = value
// ============================================================================
void slice_assign(py::array_t<double> arr, py::ssize_t start, double value);
void slice_assign(py::array_t<int> arr, py::ssize_t start, int value);
void slice_assign(py::array_t<bool> arr, py::ssize_t start, bool value);

// ============================================================================
// Additional math (element-wise)
// Python: numpy.log10, numpy.log2, numpy.arcsin, etc.
// ============================================================================
py::array_t<double> log10(const py::array_t<double> &arr);
py::array_t<double> log2(const py::array_t<double> &arr);
py::array_t<double> arcsin(const py::array_t<double> &arr);
py::array_t<double> arccos(const py::array_t<double> &arr);
py::array_t<double> arctan(const py::array_t<double> &arr);
py::array_t<double> round(const py::array_t<double> &arr);
py::array_t<double> floor(const py::array_t<double> &arr);
py::array_t<double> ceil(const py::array_t<double> &arr);
py::array_t<double> degrees(const py::array_t<double> &arr);
py::array_t<double> radians(const py::array_t<double> &arr);
py::array_t<double> sign(const py::array_t<double> &arr);

// ============================================================================
// Binary element-wise
// Python: numpy.arctan2, numpy.maximum, numpy.minimum
// ============================================================================
py::array_t<double> arctan2(const py::array_t<double> &a, const py::array_t<double> &b);
py::array_t<double> arctan2(const py::array_t<double> &a, double b);
py::array_t<double> maximum(const py::array_t<double> &a, const py::array_t<double> &b);
py::array_t<double> maximum(const py::array_t<double> &a, double b);
py::array_t<double> minimum(const py::array_t<double> &a, const py::array_t<double> &b);
py::array_t<double> minimum(const py::array_t<double> &a, double b);

// ============================================================================
// Comparison helpers
// Python: numpy.not_equal
// ============================================================================
py::array_t<bool> not_equal(const py::array_t<double> &a, double b);
py::array_t<bool> not_equal(const py::array_t<double> &a, const py::array_t<double> &b);

// ============================================================================
// Special value helpers
// Python: numpy.isnan, numpy.isinf, numpy.isfinite
// ============================================================================
py::array_t<bool> isnan(const py::array_t<double> &arr);
py::array_t<bool> isinf(const py::array_t<double> &arr);
py::array_t<bool> isfinite(const py::array_t<double> &arr);

// ============================================================================
// Array manipulation
// Python: numpy.diff, numpy.stack, numpy.concatenate, numpy.where, etc.
// ============================================================================
// Python: numpy.diff(arr, n=1, axis=-1)
py::array_t<double> diff(const py::array_t<double> &arr, int n = 1, int axis = -1);
py::array_t<double> stack(const std::vector<py::array_t<double>> &arrays);
py::array_t<double> concatenate(const std::vector<py::array_t<double>> &arrays);
py::array_t<double> vstack(const std::vector<py::array_t<double>> &arrays);
py::array_t<double> hstack(const std::vector<py::array_t<double>> &arrays);
py::array_t<double> where(const py::array_t<bool> &condition, double x, double y);
py::array_t<double> where(const py::array_t<bool> &condition, const py::array_t<double> &x, const py::array_t<double> &y);

// ============================================================================
// Statistical
// Python: numpy.std, numpy.var
// ============================================================================
double std(const py::array_t<double> &arr);
double var(const py::array_t<double> &arr);

// ============================================================================
// Set operations
// Python: numpy.isin, numpy.intersect1d
// ============================================================================
py::array_t<bool> isin(const py::array_t<double> &arr, const std::vector<double> &values);
py::array_t<double> intersect1d(const py::array_t<double> &a, const py::array_t<double> &b);

// ============================================================================
// Interpolation
// Python: numpy.interp
// ============================================================================
py::array_t<double> interp(const py::array_t<double> &x, const py::array_t<double> &xp, const py::array_t<double> &fp);

// ============================================================================
// Logical XOR
// Python: numpy.logical_xor
// ============================================================================
py::array_t<bool> logical_xor(const py::array_t<bool> &a, const py::array_t<bool> &b);

// ============================================================================
// Sorting and indexing
// Python: numpy.argsort, numpy.argmax, numpy.argmin
// ============================================================================
py::array_t<py::ssize_t> argsort(const py::array_t<double> &arr);
py::ssize_t argmax(const py::array_t<double> &arr);
py::ssize_t argmin(const py::array_t<double> &arr);

// ============================================================================
// Array transformation
// Python: numpy.roll, numpy.flip, numpy.repeat, numpy.tile
// ============================================================================
py::array_t<double> roll(const py::array_t<double> &arr, py::ssize_t shift);
py::array_t<double> flip(const py::array_t<double> &arr);
py::array_t<double> repeat(const py::array_t<double> &arr, py::ssize_t repeats);
py::array_t<double> tile(const py::array_t<double> &arr, py::ssize_t reps);

// ============================================================================
// Safe division
// ============================================================================
double safe_divide(double a, double b, double default_val = 0.0);

} // namespace numpy
