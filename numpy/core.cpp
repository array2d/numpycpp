// Python Source: numpy/core/ (top-level numpy functions)
// Line Range: numpy public API
// Alignment: strict

#include "numpy/core.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace numpy {

// ============================================================================
// Array creation
// Python: numpy.zeros_like(arr)
// ============================================================================
py::array_t<double> zeros_like(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    std::fill_n(static_cast<double *>(result.request().ptr), buf.size, 0.0);
    return result;
}

py::array_t<double> ones_like(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    std::fill_n(static_cast<double *>(result.request().ptr), buf.size, 1.0);
    return result;
}

py::array_t<double> full_like(const py::array_t<double> &arr, double fill_value)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    std::fill_n(static_cast<double *>(result.request().ptr), buf.size, fill_value);
    return result;
}

py::array_t<bool> full_like_bool(const py::array_t<double> &arr, bool fill_value)
{
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    std::fill_n(static_cast<bool *>(result.request().ptr), buf.size, fill_value);
    return result;
}

py::array_t<bool> zeros_like_bool(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    std::fill_n(static_cast<bool *>(result.request().ptr), buf.size, false);
    return result;
}

py::array_t<bool> ones_like_bool(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    std::fill_n(static_cast<bool *>(result.request().ptr), buf.size, true);
    return result;
}

py::array_t<double> zeros(const std::vector<py::ssize_t> &shape)
{
    py::array_t<double> result(shape);
    std::fill_n(static_cast<double *>(result.request().ptr), result.request().size, 0.0);
    return result;
}

py::array_t<double> ones(const std::vector<py::ssize_t> &shape)
{
    py::array_t<double> result(shape);
    std::fill_n(static_cast<double *>(result.request().ptr), result.request().size, 1.0);
    return result;
}

py::array_t<double> empty_like(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    return py::array_t<double>(buf.shape);
}

// ============================================================================
// astype
// Python: ndarray.astype(int), ndarray.astype(bool)
// ============================================================================
py::array_t<int> astype_int(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<int> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    int *dst = static_cast<int *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = static_cast<int>(src[i]);
    return result;
}

py::array_t<bool> astype_bool(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = static_cast<bool>(src[i]);
    return result;
}

py::array_t<bool> astype_bool_from_int(const py::array_t<int> &arr)
{
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    int *src = static_cast<int *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = static_cast<bool>(src[i]);
    return result;
}

py::array_t<double> truncate_to_float32(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    std::vector<float> tmp(buf.size);
    const double *src = static_cast<const double *>(buf.ptr);
    std::transform(src, src + buf.size, tmp.begin(),
                   [](double v) { return static_cast<float>(v); });
    py::array_t<double> result(buf.shape);
    double *dst = static_cast<double *>(result.mutable_data());
    std::transform(tmp.begin(), tmp.end(), dst,
                   [](float v) { return static_cast<double>(v); });
    return result;
}

// ============================================================================
// Math (element-wise)
// Python: numpy.sqrt, numpy.abs, numpy.exp, etc.
// ============================================================================
py::array_t<double> sqrt(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::sqrt(src[i]);
    return result;
}

py::array_t<double> abs(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::abs(src[i]);
    return result;
}

py::array_t<double> exp(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::exp(src[i]);
    return result;
}

py::array_t<double> log(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::log(src[i]);
    return result;
}

py::array_t<double> sin(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::sin(src[i]);
    return result;
}

py::array_t<double> cos(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::cos(src[i]);
    return result;
}

py::array_t<double> tan(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::tan(src[i]);
    return result;
}

py::array_t<double> power(const py::array_t<double> &arr, double exponent)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::pow(src[i], exponent);
    return result;
}

py::array_t<double> clip(const py::array_t<double> &arr, double min_val, double max_val)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::max(min_val, std::min(max_val, src[i]));
    return result;
}

// ============================================================================
// Reduction
// Python: numpy.sum, numpy.mean, numpy.max, numpy.min, numpy.any, numpy.all
// ============================================================================
double sum(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    double *data = static_cast<double *>(buf.ptr);
    double total = 0.0;
    for (py::ssize_t i = 0; i < buf.size; ++i)
        total += data[i];
    return total;
}

double mean(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    if (buf.size == 0)
        return 0.0;
    return sum(arr) / buf.size;
}

double max(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    double *data = static_cast<double *>(buf.ptr);
    if (buf.size == 0)
        return 0.0;
    double max_val = data[0];
    for (py::ssize_t i = 1; i < buf.size; ++i)
        if (data[i] > max_val)
            max_val = data[i];
    return max_val;
}

double min(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    double *data = static_cast<double *>(buf.ptr);
    if (buf.size == 0)
        return 0.0;
    double min_val = data[0];
    for (py::ssize_t i = 1; i < buf.size; ++i)
        if (data[i] < min_val)
            min_val = data[i];
    return min_val;
}

bool any(const py::array_t<bool> &arr)
{
    auto buf = arr.request();
    bool *data = static_cast<bool *>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        if (data[i])
            return true;
    return false;
}

bool all(const py::array_t<bool> &arr)
{
    auto buf = arr.request();
    bool *data = static_cast<bool *>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        if (!data[i])
            return false;
    return true;
}

// ============================================================================
// Comparison (element-wise)
// Python: numpy.greater, numpy.less, numpy.equal, etc.
// ============================================================================
py::array_t<bool> greater(const py::array_t<double> &a, double b)
{
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = (src[i] > b);
    return result;
}

py::array_t<bool> less(const py::array_t<double> &a, double b)
{
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = (src[i] < b);
    return result;
}

py::array_t<bool> equal(const py::array_t<double> &a, double b)
{
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = (src[i] == b);
    return result;
}

py::array_t<bool> greater_equal(const py::array_t<double> &a, double b)
{
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = (src[i] >= b);
    return result;
}

py::array_t<bool> less_equal(const py::array_t<double> &a, double b)
{
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = (src[i] <= b);
    return result;
}

// ============================================================================
// Logical (element-wise)
// Python: numpy.logical_and, numpy.logical_or, numpy.logical_not
// ============================================================================
py::array_t<bool> logical_and(const py::array_t<bool> &a, const py::array_t<bool> &b)
{
    auto buf_a = a.request();
    auto buf_b = b.request();
    py::array_t<bool> result(buf_a.shape);
    bool *src_a = static_cast<bool *>(buf_a.ptr);
    bool *src_b = static_cast<bool *>(buf_b.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    py::ssize_t n = std::min(buf_a.size, buf_b.size);
    for (py::ssize_t i = 0; i < n; ++i)
        dst[i] = src_a[i] && src_b[i];
    return result;
}

py::array_t<bool> logical_or(const py::array_t<bool> &a, const py::array_t<bool> &b)
{
    auto buf_a = a.request();
    auto buf_b = b.request();
    py::array_t<bool> result(buf_a.shape);
    bool *src_a = static_cast<bool *>(buf_a.ptr);
    bool *src_b = static_cast<bool *>(buf_b.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    py::ssize_t n = std::min(buf_a.size, buf_b.size);
    for (py::ssize_t i = 0; i < n; ++i)
        dst[i] = src_a[i] || src_b[i];
    return result;
}

py::array_t<bool> logical_not(const py::array_t<bool> &a)
{
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    bool *src = static_cast<bool *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = !src[i];
    return result;
}

// ============================================================================
// Array access
// ============================================================================
double array_get(const py::array_t<double> &arr, py::ssize_t idx)
{
    auto buf = arr.request();
    double *data = static_cast<double *>(buf.ptr);
    if (idx < 0)
        idx = buf.size + idx;
    if (idx < 0 || idx >= buf.size)
        return 0.0;
    return data[idx];
}

double array_get(const py::array_t<double> &arr, py::ssize_t i, py::ssize_t j)
{
    auto buf = arr.request();
    if (buf.ndim != 2)
        return 0.0;
    double *data = static_cast<double *>(buf.ptr);
    py::ssize_t cols = buf.shape[1];
    if (i < 0)
        i = buf.shape[0] + i;
    if (j < 0)
        j = cols + j;
    if (i < 0 || i >= buf.shape[0] || j < 0 || j >= cols)
        return 0.0;
    return data[i * cols + j];
}

bool array_get(const py::array_t<bool> &arr, py::ssize_t idx)
{
    auto buf = arr.request();
    bool *data = static_cast<bool *>(buf.ptr);
    if (idx < 0)
        idx = buf.size + idx;
    if (idx < 0 || idx >= buf.size)
        return false;
    return data[idx];
}

double array_get(const py::array &arr, py::ssize_t idx)
{
    // Python: arr[idx] — works for any numeric dtype
    return arr[py::int_(idx)].cast<double>();
}

// ============================================================================
// Dict helpers
// ============================================================================
py::array_t<double> get_array(const py::dict &d, const char *key)
{
    return d[key].cast<py::array_t<double>>();
}

void set_array(py::dict &d, const char *key, const py::array_t<double> &arr)
{
    d[key] = arr;
}

// ============================================================================
// Conversion
// Python: numpy.asarray, numpy.array
// ============================================================================
py::array_t<double> asarray(const std::vector<double> &vec)
{
    return py::array_t<double>(vec.size(), vec.data());
}

py::array_t<double> array(const std::vector<double> &vec)
{
    return py::array_t<double>(vec.size(), vec.data());
}

py::array_t<double> asarray(const py::array_t<double> &arr)
{
    return arr;
}

py::array_t<double> array(const py::array_t<double> &arr)
{
    return arr;
}

// ============================================================================
// transpose and flatten
// Python: numpy.transpose, ndarray.flatten
// ============================================================================
py::array_t<double> transpose(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    if (buf.ndim == 1)
    {
        py::array_t<double> result(buf.shape);
        double *src = static_cast<double *>(buf.ptr);
        double *dst = static_cast<double *>(result.request().ptr);
        std::memcpy(dst, src, buf.size * sizeof(double));
        return result;
    }
    if (buf.ndim == 2)
    {
        py::ssize_t rows = buf.shape[0];
        py::ssize_t cols = buf.shape[1];
        std::vector<py::ssize_t> new_shape = {cols, rows};
        py::array_t<double> result(new_shape);
        double *src = static_cast<double *>(buf.ptr);
        double *dst = static_cast<double *>(result.request().ptr);
        for (py::ssize_t i = 0; i < rows; ++i)
            for (py::ssize_t j = 0; j < cols; ++j)
                dst[j * rows + i] = src[i * cols + j];
        return result;
    }
    std::vector<py::ssize_t> shape = {buf.size};
    py::array_t<double> result(shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    std::memcpy(dst, src, buf.size * sizeof(double));
    return result;
}

py::array_t<double> flatten(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    std::vector<py::ssize_t> shape = {buf.size};
    py::array_t<double> result(shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    std::memcpy(dst, src, buf.size * sizeof(double));
    return result;
}

// ============================================================================
// mean using Eigen
// Python: numpy.mean(arr, axis=...)
// ============================================================================
py::array_t<double> mean(const py::array_t<double> &arr, int axis)
{
    auto buf = arr.request();
    if (buf.ndim < 1 || buf.ndim > 3)
        throw std::invalid_argument("Array must be 1D, 2D, or 3D");

    Eigen::Map<const Eigen::MatrixXd> eigen_map(
        static_cast<double *>(buf.ptr),
        buf.shape[0],
        (buf.ndim == 1) ? 1 : buf.shape[1] * (buf.ndim == 2 ? 1 : buf.shape[2])
    );

    if (axis == -1)
    {
        double global_mean = eigen_map.mean();
        py::array_t<double> result({1});
        auto result_buf = result.request();
        double *result_ptr = static_cast<double *>(result_buf.ptr);
        result_ptr[0] = global_mean;
        return result;
    }
    else if (buf.ndim == 1)
    {
        if (axis != 0)
            throw std::invalid_argument("For 1D array, axis must be 0 or -1");
        double mean_val = eigen_map.mean();
        py::array_t<double> result({1});
        auto result_buf = result.request();
        double *result_ptr = static_cast<double *>(result_buf.ptr);
        result_ptr[0] = mean_val;
        return result;
    }
    else if (buf.ndim == 2)
    {
        if (axis == 0)
        {
            Eigen::VectorXd col_means = eigen_map.colwise().mean();
            std::vector<double> means(col_means.data(), col_means.data() + col_means.size());
            return py::array_t<double>(means.size(), means.data());
        }
        else if (axis == 1)
        {
            Eigen::VectorXd row_means = eigen_map.rowwise().mean();
            std::vector<double> means(row_means.data(), row_means.data() + row_means.size());
            return py::array_t<double>(means.size(), means.data());
        }
        else
        {
            throw std::invalid_argument("For 2D array, axis must be 0, 1, or -1");
        }
    }
    else if (buf.ndim == 3)
    {
        int rows = buf.shape[0];
        int cols = buf.shape[1] * buf.shape[2];
        Eigen::Map<const Eigen::MatrixXd> eigen_map_3d(
            static_cast<double *>(buf.ptr), rows, cols);

        if (axis == 0)
        {
            Eigen::MatrixXd means = eigen_map_3d.rowwise().mean();
            std::vector<py::ssize_t> new_shape = {1, buf.shape[1], buf.shape[2]};
            py::array_t<double> result(new_shape);
            auto result_buf = result.request();
            double *result_ptr = static_cast<double *>(result_buf.ptr);
            for (int i = 0; i < cols; ++i)
                result_ptr[i] = means(0, i);
            return result;
        }
        else if (axis == 1)
        {
            std::vector<py::ssize_t> new_shape = {buf.shape[0], 1, buf.shape[2]};
            py::array_t<double> result(new_shape);
            auto result_buf = result.request();
            double *result_ptr = static_cast<double *>(result_buf.ptr);
            for (int i = 0; i < buf.shape[0]; ++i)
            {
                Eigen::MatrixXd slice = eigen_map_3d.block(i, 0, 1, cols);
                Eigen::VectorXd slice_mean = slice.rowwise().mean();
                for (int j = 0; j < buf.shape[2]; ++j)
                    result_ptr[i * buf.shape[2] + j] = slice_mean[j];
            }
            return result;
        }
        else if (axis == 2)
        {
            Eigen::VectorXd means(rows);
            for (int i = 0; i < rows; ++i)
            {
                Eigen::VectorXd row = eigen_map_3d.row(i);
                means[i] = row.mean();
            }
            std::vector<py::ssize_t> new_shape = {buf.shape[0], buf.shape[1], 1};
            py::array_t<double> result(new_shape);
            auto result_buf = result.request();
            double *result_ptr = static_cast<double *>(result_buf.ptr);
            for (int i = 0; i < rows; ++i)
                result_ptr[i] = means[i];
            return result;
        }
        else
        {
            throw std::invalid_argument("For 3D array, axis must be 0, 1, 2, or -1");
        }
    }

    throw std::invalid_argument("Invalid array dimensions or axis");
}

// ============================================================================
// to_vector helpers
// ============================================================================
std::vector<bool> to_vector(const py::array_t<bool> &arr)
{
    auto buf = arr.request();
    bool *ptr = static_cast<bool *>(buf.ptr);
    return std::vector<bool>(ptr, ptr + buf.size);
}

std::vector<double> to_vector(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    double *ptr = static_cast<double *>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

// ============================================================================
// Slice helpers (axis=0)
// Python: arr[start:stop]
// ============================================================================
py::array_t<double> slice(const py::array_t<double> &arr, py::ssize_t start, py::ssize_t stop)
{
    auto buf = arr.request();
    if (buf.ndim < 1)
        return py::array_t<double>{};

    py::ssize_t dim0 = buf.shape[0];
    if (start < 0) start = 0;
    if (stop > dim0) stop = dim0;
    if (start >= stop)
        return py::array_t<double>{};

    py::ssize_t n = stop - start;
    py::ssize_t trailing = 1;
    for (int i = 1; i < buf.ndim; ++i)
        trailing *= buf.shape[i];

    std::vector<py::ssize_t> new_shape;
    new_shape.reserve(buf.ndim);
    new_shape.push_back(n);
    for (int i = 1; i < buf.ndim; ++i)
        new_shape.push_back(buf.shape[i]);

    py::array_t<double> result(new_shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    std::memcpy(dst, src + start * trailing, n * trailing * sizeof(double));
    return result;
}

py::array slice(const py::array &arr, py::ssize_t start, py::ssize_t stop)
{
    auto buf = arr.request();
    if (buf.ndim < 1)
        return py::array{};

    py::ssize_t dim0 = buf.shape[0];
    if (start < 0) start = 0;
    if (stop > dim0) stop = dim0;
    if (start >= stop)
        return py::array{};

    py::ssize_t n = stop - start;
    py::ssize_t trailing = 1;
    for (int i = 1; i < buf.ndim; ++i)
        trailing *= buf.shape[i];

    py::ssize_t itemsize = buf.itemsize;

    std::vector<py::ssize_t> new_shape;
    new_shape.reserve(buf.ndim);
    new_shape.push_back(n);
    for (int i = 1; i < buf.ndim; ++i)
        new_shape.push_back(buf.shape[i]);

    py::array result(arr.dtype(), new_shape);
    const char *src = static_cast<const char *>(buf.ptr);
    char *dst = static_cast<char *>(result.request().ptr);
    std::memcpy(dst, src + start * trailing * itemsize, n * trailing * itemsize);
    return result;
}

// ============================================================================
// Column slice helpers (axis=1)
// Python: arr[:, :n]
// ============================================================================
py::array_t<double> take_cols(const py::array_t<double> &arr, py::ssize_t n)
{
    auto buf = arr.request();
    if (buf.ndim != 2 || n > buf.shape[1])
        return arr.attr("copy")().cast<py::array_t<double>>();

    py::ssize_t rows = buf.shape[0];
    py::ssize_t src_cols = buf.shape[1];
    std::vector<py::ssize_t> new_shape = {rows, n};

    py::array_t<double> result(new_shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);

    for (py::ssize_t i = 0; i < rows; ++i)
    {
        for (py::ssize_t j = 0; j < n; ++j)
        {
            dst[i * n + j] = src[i * src_cols + j];
        }
    }
    return result;
}

py::array_t<float> take_cols(const py::array_t<float> &arr, py::ssize_t n)
{
    auto buf = arr.request();
    if (buf.ndim != 2 || n > buf.shape[1])
        return arr.attr("copy")().cast<py::array_t<float>>();

    py::ssize_t rows = buf.shape[0];
    py::ssize_t src_cols = buf.shape[1];
    std::vector<py::ssize_t> new_shape = {rows, n};

    py::array_t<float> result(new_shape);
    float *src = static_cast<float *>(buf.ptr);
    float *dst = static_cast<float *>(result.request().ptr);

    for (py::ssize_t i = 0; i < rows; ++i)
    {
        for (py::ssize_t j = 0; j < n; ++j)
        {
            dst[i * n + j] = src[i * src_cols + j];
        }
    }
    return result;
}

// ============================================================================
// Slice assignment helpers (1D)
// Python: arr[start:] = value
// ============================================================================
void slice_assign(py::array_t<double> arr, py::ssize_t start, double value)
{
    auto buf = arr.request();
    if (buf.ndim < 1 || start >= buf.shape[0])
        return;
    if (start < 0) start = 0;
    double *ptr = static_cast<double *>(buf.ptr);
    std::fill(ptr + start, ptr + buf.size, value);
}

void slice_assign(py::array_t<int> arr, py::ssize_t start, int value)
{
    auto buf = arr.request();
    if (buf.ndim < 1 || start >= buf.shape[0])
        return;
    if (start < 0) start = 0;
    int *ptr = static_cast<int *>(buf.ptr);
    std::fill(ptr + start, ptr + buf.size, value);
}

void slice_assign(py::array_t<bool> arr, py::ssize_t start, bool value)
{
    auto buf = arr.request();
    if (buf.ndim < 1 || start >= buf.shape[0])
        return;
    if (start < 0) start = 0;
    bool *ptr = static_cast<bool *>(buf.ptr);
    std::fill(ptr + start, ptr + buf.size, value);
}

// ============================================================================
// Additional math (element-wise)
// Python: numpy.log10, numpy.log2, numpy.arcsin, etc.
// ============================================================================
py::array_t<double> log10(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::log10(src[i]);
    return result;
}

py::array_t<double> log2(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::log2(src[i]);
    return result;
}

py::array_t<double> arcsin(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::asin(src[i]);
    return result;
}

py::array_t<double> arccos(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::acos(src[i]);
    return result;
}

py::array_t<double> arctan(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::atan(src[i]);
    return result;
}

py::array_t<double> round(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::round(src[i]);
    return result;
}

py::array_t<double> floor(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::floor(src[i]);
    return result;
}

py::array_t<double> ceil(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::ceil(src[i]);
    return result;
}

py::array_t<double> degrees(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = src[i] * 180.0 / M_PI;
    return result;
}

py::array_t<double> radians(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = src[i] * M_PI / 180.0;
    return result;
}

py::array_t<double> sign(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = (src[i] > 0) - (src[i] < 0);
    return result;
}

// ============================================================================
// Binary element-wise
// Python: numpy.arctan2, numpy.maximum, numpy.minimum
// ============================================================================
py::array_t<double> arctan2(const py::array_t<double> &a, const py::array_t<double> &b)
{
    auto buf_a = a.request(), buf_b = b.request();
    py::ssize_t size = std::min(buf_a.size, buf_b.size);
    py::array_t<double> result(buf_a.shape);
    double *pa = static_cast<double *>(buf_a.ptr);
    double *pb = static_cast<double *>(buf_b.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < size; ++i)
        dst[i] = std::atan2(pa[i], pb[i]);
    return result;
}

py::array_t<double> arctan2(const py::array_t<double> &a, double b)
{
    auto buf_a = a.request();
    py::array_t<double> result(buf_a.shape);
    double *pa = static_cast<double *>(buf_a.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf_a.size; ++i)
        dst[i] = std::atan2(pa[i], b);
    return result;
}

py::array_t<double> maximum(const py::array_t<double> &a, const py::array_t<double> &b)
{
    auto buf_a = a.request(), buf_b = b.request();
    py::ssize_t size = std::min(buf_a.size, buf_b.size);
    py::array_t<double> result(buf_a.shape);
    double *pa = static_cast<double *>(buf_a.ptr);
    double *pb = static_cast<double *>(buf_b.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < size; ++i)
        dst[i] = std::max(pa[i], pb[i]);
    return result;
}

py::array_t<double> maximum(const py::array_t<double> &a, double b)
{
    auto buf_a = a.request();
    py::array_t<double> result(buf_a.shape);
    double *pa = static_cast<double *>(buf_a.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf_a.size; ++i)
        dst[i] = std::max(pa[i], b);
    return result;
}

py::array_t<double> minimum(const py::array_t<double> &a, const py::array_t<double> &b)
{
    auto buf_a = a.request(), buf_b = b.request();
    py::ssize_t size = std::min(buf_a.size, buf_b.size);
    py::array_t<double> result(buf_a.shape);
    double *pa = static_cast<double *>(buf_a.ptr);
    double *pb = static_cast<double *>(buf_b.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < size; ++i)
        dst[i] = std::min(pa[i], pb[i]);
    return result;
}

py::array_t<double> minimum(const py::array_t<double> &a, double b)
{
    auto buf_a = a.request();
    py::array_t<double> result(buf_a.shape);
    double *pa = static_cast<double *>(buf_a.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf_a.size; ++i)
        dst[i] = std::min(pa[i], b);
    return result;
}

// ============================================================================
// Comparison helpers
// Python: numpy.not_equal
// ============================================================================
py::array_t<bool> not_equal(const py::array_t<double> &a, double b)
{
    auto buf = a.request();
    py::array_t<bool> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = (src[i] != b);
    return result;
}

py::array_t<bool> not_equal(const py::array_t<double> &a, const py::array_t<double> &b)
{
    auto buf_a = a.request(), buf_b = b.request();
    py::ssize_t size = std::min(buf_a.size, buf_b.size);
    py::array_t<bool> result(buf_a.shape);
    double *pa = static_cast<double *>(buf_a.ptr);
    double *pb = static_cast<double *>(buf_b.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < size; ++i)
        dst[i] = (pa[i] != pb[i]);
    return result;
}

// ============================================================================
// Special value helpers
// Python: numpy.isnan, numpy.isinf, numpy.isfinite
// ============================================================================
py::array_t<bool> isnan(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::isnan(src[i]);
    return result;
}

py::array_t<bool> isinf(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::isinf(src[i]);
    return result;
}

py::array_t<bool> isfinite(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = std::isfinite(src[i]);
    return result;
}

// ============================================================================
// Array manipulation
// Python: numpy.diff, numpy.stack, numpy.concatenate, numpy.where
// ============================================================================
py::array_t<double> diff(const py::array_t<double> &arr, int n, int axis)
{
    auto buf = arr.request();
    if (buf.ndim == 0 || buf.size < 2)
        return py::array_t<double>{};

    // 1D array: axis is always 0
    if (buf.ndim == 1) {
        std::vector<py::ssize_t> shape = {buf.size - 1};
        py::array_t<double> result(shape);
        const double *src = static_cast<const double *>(buf.ptr);
        double *dst = static_cast<double *>(result.request().ptr);
        for (py::ssize_t i = 0; i < buf.size - 1; ++i)
            dst[i] = src[i + 1] - src[i];
        return result;
    }

    // 2D array: handle axis
    if (buf.ndim == 2) {
        py::ssize_t rows = buf.shape[0];
        py::ssize_t cols = buf.shape[1];
        const double *src = static_cast<const double *>(buf.ptr);

        // Resolve axis: -1 -> last axis (1 for 2D)
        int ax = (axis == -1) ? 1 : axis;

        if (ax == 0) {
            // diff along rows: output shape (rows-1, cols)
            py::array_t<double> result({rows - 1, cols});
            double *dst = static_cast<double *>(result.request().ptr);
            for (py::ssize_t i = 0; i < rows - 1; ++i) {
                for (py::ssize_t j = 0; j < cols; ++j) {
                    dst[i * cols + j] = src[(i + 1) * cols + j] - src[i * cols + j];
                }
            }
            return result;
        } else {
            // diff along columns (axis=1): output shape (rows, cols-1)
            py::array_t<double> result({rows, cols - 1});
            double *dst = static_cast<double *>(result.request().ptr);
            for (py::ssize_t i = 0; i < rows; ++i) {
                for (py::ssize_t j = 0; j < cols - 1; ++j) {
                    dst[i * (cols - 1) + j] = src[i * cols + j + 1] - src[i * cols + j];
                }
            }
            return result;
        }
    }

    // Fallback for higher dims: flatten and diff
    std::vector<py::ssize_t> shape = {buf.size - 1};
    py::array_t<double> result(shape);
    const double *src = static_cast<const double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size - 1; ++i)
        dst[i] = src[i + 1] - src[i];
    return result;
}

py::array_t<double> stack(const std::vector<py::array_t<double>> &arrays)
{
    if (arrays.empty())
        return py::array_t<double>{};
    auto buf0 = arrays[0].request();
    py::ssize_t total_size = buf0.size * arrays.size();
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(arrays.size()), buf0.size};
    py::array_t<double> result(shape);
    double *dst = static_cast<double *>(result.request().ptr);
    py::ssize_t offset = 0;
    for (const auto &arr : arrays)
    {
        auto buf = arr.request();
        double *src = static_cast<double *>(buf.ptr);
        for (py::ssize_t i = 0; i < buf.size; ++i)
            dst[offset++] = src[i];
    }
    return result;
}

py::array_t<double> concatenate(const std::vector<py::array_t<double>> &arrays)
{
    if (arrays.empty())
        return py::array_t<double>{};
    py::ssize_t total_size = 0;
    for (const auto &arr : arrays)
        total_size += arr.request().size;
    std::vector<py::ssize_t> shape = {total_size};
    py::array_t<double> result(shape);
    double *dst = static_cast<double *>(result.request().ptr);
    py::ssize_t offset = 0;
    for (const auto &arr : arrays)
    {
        auto buf = arr.request();
        double *src = static_cast<double *>(buf.ptr);
        for (py::ssize_t i = 0; i < buf.size; ++i)
            dst[offset++] = src[i];
    }
    return result;
}

py::array_t<double> vstack(const std::vector<py::array_t<double>> &arrays)
{
    return stack(arrays);
}

py::array_t<double> hstack(const std::vector<py::array_t<double>> &arrays)
{
    return concatenate(arrays);
}

py::array_t<double> where(const py::array_t<bool> &condition, double x, double y)
{
    auto buf = condition.request();
    py::array_t<double> result(buf.shape);
    bool *cond = static_cast<bool *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = cond[i] ? x : y;
    return result;
}

py::array_t<double> where(const py::array_t<bool> &condition, const py::array_t<double> &x, const py::array_t<double> &y)
{
    auto buf_c = condition.request();
    auto buf_x = x.request();
    auto buf_y = y.request();
    py::array_t<double> result(buf_c.shape);
    bool *cond = static_cast<bool *>(buf_c.ptr);
    double *px = static_cast<double *>(buf_x.ptr);
    double *py_ = static_cast<double *>(buf_y.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    py::ssize_t n = std::min({buf_c.size, buf_x.size, buf_y.size});
    for (py::ssize_t i = 0; i < n; ++i)
        dst[i] = cond[i] ? px[i] : py_[i];
    return result;
}

// ============================================================================
// Statistical
// Python: numpy.std, numpy.var
// ============================================================================
double std(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    if (buf.size == 0)
        return 0.0;
    double mean_val = mean(arr);
    double *data = static_cast<double *>(buf.ptr);
    double sum_sq = 0.0;
    for (py::ssize_t i = 0; i < buf.size; ++i)
    {
        double diff = data[i] - mean_val;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / buf.size);
}

double var(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    if (buf.size == 0)
        return 0.0;
    double mean_val = mean(arr);
    double *data = static_cast<double *>(buf.ptr);
    double sum_sq = 0.0;
    for (py::ssize_t i = 0; i < buf.size; ++i)
    {
        double diff = data[i] - mean_val;
        sum_sq += diff * diff;
    }
    return sum_sq / buf.size;
}

// ============================================================================
// Set operations
// Python: numpy.isin, numpy.intersect1d
// ============================================================================
py::array_t<bool> isin(const py::array_t<double> &arr, const std::vector<double> &values)
{
    auto buf = arr.request();
    py::array_t<bool> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    std::unordered_set<double> value_set(values.begin(), values.end());
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = value_set.count(src[i]) > 0;
    return result;
}

py::array_t<double> intersect1d(const py::array_t<double> &a, const py::array_t<double> &b)
{
    auto buf_a = a.request();
    auto buf_b = b.request();
    double *pa = static_cast<double *>(buf_a.ptr);
    double *pb = static_cast<double *>(buf_b.ptr);
    std::unordered_set<double> set_a(pa, pa + buf_a.size);
    std::unordered_set<double> set_b(pb, pb + buf_b.size);
    std::vector<double> intersection;
    for (double val : set_a)
        if (set_b.count(val))
            intersection.push_back(val);
    return py::array_t<double>(intersection.size(), intersection.data());
}

// ============================================================================
// Interpolation
// Python: numpy.interp
// ============================================================================
py::array_t<double> interp(const py::array_t<double> &x, const py::array_t<double> &xp, const py::array_t<double> &fp)
{
    auto buf_x = x.request();
    auto buf_xp = xp.request();
    auto buf_fp = fp.request();
    py::array_t<double> result(buf_x.shape);
    double *px = static_cast<double *>(buf_x.ptr);
    double *pxp = static_cast<double *>(buf_xp.ptr);
    double *pfp = static_cast<double *>(buf_fp.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    py::ssize_t n = buf_xp.size;
    for (py::ssize_t i = 0; i < buf_x.size; ++i)
    {
        double xi = px[i];
        if (xi <= pxp[0])
        {
            dst[i] = pfp[0];
        }
        else if (xi >= pxp[n - 1])
        {
            dst[i] = pfp[n - 1];
        }
        else
        {
            py::ssize_t j = 1;
            while (j < n && pxp[j] < xi)
                ++j;
            double t = (xi - pxp[j - 1]) / (pxp[j] - pxp[j - 1]);
            dst[i] = pfp[j - 1] + t * (pfp[j] - pfp[j - 1]);
        }
    }
    return result;
}

// ============================================================================
// Logical XOR
// Python: numpy.logical_xor
// ============================================================================
py::array_t<bool> logical_xor(const py::array_t<bool> &a, const py::array_t<bool> &b)
{
    auto buf_a = a.request();
    auto buf_b = b.request();
    py::ssize_t size = std::min(buf_a.size, buf_b.size);
    py::array_t<bool> result(buf_a.shape);
    bool *pa = static_cast<bool *>(buf_a.ptr);
    bool *pb = static_cast<bool *>(buf_b.ptr);
    bool *dst = static_cast<bool *>(result.request().ptr);
    for (py::ssize_t i = 0; i < size; ++i)
        dst[i] = pa[i] ^ pb[i];
    return result;
}

// ============================================================================
// Sorting and indexing
// Python: numpy.argsort, numpy.argmax, numpy.argmin
// ============================================================================
py::array_t<py::ssize_t> argsort(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    double *data = static_cast<double *>(buf.ptr);
    std::vector<py::ssize_t> indices(buf.size);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        indices[i] = i;
    std::stable_sort(indices.begin(), indices.end(),
                     [&](py::ssize_t a, py::ssize_t b)
                     { return data[a] < data[b]; });
    return py::array_t<py::ssize_t>(indices.size(), indices.data());
}

py::ssize_t argmax(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    if (buf.size == 0)
        return -1;
    double *data = static_cast<double *>(buf.ptr);
    py::ssize_t max_idx = 0;
    for (py::ssize_t i = 1; i < buf.size; ++i)
        if (data[i] > data[max_idx])
            max_idx = i;
    return max_idx;
}

py::ssize_t argmin(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    if (buf.size == 0)
        return -1;
    double *data = static_cast<double *>(buf.ptr);
    py::ssize_t min_idx = 0;
    for (py::ssize_t i = 1; i < buf.size; ++i)
        if (data[i] < data[min_idx])
            min_idx = i;
    return min_idx;
}

// ============================================================================
// Array transformation
// Python: numpy.roll, numpy.flip, numpy.repeat, numpy.tile
// ============================================================================
py::array_t<double> roll(const py::array_t<double> &arr, py::ssize_t shift)
{
    auto buf = arr.request();
    if (buf.size == 0)
        return py::array_t<double>{};
    shift = shift % buf.size;
    if (shift < 0)
        shift += buf.size;
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[(i + shift) % buf.size] = src[i];
    return result;
}

py::array_t<double> flip(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    py::array_t<double> result(buf.shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        dst[i] = src[buf.size - 1 - i];
    return result;
}

py::array_t<double> repeat(const py::array_t<double> &arr, py::ssize_t repeats)
{
    auto buf = arr.request();
    std::vector<py::ssize_t> shape = {buf.size * repeats};
    py::array_t<double> result(shape);
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i)
        for (py::ssize_t r = 0; r < repeats; ++r)
            dst[i * repeats + r] = src[i];
    return result;
}

py::array_t<double> tile(const py::array_t<double> &arr, py::ssize_t reps)
{
    return repeat(arr, reps);
}

// ============================================================================
// Safe division
// ============================================================================
double safe_divide(double a, double b, double default_val)
{
    if (b == 0.0)
        return default_val;
    return a / b;
}

} // namespace numpy
