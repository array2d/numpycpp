// Python Source: numpy/linalg/ (numpy.linalg module) + numpy.dot
// Line Range: numpy.linalg.norm, numpy.dot
// Alignment: strict

#include "numpy/linalg.h"
#include <cmath>
#include <algorithm>

namespace numpy {
namespace linalg {

// Python: numpy.linalg.norm(arr)
// Line: linalg.py
double norm(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    double *data = static_cast<double *>(buf.ptr);
    double sum = 0.0;
    for (py::ssize_t i = 0; i < buf.size; ++i)
        sum += data[i] * data[i];
    return std::sqrt(sum);
}

// Python: numpy.linalg.norm(arr) with float32 input
// Line: linalg.py
// Internal computation in float32 to match Python's precision path.
// Python: float32(x) → float32(x²) → float32(sum) → float32(sqrt) → float64 return.
double norm(const py::array_t<float> &arr)
{
    auto buf = arr.request();
    float *data = static_cast<float *>(buf.ptr);
    float sum = 0.0f;
    for (py::ssize_t i = 0; i < buf.size; ++i)
        sum += data[i] * data[i];
    return static_cast<double>(std::sqrt(sum));
}

// Python: numpy.linalg.norm(arr, axis=1)
// Line: linalg.py
py::array_t<double> norm_axis1(const py::array_t<double> &arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2)
    {
        py::array_t<double> result({1});
        double *dst = static_cast<double *>(result.request().ptr);
        double *data = static_cast<double *>(buf.ptr);
        double sum = 0.0;
        for (py::ssize_t i = 0; i < buf.size; ++i)
            sum += data[i] * data[i];
        dst[0] = std::sqrt(sum);
        return result;
    }
    py::ssize_t rows = buf.shape[0];
    py::ssize_t cols = buf.shape[1];
    py::array_t<double> result({rows});
    double *src = static_cast<double *>(buf.ptr);
    double *dst = static_cast<double *>(result.request().ptr);
    for (py::ssize_t i = 0; i < rows; ++i)
    {
        double sum = 0.0;
        for (py::ssize_t j = 0; j < cols; ++j)
            sum += src[i * cols + j] * src[i * cols + j];
        dst[i] = std::sqrt(sum);
    }
    return result;
}

} // namespace linalg

// Python: numpy.dot(a, b)
// Line: core.py
double dot(const py::array_t<double> &a, const py::array_t<double> &b)
{
    auto buf_a = a.request();
    auto buf_b = b.request();
    double *src_a = static_cast<double *>(buf_a.ptr);
    double *src_b = static_cast<double *>(buf_b.ptr);
    py::ssize_t n = std::min(buf_a.size, buf_b.size);
    double sum = 0.0;
    for (py::ssize_t i = 0; i < n; ++i)
        sum += src_a[i] * src_b[i];
    return sum;
}

} // namespace numpy
