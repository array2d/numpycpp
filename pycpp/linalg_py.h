// Pybind11 wrappers for linalg native functions.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../numpy/linalg.h"

namespace py = pybind11;

namespace numpy {
namespace linalg {

/// numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)
template<typename T>
T norm(const py::array_t<T>& arr) {
    auto buf = arr.request();
    return norm(static_cast<const T*>(buf.ptr), buf.size);
}

/// numpy.linalg.norm(x, ord=None, axis=N, keepdims=False) — N-D with axis
template<typename T>
py::array_t<T> norm(const py::array_t<T>& arr, int axis = -1) {
    auto buf = arr.request();
    int ndim = static_cast<int>(buf.ndim);
    int ax = (axis == -1) ? ndim - 1 : axis;

    std::vector<ptrdiff_t> shape(buf.shape.begin(), buf.shape.end());

    // Output shape: drop the reduced axis
    std::vector<ptrdiff_t> out_shape;
    for (int d = 0; d < ndim; ++d)
        if (d != ax) out_shape.push_back(shape[d]);
    if (out_shape.empty()) out_shape.push_back(1);

    std::vector<py::ssize_t> py_out_shape(out_shape.begin(), out_shape.end());
    py::array_t<T> result(py_out_shape);
    numpy::norm_axis(static_cast<const T*>(buf.ptr),
                            static_cast<T*>(result.request().ptr),
                            shape.data(), ndim, ax);
    return result;
}

} // namespace linalg

/// numpy.dot(a, b, out=None)
template<typename T>
T dot(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    return numpy::dot(static_cast<const T*>(ba.ptr),
                             static_cast<const T*>(bb.ptr),
                             std::min(ba.size, bb.size));
}

/// numpy.matmul(a, b) — bit-exact via cblas_sgemm64_ (same kernel as numpy)
/// Supported shapes (mirrors numpy.matmul rules):
///   2D × 2D:  (M,K) @ (K,N) → (M,N)
///   1D × 2D:  (K,)  @ (K,N) → (N,)      [treated as (1,K) @ (K,N), result squeezed]
///   2D × 1D:  (M,K) @ (K,)  → (M,)      [treated as (M,K) @ (K,1), result squeezed]
///   3D × 3D:  (B,M,K) @ (B,K,N) → (B,M,N)  [batched loop, one gemm per batch]
template<typename T>
py::array_t<T> matmul(const py::array_t<T>& a, const py::array_t<T>& b) {
    auto ba = a.request(), bb = b.request();
    const T* A = static_cast<const T*>(ba.ptr);
    const T* B = static_cast<const T*>(bb.ptr);

    // 2D × 2D
    if (ba.ndim == 2 && bb.ndim == 2) {
        size_t M = ba.shape[0], K = ba.shape[1], N = bb.shape[1];
        py::array_t<T> out({(py::ssize_t)M, (py::ssize_t)N});
        T* C = static_cast<T*>(out.request().ptr);
        // matmul_slice mirrors numpy's sdot/gemv/gemm dispatch exactly
        numpy::linalg::matmul(A, B, C, M, K, N);
        return out;
    }
    // 1D × 2D: (K,) @ (K,N) → (N,)   uses cblas_*gemv64_ Trans
    if (ba.ndim == 1 && bb.ndim == 2) {
        size_t K = ba.shape[0], N = bb.shape[1];
        py::array_t<T> out({(py::ssize_t)N});
        numpy::linalg::matmul_vm(A, B, static_cast<T*>(out.request().ptr), K, N);
        return out;
    }
    // 2D × 1D: (M,K) @ (K,) → (M,)   uses cblas_*gemv64_ NoTrans
    if (ba.ndim == 2 && bb.ndim == 1) {
        size_t M = ba.shape[0], K = ba.shape[1];
        py::array_t<T> out({(py::ssize_t)M});
        numpy::linalg::matmul_mv(A, B, static_cast<T*>(out.request().ptr), M, K);
        return out;
    }
    // batched 3D × 3D: (B,M,K) @ (B,K,N) → (B,M,N)
    if (ba.ndim == 3 && bb.ndim == 3) {
        size_t batch = ba.shape[0], M = ba.shape[1], K = ba.shape[2], N = bb.shape[2];
        py::array_t<T> out({(py::ssize_t)batch, (py::ssize_t)M, (py::ssize_t)N});
        numpy::linalg::matmul(A, B, static_cast<T*>(out.request().ptr), batch, M, K, N);
        return out;
    }
    throw std::invalid_argument("matmul: unsupported ndim combination");
}

} // namespace numpy
