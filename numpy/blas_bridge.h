// OpenBLAS bridge — resolve cblas_*dot from numpy's loaded modules.
// Same approach as svml_bridge.h: resolve symbols from numpy's
// already-loaded _multiarray_umath.so. dlsym searches transitive
// dependencies, so cblas_*dot symbols from OpenBLAS are found.
// No hardcoded .so paths.
//
// Call blas_init(path_to_multiarray_umath_so) before first use.

#pragma once

#include <cstdint>
#include <dlfcn.h>
#include <cstdio>

namespace numpy {
namespace blas {

inline void* g_blas_handle = nullptr;

inline bool blas_init(const char* umath_path) {
    static bool initialized = false;
    if (initialized) return g_blas_handle != nullptr;
    initialized = true;

    if (umath_path && umath_path[0]) {
        // Get RTLD_NOLOAD handle to numpy's _multiarray_umath.so.
        // dlsym on this handle searches transitive deps (OpenBLAS).
        g_blas_handle = dlopen(umath_path, RTLD_NOLOAD | RTLD_LAZY);
        fprintf(stderr, "[BLAS] blas_init(%s) -> handle=%p dlerror=%s\n",
                umath_path, g_blas_handle, dlerror());
        if (g_blas_handle) {
            void* test = dlsym(g_blas_handle, "cblas_sdot64_");
            fprintf(stderr, "[BLAS] dlsym(cblas_sdot64_) -> %p dlerror=%s\n", test, dlerror());
            if (test) return true;

            test = dlsym(g_blas_handle, "cblas_ddot64_");
            fprintf(stderr, "[BLAS] dlsym(cblas_ddot64_) -> %p dlerror=%s\n", test, dlerror());
            if (test) return true;
        }
    }

    // Fallback: try RTLD_DEFAULT
    void* test = dlsym(RTLD_DEFAULT, "cblas_sdot64_");
    if (test) {
        g_blas_handle = RTLD_DEFAULT;
        fprintf(stderr, "[BLAS] found cblas_sdot64_ via RTLD_DEFAULT\n");
        return true;
    }
    fprintf(stderr, "[BLAS] BLAS not available; dot will use sequential fallback\n");
    return false;
}

// cblas_sdot64_ — ILP64 interface (blasint = int64_t)
inline float cblas_sdot(int64_t n, const float* x, int64_t incx, const float* y, int64_t incy) {
    static auto fn = (float (*)(int64_t, const float*, int64_t, const float*, int64_t))
        dlsym(g_blas_handle, "cblas_sdot64_");
    if (fn) return fn(n, x, incx, y, incy);
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += x[i * incx] * y[i * incy];
    return sum;
}

inline double cblas_ddot(int64_t n, const double* x, int64_t incx, const double* y, int64_t incy) {
    static auto fn = (double (*)(int64_t, const double*, int64_t, const double*, int64_t))
        dlsym(g_blas_handle, "cblas_ddot64_");
    if (fn) return fn(n, x, incx, y, incy);
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += x[i * incx] * y[i * incy];
    return sum;
}

} // namespace blas
} // namespace numpy
