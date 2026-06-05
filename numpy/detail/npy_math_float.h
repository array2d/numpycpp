// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  INTERNAL HEADER — DIRECT INCLUSION IS A COMPILE ERROR                 ║
// ║                                                                          ║
// ║  This file implements arch/OS-specific float32 polynomial kernels that  ║
// ║  are tied to numpy's internal SIMD constants.  The API is UNSTABLE and  ║
// ║  subject to change without notice.                                       ║
// ║                                                                          ║
// ║  ✗  #include "numpy/detail/npy_math_float.h"   ← compile error                ║
// ║  ✓  #include "numpy/core.h"             ← only correct entry point      ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#pragma once

#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  error "npy_math_float.h is an internal header — do not include directly. \
Use #include \"numpy/core.h\" instead."
#endif

#include <cstdint>
#include <cstring>
#include <cmath>

namespace numpy {
namespace detail {
// Float32 bit-exact polynomial approximations — internal detail, do not use directly.
// These replicate numpy's simd_exp_FLOAT, simd_log_FLOAT, simd_sincos_f32 algorithms.
// Public API: use numpy::exp() etc. from core.h.

namespace {

inline float uint32_to_float(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint32_t float_to_uint32(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

// ============================================================================
// EXP — Cody-Waite reduction + Pade rational polynomial + ldexp scaling
// ============================================================================

inline float npy_expf(float x) {
    const float xmax       = 88.72283935546875f;
    const float xmin       = -103.97208404541015625f;
    const float log2e      = uint32_to_float(0x3fb8aa3bu);
    const float ln2_high   = uint32_to_float(0xbf317200u);
    const float ln2_low    = uint32_to_float(0xb5bfbe8eu);
    const float magic      = uint32_to_float(0x4b400000u);

    const float p0 = uint32_to_float(0x3f800000u);
    const float p1 = uint32_to_float(0x3f39cbd5u);
    const float p2 = uint32_to_float(0x3e7d4c58u);
    const float p3 = uint32_to_float(0x3d517d8cu);
    const float p4 = uint32_to_float(0x3bdd7159u);
    const float p5 = uint32_to_float(0x3a053dd8u);
    const float q0 = uint32_to_float(0x3f800000u);
    const float q1 = uint32_to_float(0xbe8c6857u);
    const float q2 = uint32_to_float(0x3cb0e832u);

    uint32_t bits = float_to_uint32(x);
    uint32_t exp_bits = (bits >> 23) & 0xff;

    // NaN
    if (exp_bits == 0xff && (bits & 0x7fffff) != 0) return x;
    // +Inf
    if (bits == 0x7f800000u) return x;
    // -Inf -> 0
    if (bits == 0xff800000u) return 0.0f;
    // Overflow
    if (x >= xmax) return uint32_to_float(0x7f800000u);
    // Underflow
    if (x <= xmin) return 0.0f;

    // quadrant = rint(x * log2(e))
    float qf = x * log2e;
    qf = qf + magic;
    qf = qf - magic;

    // Cody-Waite
    float r = x + qf * ln2_high;
    r = r + qf * ln2_low;

    // P(r)/Q(r)
    float num = std::fma(p5, r, p4);
    num = std::fma(num, r, p3);
    num = std::fma(num, r, p2);
    num = std::fma(num, r, p1);
    num = std::fma(num, r, p0);

    float den = std::fma(q2, r, q1);
    den = std::fma(den, r, q0);

    float poly = num / den;

    // Scale
    int qi = static_cast<int>(qf);
    return std::ldexp(poly, qi);
}

// ============================================================================
// LOG — exponent extraction + polynomial in (m-1) + k*ln(2)
// ============================================================================

inline float npy_logf(float x) {
    uint32_t bits = float_to_uint32(x);
    uint32_t exp_field = (bits >> 23) & 0xff;

    // NaN
    if (exp_field == 0xff && (bits & 0x7fffff) != 0) return x;
    // x == 0 or -0 -> -inf  (must check before sign bit to handle -0.0 correctly)
    if ((bits & 0x7fffffffu) == 0) return uint32_to_float(0xff800000u);
    // x < 0 -> -NaN
    if (bits & 0x80000000u) return uint32_to_float(0xffc00000u);
    // x == +inf -> +inf
    if (bits == 0x7f800000u) return x;

    // x = 2^k * m, where m in [0.5, 1) — matches numpy's getexp/getmant
    // numpy: k = getexp(x) + 1, m = getmant(x) in [0.5, 1)
    int k = static_cast<int>(exp_field) - 126;  // unbiased exp + 1
    uint32_t mant_bits = (bits & 0x7fffffu) | 0x3f000000u;
    float m = uint32_to_float(mant_bits);  // m in [0.5, 1)
    float exponent = static_cast<float>(k);

    // if m <= sqrt(1/2): m *= 2; exponent -= 1
    const float sqrt_half = uint32_to_float(0x3f3504f3u);
    if (m <= sqrt_half) {
        m = m * 2.0f;
        exponent = exponent - 1.0f;
    }

    float y = m - 1.0f;  // y in [sqrt(1/2)-1, 1)

    // Polynomial coeffs from numpy's NPY_COEFF_P*_LOGf
    const float p0 = uint32_to_float(0x00000000u);
    const float p1 = uint32_to_float(0x3f800000u);
    const float p2 = uint32_to_float(0x4007361cu);
    const float p3 = uint32_to_float(0x3fbd70a9u);
    const float p4 = uint32_to_float(0x3ec30333u);
    const float p5 = uint32_to_float(0x3cd42bcdu);
    const float q0 = uint32_to_float(0x3f800000u);
    const float q1 = uint32_to_float(0x4027361cu);
    const float q2 = uint32_to_float(0x401cfe0du);
    const float q3 = uint32_to_float(0x3f7c8ae4u);
    const float q4 = uint32_to_float(0x3e1e5bf3u);
    const float q5 = uint32_to_float(0x3bc083dfu);

    float num = std::fma(p5, y, p4);
    num = std::fma(num, y, p3);
    num = std::fma(num, y, p2);
    num = std::fma(num, y, p1);
    num = std::fma(num, y, p0);

    float den = std::fma(q5, y, q4);
    den = std::fma(den, y, q3);
    den = std::fma(den, y, q2);
    den = std::fma(den, y, q1);
    den = std::fma(den, y, q0);

    float poly = num / den;

    const float ln2 = uint32_to_float(0x3f317218u);
    return std::fma(exponent, ln2, poly);
}

// ============================================================================
// SIN/COS — Cody-Waite by PI/2 + polynomial approximation
// ============================================================================

inline void npy_sincosf(float x, float* sin_out, float* cos_out) {
    const float two_over_pi = uint32_to_float(0x3f22f983u);
    const float pio2_high   = uint32_to_float(0xbfc90fd8u);
    const float pio2_med    = uint32_to_float(0xb4a8885au);
    const float pio2_low    = uint32_to_float(0xa7c234c4u);
    const float magic       = uint32_to_float(0x4b400000u);
    const float max_sin     = 117435.992f;
    const float max_cos     = 71476.0625f;

    const float cos_c0 = uint32_to_float(0x3f800000u);
    const float cos_c2 = uint32_to_float(0xbf000000u);
    const float cos_c4 = uint32_to_float(0x3d2aaa9eu);
    const float cos_c6 = uint32_to_float(0xbab6036eu);
    const float cos_c8 = uint32_to_float(0x37cc730bu);

    const float sin_s3 = uint32_to_float(0xbe2aaaabu);
    const float sin_s5 = uint32_to_float(0x3c0888cdu);
    const float sin_s7 = uint32_to_float(0xb95035ddu);
    const float sin_s9 = uint32_to_float(0x363e9ddeu);

    uint32_t bits = float_to_uint32(x);
    uint32_t exp_bits = (bits >> 23) & 0xff;
    if (exp_bits == 0xff && (bits & 0x7fffff) != 0) {
        *sin_out = x;
        *cos_out = x;
        return;
    }

    float abs_x = (x < 0.0f) ? -x : x;

    // ---- SIN ----
    float sin_r;
    if (abs_x <= max_sin) {
        float qf = x * two_over_pi;
        qf = qf + magic;
        qf = qf - magic;

        float r = x + qf * pio2_high;
        r = r + qf * pio2_med;
        r = r + qf * pio2_low;

        float r2 = r * r;

        float cp = std::fma(cos_c8, r2, cos_c6);
        cp = std::fma(cp, r2, cos_c4);
        cp = std::fma(cp, r2, cos_c2);
        cp = std::fma(cp, r2, cos_c0);

        float sp = std::fma(sin_s9, r2, sin_s7);
        sp = std::fma(sp, r2, sin_s5);
        sp = std::fma(sp, r2, sin_s3);
        sp = sp * r2;  // +0
        sp = std::fma(sp, r, r);

        int iq = static_cast<int>(qf);
        if ((iq & 1) == 0) {
            sin_r = sp;
        } else {
            sin_r = cp;
        }
        if ((iq & 2) != 0) {
            sin_r = -sin_r;
        }
    } else {
        sin_r = std::sin(x);
    }
    *sin_out = sin_r;

    // ---- COS ----
    float cos_r;
    if (abs_x <= max_cos) {
        float qf = x * two_over_pi;
        qf = qf + magic;
        qf = qf - magic;

        int iq = static_cast<int>(qf) + 1;

        float r = x + qf * pio2_high;
        r = r + qf * pio2_med;
        r = r + qf * pio2_low;

        float r2 = r * r;

        float cp = std::fma(cos_c8, r2, cos_c6);
        cp = std::fma(cp, r2, cos_c4);
        cp = std::fma(cp, r2, cos_c2);
        cp = std::fma(cp, r2, cos_c0);

        float sp = std::fma(sin_s9, r2, sin_s7);
        sp = std::fma(sp, r2, sin_s5);
        sp = std::fma(sp, r2, sin_s3);
        sp = sp * r2;  // +0
        sp = std::fma(sp, r, r);

        if ((iq & 1) == 0) {
            cos_r = sp;
        } else {
            cos_r = cp;
        }
        if ((iq & 2) != 0) {
            cos_r = -cos_r;
        }
    } else {
        cos_r = std::cos(x);
    }
    *cos_out = cos_r;
}

inline float npy_sinf(float x) {
    float s, c;
    npy_sincosf(x, &s, &c);
    return s;
}

inline float npy_cosf(float x) {
    float s, c;
    npy_sincosf(x, &s, &c);
    return c;
}

} // anonymous namespace
} // namespace detail
} // namespace numpy
