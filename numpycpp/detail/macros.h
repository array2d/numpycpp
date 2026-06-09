// ════════════════════════════════════════════════════════════════════════════
//  numpycpp internal — detail/macros.h
//  Shared preprocessor macros used across elementwise.h, manipulation.h, etc.
//  No NUMPYCPP_INTERNAL_INCLUDE guard required (pure macros, no symbols).
// ════════════════════════════════════════════════════════════════════════════
#pragma once

// Stack-allocation threshold for small-array optimizations
#ifndef NUMPY_SMALL_STACK
#  define NUMPY_SMALL_STACK 128
#endif

// 4x loop unrolling to reduce branch overhead.
// All calls are inlined → identical codegen to hand-written unrolled loops.
#ifndef NUMPY_UNROLL4
#define NUMPY_UNROLL4(dst_i, body)        \
    do { size_t _i = 0;                   \
        for (; _i + 3 < n; _i += 4) {     \
            size_t dst_i = _i + 0; body;  \
            dst_i = _i + 1; body;         \
            dst_i = _i + 2; body;         \
            dst_i = _i + 3; body;         \
        }                                 \
        for (; _i < n; ++_i) {            \
            size_t dst_i = _i; body;      \
        }                                 \
    } while(0)
#endif
