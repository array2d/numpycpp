#!/usr/bin/env python3
"""
bench/bench.py — numpycpp vs numpy: throughput (GB/s) and ULP accuracy.

CSV columns (stdout):
  func, dtype, N, numpycpp_ms, numpy_ms, speedup_x, numpycpp_GBps, numpy_GBps, max_ulp_vs_numpy

Usage:
  # build first (from project root):
  #   cmake -S tests -B tests/build && cmake --build tests/build

  PYTHONPATH=tests/build python3 bench/bench.py
  PYTHONPATH=tests/build python3 bench/bench.py > results.csv
"""

import sys
import csv
import timeit
import numpy as np

try:
    import numpycpp
except ImportError:
    sys.exit("numpycpp not found — set PYTHONPATH=tests/build (or wherever the .so lives)")

# ---------------------------------------------------------------------------
# Function table: (name, numpy_fn, numpycpp_fn, input_lo, input_hi)
# ---------------------------------------------------------------------------
FUNCS = [
    ("sqrt",   np.sqrt,   numpycpp.sqrt,   0.1, 10.0),
    ("abs",    np.abs,    numpycpp.abs,   -5.0,  5.0),
    ("exp",    np.exp,    numpycpp.exp,    0.1,  5.0),
    ("log",    np.log,    numpycpp.log,    0.1, 10.0),
    ("sin",    np.sin,    numpycpp.sin,    0.1,  5.0),
    ("cos",    np.cos,    numpycpp.cos,    0.1,  5.0),
    ("tan",    np.tan,    numpycpp.tan,    0.1,  1.0),
    ("arcsin", np.arcsin, numpycpp.arcsin,-0.9,  0.9),
    ("arccos", np.arccos, numpycpp.arccos,-0.9,  0.9),
    ("arctan", np.arctan, numpycpp.arctan, 0.1,  5.0),
    ("cbrt",   np.cbrt,   numpycpp.cbrt,   0.1, 10.0),
    ("expm1",  np.expm1,  numpycpp.expm1,  0.1,  1.0),
    ("log1p",  np.log1p,  numpycpp.log1p,  0.1, 10.0),
    ("log10",  np.log10,  numpycpp.log10,  0.1, 10.0),
    ("log2",   np.log2,   numpycpp.log2,   0.1, 10.0),
]

DTYPES  = ["float64", "float32"]
# sizes from 2^10 to 2^19 (1K … 512K elements)
SIZES   = [1 << k for k in range(10, 20, 3)]   # 1024, 8192, 65536, 524288
REPS    = 50    # timeit repeats (take min — eliminates OS scheduling noise)
NUMBER  = 5     # calls per repeat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(42)

def _make(lo: float, hi: float, N: int, dtype: str) -> np.ndarray:
    return np.ascontiguousarray(_rng.uniform(lo, hi, N).astype(dtype))

def _bench_ms(fn, arr: np.ndarray) -> float:
    """Return minimum wall-time per call in milliseconds."""
    t = timeit.repeat(lambda: fn(arr), repeat=REPS, number=NUMBER)
    return min(t) / NUMBER * 1e3

def _max_ulp(ref: np.ndarray, got: np.ndarray) -> int:
    """Max absolute ULP difference between two same-dtype arrays."""
    if ref.dtype == np.float64:
        return int(np.max(np.abs(ref.view(np.int64) - got.view(np.int64))))
    return int(np.max(np.abs(ref.view(np.int32) - got.view(np.int32))))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    w = csv.writer(sys.stdout)
    w.writerow([
        "func", "dtype", "N",
        "numpycpp_ms", "numpy_ms", "speedup_x",
        "numpycpp_GBps", "numpy_GBps",
        "max_ulp_vs_numpy",
    ])

    for fn_name, np_fn, cpp_fn, lo, hi in FUNCS:
        for dtype in DTYPES:
            itemsize = 8 if dtype == "float64" else 4
            for N in SIZES:
                arr = _make(lo, hi, N, dtype)

                # warm-up — fill caches, JIT the pybind11 dispatch path
                cpp_fn(arr); np_fn(arr)

                t_cpp = _bench_ms(cpp_fn, arr)
                t_np  = _bench_ms(np_fn,  arr)

                gb_cpp  = N * itemsize / t_cpp / 1e6
                gb_np   = N * itemsize / t_np  / 1e6
                speedup = t_np / t_cpp
                ulp     = _max_ulp(np_fn(arr), cpp_fn(arr))

                w.writerow([
                    fn_name, dtype, N,
                    f"{t_cpp:.4f}", f"{t_np:.4f}",
                    f"{speedup:.3f}",
                    f"{gb_cpp:.2f}", f"{gb_np:.2f}",
                    ulp,
                ])
                sys.stdout.flush()   # stream CSV line-by-line

if __name__ == "__main__":
    main()
