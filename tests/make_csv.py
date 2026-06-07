#!/usr/bin/env python3
"""Generate tests/ulp_precision.csv — ULP differences: numpycpp vs numpy.

Usage:
    make csv          # from tests/ directory
    python3 tests/make_csv.py   # from repo root
"""

import os, sys, struct, csv
import numpy as np
import importlib

# Ensure the tests directory is on sys.path so we can import the C++ module
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
cpp = importlib.import_module("numpycpp")


def ulp_f64(a: float, b: float) -> int:
    """Signed ULP distance between two float64 values."""
    if a == b:
        return 0
    if np.isnan(a) or np.isnan(b):
        return 2**63  # sentinel
    pa = struct.unpack("q", struct.pack("d", float(a)))[0]
    pb = struct.unpack("q", struct.pack("d", float(b)))[0]
    if pa < 0: pa = (-pa) ^ 0x7FFFFFFFFFFFFFFF
    if pb < 0: pb = (-pb) ^ 0x7FFFFFFFFFFFFFFF
    return abs(pa - pb)


def ulp_f32(a: float, b: float) -> int:
    """Signed ULP distance between two float32 values."""
    fa, fb = np.float32(a), np.float32(b)
    if fa == fb:
        return 0
    if np.isnan(fa) or np.isnan(fb):
        return 2**31  # sentinel
    pa = struct.unpack("i", struct.pack("f", float(fa)))[0]
    pb = struct.unpack("i", struct.pack("f", float(fb)))[0]
    if pa < 0: pa = (-pa) ^ 0x7FFFFFFF
    if pb < 0: pb = (-pb) ^ 0x7FFFFFFF
    return abs(pa - pb)


def measure_unary(cpp_fn, np_fn, prep, dt, ulf, rng, n=100_000):
    a = rng.randn(n).astype(dt)
    a = prep(a)
    cr = np.asarray(getattr(cpp, cpp_fn)(a))
    pr = np_fn(a)
    max_u, n_diff = 0, 0
    for i in range(cr.size):
        if cr.flat[i] != pr.flat[i]:
            u = ulf(cr.flat[i], pr.flat[i])
            if u > max_u:
                max_u = u
            n_diff += 1
    return max_u, n_diff


def main():
    rng = np.random.RandomState(42)
    N = 100_000
    ULP_F64 = f"{2**-52:.2e}"
    ULP_F32 = f"{2**-23:.2e}"

    header = [
        "function", "dtype", "max_ulp", "n_diff", "total",
        "category", "ulp_value_f64", "ulp_value_f32",
    ]
    rows = []

    # --- Transcendental unary ---
    TRANS = [
        ("exp",    np.exp,    lambda a: a),
        ("log",    np.log,    lambda a: np.abs(a) + 0.1),
        ("sin",    np.sin,    lambda a: a),
        ("cos",    np.cos,    lambda a: a),
        ("tan",    np.tan,    lambda a: a * 0.5),
        ("cbrt",   np.cbrt,   lambda a: a),
        ("expm1",  np.expm1,  lambda a: a * 2.0),
        ("log1p",  np.log1p,  lambda a: np.abs(a) + 0.1),
        ("log10",  np.log10,  lambda a: np.abs(a) + 0.1),
        ("log2",   np.log2,   lambda a: np.abs(a) + 0.1),
        ("arcsin", np.arcsin, lambda a: np.clip(a * 0.5, -1, 1)),
        ("arccos", np.arccos, lambda a: np.clip(a * 0.5, -1, 1)),
        ("arctan", np.arctan, lambda a: a),
    ]

    for cfn, nfn, prep in TRANS:
        for dt, name, ulf in [
            (np.float64, "float64", ulp_f64),
            (np.float32, "float32", ulp_f32),
        ]:
            mu, nd = measure_unary(cfn, nfn, prep, dt, ulf, rng, N)
            rows.append([cfn, name, mu, nd, N, "transcendental", ULP_F64, ULP_F32])

    # --- Element-wise (should be bit-exact) ---
    ELEM = [
        ("sqrt",    np.sqrt,    lambda a: np.abs(a)),
        ("abs",     np.abs,     lambda a: a),
        ("sign",    np.sign,    lambda a: a),
        ("round",   np.round,   lambda a: a * 100),
        ("floor",   np.floor,   lambda a: a * 100),
        ("ceil",    np.ceil,    lambda a: a * 100),
        ("degrees", np.degrees, lambda a: a),
        ("radians", np.radians, lambda a: a),
    ]

    for cfn, nfn, prep in ELEM:
        for dt, name, ulf in [
            (np.float64, "float64", ulp_f64),
            (np.float32, "float32", ulp_f32),
        ]:
            mu, nd = measure_unary(cfn, nfn, prep, dt, ulf, rng, N)
            rows.append([cfn, name, mu, nd, N, "element-wise", ULP_F64, ULP_F32])

    # --- Binary ---
    BIN = [
        ("power",   np.power,   "scalar exponent 2.0"),
        ("arctan2", np.arctan2,  "scalar 1.0 denominator"),
        ("hypot",   np.hypot,    "two arrays"),
    ]

    for cfn, nfn, _desc in BIN:
        for dt, name, ulf in [
            (np.float64, "float64", ulp_f64),
            (np.float32, "float32", ulp_f32),
        ]:
            a = rng.randn(N).astype(dt)
            if cfn == "hypot":
                b = np.abs(rng.randn(N).astype(dt)) + dt(0.1)
            elif cfn == "power":
                b = dt(2.0)
                a = np.abs(a) + dt(0.01)  # keep positive for fractional exponent
            else:
                b = dt(1.0)

            cr = np.asarray(getattr(cpp, cfn)(a, b))
            pr = nfn(a, b)
            max_u, n_diff = 0, 0
            for i in range(cr.size):
                if cr.flat[i] != pr.flat[i]:
                    u = ulf(cr.flat[i], pr.flat[i])
                    if u > max_u:
                        max_u = u
                    n_diff += 1
            rows.append([cfn, name, max_u, n_diff, N, "binary", ULP_F64, ULP_F32])

    csv_path = os.path.join(_here, "ulp_precision.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
