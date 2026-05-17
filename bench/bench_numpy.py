"""Reference numpy benchmarks for comparison with numpcpp bench_core.

Usage: python bench_numpy.py
"""
import time
import numpy as np

SIZES = [1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22]
REPEAT = 10

rng = np.random.default_rng(42)


def make_data(n):
    return rng.uniform(1.0, 100.0, size=n)


def bench(name, fn, n, src, *args):
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        result = fn(src, *args)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) / REPEAT * 1000
    print(f"  {name:12s}  n={n:8d}  {elapsed_ms:10.4f} ms  ({n / elapsed_ms * 1000 / 1e6:.2f} Melem/s)")


elementwise = [
    ("sqrt", np.sqrt),
    ("abs",  np.abs),
    ("exp",  np.exp),
    ("log",  np.log),
    ("sin",  np.sin),
    ("cos",  np.cos),
]

reductions = [
    ("sum",  np.sum),
    ("mean", np.mean),
    ("max",  np.max),
]

print("=== element-wise ===")
for n in SIZES:
    src = make_data(n)
    for name, fn in elementwise:
        bench(name, fn, n, src)

print("\n=== reduction ===")
for n in SIZES:
    src = make_data(n)
    for name, fn in reductions:
        bench(name, fn, n, src)

print("\n=== dot (1D) ===")
for n in SIZES:
    a = make_data(n)
    b = make_data(n)
    bench("dot", np.dot, n, a, b)

print("\n=== norm (L2) ===")
for n in SIZES:
    src = make_data(n)
    bench("norm", np.linalg.norm, n, src)
