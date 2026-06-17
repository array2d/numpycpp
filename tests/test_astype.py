"""numpy::astype C++ 单测 — 全部 dtype 转换 + 递归安全性。
运行:  pytest tests/test_astype.py -v
"""
import numpy as np, pytest, numpycpp as cpp

_SRC = {"f64": lambda: np.array([1.5,-2.7,3.1], dtype=np.float64),
        "f32": lambda: np.array([1.5,-2.7,3.1], dtype=np.float32),
        "i32": lambda: np.array([1,-2,3], dtype=np.int32),
        "i64": lambda: np.array([1,-2,3], dtype=np.int64),
        "bool": lambda: np.array([True,False,True])}
# 目标必须显式指定精度; "float" 已移除
_EXPECT = {
    "float64": np.float64, "double": np.float64,
    "float32": np.float32,
    "int":     np.int32,   "int32":  np.int32,
    "int64":   np.int64,   "bool":   np.bool_,
}

@pytest.mark.parametrize("src", _SRC)
@pytest.mark.parametrize("dst,exp_dt", list(_EXPECT.items()))
def test_astype(src, dst, exp_dt):
    a = _SRC[src]()
    r = np.asarray(cpp.astype(a, dst))
    assert r.dtype == exp_dt, f"{src}→{dst}: got {r.dtype}, expected {exp_dt}"
    assert np.array_equal(r, a.astype(exp_dt)), f"{src}→{dst}: value mismatch"

def test_float_rejected():
    """'float' 作为目标必须被拒绝（必须显式指定 float32/float64）"""
    a = np.array([1.5], dtype=np.float64)
    with pytest.raises(RuntimeError, match="unsupported.*target"):
        cpp.astype(a, "float")

@pytest.mark.parametrize("label,arr,expect_fallback", [
    ("f32 LE", np.array([1.5],dtype=np.float32), True),
    ("f32 BE", np.array([1.5],dtype='>f4'), True),
    ("f64 LE", np.array([1.5],dtype=np.float64), True),
    ("i32",    np.array([1],dtype=np.int32), False),
    ("bool",   np.array([True]), False),
])
def test_dtype_diag(label, arr, expect_fallback):
    info = cpp._diag_astype_dtype(arr)
    assert info["fallback_works"] == expect_fallback, f"{label}: {info['fallback_works']}"

def test_no_recursion_32():
    for i in range(32):
        a = np.random.RandomState(i).randn(4,2).astype(np.float32)
        assert np.asarray(cpp.astype(a,"float64")).dtype == np.float64

def test_no_recursion_1k():
    a = np.array([1.5,2.7,3.1], dtype=np.float32)
    for _ in range(1000):
        r = np.asarray(cpp.astype(a,"float64"))
    assert r.dtype == np.float64

def test_empty():
    assert np.asarray(cpp.astype(np.array([],dtype=np.float32),"float64")).size == 0

def test_unsupported_raises():
    with pytest.raises(RuntimeError): cpp.astype(np.array([1.0]),"complex64")

def test_self_noop():
    for dt,val in [(np.float64,1.5),(np.float32,1.5),(np.int32,1),(np.int64,1),(bool,True)]:
        assert np.asarray(cpp.astype(np.array([val],dtype=dt),str(np.dtype(dt)))).dtype == dt

if __name__ == "__main__":
    import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
    sys.exit(pytest.main([__file__,"-v","--tb=short"]))
