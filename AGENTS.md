# Agent Instructions for numpycpp

## 零容错原则 — 暴露错误，禁止吞错

**为什么暴露错误这么重要：**

吞错误（fallback / silent failure / 返回假值）会导致查找一个很小的 ULP 问题耗费无穷漫长的时间和人力。

真实案例：`blas_bridge.h` 中的 `find_openblas_path()` 匹配到了 scipy 的 LP64
OpenBLAS 而非 numpy 的 ILP64 OpenBLAS。`dlsym("dgesv_64_")` 返回 nullptr，
但代码静默 fallback 到 `return false` / 返回零矩阵，导致 5000 个矩阵的 inv
全部错误 —— 没有任何异常、没有任何报错。排查这个 0‑ULP 偏差的根因花费了数周。

**规则：**

1. **BLAS / LAPACK 符号解析失败 → 立刻 `throw std::runtime_error`，进程退出。**
   不返回 nullptr，不 fallback 到纯 C++ 循环，不返回假值。

2. **任何系统级前提条件不满足（OpenBLAS 未加载、ILP64 ABI 不匹配、
   符号缺失）→ 立刻 throw，附带清晰的错误消息说明原因和修复方法。**

3. **禁止 silent failure。** 一个静默错误浪费的排查时间远超一个清晰的
   crash + stack trace。

4. **数学意义上的失败（如奇异矩阵）也要 throw**，不要 return false/zero。
   调用方如果期望处理奇异矩阵，应当显式 catch。

## 代码风格

- BLAS 函数指针用 function-local static 缓存，首次调用时通过 `resolve_blas()` 解析。
  `resolve_blas()` 要么返回有效指针要么 throw——绝无 nullptr 返回。
- 不做 fallback。`blas_bridge.h` 不存在任何 "if BLAS unavailable, do X" 的代码路径。
