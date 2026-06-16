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

## C++ API 与 numpy 严格对齐规则

**核心原则：C++ API 必须与 numpy 在接口层面完全一致。禁止为绕过对齐而修改测试代码。**

### 1. 参数顺序必须与 numpy 一致

- **错误示例：** `compress(arr, mask)` — C++ 把数组放前面
- **正确：** `compress(condition, a)` — 严格匹配 `numpy.compress(condition, a)`

### 2. 默认值语义必须与 numpy 一致

- **错误示例：** `eye(N, M=-1, k=0)` — 用 `-1` 作哨兵值表示"使用 N"
- **正确：** `eye(N, M=py::none(), k=0)` — 匹配 `numpy.eye(N, M=None, k=0)`
- **规则：** 禁止使用自定义哨兵值（如 `-1`）。如果 numpy 用 `None`，C++ 必须用 `py::none()`。

### 3. 数学函数必须经 SVML 桥，禁止直接调用 std::

- **禁止：** `std::pow(base, x)` / `std::atan2(y, x)` / `std::exp(x)` — 这些与 numpy 的 SVML 实现差 1-3 ULP
- **正确：** 通过 `detail::pow()` / `detail::atan2()` / `detail::exp()` 调用 — 这些在 bit_exact 模式下经 dlsym 解析 numpy 的 SVML 向量符号

### 4. SVML 符号必须用向量版本，不能用标量 fallback

- **错误：** `resolve_svml("npy_atan2")` — 解析到 libm 的标量 `atan2`，与 numpy 内部使用的 `__svml_atan28` 差 1 ULP
- **正确：** `resolve_svml("__svml_atan28")` — 解析 numpy 实际使用的 Intel SVML 向量符号
- **关键符号对照表：**

| 函数 | 错误符号（标量 libm） | 正确符号（Intel SVML） |
|------|----------------------|----------------------|
| pow f64 | `npy_pow` | `__svml_pow8` |
| pow f32 | `npy_powf` | `__svml_powf16` |
| atan2 f64 | `npy_atan2` | `__svml_atan28` |
| atan2 f32 | `npy_atan2f` | `__svml_atan2f16` |

SVML 向量函数签名：`__m512d (*)(__m512d, __m512d)` (f64) / `__m512 (*)(__m512, __m512)` (f32)。
调用方式：标量 → `_mm512_set1_pd(x)` 广播 → SVML → `_mm512_cvtsd_f64()` 提取。

### 5. 头文件 include 顺序

`numpy.h` 必须在所有子模块头文件之前加载 `svml_bridge.h`，确保 `detail::pow` 等符号在 `init.h` 的 `logspace` 等函数中可用。

### 6. 修复流程：诊断 → 定位根因 → 修 C++ → 严禁改测试

当发现 bit_exact 测试失败时：

1. **诊断：** `python3 -c "import numpycpp, numpy as np; ..."` 确认差异和 ULP 量级
2. **定位根因：** 追溯 C++ 调用链，找出哪一步使用了非 numpy 等价的实现（`std::`、标量 `npy_*`、错误参数顺序等）
3. **修 C++ 源码：** 修改头文件中的实现，使其严格匹配 numpy 的等效路径
4. **严禁改测试：** 测试代码是"黄金标准"——它反映 numpy 的真实行为。修改测试来绕过 C++ 实现缺陷等同于作弊

## 编译模式

### bit_exact 模式（默认）

```cmake
# cmake -S tests -B tests/build
```

关键 flag：
- `-O2 -ffp-contract=off` — 禁用 FMA 融合
- `-fno-builtin-{exp,log,sin,cos,tan,pow,atan2,sqrt,...}` — 阻止 GCC 替换 dlsym 路径
- `-msse4.1 -mfma` — SIMD 支持
- 无 `-march=native` — 避免编译器自动矢量化改变计算顺序
- 通过 `dlsym` 从 numpy 的 `_multiarray_umath.so` 解析 SVML 符号

### std 模式

```cmake
# cmake -S tests -B tests/build_std -DNUMPYCPP_STD_ONLY=ON
```

- `-O3 -march=native` — 无精度约束，允许编译器自由优化
- 无 `-fno-builtin-*` — 使用 GCC 内置数学函数
- 与 numpy 的 SVML 实现预期差 0-2 ULP
