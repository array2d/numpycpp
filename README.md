# numpcpp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-%3E%3D3.16-green.svg)](https://cmake.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## 背景

NumPy 的性能已经很快了，但它的上限被 Python 锁死了。

我们创建了 `numpycpp` — 保留 NumPy 的使用习惯，但允许 C++ 突破 Python 的性能天花板，进一步加速你的代码。

## 概述

`numpycpp` 是一个 **header-only C++ 库**，用 raw pointer + size 接口实现了 numpy 核心 API（`numpy.*`、`numpy.linalg.*`、`numpy.einsum`）的像素级对齐。零外部依赖，纯 C++17 标准库即可编译。

## 快速开始

### 依赖

- **C++17** 编译器 (GCC >= 9, Clang >= 7, MSVC >= 2019)

### 使用

```cpp
#include "numpy/core.h"

// 纯 C++ 调用，无 pybind11 依赖
std::vector<double> data = {1.0, 4.0, 9.0};
std::vector<double> result(data.size());

numpy::sqrt(data.data(), result.data(), data.size());
// result → {1.0, 2.0, 3.0}

double s = numpy::sum(data.data(), data.size());
// s → 14.0
```

### 安装

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make deb                # 生成 numpcpp-dev-1.21.2-Linux.deb
sudo dpkg -i numpcpp-dev-1.21.2-Linux.deb
# 头文件安装至 /usr/include/numpycpp/
```

### 精度测试

测试套件验证每个 C++ 函数与 Python numpy 的**像素级精度对齐**。

```bash
cd tests
make                    # 编译测试模块
make test               # 运行全部 336 个测试
```

## 项目结构

```
numpycpp/
├── numpy/              # 纯 C++ 头文件（零依赖）
│   ├── core.h          # numpy.* 等价函数（~80个）
│   ├── linalg.h        # numpy.linalg.* 等价函数
│   └── einsum.h        # numpy.einsum 等价实现
├── pycpp/              # pybind11 包装层（可选引用）
│   ├── core_py.h
│   ├── linalg_py.h
│   └── einsum_py.h
├── tests/              # Python 精度测试 + 测试用模块
│   ├── module.cpp      # 测试用 pybind11 模块
│   ├── Makefile
│   └── test_*.py
├── CMakeLists.txt      # 构建 & .deb 打包
└── README.md
```

## 许可

[MIT](LICENSE)
