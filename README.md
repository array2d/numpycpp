# numpcpp

C++ pixel-level alignment of Python numpy, powered by Eigen.

## Overview

`numpycpp` provides a C++ implementation of numpy's core API (`numpy.*`, `numpy.linalg.*`, `numpy.einsum`), designed for seamless pybind11 integration. Every function mirrors Python numpy's signature and semantics.

## Dependencies

- C++20
- [pybind11](https://github.com/pybind/pybind11)
- [Eigen3](https://eigen.tuxfamily.org/) >= 3.3

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

```python
import numpcpp
```

## License

MIT
