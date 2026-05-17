# numpcpp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-%3E%3D3.16-green.svg)](https://cmake.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

C++ pixel-level alignment of Python numpy, powered by [Eigen](https://eigen.tuxfamily.org/).

## Overview

`numpycpp` provides a C++ implementation of numpy's core API (`numpy.*`, `numpy.linalg.*`, `numpy.einsum`), designed for seamless [pybind11](https://github.com/pybind/pybind11) integration. Every function mirrors Python numpy's signature and semantics — what you write in Python, you get in C++.

## Quick Start

### Dependencies

- **C++20** compiler (GCC >= 11, Clang >= 14, MSVC >= 2022)
- **[Eigen3](https://eigen.tuxfamily.org/)** >= 3.3
- **[pybind11](https://github.com/pybind/pybind11)**
- **CMake** >= 3.16

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Usage

```python
import numpcpp
```

## Project Structure

```
numpycpp/
├── numpy/           # C++ source and headers
│   ├── core.h/.cpp  # numpy.* equivalents
│   ├── linalg.h/.cpp# numpy.linalg.* equivalents
│   └── einsum.h/.cpp# numpy.einsum
├── tests/           # Python test suite
├── CMakeLists.txt   # Build configuration
└── README.md
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## Security

See [SECURITY.md](SECURITY.md) for our security policy and vulnerability reporting process.

## License

[MIT](LICENSE)
