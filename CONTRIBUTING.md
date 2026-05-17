# Contributing to numpcpp

Thanks for your interest in contributing!

## Getting Started

1. Fork the repository and clone it locally.
2. Install dependencies: C++20 compiler, CMake >= 3.16, Eigen3 >= 3.3, pybind11.
3. Build the project:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Development Workflow

- Create a branch for your change.
- Make your changes, following the existing code style.
- Add tests in `tests/` for new functionality.
- Verify the build succeeds before submitting.

## Pull Requests

- Keep PRs focused on a single change.
- Reference any related issues in the PR description.
- Ensure all tests pass.

## Code Style

- C++20, no compiler extensions.
- Follow existing naming conventions in the codebase.
- Header-only where practical for pybind11 integration.

## Reporting Issues

Use the [issue tracker](https://github.com/array2d/numpycpp/issues) and include:

- Steps to reproduce
- Expected vs actual behavior
- Compiler version, OS, and dependency versions
