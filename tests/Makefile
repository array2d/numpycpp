# Build numpycpp Python module for testing.
# Requires: pybind11, Eigen3, Python3 development headers.
#
# Alignment: BIT-LEVEL — all tests verify bit-exact C++ vs numpy results.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -fPIC -fopenmp -ffp-contract=off \
	-ffloat-store -msse4.1 -mavx512f -mfma \
	-fno-builtin-exp -fno-builtin-log -fno-builtin-sin \
	-fno-builtin-cos -fno-builtin-tan -fno-builtin-pow \
	-fno-builtin-sqrt -fno-builtin-atan2 -fno-builtin-log2 \
	-fno-builtin-log10 -fno-builtin-asin -fno-builtin-acos \
	-fno-builtin-atan -fno-builtin-exp2 \
	-fno-builtin-cbrt -fno-builtin-expm1 -fno-builtin-log1p
INCLUDES  = -I.. -I../pycpp $(shell python3 -m pybind11 --includes) $(shell pkg-config --cflags eigen3 2>/dev/null || echo)
LDFLAGS   = -shared -ldl

MODULE    = numpycpp.so
SRC       = module.cpp

.PHONY: all clean test

all: $(MODULE)

$(MODULE): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

test: $(MODULE)
	@cd .. && PYTHONPATH=tests:$$PYTHONPATH python3 -m pytest tests/test_all.py -q --tb=short --no-header

clean:
	rm -f $(MODULE)
