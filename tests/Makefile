# Build numpycpp Python module for testing.
# Requires: pybind11, Eigen3, Python3 development headers.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -fPIC
INCLUDES  = -I.. -I../pycpp $(shell python3 -m pybind11 --includes) $(shell pkg-config --cflags eigen3 2>/dev/null || echo)
LDFLAGS   = -shared

MODULE    = numpycpp.so
SRC       = module.cpp

.PHONY: all clean test

all: $(MODULE)

$(MODULE): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

test: $(MODULE)
	cd .. && PYTHONPATH=tests:$$PYTHONPATH python3 -m pytest tests/ -v

clean:
	rm -f $(MODULE)
