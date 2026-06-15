# Benchmarks

Microbenchmarks and regression benchmarks for persistence computation kernels.

## Overview

The benchmarks directory contains C++ microbenchmarks for individual
computation kernels (distance computation, matrix reduction, cohomology,
etc.) as well as end-to-end persistence benchmarks. These benchmarks
are used for:

- **Regression detection**: ensuring performance doesn't regress across
  code changes
- **Hardware characterization**: measuring performance across different
  CPU and GPU architectures
- **Algorithm comparison**: comparing the performance of different
  reduction strategies (standard, cohomology, PH4/PH5/PH6)

## Running Benchmarks

Benchmarks are built and run via CMake. See the C++ build documentation
for details.

## C++ API

See the C++ API reference for detailed benchmark function documentation.
