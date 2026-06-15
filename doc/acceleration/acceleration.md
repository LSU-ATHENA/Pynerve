# Acceleration

Acceleration layer providing adaptive kernel dispatch, performance monitoring,
and fallback strategies for the persistence pipeline.

## Overview

The acceleration framework sits between the high-level persistence API and the
concrete CPU/GPU/SIMD backends. It provides:

- **Adaptive dispatch**: runtime selection of the fastest available kernel
  based on input characteristics and hardware capabilities
- **Performance monitoring**: real-time tracking of kernel execution time,
  cache misses, and memory bandwidth utilization
- **Graceful fallback**: automatic fallback to a slower but available kernel
  when the preferred kernel is unavailable

## C++ API

See the C++ API reference for detailed documentation of the acceleration
layer classes and functions.
