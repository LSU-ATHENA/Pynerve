# Overview

## Namespace Structure

```
nerve
|-- nerve::persistence        -- PH computation, diagrams, pairs
|-- nerve::algorithms         -- Distance, KNN, vectorization
|-- nerve::algebra            -- Simplex, simplicial complex, chain complex
|-- nerve::core               -- Thread pools, memory pools, determinism
|-- nerve::memory             -- Slab allocator, thread-local pool, RawArrayPool
|-- nerve::determinism        -- Seed management, GPU reduction, MPI reduction
|-- nerve::gpu                -- DeviceMemoryPool, CUDA error handling
|-- nerve::cpu::simd          -- CPUFeatureDetector, SIMD dispatch
|-- nerve::distributed        -- MPI persistence, NCCL/MPI bridge, NVSHMEM
|-- nerve::streaming          -- Approximate/Exact streaming PH
|-- nerve::nn                 -- Diagram convolutions
|-- nerve::sheaf              -- Sheaf Laplacian, sheaf learning
|-- nerve::graphs             -- Graph sheaf, graph homology, GNN layers
|-- nerve::torch              -- PyTorch C++ integration (SimplexTree, Filtration, Mapper)
|-- nerve::errors             -- Error codes, ErrorResult<T>
|-- nerve::io                 -- Async IO, NPY/HDF5 readers
|-- nerve::autodiff           -- Differentiable persistence
|-- nerve::serialization      -- Flaterabytesuffers, Arrow serialization
|-- nerve::validation         -- Replay, microbenchmarks
|-- nerve::instrumentation    -- Metrics, stability certificates
|-- nerve::spectral           -- Laplacians, Dirac operator, eigensolver
|-- nerve::dmt                -- Discrete Morse Theory
|-- nerve::filtration         -- VR filtration, level set, sparse VR
|-- nerve::streaming          -- Windowed PH, lock-free streaming
|-- nerve::math               -- Finite fields, precision, tolerance
|-- nerve::probabilistic      -- Randomized algorithms
```


## Core Types

```cpp
#include <nerve/types.hpp>

namespace nerve;

using Index = std::int32_t;
using Size = std::size_t;
using Dimension = std::int32_t;
using Field = double;

struct Simplex {
    Vector<Index> vertices;
    Field value;
    Index simplex_index;

    Dimension dimension() const noexcept;
    Field filtration_value() const noexcept;
    Span<const Index> asSpan() const noexcept;
};

enum class MemoryLocation { Host, Device, Pinned };

struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t available_memory;
    int compute_capability;
};
```

**Cost semantics:**

**Simplex**: Construction and copy are O(k) where k is the number of vertices, move is O(1), and destruction is O(k). **Index** and **Field** are O(1) for all operations.

<- [C++ API Overview](index.md)
