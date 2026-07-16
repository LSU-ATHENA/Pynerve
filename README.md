# Pynerve

**Topological Data Analysis for Python and C++.**

Pynerve is a high-performance framework for Topological Data Analysis (TDA), providing persistent homology, filtration construction, topological feature extraction, differentiable topology, and large-scale computational pipelines through a unified Python and C++ interface.

Built around a modern C++ core and designed for both research and production environments, Pynerve scales from exploratory analysis on a laptop to distributed workloads spanning multiple nodes and accelerators.

```python
import pynerve
import numpy as np

points = np.random.randn(10000, 3)

result = pynerve.compute_persistence(
    points,
    max_dim=2,
    max_radius=1.5,
)

print(result.betti_numbers)
```


## Why Pynerve?

Topological Data Analysis has matured into an important tool for understanding complex datasets, yet many existing workflows remain fragmented.

Researchers often combine multiple libraries for filtration construction, persistent homology, feature extraction, visualization, machine learning integration, distributed execution, and large-scale processing.

Pynerve was created to provide a unified environment for these tasks.

The project combines a high-performance computational core with an accessible Python interface while remaining suitable for large-scale scientific computing, machine learning, and research workflows.


## Core Capabilities

### Persistent Homology

Compute persistence diagrams and barcodes from a variety of filtration types.

Supported workflows include:

* Vietoris-Rips complexes
* Sparse Vietoris-Rips filtrations
* Witness constructions
* Alpha-style geometric filtrations
* Cubical filtrations
* Graph filtrations
* User-defined filtrations

### Filtration Construction

Build filtrations from:

* Point clouds
* Distance matrices
* k-nearest-neighbor graphs
* Weighted graphs
* Scalar fields
* Volumetric data
* Time-varying datasets

### Topological Feature Extraction

Transform persistence information into machine-learning-ready representations.

Available methods include:

* Persistence images
* Persistence landscapes
* Persistence silhouettes
* Betti curves
* Persistence statistics
* Vectorized descriptors

### Machine Learning Integration

Pynerve includes native support for modern machine learning workflows.

Features include:

* PyTorch tensor interoperability
* Differentiable topological operators
* Topological regularization
* Learned filtration layers
* Topological loss functions
* GPU-aware tensor pipelines

### Large-Scale Computing

Pynerve is designed for datasets that exceed the limits of traditional in-memory workflows.

Capabilities include:

* Streaming computations
* Out-of-core processing
* Distributed execution
* Multi-node workflows
* Multi-GPU execution
* Parallel persistence pipelines


## Design Goals

Pynerve is guided by several principles:

### Performance

Algorithms should remain practical on real datasets rather than only on benchmark examples.

### Scalability

Methods should continue to operate as datasets grow from thousands to millions of observations.

### Reproducibility

Scientific results should be deterministic and reproducible across platforms and environments.

### Accessibility

Advanced topology should be available through clear Python APIs without sacrificing low-level control.

### Extensibility

Researchers should be able to implement new filtrations, descriptors, and workflows without modifying the core library.


## Quick Start

### Persistent Homology from a Point Cloud

```python
import pynerve
import numpy as np

points = np.random.normal(size=(5000, 3))

result = pynerve.compute_persistence(
    points,
    max_dim=2,
    max_radius=2.0,
)

print(f"Betti numbers: {result.betti_numbers}")
print(f"Found {len(result.pairs)} persistence pairs")
```

### Distance Matrix Workflow

```python
import pynerve
import numpy as np

points = np.random.randn(200, 3)
D = np.linalg.norm(points[:, None] - points[None, :], axis=2)

result = pynerve.compute_persistence(
    D,
    max_dim=1,
    max_radius=2.0,
)
```

### Persistence Images

```python
import pynerve
import numpy as np

diagram = np.array([
    [0.0, 1.0, 0],
    [0.5, 2.0, 1],
    [1.0, 3.0, 2],
])

image = pynerve.persistence_image(
    diagram,
    resolution=(64, 64),
    sigma=0.1,
)
```

### PyTorch Integration

```python
import torch
import pynerve

x = torch.randn(1024, 3)

result = pynerve.compute_persistence(x, max_dim=2, max_radius=1.0)
```



## Architecture

### Core Library

A modern C++ implementation containing topology algorithms, filtration machinery, numerical kernels, and execution infrastructure.

### Python Bindings

A Python interface exposing the majority of library functionality while preserving performance-critical execution paths.

### Machine Learning Components

PyTorch-integrated modules for differentiable topology and topological deep learning.

### Distributed Runtime

Infrastructure for large-scale and multi-node computations.


## Documentation

Documentation is organized into several sections.

| Section               | Description                     |
| --------------------- | ------------------------------- |
| Getting Started       | Installation and first examples |
| Tutorials             | Guided workflows                |
| API Reference         | Complete API documentation      |
| Algorithms            | Mathematical background         |
| Machine Learning      | PyTorch integration             |
| Distributed Computing | Large-scale execution           |
| Developer Guide       | Building and extending Pynerve    |


## Installation

### Pip

```bash
pip install pynerve
```

### Optional PyTorch Support

```bash
pip install "pynerve[torch]"
```

### Full Installation

```bash
pip install "pynerve[all]"
```

### Conda(not supported yet)


### Build From Source

Requirements:

* C++20 compiler
* CMake 3.20+
* Python 3.10+

```bash
git clone https://github.com/LSU-ATHENA/Pynerve
cd Pynerve

pip install -e ./python
```


## Platform Support

| Platform | Status |
| -------- | ------ |
| Linux    | Supported (primary target) |
| macOS    | Supported |
| Windows  | Supported (native via `nerve::sys` platform abstraction layer - MSVC, Clang-cl) |


## GPU Support

Pynerve requires an NVIDIA GPU with compute capability 7.5 or higher (GeForce RTX 20xx / Turing or later).

| Generation | Compute Capability | Level of Support |
| ---------- | ------------------ | ---------------- |
| Turing (20xx) | 7.5 | Full support, basic GPU acceleration |
| Ampere (30xx) / newer | 8.0+ | Full support with architecture-specific optimizations |

GPUs older than the Turing architecture (compute capability < 7.5) are not supported.


## Project Status

Pynerve is currently maintained by a single developer and is open to future contributors.

Current work focuses on:

* Expanded filtration support
* Additional topological descriptors
* Improved distributed workflows
* Enhanced machine learning integration
* Documentation and tutorial coverage

The public API is stabilizing, although some advanced modules may continue to evolve between releases.


## Contributing

Contributions, bug reports, feature requests, and research collaborations are welcome.

Please see the contributor documentation for development setup, coding standards, and submission guidelines.


## Development History

Pynerve was initially developed and validated on a consumer workstation:

- AMD Ryzen 7 2700X
- NVIDIA GTX 1070
- 16GB system memory

The project was designed to support accessible development environments while
scaling to larger computational workloads through GPU acceleration and HPC
resources.



## Acknowledgements

Pynerve development and benchmarking were supported in part by HPC
resources provided by Louisiana State University.

The author gratefully acknowledges access to these computational
resources, which enabled large-scale experimentation and validation.


## Citation

If Pynerve contributes to published research, please cite:

```bibtex
@software{Pynerve,
  title={Pynerve},
  version={1.0.0},
  year={2026},
  author={Pradip Debnath and Keith G. Mills},
  url={https://github.com/LSU-ATHENA/Pynerve}
}
```


## License

MIT License.
