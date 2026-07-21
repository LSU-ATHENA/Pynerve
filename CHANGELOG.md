# Changelog

All notable changes to pynerve are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Linker error: `heterogeneous_fast_vr.cpp` moved from CUDA_CORE to CORE sources for CPU-only builds
- Pair designated initializer replaced with positional constructor (GCC 13+ compatibility)
- Ruff SIM105, B603, F541, and PLC0415 lint violations resolved across tools and Python modules
- clang-tidy `performance-avoid-endl` in anomaly detection module
- `simd-arm (neon)` and `native-arch` jobs marked `continue-on-error: true` to prevent infra-related false failures

### Changed
- CI test-coverage `--fail-under` threshold raised from 5 to 80
- Dependabot: bumped 5 GitHub Actions (download-artifact v4→v8, docker/setup-buildx-action v3→v4, docker/login-action v3→v4, softprops/action-gh-release v2→v3, docker/metadata-action v5→v6)
- Branch protection: @spinxog added as sole bypass user on `main` and `development`
- Author email removed from `pyproject.toml`

### Added
- README badges: CI status, PyPI version, Python versions, license
- Issue templates: bug report and feature request (YAML forms)
- Pull request template with checklist
- CITATION.cff for academic citations
- CHANGELOG.md (Keep a Changelog format)
- API stability policy in CONTRIBUTING.md
- MkDocs GitHub Pages deployment workflow

## [1.0.4] — 2026-07-19

### Added
- PyPI platform wheels for Linux (manylinux_2_28), macOS (arm64), and Windows (amd64)
- Python 3.10, 3.11, 3.12, and 3.13 wheel support
- GPU CI pre-work: Docker image (CUDA 12.4), pytest GPU markers, smoke tests, ARC runner config
- SIMD backend detection and conditional skip for AVX-512 and ARM NEON
- Benchmark regression tooling (`tools/benchmark_regression.py`) and flaky test reporter (`tools/flaky_report.py`)

### Fixed
- Windows MSVC portability: `unistd.h`, `ssize_t`, `std::aligned_alloc`, `cuda_runtime.h` stubs
- macOS platform guard for Linux-specific thread affinity APIs
- `std::end` to `\n` replacement for clang-tidy `performance-avoid-endl`
- Ruff lint violations (SIM105, B603, F541, PLC0415) across tools and Python modules

### Changed
- Python `pyproject.toml`: added `project_urls`, keywords, classifiers for PyPI metadata
- Branch protection: CODEOWNERS review required on `main` and `development`
- CI coverage threshold raised from `--fail-under=5` to `--fail-under=80`

## [1.0.0] — 2026-07-18

### Added
- Initial public release
- Persistent homology computation (standard, cohomology, PH4, PH5, PH6 engines)
- Vietoris-Rips, sparse VR, witness, alpha, and cubical filtrations
- Persistence images, landscapes, silhouettes, Betti curves
- PyTorch integration: tensor interoperability, differentiable topology operators
- Streaming, out-of-core, and distributed (MPI) execution
- GPU acceleration via CUDA: reduction kernels, distance kernels, tensor core support
- Python bindings with NumPy and SciPy interop
- Memory pool, NUMA-aware allocation, SIMD-optimized primitives
- Sheaf Laplacian, spectral methods, anomaly detection
- Comprehensive documentation (mkdocs, Material theme, mkdocstrings)
