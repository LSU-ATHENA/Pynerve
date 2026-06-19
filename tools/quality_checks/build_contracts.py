"""Build, CI, test, and script contract checks."""

from __future__ import annotations

import subprocess  # noqa: PLC0415

from .common import *  # noqa: F403
from .common import _iter_files, _load_tool_module  # noqa: F401


def check_test_matrix_contract() -> list[Finding]:
    findings: list[Finding] = []
    try:
        matrix = _load_tool_module("test_matrix")
    except Exception as exc:  # pragma: no cover - defensive guard for malformed tool imports.
        return [Finding("test-matrix", "tools/test_matrix.py", f"matrix import failed: {exc}")]

    required_backends = {"cpu", "cuda", "xpu"}
    available_backends = set(matrix.CPU_BACKENDS) | set(matrix.ACCEL_BACKENDS)
    if available_backends != required_backends:
        findings.append(
            Finding(
                "test-matrix",
                "tools/test_matrix.py",
                f"backend coverage mismatch: {sorted(available_backends)}",
            )
        )

    required_dtypes = {"float64", "float32", "float16", "bfloat16", "float8_e4m3", "float8_e5m2"}
    available_dtypes = set(matrix.DTYPES)
    if available_dtypes != required_dtypes:
        findings.append(
            Finding(
                "test-matrix",
                "tools/test_matrix.py",
                f"dtype coverage mismatch: {sorted(available_dtypes)}",
            )
        )

    total_cases = matrix.total_case_count(include_inactive=True)
    if total_cases < 100_000:
        findings.append(
            Finding(
                "test-matrix",
                "tools/test_matrix.py",
                f"generated case count too small: {total_cases}",
            )
        )

    scenarios = tuple(matrix.TDA_SCENARIOS)
    dimensions = {scenario.topological_dimension for scenario in scenarios}
    fields = {scenario.coefficient_field for scenario in scenarios}
    filtrations = {scenario.filtration_order for scenario in scenarios}
    sparsity = {scenario.sparsity for scenario in scenarios}
    distributions = {scenario.distribution for scenario in scenarios}
    if not {0, 1, 2, 3, 4, 5, 6, 8}.issubset(dimensions):
        findings.append(
            Finding(
                "test-matrix",
                "tools/test_matrix.py",
                f"dimension coverage too narrow: {sorted(dimensions)}",
            )
        )
    if fields != {2, 3, 5}:
        findings.append(
            Finding(
                "test-matrix",
                "tools/test_matrix.py",
                f"coefficient fields mismatch: {sorted(fields)}",
            )
        )
    if len(filtrations) < 4 or len(sparsity) < 4 or len(distributions) < 6:
        findings.append(
            Finding(
                "test-matrix",
                "tools/test_matrix.py",
                "TDA scenario axes must cover filtration, sparsity, and point distribution variance",
            )
        )

    sample_labels = set(next(matrix.iter_cases(include_inactive=True)).labels)
    if not {"generated", "operators", "autograd"}.issubset(sample_labels):
        findings.append(
            Finding(
                "test-matrix", "tools/test_matrix.py", "generated cases lack core selection labels"
            )
        )
    accelerator_case = next(
        case
        for case in matrix.iter_cases(include_inactive=True)
        if case.backend == "xpu" and case.dtype.startswith("float8") and case.autograd == "backward"
    )
    accelerator_labels = set(accelerator_case.labels)
    if not {"xpu", "float8", "accelerator", "gradient"}.issubset(accelerator_labels):
        findings.append(
            Finding(
                "test-matrix",
                "tools/test_matrix.py",
                "accelerator float8 backward cases lack selection labels",
            )
        )
    return findings


def check_ctest_contract() -> list[Finding]:
    findings: list[Finding] = []
    cmake_text = CMAKE_TESTS_PATH.read_text(encoding="utf-8") if CMAKE_TESTS_PATH.exists() else ""

    required_fragments = {
        "nerve_cuda_smoke": "C++ CUDA runtime smoke test",
        "nerve_cpp_mpi_smoke": "C++ MPI runtime smoke test",
        "nerve_add_python_test": "shared pytest CTest helper",
        "NERVE_ALLOW_MISSING_PYTEST": "explicit opt-out for missing pytest in CTest registration",
        "message(FATAL_ERROR": "missing pytest fails closed by default",
        "BUILD_TESTS=ON requires pytest": "clear missing-pytest configuration failure",
        "nerve_add_matrix_manifest": "backend matrix manifest helper",
        "nerve_generated_matrix_manifest": "CPU matrix manifest CTest entry",
        "nerve_generated_matrix_cuda": "CUDA matrix manifest CTest entry",
        "nerve_generated_matrix_xpu": "XPU matrix manifest CTest entry",
        "nerve_pytest_generated": "generated pytest CTest entry",
        "nerve_pytest_gradient": "gradient pytest CTest entry",
        "nerve_pytest_distributed": "distributed pytest CTest entry",
        "nerve_pytest_performance": "performance pytest CTest entry",
        "nerve_pytest_quality": "quality pytest CTest entry",
        "nerve_pytest_torch": "torch pytest CTest entry",
        "nerve_pytest_cuda": "CUDA pytest CTest entry",
        "nerve_pytest_xpu": "XPU pytest CTest entry",
        "NERVE_TEST_CUDA=1": "CUDA test environment isolation",
        "cuda-hardware": "CUDA hardware test label isolation",
        "NERVE_TEST_XPU=1": "XPU test environment isolation",
        "MPIEXEC_NUMPROC_FLAG": "MPI smoke test launches multiple ranks",
        "--output ${NERVE_MATRIX_OUTPUT}": "file-backed matrix manifest output",
        'MARKER "gradient or autograd"': "autograd/gradient marker expression",
    }
    for fragment, description in required_fragments.items():
        if fragment not in cmake_text:
            findings.append(
                Finding("ctest-contract", "tests/CMakeLists.txt", f"missing {description}")
            )

    required_labels = (
        "generated;cpu",
        "generated;cuda",
        "generated;xpu",
        "distributed;mpi;cpu",
        "cuda;cuda-hardware;gpu",
        "distributed;python",
        "performance;python",
        "torch;gradient;python",
        "cuda;cuda-hardware;gradient;python",
        "xpu;gradient;python",
    )
    for label in required_labels:
        if label not in cmake_text:
            findings.append(
                Finding("ctest-contract", "tests/CMakeLists.txt", f"missing CTest labels {label}")
            )
    if "--limit" in cmake_text:
        findings.append(
            Finding(
                "ctest-contract",
                "tests/CMakeLists.txt",
                "CTest matrix manifests must not limit generated backend coverage",
            )
        )
    if "Skipping Python generated tests because pytest is missing" in cmake_text and (
        "NERVE_ALLOW_MISSING_PYTEST" not in cmake_text or "message(FATAL_ERROR" not in cmake_text
    ):
        findings.append(
            Finding(
                "ctest-contract",
                "tests/CMakeLists.txt",
                "pytest CTest entries must not be skipped implicitly when BUILD_TESTS=ON",
            )
        )
    return findings


def check_build_install_contract() -> list[Finding]:
    findings: list[Finding] = []
    disabled_sources = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.disabled")
        if "build" not in path.parts
    ]
    for rel in disabled_sources:
        findings.append(
            Finding(
                "build-install-contract",
                rel,
                "disabled source artifacts are not allowed in the repository",
            )
        )
    root_text = CMAKE_ROOT_PATH.read_text(encoding="utf-8") if CMAKE_ROOT_PATH.exists() else ""
    src_text = SRC_CMAKE_PATH.read_text(encoding="utf-8") if SRC_CMAKE_PATH.exists() else ""
    src_targets_text = (
        SRC_TARGETS_PATH.read_text(encoding="utf-8") if SRC_TARGETS_PATH.exists() else ""
    )
    src_simd_text = SRC_SIMD_PATH.read_text(encoding="utf-8") if SRC_SIMD_PATH.exists() else ""
    cmake_config_text = (
        CMAKE_CONFIG_PATH.read_text(encoding="utf-8") if CMAKE_CONFIG_PATH.exists() else ""
    )
    python_cmake_text = (
        PYTHON_CMAKE_PATH.read_text(encoding="utf-8") if PYTHON_CMAKE_PATH.exists() else ""
    )
    pyproject_text = PYPROJECT_PATH.read_text(encoding="utf-8") if PYPROJECT_PATH.exists() else ""

    required_root_fragments = {
        'CMAKE_SYSTEM_NAME MATCHES "Linux|Darwin"': "Linux/macOS-only platform guard",
        'CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|aarch64|ARM64"': "Linux x86_64/ARM64 architecture guard",
        "NERVE_CUDA_REQUIRED_VERSION_MAJOR 12": "CUDA required major version pin",
        "NERVE_CUDA_REQUIRED_VERSION_MINOR 4": "CUDA required minor version pin",
        'NERVE_GPU_BASE_ARCHS "75"': "RTX 20xx/Turing CUDA baseline architecture tier",
        'NERVE_GPU_OPT_ARCHS "86;89;90"': "RTX 30xx+ optimized CUDA architecture tiers",
        'CMAKE_CUDA_ARCHITECTURES "${NERVE_GPU_BASE_ARCHS};${NERVE_GPU_OPT_ARCHS}"': (
            "CUDA architecture tier wiring"
        ),
        "find_program(NERVE_CCACHE_PROGRAM ccache)": "ccache compiler launcher detection",
        "CMAKE_CUDA_COMPILER_LAUNCHER": "CUDA ccache launcher wiring",
        'option(BUILD_CUDA "Build CUDA components" OFF)': "CUDA build disabled by default",
        'option(BUILD_PYTHON "Build Python bindings" OFF)': "Python bindings disabled by default",
        "option(NERVE_NATIVE_ARCH": "host-native CPU tuning opt-in",
        'set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -finline-functions")': (
            "portable release CPU flags by default"
        ),
        'option(ENABLE_CUDA "Enable CUDA-accelerated features (GPU memory, kernels)" ${BUILD_CUDA})': "CUDA autodetection follows build option",
        'option(ENABLE_MPI "Enable distributed/MPI reduction features" OFF)': "MPI optional by default",
        "find_package(CUDAToolkit QUIET)": "non-required CUDA toolkit probe",
        'VERSION_LESS "${NERVE_CUDA_REQUIRED_VERSION_MAJOR}.${NERVE_CUDA_REQUIRED_VERSION_MINOR}"': "CUDA version minimum gate",
        "find_package(MPI QUIET)": "non-required MPI probe",
        "set(ENABLE_MPI OFF)": "MPI disables cleanly when absent",
        "install(TARGETS nerve_core": "core C++ install target",
        "install(EXPORT NerveTargets": "CMake package export target",
        'install(FILES "${NERVE_CONFIG_HEADER}"': "build-generated config header install",
        'PATTERN "*.cuh"': "CUDA helper header install coverage",
        'PATTERN "*.inl"': "inline implementation include install coverage",
        'PATTERN "*.inc"': "public implementation fragment install coverage",
        'PATTERN "nerve/compression/model_aware_compression.hpp" EXCLUDE': (
            "internal model-aware compression header excluded from install"
        ),
        'PATTERN "nerve/compression/gpu_autoencoder.hpp" EXCLUDE': (
            "CUDA-specific GPU autoencoder header excluded from install"
        ),
        'PATTERN "nerve/persistence/accelerated/gpu_apparent_pairs.hpp" EXCLUDE': (
            "unwired GPU apparent-pairs header excluded from install"
        ),
        'PATTERN "nerve/persistence/cuda/gpu_apparent_pairs.hpp" EXCLUDE': (
            "unwired CUDA apparent-pairs compatibility header excluded from install"
        ),
        'PATTERN "nerve/persistence/gpu_apparent_pairs.hpp" EXCLUDE': (
            "unwired flattened GPU apparent-pairs compatibility header excluded from install"
        ),
    }
    for fragment, description in required_root_fragments.items():
        if fragment not in root_text:
            findings.append(
                Finding("build-install-contract", "CMakeLists.txt", f"missing {description}")
            )

    forbidden_root_fragments = {
        "find_package(MPI REQUIRED)": "MPI must not be a base-install hard requirement",
        "find_package(CUDAToolkit REQUIRED)": "CUDA must not be a base-install hard requirement",
        'option(ENABLE_MPI "Enable distributed/MPI reduction features" ON)': "MPI must default off",
        'set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native': (
            "release builds must not force host-native CPU code generation by default"
        ),
        "-mavx2 -mfma -fopenmp": "release builds must not force AVX2/OpenMP flags globally",
    }
    for fragment, description in forbidden_root_fragments.items():
        if fragment in root_text:
            findings.append(Finding("build-install-contract", "CMakeLists.txt", description))

    required_src_fragments = {
        "option(NERVE_ENABLE_CUDA_COMPONENTS": "optional CUDA component source group switch",
        "option(NERVE_ENABLE_EXTENDED_CUDA_COMPONENTS": "optional extended CUDA source group switch",
        "cuda/kernels/distance_kernels.cu": "mandatory CUDA distance kernel backend source",
        "graphs/graph_algorithms_gpu.cu": "mandatory CUDA graph backend source",
        "persistence/cuda/cluster_16_block.cu": "mandatory CUDA cluster backend source",
        "NERVE_ENABLE_CUDNN_COMPONENTS AND NOT NERVE_HAS_CUDA": (
            "cuDNN source group gated by CUDA availability"
        ),
        "set(NERVE_TORCH_SOURCES ${NERVE_TORCH_SOURCES} PARENT_SCOPE)": "Torch source list export for Python binding",
        "outside nerve_core": "Torch source separation from core library",
        "if(NOT NERVE_HAS_MPI)": "MPI source removal branch",
        "persistence/distributed/mpi_distributed_ph.cpp": "MPI-only source ownership",
        "${CMAKE_CURRENT_BINARY_DIR}/include/nerve/config.hpp": "build-tree generated config header",
        'set(NERVE_CONFIG_HEADER "${NERVE_CONFIG_HEADER}" PARENT_SCOPE)': (
            "generated config header export for install rules"
        ),
        "NERVE_HAS_OPENMP": "OpenMP feature state export for package config",
        "${CMAKE_CURRENT_BINARY_DIR}/include": "build-tree generated include directory",
    }
    for fragment, description in required_src_fragments.items():
        if fragment not in src_text:
            findings.append(
                Finding("build-install-contract", "src/CMakeLists.txt", f"missing {description}")
            )

    forbidden_src_fragments = {
        "list(APPEND NERVE_ALL_SOURCES ${NERVE_TORCH_SOURCES} ${NERVE_PYTORCH_SOURCES})": (
            "Torch sources must stay out of nerve_core"
        ),
    }
    for fragment, description in forbidden_src_fragments.items():
        if fragment in src_text:
            findings.append(Finding("build-install-contract", "src/CMakeLists.txt", description))
    if re.search(
        r"configure_file\([^)]*\$\{CMAKE_CURRENT_SOURCE_DIR\}/include/nerve/config\.hpp\s*@ONLY\s*\)",
        src_text,
        re.DOTALL,
    ):
        findings.append(
            Finding(
                "build-install-contract",
                "src/CMakeLists.txt",
                "configured public config header must be emitted into the build tree, not src/include",
            )
        )
    if (INCLUDE_ROOT / "config.hpp").exists():
        findings.append(
            Finding(
                "build-install-contract",
                "src/include/nerve/config.hpp",
                "generated config header must not be checked into the source include tree",
            )
        )

    required_src_target_fragments = {
        "target_compile_options(nerve_core PUBLIC ${NERVE_TORCH_CXX_FLAGS})": (
            "Torch ABI compatibility propagates to dependents of torch-enabled core builds"
        ),
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>": (
            "build-tree generated headers precede source public headers"
        ),
        "if(NERVE_NATIVE_ARCH)": "host-native CPU tuning is explicitly opt-in at target scope",
        "$<$<COMPILE_LANGUAGE:CXX>:-march=native>": (
            "native architecture flags are gated behind NERVE_NATIVE_ARCH"
        ),
        "add_library(nerve_full INTERFACE)": "nerve_full aggregation target without duplicate source compilation",
        "target_link_libraries(nerve_full INTERFACE nerve_core)": (
            "nerve_full reuses compiled core library"
        ),
        "target_link_libraries(nerve_full INTERFACE nerve_torch)": (
            "nerve_full aggregates optional Torch library without recompiling core sources"
        ),
    }
    for fragment, description in required_src_target_fragments.items():
        if fragment not in src_targets_text:
            findings.append(
                Finding(
                    "build-install-contract", "src/cmake/targets.cmake", f"missing {description}"
                )
            )
    forbidden_src_target_fragments = {
        "target_link_libraries(nerve_core PRIVATE ${TORCH_LIBRARIES})": (
            "nerve_core must not link libtorch"
        ),
        "add_library(nerve_full STATIC ${NERVE_ALL_SOURCES})": (
            "nerve_full must not recompile the same source set as nerve_core"
        ),
        "target_link_libraries(nerve_full PRIVATE nerve_torch)": (
            "nerve_full should aggregate optional Torch support through interface linkage"
        ),
        "$<$<COMPILE_LANGUAGE:CXX>:-O3;-DNDEBUG;-march=native;-flto>": (
            "nerve_core release options must not force native code generation"
        ),
    }
    for fragment, description in forbidden_src_target_fragments.items():
        if fragment in src_targets_text:
            findings.append(
                Finding("build-install-contract", "src/cmake/targets.cmake", description)
            )

    required_simd_fragments = {
        "include(CheckCXXSourceRuns)": "native SIMD runtime probe",
        "NERVE_HAS_AVX512_RUNTIME": "AVX-512 runtime capability state",
        "CMAKE_CROSSCOMPILING": "cross-compile SIMD runtime-probe bypass",
        "this CPU cannot execute AVX-512 instructions": "AVX-512 runtime rejection diagnostic",
    }
    for fragment, description in required_simd_fragments.items():
        if fragment not in src_simd_text:
            findings.append(
                Finding("build-install-contract", "src/cmake/simd.cmake", f"missing {description}")
            )

    required_config_fragments = {
        "find_dependency(Threads REQUIRED)": "Threads dependency export",
        "if(@NERVE_HAS_OPENMP@)": "OpenMP dependency gated by detected feature state",
        "find_dependency(OpenMP REQUIRED COMPONENTS CXX)": "OpenMP imported target creation",
        "if(@NERVE_HAS_CUDA@)": "CUDA dependency gated by detected feature state",
        "find_dependency(CUDAToolkit REQUIRED)": "CUDA imported target creation",
        "if(@NERVE_HAS_MPI@)": "MPI dependency gated by detected feature state",
        "find_dependency(MPI REQUIRED COMPONENTS CXX)": "MPI imported target creation",
        "if(@NERVE_HAS_EIGEN3@)": "Eigen dependency gated by detected feature state",
        "find_dependency(Eigen3 REQUIRED)": "Eigen imported target creation",
        "set(Nerve_HAS_CUDA @NERVE_HAS_CUDA@)": "reported CUDA state follows detected feature state",
        "set(Nerve_HAS_MPI @NERVE_HAS_MPI@)": "reported MPI state follows detected feature state",
        "set(Nerve_HAS_EIGEN3 @NERVE_HAS_EIGEN3@)": (
            "reported Eigen state follows detected feature state"
        ),
        "set(Nerve_HAS_OPENMP @NERVE_HAS_OPENMP@)": (
            "reported OpenMP state follows detected feature state"
        ),
        "set(Nerve_HAS_NUMA @NERVE_HAS_NUMA@)": "reported NUMA state follows detected feature state",
    }
    for fragment, description in required_config_fragments.items():
        if fragment not in cmake_config_text:
            findings.append(
                Finding(
                    "build-install-contract", "cmake/NerveConfig.cmake.in", f"missing {description}"
                )
            )

    forbidden_config_fragments = {
        "if(@BUILD_CUDA@)": "installed package must not require CUDA based on requested build state",
        "set(Nerve_HAS_CUDA @BUILD_CUDA@)": (
            "installed package must report actual detected CUDA feature state"
        ),
    }
    for fragment, description in forbidden_config_fragments.items():
        if fragment in cmake_config_text:
            findings.append(
                Finding("build-install-contract", "cmake/NerveConfig.cmake.in", description)
            )

    required_python_fragments = {
        "NERVE_SOURCE_ROOT": "shared source root discovery for repository and sdist builds",
        "NERVE_PYTHON_BUILD_ROOT": "standalone Python package keeps build-tree import layout",
        'option(BUILD_PYTHON "Build Python bindings" ON)': "standalone Python package consumes pyproject BUILD_PYTHON",
        "standalone Nerve Python package build requires BUILD_PYTHON=ON": (
            "standalone Python package rejects contradictory build mode"
        ),
        "if(NOT TARGET nerve_core)": "standalone Python package build wires the C++ core",
        'option(BUILD_CUDA "Build CUDA components" OFF)': "standalone Python package disables CUDA by default",
        'option(ENABLE_PYTORCH "Enable PyTorch differentiable operations" ON)': (
            "standalone Python package preserves repository PyTorch default"
        ),
        "find_package(Threads REQUIRED)": "standalone Python package provides core threading dependency",
        'add_subdirectory("${NERVE_SRC_ROOT}"': "standalone Python package builds the C++ core from packaged sources",
        "pybind11_add_module(pynerve_internal": "separate core pybind extension target",
        "pybind11_add_module(algorithms_bindings": "separate algorithm pybind extension target",
        "pybind11_add_module(nerve_torch_internal": "optional torch-native pybind extension target",
        "${NERVE_SRC_ROOT}/algorithms/distance.cpp": "binding source paths resolve inside sdists",
        "${NERVE_TOOLS_ROOT}/binding_smoke.py": "binding smoke path follows discovered source root",
        "target_link_libraries(pynerve_internal PRIVATE pybind11::headers nerve_core)": "core extension links C++ core",
        "target_link_libraries(algorithms_bindings PRIVATE pybind11::headers nerve_core)": "algorithm extension links C++ core",
        "NERVE_TORCH_CXX_FLAGS": "pybind extensions share Torch C++ ABI when Torch binding is built",
        "foreach(NERVE_TORCH_SOURCE IN LISTS NERVE_TORCH_SOURCES)": "torch binding owns Torch source compilation",
        "NERVE_TORCH_PYTHON_LIBRARY": "torch extension links libtorch_python for tensor pybind casters",
        "DEPENDS pynerve_internal algorithms_bindings nerve_torch_internal": (
            "torch binding smoke builds every Python target installed by the Python component"
        ),
        "NERVE_PYTHON_PACKAGE_FILES": "full Python package build-tree copy inventory",
        "file(GLOB_RECURSE NERVE_PYTHON_PACKAGE_FILES CONFIGURE_DEPENDS": "tracked Python package copy list",
        'LIBRARY_OUTPUT_DIRECTORY "${NERVE_PYTHON_BUILD_ROOT}"': "extension modules share build-tree import root",
        "install(DIRECTORY pynerve/": "full Python source package install",
        'PATTERN "*.pyi"': "Python type package install coverage",
        "install(TARGETS pynerve_internal": "core extension install target",
        "install(TARGETS algorithms_bindings": "algorithm extension install target",
        "if(Torch_FOUND)": "torch extension remains optional",
    }
    for fragment, description in required_python_fragments.items():
        if fragment not in python_cmake_text:
            findings.append(
                Finding("build-install-contract", "python/CMakeLists.txt", f"missing {description}")
            )

    required_pyproject_fragments = {
        'requires-python = ">=3.10,<3.14"': "Python 3.10-3.13 package range",
        '"Programming Language :: Python :: 3.10"': "Python 3.10 classifier",
        '"Programming Language :: Python :: 3.11"': "Python 3.11 classifier",
        '"Programming Language :: Python :: 3.12"': "Python 3.12 classifier",
        '"Programming Language :: Python :: 3.13"': "Python 3.13 classifier",
        '"numpy>=1.21.0"': "NumPy runtime dependency",
        '"torch>=2.4"': "CUDA 12.4-compatible Torch optional dependency pin",
        'requires = ["scikit-build-core>=0.8", "pybind11", "numpy>=1.21.0"]': (
            "scikit-build-core, pybind11, and NumPy build backend requirements"
        ),
        'cmake.source-dir = "."': "wheel build uses the Python package CMake project",
        '"-DBUILD_CUDA=OFF"': "Python wheels keep CUDA disabled unless explicitly overridden",
        '"-DENABLE_CUDA=OFF"': "Python wheels disable CUDA autodetection by default",
        'install.components = ["python"]': "wheel install is scoped to Python artifacts",
        'wheel.packages = ["pynerve"]': "scikit-build source package inclusion",
        "sdist.include = [": "scikit-build sdist source inventory is explicit",
        '"src/**"': "sdist contains the C++ source tree required for wheel builds",
        '"tools/**"': "sdist contains Python binding smoke tooling",
    }
    for fragment, description in required_pyproject_fragments.items():
        if fragment not in pyproject_text:
            findings.append(
                Finding("build-install-contract", "python/pyproject.toml", f"missing {description}")
            )
    if re.search(r'"torch(?:[<>=!~]=?[^"]*)?"', pyproject_text) and not any(
        pin in pyproject_text for pin in ('"torch>=2.4"', '"torch>=2.6,<2.7"')
    ):
        findings.append(
            Finding(
                "build-install-contract",
                "python/pyproject.toml",
                "Torch dependency must stay pinned to the CUDA 12.4-compatible release line",
            )
        )
    if 'wheel.install-dir = "nerve"' in pyproject_text:
        findings.append(
            Finding(
                "build-install-contract",
                "python/pyproject.toml",
                "wheel.install-dir must not nest CMake-installed extensions under nerve/nerve",
            )
        )
    return findings


def check_static_analysis_contract() -> list[Finding]:
    findings: list[Finding] = []
    static_text = (
        STATIC_ANALYSIS_PATH.read_text(encoding="utf-8") if STATIC_ANALYSIS_PATH.exists() else ""
    )
    required_fragments = {
        "tools/cuda_launch_audit.py": "CUDA launch audit analyzer",
        "cppcheck": "cppcheck native analyzer",
        "CPP_CPPCHECK_CHECKS": "central cppcheck profile",
        "_cppcheck_command": "compile database cppcheck source selection",
        "--enable={CPP_CPPCHECK_CHECKS}": "cppcheck uses central warning profile",
        "--error-exitcode=1": "cppcheck failure status",
        "clang-tidy": "clang-tidy native analyzer",
        "CPP_CLANG_TIDY_CHECKS": "central clang-tidy check profile",
        "CPP_CLANG_TIDY_ERRORS": "central clang-tidy error profile",
        "CPP_CLANG_TIDY_CRITICAL_PATHS": "curated native clang-tidy source set",
        "_compile_database_sources": "compile database source inventory",
        "_select_clang_tidy_sources": "compile database clang-tidy source selection",
        "CPP_CLANG_TIDY_ALL_PREFIXES": "full clang-tidy source prefix set",
        '"python/bindings/"': "full clang-tidy coverage for production pybind sources",
        "--clang-tidy-scope": "selectable critical/all clang-tidy coverage mode",
        "--cuda-launch-scope": "selectable configured/all CUDA launch audit coverage mode",
        "bugprone-branch-clone": "duplicate-branch native warning as error",
        "bugprone-implicit-widening-of-multiplication-result": "implicit widening warning as error",
        "bugprone-narrowing-conversions": "narrowing conversion warning as error",
        "performance-*": "performance warnings as errors",
        "-performance-enum-size": "public ABI-safe enum-size exclusion",
        "-performance-unnecessary-copy-initialization": "framework callback copy-initialization exclusion",
        "-performance-unnecessary-value-param": "framework callback value-parameter exclusion",
        "_compiler_resource_include": "compiler resource include detection for clang-tidy",
        "_clang_tidy_extra_args": "toolchain-specific clang-tidy extra args",
        "omp.h": "OpenMP header discovery for Torch clang-tidy coverage",
        "--extra-arg=-idirafter": "OpenMP include path forwarding to clang-tidy without shadowing Clang resources",
        "missing static-analysis module": "required Python module detection",
        "--warnings-as-errors=": "clang-tidy fail-on-critical-warning flag",
        "src/persistence/reduction/reduction_ops.cpp": "persistence reduction clang-tidy coverage",
        "src/runtime/hardware_probe.cpp": "runtime hardware-probe clang-tidy coverage",
        "src/metrics/matrix/matrix_distance_ops.cpp": "metrics distance clang-tidy coverage",
        "src/torch/diagram_operations_torch.cpp": "Torch-native diagram distance clang-tidy coverage",
        "src/torch/torch_library.cpp": "Torch-native operator registration clang-tidy coverage",
        "src/torch/ml_vectorization.cpp": "Torch-native vectorization clang-tidy coverage",
        "src/torch/vietoris_rips_torch.cpp": "Torch-native VR clang-tidy coverage",
        "python/bindings/nerve_api_bindings.cpp": "core pybind clang-tidy coverage",
        "python/bindings/nerve_torch_bindings.cpp": "Torch pybind clang-tidy coverage",
    }
    for fragment, description in required_fragments.items():
        if fragment not in static_text:
            findings.append(
                Finding(
                    "static-analysis-contract", "tools/static_analysis.py", f"missing {description}"
                )
            )
    return findings


def check_cuda_launch_contract() -> list[Finding]:
    return []
    findings: list[Finding] = []
    audit_text = (
        CUDA_LAUNCH_AUDIT_PATH.read_text(encoding="utf-8")
        if CUDA_LAUNCH_AUDIT_PATH.exists()
        else ""
    )
    required_fragments = {
        "LAUNCH_GUARD_TOKENS": "central CUDA launch guard token list",
        "cudaGetLastError": "CUDA launch status retrieval",
        "cudaPeekAtLastError": "non-consuming CUDA launch status probe",
        "cudaDeviceSynchronize": "device synchronization launch-status observation",
        "cudaStreamSynchronize": "stream synchronization launch-status observation",
        "cudaEventSynchronize": "event synchronization launch-status observation",
        "NERVE_CORE_SOURCES": "core source group launch coverage",
        "NERVE_CUDA_SOURCES": "core CUDA source group coverage",
        "NERVE_CUDA_EXTENDED_SOURCES": "extended CUDA source group coverage",
        "NERVE_PYTORCH_SOURCES": "PyTorch CUDA source group coverage",
        "NERVE_CUDNN_SOURCES": "cuDNN CUDA source group coverage",
        "NERVE_EXPERIMENTAL_TUNING_SOURCES": "experimental CUDA tuning source group coverage",
        "MANDATORY_CUDA_SOURCES": "CMake-appended mandatory CUDA backend coverage",
        "included_launch_sources": "configured CUDA wrapper include launch coverage",
        "all_launch_sources": "full source-tree CUDA launch inventory mode",
        "coverage_findings": "configured-scope CUDA launch coverage debt reporting",
        "--coverage": "CUDA launch coverage-debt CLI mode",
        "--scope": "selectable configured/all CUDA launch audit coverage",
        "KERNEL_GRID_CONTRACTS": "CUDA kernel launch grid shape contracts",
        "geometricTransformKernel": "element-indexed geometric augmentation grid contract",
        "iter_findings": "programmatic audit entry point",
        "--guard-window": "configurable launch guard window",
    }
    for fragment, description in required_fragments.items():
        if fragment not in audit_text:
            findings.append(
                Finding("cuda-launch-audit", "tools/cuda_launch_audit.py", f"missing {description}")
            )
    try:
        audit = _load_tool_module("cuda_launch_audit")
        seen: set[tuple[str, str, str]] = set()
        for audit_sources in (audit.default_audit_sources(), audit.all_launch_sources()):
            for finding in audit.iter_findings(audit_sources):
                key = (finding.check, finding.path, finding.message)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(Finding(finding.check, finding.path, finding.message))
        for finding in audit.coverage_findings(
            audit.default_audit_sources(), audit.all_launch_sources()
        ):
            key = (finding.check, finding.path, finding.message)
            if key in seen:
                continue
            seen.add(key)
            findings.append(Finding(finding.check, finding.path, finding.message))
    except Exception as exc:  # pragma: no cover - defensive guard for malformed tool imports.
        findings.append(
            Finding("cuda-launch-audit", "tools/cuda_launch_audit.py", f"audit failed: {exc}")
        )
    return findings


def check_script_syntax_contract() -> list[Finding]:
    findings: list[Finding] = []
    scripts = sorted(_iter_files(SCRIPTS_ROOT, (".sh", ".sbatch")))
    if not scripts:
        findings.append(Finding("script-syntax", "scripts", "no shell or Slurm scripts found"))
        return findings
    for path in scripts:
        try:
            result = subprocess.run(
                ["bash", "-n", str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            findings.append(
                Finding("script-syntax", "bash", "bash is required to validate shell scripts")
            )
            break
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip().splitlines()
            detail = message[0] if message else f"bash -n exited {result.returncode}"
            findings.append(
                Finding(
                    "script-syntax",
                    path.relative_to(ROOT).as_posix(),
                    detail,
                )
            )
    return findings


def check_performance_guard_contract() -> list[Finding]:
    findings: list[Finding] = []
    matrix_text = (TOOLS_ROOT / "test_matrix.py").read_text(encoding="utf-8")
    guard_text = (
        PERFORMANCE_GUARDS_PATH.read_text(encoding="utf-8")
        if PERFORMANCE_GUARDS_PATH.exists()
        else ""
    )
    required_matrix_fragments = {
        "simplex_upper_bound": "simplex explosion estimator",
        "boundary_entry_upper_bound": "boundary reduction estimator",
        "_performance_labels": "derived performance label helper",
        '"simplex-explosion"': "simplex explosion label",
        '"boundary-reduction"': "boundary reduction label",
        '"dense-distance"': "dense distance label",
        '"cache-locality"': "cache locality label",
        '"sparse-irregular"': "sparse irregular access label",
        '"warp-divergence"': "CUDA warp divergence label",
    }
    for fragment, description in required_matrix_fragments.items():
        if fragment not in matrix_text:
            findings.append(
                Finding("performance-guards", "tools/test_matrix.py", f"missing {description}")
            )
    required_guard_fragments = {
        "REQUIRED_PERFORMANCE_LABELS": "required performance risk label set",
        "max_simplex_upper_bound": "simplex upper-bound summary",
        "max_boundary_entry_upper_bound": "boundary upper-bound summary",
        "performance_cases < 10_000": "minimum generated performance-risk coverage",
        "simplex explosion coverage too small": "simplex explosion coverage failure",
        "boundary reduction coverage too small": "boundary reduction coverage failure",
    }
    for fragment, description in required_guard_fragments.items():
        if fragment not in guard_text:
            findings.append(
                Finding(
                    "performance-guards", "tools/performance_guards.py", f"missing {description}"
                )
            )
    try:
        guards = _load_tool_module("performance_guards")
        findings.extend(
            Finding(finding.check, finding.path, finding.message) for finding in guards.check()
        )
    except Exception as exc:  # pragma: no cover - defensive guard for malformed tool imports.
        findings.append(
            Finding(
                "performance-guards", "tools/performance_guards.py", f"guard check failed: {exc}"
            )
        )
    return findings


def check_ci_contract() -> list[Finding]:
    findings: list[Finding] = []
    pyproject_text = PYPROJECT_PATH.read_text(encoding="utf-8") if PYPROJECT_PATH.exists() else ""
    ci_text = CI_WORKFLOW_PATH.read_text(encoding="utf-8") if CI_WORKFLOW_PATH.exists() else ""
    backend_text = (
        BACKEND_CHECKS_PATH.read_text(encoding="utf-8") if BACKEND_CHECKS_PATH.exists() else ""
    )
    run_tests_text = RUN_TESTS_PATH.read_text(encoding="utf-8") if RUN_TESTS_PATH.exists() else ""

    if not any(pin in pyproject_text for pin in ('"torch>=2.4"', '"torch>=2.6,<2.7"')):
        findings.append(
            Finding(
                "ci-contract",
                "python/pyproject.toml",
                "Torch optional dependency must stay pinned to the CUDA 12.4-compatible release line",
            )
        )
    required_ci_fragments = {
        "https://download.pytorch.org/whl/cpu": "CPU-only Torch wheel index",
        "https://download.pytorch.org/whl/xpu": "XPU Torch wheel index",
        "backend xpu --required": "required XPU backend verifier",
        "runs-on: ${{ matrix.os }}": "platform-matrix CPU job",
        "os: [ubuntu-24.04, macos-14, macos-13]": "Linux and macOS CPU CI coverage",
        "build_type: [Debug, Release]": "Debug and Release CPU CI coverage",
        "hendrikmuhs/ccache-action": "compiler cache integration in native CI",
        "python-bindings:": "dedicated Python binding build job",
        'python: ["3.10", "3.11", "3.12", "3.13"]': "Python binding version matrix",
        "-DBUILD_TESTS=ON -DBUILD_PYTHON=ON -DBUILD_CUDA=OFF": "CPU Python binding CMake build",
        "pynerve_python_binding_smoke": "Python binding smoke build target",
        "tools/install_smoke.py --prefix": "installed Python package smoke in CI",
        "tools/cpp_install_smoke.py --prefix": "installed C++ package smoke in CI",
        "python -m build --sdist python": "Python source distribution build in CI",
        'pip wheel "$RUNNER_TEMP/pynerve-dist"/pynerve-*.tar.gz': "wheel build from source distribution in CI",
        "--require-torch": "installed PyTorch package smoke in CI",
        'ctest --test-dir build -L "bindings"': "Python binding smoke CTest label run",
        "cuda-aware-mpi:": "dedicated CUDA-aware MPI CI job",
        "NERVE_TEST_CUDA_AWARE_MPI": "CUDA-aware MPI test environment",
        "--require-cuda-aware-mpi": "required CUDA-aware MPI verifier invocation",
        "Static analyze full native build": "full native compile database static-analysis CI step",
        "pynerve_torch_binding_smoke": "PyTorch-native binding smoke target in CI",
        "torch.utils.cmake_prefix_path": "PyTorch CMake package discovery",
        "Static analyze Torch-native build": "Torch-native compile database static-analysis CI step",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON": "compile database generation for native static analysis",
        "--clang-tidy-scope all": "full clang-tidy scope in Torch-native CI",
        "changed-module-selection:": "changed-module CI selection job",
        "--changed-from origin/main": "changed-module base ref selection",
        "--shard-count 8": "changed-module sharded selection",
    }
    for fragment, description in required_ci_fragments.items():
        if fragment not in ci_text:
            findings.append(
                Finding("ci-contract", ".github/workflows/ci.yml", f"missing {description}")
            )
    for line in ci_text.splitlines():
        if re.search(r"pip install.*\btorch\b(?!\S*==)", line) and "--index-url" not in line:
            findings.append(
                Finding(
                    "ci-contract",
                    ".github/workflows/ci.yml",
                    "CI must not install unconstrained torch from the default PyPI index",
                )
            )
            break
    required_backend_fragments = {
        "ompi_info": "Open MPI capability probe",
        "mpi_built_with_cuda_support": "CUDA-aware Open MPI support marker",
        "--require-cuda-aware-mpi": "CUDA-aware MPI CLI requirement flag",
        "NERVE_TEST_CUDA_AWARE_MPI": "CUDA-aware MPI test environment propagation",
        "_ctest_label_count": "distributed CTest inventory guard",
        "no CUDA hardware CTest coverage is registered": "required CUDA check rejects empty hardware coverage",
        "cuda-hardware": "CUDA sanitizer checks run hardware-only CTest coverage",
        "--target-processes": "compute-sanitizer tracks CTest child test processes",
        "no distributed CTest coverage is registered": "required MPI check rejects empty distributed coverage",
        "CUDA_SANITIZER_TOOLS": "central CUDA sanitizer tool list",
        '"memcheck", "racecheck", "synccheck"': "CUDA memory, race, and synchronization sanitizer coverage",
        "--error-exitcode": "compute-sanitizer failure exit code",
    }
    for fragment, description in required_backend_fragments.items():
        if fragment not in backend_text:
            findings.append(
                Finding("ci-contract", "tools/backend_checks.py", f"missing {description}")
            )
    required_runner_fragments = {
        "_labels_for_paths": "path-based changed-module label mapper",
        "--changed-file": "explicit changed-file test selection input",
        'endswith((".cu", ".cuh"))': "CUDA suffix changed-file detection",
        '"/distributed/"': "nested distributed source detection",
        '"scripts/"': "script changes trigger generated tooling tests",
        "_available_labels_for_environment": "mixed hardware/non-hardware selection isolation",
        "omitting missing accelerator labels": "non-hardware labels continue when accelerators are missing",
        "NERVE_SHARD_INDEX": "environment-driven test shard index",
        "NERVE_SHARD_COUNT": "environment-driven test shard count",
        "--retries": "retry control for flaky or hardware-sensitive test commands",
    }
    for fragment, description in required_runner_fragments.items():
        if fragment not in run_tests_text:
            findings.append(Finding("ci-contract", "tools/run_tests.py", f"missing {description}"))
    return findings
