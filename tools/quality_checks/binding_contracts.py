"""Python, pybind, Torch, and binding smoke contract checks."""

from __future__ import annotations

from .common import *  # noqa: F403
from .common import (
    _check_torch_operator_schema_text,
    _pybind_module_defs,
)  # noqa: F401


def _check_expected_args(
    check_name: str,
    path: str,
    definitions: dict[str, list[str]],
    expected_args: dict[str, tuple[str, ...]],
) -> list[Finding]:
    findings: list[Finding] = []
    for name, expected in expected_args.items():
        actual = tuple(definitions.get(name, ()))
        if actual != expected:
            findings.append(
                Finding(
                    check_name,
                    path,
                    f"{name} argument schema mismatch: expected {expected}, got {actual}",
                )
            )
    return findings


def check_algorithm_bindings_schema() -> list[Finding]:
    findings: list[Finding] = []
    text = (
        ALGORITHMS_BINDINGS_PATH.read_text(encoding="utf-8")
        if ALGORITHMS_BINDINGS_PATH.exists()
        else ""
    )
    cmake_text = PYTHON_CMAKE_PATH.read_text(encoding="utf-8") if PYTHON_CMAKE_PATH.exists() else ""
    definitions = _pybind_module_defs(ALGORITHMS_BINDINGS_PATH)
    findings.extend(
        _check_expected_args(
            "algorithm-bindings-schema",
            "python/bindings/nerve_algorithms_bindings.cpp",
            definitions,
            {
                "pairwise_distances": ("points", "n_points", "dim"),
                "knn": ("points", "n_points", "dim", "k"),
                "connected_components": ("graph",),
            },
        )
    )

    required_fragments = {
        "PYBIND11_MODULE(algorithms_bindings, m)": "algorithms pybind module",
        'bind_distance_computer<float>(m, "DistanceMatrixComputerF")': "float distance class binding",
        'bind_distance_computer<double>(m, "DistanceMatrixComputerD")': "double distance class binding",
        'bind_knn_computer<float>(m, "KNNComputerF")': "float KNN class binding",
        'bind_knn_computer<double>(m, "KNNComputerD")': "double KNN class binding",
        'bind_mapper<float>(m, "MapperAlgorithmF")': "float mapper class binding",
        'bind_mapper<double>(m, "MapperAlgorithmD")': "double mapper class binding",
        'bind_diagram_conv<float>(m, "DiagramConv1DF")': "float diagram convolution binding",
        'bind_diagram_conv<double>(m, "DiagramConv1DD")': "double diagram convolution binding",
        "copy_to_array(result.distances, {n, result.k})": "KNN convenience output uses clamped result.k",
        "copy_to_array(result.indices, {n, result.k})": "KNN index output uses clamped result.k",
        "py::array::c_style | py::array::forcecast": (
            "algorithm bindings normalize NumPy arrays to contiguous typed buffers"
        ),
        "validate_point_array": "algorithm bindings validate point array layouts",
        "must contain only finite values": ("algorithm bindings reject non-finite numeric inputs"),
        "size must equal rows * dim": (
            "algorithm bindings reject mismatched n_points/dim metadata"
        ),
        "batch_size must be positive": ("diagram convolution binding rejects zero batch layouts"),
        "validate_exact_array": ("diagram convolution binding validates diagram layout size"),
    }
    for fragment, description in required_fragments.items():
        if fragment not in text:
            findings.append(
                Finding(
                    "algorithm-bindings-schema",
                    "python/bindings/nerve_algorithms_bindings.cpp",
                    f"missing {description}",
                )
            )
    required_cmake_fragments = {
        "pybind11_add_module(algorithms_bindings": "separate algorithms_bindings extension target",
        "bindings/nerve_algorithms_bindings.cpp": "algorithm binding source on its own target",
        "target_link_libraries(algorithms_bindings": "algorithm binding linkage",
        "OUTPUT_NAME algorithms_bindings": "algorithm binding artifact name",
        "install(TARGETS algorithms_bindings": "algorithm binding install target",
    }
    for fragment, description in required_cmake_fragments.items():
        if fragment not in cmake_text:
            findings.append(
                Finding(
                    "algorithm-bindings-schema", "python/CMakeLists.txt", f"missing {description}"
                )
            )
    nerve_internal_sources = re.search(
        r"set\(\s*PYTHON_API_SOURCES(?P<body>.*?)\)\s*pybind11_add_module\(nerve_internal",
        cmake_text,
        re.DOTALL,
    )
    if (
        nerve_internal_sources
        and "bindings/nerve_algorithms_bindings.cpp" in nerve_internal_sources.group("body")
    ):
        findings.append(
            Finding(
                "algorithm-bindings-schema",
                "python/CMakeLists.txt",
                "nerve_internal must not compile the algorithms_bindings PYBIND11_MODULE source",
            )
        )
    return findings

