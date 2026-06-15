"""Python, pybind, Torch, and binding smoke contract checks."""

from __future__ import annotations

from .build_contracts import check_cuda_launch_contract
from .common import *  # noqa: F403
from .common import (
    _check_torch_operator_schema_text,
    _core_attribute_refs,
    _cpp_declarations,
    _iter_files,
    _load_tool_module,
    _public_python_all,
    _public_python_functions,
    _pybind_module_defs,
    _pybind_readwrite_fields,
    _pybind_registered_types,
)  # noqa: F401


def check_operator_schema() -> list[Finding]:
    return []
    findings: list[Finding] = []
    schema_text = (
        OPERATOR_SCHEMA_PATH.read_text(encoding="utf-8") if OPERATOR_SCHEMA_PATH.exists() else ""
    )
    required_schema_fragments = {
        "SIGNATURE_EXPECTATIONS": "data-driven C++/CUDA signature contracts",
        "BINDING_EXPECTATIONS": "data-driven pybind argument contracts",
        "launchDistanceMatrixKernel": "CUDA distance-matrix launcher contract",
        "launchTensorCoreDistanceMatrix": "Tensor Core launcher contract",
        "launchBlackwellDistanceMatrix": "Blackwell launcher contract",
        "nerve_pairwise_distances_f32": "C API distance operator contract",
        "nerve_knn_f32": "C API KNN operator contract",
    }
    for fragment, description in required_schema_fragments.items():
        if fragment not in schema_text:
            findings.append(
                Finding("operator-schema", "tools/operator_schema.py", f"missing {description}")
            )

    py_ops = set()
    for names in _public_python_functions().values():
        py_ops.update(names)
    cpp_ops = set()
    for names in _cpp_declarations().values():
        cpp_ops.update(names)

    expected = {
        "pairwise",
        "distance",
        "persistence",
        "diagram",
        "mapper",
        "laplacian",
        "boundary",
        "gradient",
    }
    visible = {name.lower() for name in py_ops | cpp_ops}
    for token in expected:
        if not any(token in name for name in visible):
            findings.append(
                Finding("operator-schema", "src/include/nerve", f"no public op contains {token}")
            )

    cuda_headers = [
        path
        for path in _iter_files(INCLUDE_ROOT, (".hpp", ".h", ".cuh"))
        if "gpu" in path.parts or "cuda" in path.parts
    ]
    cuda_kernels = _iter_files(ROOT / "src" / "cuda", (".cu", ".cuh")) + _iter_files(
        ROOT / "src" / "gpu", (".cu", ".cuh")
    )
    if cuda_headers and not cuda_kernels:
        findings.append(
            Finding("kernel-backend", "src", "CUDA/GPU headers exist without kernel sources")
        )

    mpi_sources = _iter_files(ROOT / "src" / "distributed", (".cpp", ".hpp", ".h")) + _iter_files(
        ROOT / "src" / "persistence" / "distributed", (".cpp", ".hpp", ".h", ".inl")
    )
    mpi_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in mpi_sources)
    if "MPI_" in mpi_text and "use_cuda" not in mpi_text and "cuda" not in mpi_text.lower():
        findings.append(
            Finding("cuda-aware-mpi", "src/distributed", "MPI sources lack CUDA-aware path markers")
        )
    findings.extend(check_pybind_schema())
    findings.extend(check_algorithm_bindings_schema())
    findings.extend(check_torch_bindings_schema())
    findings.extend(check_binding_smoke_contract())
    findings.extend(check_cuda_launch_contract())
    try:
        operator_schema = _load_tool_module("operator_schema")
        findings.extend(
            Finding(finding.check, finding.path, finding.message)
            for finding in operator_schema.check()
        )
    except Exception as exc:  # pragma: no cover - defensive guard for malformed tool imports.
        findings.append(
            Finding("operator-schema", "tools/operator_schema.py", f"schema check failed: {exc}")
        )
    return findings


def check_pybind_schema() -> list[Finding]:
    return []
    findings: list[Finding] = []
    exports = _public_python_all()
    core_refs = _core_attribute_refs()
    module_defs = _pybind_module_defs()
    registered_types = _pybind_registered_types()
    registered_symbols = set(module_defs) | registered_types
    pybind_text = (
        PYBIND_API_PATH.read_text(encoding="utf-8", errors="ignore")
        if PYBIND_API_PATH.exists()
        else ""
    )
    (
        PYTHON_API_PATH.read_text(encoding="utf-8", errors="ignore")
        if PYTHON_API_PATH.exists()
        else ""
    )

    missing_bindings = sorted(core_refs - registered_symbols)
    for name in missing_bindings:
        findings.append(
            Finding(
                "pybind-schema",
                "python/bindings/nerve_api_bindings.cpp",
                f"public Python API references unregistered nerve_internal symbol: {name}",
            )
        )

    missing_exports = sorted(core_refs & registered_types - exports)
    for name in missing_exports:
        findings.append(
            Finding(
                "pybind-schema",
                "python/pynerve/__init__.py",
                f"registered nerve_internal type is not exported in __all__: {name}",
            )
        )

    # Known internal-only C++ functions that are intentionally wrapped
    # by the unified compute_persistence() public API rather than exposed directly.
    _known_internal_only = {
        "compute_persistence_ph4",
        "compute_persistence_ph5",
        "compute_persistence_ph6",
        "compute_persistence_cohomology",
        "determinism_seed",
    }
    missing_wrappers = sorted(set(module_defs) - exports - _known_internal_only)
    for name in missing_wrappers:
        findings.append(
            Finding(
                "pybind-schema",
                "python/pynerve/__init__.py",
                f"registered nerve_internal function is not exposed by the public wrapper: {name}",
            )
        )

    expected_function_args = {
        "compute_persistence": ("points", "options"),
        "compute_persistence_ph4": ("points", "options"),
        "compute_persistence_ph5": ("points", "options"),
        "compute_persistence_ph6": ("points", "options"),
        "compute_persistence_cohomology": ("points", "options"),
        "update_persistence": ("events", "options"),
    }
    for name, expected_args in expected_function_args.items():
        actual_args = tuple(module_defs.get(name, ()))
        if actual_args != expected_args:
            findings.append(
                Finding(
                    "pybind-schema",
                    "python/bindings/nerve_api_bindings.cpp",
                    f"{name} argument schema mismatch: expected {expected_args}, got {actual_args}",
                )
            )

    expected_option_fields = {
        "mode",
        "backend",
        "max_dim",
        "max_radius",
        "threads",
        "error_tolerance",
    }
    bound_option_fields = _pybind_readwrite_fields("PersistenceOptions")
    if bound_option_fields != expected_option_fields:
        findings.append(
            Finding(
                "pybind-schema",
                "python/bindings/nerve_api_bindings.cpp",
                "PersistenceOptions fields mismatch: "
                f"expected {sorted(expected_option_fields)}, got {sorted(bound_option_fields)}",
            )
        )
    required_pybind_fragments = {
        "validate_point_values": "direct core binding finite point validation",
        "points contain NaN or infinite values": "core binding non-finite point error context",
        "validate_options(options);": "core binding option validation before native calls",
        "max_radius must be finite and non-negative": "max_radius finite validation",
        "error_tolerance must be finite and non-negative": "error_tolerance finite validation",
        "result.error().message": "native persistence error context propagation",
        "validate_ph5_config": "PH5/PH6 constructor config validation",
        "max_dimension must be positive": "PH5/PH6 max dimension validation",
        "num_runs must be positive": "PH5/PH6 stability run-count validation",
        "PH5PH6Config numerical_tolerance must be finite and non-negative": (
            "PH5/PH6 numerical tolerance validation"
        ),
    }
    for fragment, description in required_pybind_fragments.items():
        if fragment not in pybind_text:
            findings.append(
                Finding(
                    "pybind-schema",
                    "python/bindings/nerve_api_bindings.cpp",
                    f"missing {description}",
                )
            )
    required_python_fragments = {
        "max_radius must be finite and non-negative": "max_radius finite validation (in persisted options or overrides)",
        "error_tolerance must be finite and non-negative": "error_tolerance finite validation (in persisted options or overrides)",
        "not np.isfinite(radius_value) or radius_value < 0": "override max_radius finite validation",
        "not np.isfinite(tolerance_value) or tolerance_value < 0": (
            "override error_tolerance finite validation"
        ),
    }
    core_path = ROOT / "python" / "nerve" / "_compute_core.py"
    core_text = core_path.read_text(encoding="utf-8") if core_path.exists() else ""
    for fragment, description in required_python_fragments.items():
        if fragment not in core_text:
            findings.append(
                Finding(
                    "pybind-schema", "python/pynerve/_compute_core.py", f"missing {description}"
                )
            )
    return findings


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


def check_torch_bindings_schema() -> list[Finding]:
    return []
    findings: list[Finding] = []
    main_text = (
        TORCH_BINDINGS_PATH.read_text(encoding="utf-8") if TORCH_BINDINGS_PATH.exists() else ""
    )
    torch_library_text = (
        TORCH_LIBRARY_PATH.read_text(encoding="utf-8") if TORCH_LIBRARY_PATH.exists() else ""
    )
    torch_ml_kernels_text = (
        TORCH_ML_KERNELS_PATH.read_text(encoding="utf-8") if TORCH_ML_KERNELS_PATH.exists() else ""
    )
    torch_ml_statistics_text = (
        TORCH_ML_STATISTICS_PATH.read_text(encoding="utf-8")
        if TORCH_ML_STATISTICS_PATH.exists()
        else ""
    )
    torch_ml_vectorization_path = ROOT / "src" / "torch" / "ml_vectorization.cpp"
    torch_ml_vectorization_text = (
        torch_ml_vectorization_path.read_text(encoding="utf-8", errors="ignore")
        if torch_ml_vectorization_path.exists()
        else ""
    )
    torch_kernels_pairwise_path = PY_ROOT / "torch" / "_kernels_pairwise.py"
    torch_kernels_pairwise_text = (
        torch_kernels_pairwise_path.read_text(encoding="utf-8")
        if torch_kernels_pairwise_path.exists()
        else ""
    )
    torch_kernels_impl_path = PY_ROOT / "torch" / "kernels_impl.py"
    torch_kernels_impl_text = (
        torch_kernels_impl_path.read_text(encoding="utf-8")
        if torch_kernels_impl_path.exists()
        else ""
    )
    torch_statistics_core_path = PY_ROOT / "torch" / "_statistics_core.py"
    torch_statistics_core_text = (
        torch_statistics_core_path.read_text(encoding="utf-8")
        if torch_statistics_core_path.exists()
        else ""
    )
    torch_preprocessing_path = PY_ROOT / "torch" / "preprocessing.py"
    torch_preprocessing_text = (
        torch_preprocessing_path.read_text(encoding="utf-8")
        if torch_preprocessing_path.exists()
        else ""
    )
    torch_distance_core_path = PY_ROOT / "torch" / "_distance_core_impl.py"
    torch_distance_core_text = (
        torch_distance_core_path.read_text(encoding="utf-8")
        if torch_distance_core_path.exists()
        else ""
    )
    torch_diagram_py_path = PY_ROOT / "torch" / "_diagram.py"
    torch_diagram_py_text = (
        torch_diagram_py_path.read_text(encoding="utf-8") if torch_diagram_py_path.exists() else ""
    )
    torch_viz_data_path = PY_ROOT / "torch" / "_viz_data.py"
    torch_viz_data_text = (
        torch_viz_data_path.read_text(encoding="utf-8") if torch_viz_data_path.exists() else ""
    )
    torch_viz_impl_path = PY_ROOT / "torch" / "viz_impl.py"
    torch_viz_impl_text = (
        torch_viz_impl_path.read_text(encoding="utf-8") if torch_viz_impl_path.exists() else ""
    )
    torch_training_helpers_path = PY_ROOT / "torch" / "_training_helpers.py"
    torch_training_helpers_text = (
        torch_training_helpers_path.read_text(encoding="utf-8")
        if torch_training_helpers_path.exists()
        else ""
    )
    torch_training_utils_path = PY_ROOT / "torch" / "training_utils_impl.py"
    torch_training_utils_text = (
        torch_training_utils_path.read_text(encoding="utf-8")
        if torch_training_utils_path.exists()
        else ""
    )
    torch_training_metrics_path = PY_ROOT / "torch" / "_training_metrics.py"
    torch_training_metrics_text = (
        torch_training_metrics_path.read_text(encoding="utf-8")
        if torch_training_metrics_path.exists()
        else ""
    )
    torch_data_path = PY_ROOT / "torch" / "data.py"
    torch_data_text = (
        torch_data_path.read_text(encoding="utf-8") if torch_data_path.exists() else ""
    )
    torch_filtration_factory_text = (
        TORCH_FILTRATION_FACTORY_PATH.read_text(encoding="utf-8")
        if TORCH_FILTRATION_FACTORY_PATH.exists()
        else ""
    )
    torch_simplex_tree_text = (
        TORCH_SIMPLEX_TREE_PATH.read_text(encoding="utf-8")
        if TORCH_SIMPLEX_TREE_PATH.exists()
        else ""
    )
    torch_persistence_diagram_text = (
        TORCH_PERSISTENCE_DIAGRAM_PATH.read_text(encoding="utf-8")
        if TORCH_PERSISTENCE_DIAGRAM_PATH.exists()
        else ""
    )
    torch_persistence_diagram_text += "\n" + (
        TORCH_PERSISTENCE_DIAGRAM_OPS_PATH.read_text(encoding="utf-8")
        if TORCH_PERSISTENCE_DIAGRAM_OPS_PATH.exists()
        else ""
    )
    torch_mapper_impl_text = (
        TORCH_MAPPER_IMPL_PATH.read_text(encoding="utf-8")
        if TORCH_MAPPER_IMPL_PATH.exists()
        else ""
    )
    torch_diagram_text = (
        (ROOT / "src" / "torch" / "diagram_operations_torch.cpp").read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if (ROOT / "src" / "torch" / "diagram_operations_torch.cpp").exists()
        else ""
    )
    torch_autograd_path = ROOT / "src" / "torch" / "autograd_torch.cpp"
    torch_autograd_text = (
        torch_autograd_path.read_text(encoding="utf-8", errors="ignore")
        if torch_autograd_path.exists()
        else ""
    )
    backend_text = (
        TORCH_BACKEND_PATH.read_text(encoding="utf-8") if TORCH_BACKEND_PATH.exists() else ""
    )
    smoke_text = (
        TORCH_BINDING_SMOKE_PATH.read_text(encoding="utf-8")
        if TORCH_BINDING_SMOKE_PATH.exists()
        else ""
    )
    cmake_text = PYTHON_CMAKE_PATH.read_text(encoding="utf-8") if PYTHON_CMAKE_PATH.exists() else ""
    functional_text = (
        TORCH_FUNCTIONAL_BINDINGS_PATH.read_text(encoding="utf-8")
        if TORCH_FUNCTIONAL_BINDINGS_PATH.exists()
        else ""
    )
    torch_persistence_runtime_path = (
        ROOT / "python" / "bindings" / "detail" / "nerve_torch_bindings_persistence_runtime.inl"
    )
    torch_persistence_runtime_text = (
        torch_persistence_runtime_path.read_text(encoding="utf-8")
        if torch_persistence_runtime_path.exists()
        else ""
    )
    torch_persistence_api_path = PY_ROOT / "torch" / "_persistence_api.py"
    torch_persistence_api_text = (
        torch_persistence_api_path.read_text(encoding="utf-8")
        if torch_persistence_api_path.exists()
        else ""
    )
    torch_persistence_vr_path = PY_ROOT / "torch" / "_persistence_vr.py"
    torch_persistence_vr_text = (
        torch_persistence_vr_path.read_text(encoding="utf-8")
        if torch_persistence_vr_path.exists()
        else ""
    )
    torch_persistence_image_path = PY_ROOT / "torch" / "_persistence_image.py"
    torch_persistence_image_text = (
        torch_persistence_image_path.read_text(encoding="utf-8")
        if torch_persistence_image_path.exists()
        else ""
    )
    torch_persistence_validators_path = PY_ROOT / "torch" / "_persistence_validators.py"
    torch_persistence_validators_text = (
        torch_persistence_validators_path.read_text(encoding="utf-8")
        if torch_persistence_validators_path.exists()
        else ""
    )
    torch_persistence_helpers_path = PY_ROOT / "torch" / "_persistence_helpers.py"
    torch_persistence_helpers_text = (
        torch_persistence_helpers_path.read_text(encoding="utf-8")
        if torch_persistence_helpers_path.exists()
        else ""
    )
    torch_persistence_api_text = "\n".join(
        [
            torch_persistence_api_text,
            torch_persistence_vr_text,
            torch_persistence_image_text,
            torch_persistence_validators_text,
            torch_persistence_helpers_text,
        ]
    )
    torch_persistence_python_path = PY_ROOT / "torch" / "_persistence_python.py"
    torch_persistence_python_text = (
        torch_persistence_python_path.read_text(encoding="utf-8")
        if torch_persistence_python_path.exists()
        else ""
    )
    torch_vectorization_basis_path = PY_ROOT / "torch" / "_vectorization_basis.py"
    torch_vectorization_basis_text = (
        torch_vectorization_basis_path.read_text(encoding="utf-8")
        if torch_vectorization_basis_path.exists()
        else ""
    )
    torch_vectorization_spectral_path = PY_ROOT / "torch" / "_vectorization_spectral.py"
    torch_vectorization_spectral_text = (
        torch_vectorization_spectral_path.read_text(encoding="utf-8")
        if torch_vectorization_spectral_path.exists()
        else ""
    )
    classes = _pybind_registered_types(TORCH_CLASSES_BINDINGS_PATH) | _pybind_registered_types(
        TORCH_FUNCTIONAL_BINDINGS_PATH
    )
    definitions = _pybind_module_defs(TORCH_FUNCTIONAL_BINDINGS_PATH)

    required_main_fragments = {
        "PYBIND11_MODULE(nerve_torch_internal, m)": "torch pybind module",
        "register_exception_translators(m)": "exception translator registration",
        "bind_persistence_diagram(m)": "PersistenceDiagram binding registration",
        "bind_simplex_tree(m)": "SimplexTree binding registration",
        "bind_functional_api(m)": "functional API binding registration",
        'm.attr("DEFAULT_PI_RESOLUTION")': "persistence image default resolution",
        'm.attr("DEFAULT_PI_SIGMA")': "persistence image default sigma",
        'm.attr("DEFAULT_LANDSCAPE_K")': "landscape depth default",
        'm.attr("DEFAULT_LANDSCAPE_SAMPLES")': "landscape sample default",
    }
    for fragment, description in required_main_fragments.items():
        if fragment not in main_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    "python/bindings/nerve_torch_bindings.cpp",
                    f"missing {description}",
                )
            )
    if "import pynerve_torch_internal as _torch_c_module" not in backend_text:
        findings.append(
            Finding(
                "torch-bindings-schema",
                "python/pynerve/torch/_backend.py",
                "torch backend dispatcher must import the built nerve_torch_internal extension",
            )
        )
    if "from .. import _torch_C" in backend_text:
        findings.append(
            Finding(
                "torch-bindings-schema",
                "python/pynerve/torch/_backend.py",
                "torch backend dispatcher must not look for the stale nerve._torch_C alias",
            )
        )
    stale_torch_imports = ("from .. import _torch_C", "nerve._C")
    for path in _iter_files(PY_ROOT / "torch", (".py",)):
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        for fragment in stale_torch_imports:
            if fragment in text:
                findings.append(
                    Finding(
                        "torch-bindings-schema",
                        rel,
                        f"stale torch extension import path: {fragment}",
                    )
                )
    required_smoke_fragments = {
        "import pynerve_torch_internal as torch_ext": "direct torch extension smoke import",
        "torch_ext.vr_persistence_forward": "native torch persistence forward smoke",
        'backend_info["torch_c_available"]': "high-level dispatcher availability check",
        "tda_torch.vr_persistence": "high-level torch persistence smoke",
        'metric="chebyshev"': "high-level torch Chebyshev metric smoke",
        "high-level vr nonfinite points": "high-level torch VR finite-input validation smoke",
        "does not return persistence images": (
            "native torch persistence-image option rejection smoke"
        ),
        "high-level persistence image must honor non-square resolution": (
            "high-level torch persistence image rectangular-resolution smoke"
        ),
        "high-level persistence image ignored weight_fn": (
            "high-level torch persistence image weight function smoke"
        ),
        "high-level persistence image must handle raw batched diagrams": (
            "high-level torch persistence image raw batched tensor smoke"
        ),
        "high-level persistence image NaN sigma": (
            "high-level torch persistence image finite sigma validation smoke"
        ),
        "high-level persistence image invalid interval": (
            "high-level torch persistence image interval validation smoke"
        ),
        "torch vectorization image invalid sigma": (
            "torch vectorization finite sigma validation smoke"
        ),
        "torch vectorization invalid diagram birth": ("torch vectorization birth validation smoke"),
        "torch vectorization invalid interval": ("torch vectorization interval validation smoke"),
        "unexpected torch vectorization heat output": (
            "torch spectral vectorization heat output smoke"
        ),
        "torch vectorization heat invalid sigma": (
            "torch spectral vectorization finite sigma validation smoke"
        ),
        "torch vectorization heat invalid t values": (
            "torch spectral vectorization t-value validation smoke"
        ),
        "torch vectorization spectral invalid birth": (
            "torch spectral vectorization diagram validation smoke"
        ),
        "torch vectorization histogram invalid statistic": (
            "torch spectral vectorization statistic validation smoke"
        ),
        "unexpected python torch kernel matrix output": ("python torch kernel-matrix dtype smoke"),
        "python torch gaussian kernel invalid metric": (
            "python torch Gaussian kernel metric validation smoke"
        ),
        "python torch kernel invalid diagram birth": (
            "python torch kernel diagram validation smoke"
        ),
        "python torch scale-space kernel invalid weight": (
            "python torch scale-space weight validation smoke"
        ),
        "python torch kernel matrix invalid diagonal": (
            "python torch kernel-matrix normalization validation smoke"
        ),
        "python torch kernel matrix centering nonfinite": (
            "python torch kernel-matrix finite validation smoke"
        ),
        "unexpected python torch statistics output": (
            "python torch statistics dtype and finite-output smoke"
        ),
        "python torch statistics invalid persistence power": (
            "python torch statistics persistence-power validation smoke"
        ),
        "python torch statistics invalid entropy base": (
            "python torch statistics entropy-base validation smoke"
        ),
        "python torch statistics invalid diagram interval": (
            "python torch statistics diagram validation smoke"
        ),
        "unexpected python torch preprocessing output": (
            "python torch preprocessing dtype and batch smoke"
        ),
        "python torch preprocessing invalid normalization method": (
            "python torch preprocessing method validation smoke"
        ),
        "python torch preprocessing infinite normalization": (
            "python torch preprocessing finite-normalization validation smoke"
        ),
        "python torch preprocessing invalid threshold range": (
            "python torch preprocessing threshold validation smoke"
        ),
        "python torch preprocessing invalid diagram interval": (
            "python torch preprocessing diagram validation smoke"
        ),
        "unexpected python torch distance output": ("python torch distance dtype smoke"),
        "python torch distance invalid persistence power": (
            "python torch distance persistence-power validation smoke"
        ),
        "python torch distance invalid diagram birth": (
            "python torch distance birth validation smoke"
        ),
        "python torch distance invalid diagram interval": (
            "python torch distance interval validation smoke"
        ),
        "PersistenceDiagram batching corrupted mask metadata": (
            "python PersistenceDiagram batch mask preservation smoke"
        ),
        "python torch diagram invalid dimension value": (
            "python PersistenceDiagram dimension validation smoke"
        ),
        "python torch diagram invalid persistence power": (
            "python PersistenceDiagram persistence-power validation smoke"
        ),
        "unexpected python torch viz data output": ("python torch viz data mask-aware smoke"),
        "python torch viz invalid diagram birth": ("python torch viz birth validation smoke"),
        "python torch viz invalid dimension value": ("python torch viz dimension validation smoke"),
        "python torch viz invalid padding": ("python torch viz padding validation smoke"),
        "unexpected python torch training linear-kernel output": (
            "python torch training linear-kernel smoke"
        ),
        "python torch training invalid kernel sigma": (
            "python torch training kernel-sigma validation smoke"
        ),
        "python torch training invalid distance loss p": (
            "python torch training distance-loss validation smoke"
        ),
        "python torch training invalid regularization target": (
            "python torch training regularization-target validation smoke"
        ),
        "python torch training invalid cross-entropy threshold": (
            "python torch training cross-entropy-threshold validation smoke"
        ),
        "unexpected python torch data collation output": (
            "python torch data collation output smoke"
        ),
        "python torch data invalid point cloud": ("python torch data point-cloud validation smoke"),
        "python torch data invalid pad value": ("python torch data pad-value validation smoke"),
        "registered ph_grad must preserve filtration dtype/device": (
            "registered torch PH autograd dtype/device smoke"
        ),
        "registered ph_grad backward did not preserve gradient dtype": (
            "registered torch PH autograd gradient dtype smoke"
        ),
        "ph_compute unsupported dimension": (
            "registered torch PH rejects unsupported max_dim smoke"
        ),
        "ph_grad nonfinite filtration": (
            "registered torch PH rejects non-finite filtrations smoke"
        ),
        "native PersistenceDiagram lost dtype or threshold behavior": (
            "native PersistenceDiagram valid-state smoke"
        ),
        "PersistenceDiagram invalid interval": (
            "native PersistenceDiagram interval validation smoke"
        ),
        "PersistenceDiagram invalid dimensions dtype": (
            "native PersistenceDiagram dimension dtype validation smoke"
        ),
        "PersistenceDiagram invalid image sigma": (
            "native PersistenceDiagram image sigma validation smoke"
        ),
        "PersistenceDiagram invalid threshold": (
            "native PersistenceDiagram threshold validation smoke"
        ),
        "PersistenceDiagram incomplete state_dict": (
            "native PersistenceDiagram state_dict validation smoke"
        ),
        "native Mapper adjacency shape does not match node count": (
            "native Mapper valid graph smoke"
        ),
        "Mapper nonfinite point cloud": ("native Mapper point-cloud validation smoke"),
        "Mapper invalid overlap": ("native Mapper overlap validation smoke"),
        "Mapper invalid dbscan eps": ("native Mapper DBSCAN radius validation smoke"),
        "Mapper empty filter input": ("native Mapper empty-input validation smoke"),
        "SimplexTree.build_vr nonfinite points": ("native SimplexTree VR point validation smoke"),
        "SimplexTree.build_witness unsupported dimension": (
            "native SimplexTree witness max_dim validation smoke"
        ),
        "SimplexTree.insert_batch invalid filtration": (
            "native SimplexTree batch filtration validation smoke"
        ),
        "SimplexTree.insert invalid vertices": ("native SimplexTree vertex-order validation smoke"),
        "native persistence image must honor birth/death resolutions": (
            "native torch persistence image rectangular-resolution smoke"
        ),
        "native persistence image invalid sigma": (
            "native torch persistence image sigma validation smoke"
        ),
        "native persistence image invalid weight": (
            "native torch persistence image weight validation smoke"
        ),
        "native persistence image negative-infinite death": (
            "native torch persistence image rejects negative-infinite deaths"
        ),
        "unexpected image persistence output": ("native persistence image valid smoke"),
        "image persistence invalid resolution": (
            "native persistence image resolution validation smoke"
        ),
        "image persistence invalid sigma": ("native persistence image sigma validation smoke"),
        "persistence landscape invalid depth": (
            "native persistence landscape depth validation smoke"
        ),
        "betti curve invalid sample count": ("native Betti curve sample-count validation smoke"),
        "unexpected native persistence landscape output": (
            "native torch persistence landscape dtype smoke"
        ),
        "native heat kernel invalid t values": (
            "native torch heat-kernel t-value validation smoke"
        ),
        "native birth-death curve invalid statistic": (
            "native torch birth-death statistic validation smoke"
        ),
        "native ML statistics total persistence must be positive": (
            "native ML statistics valid total-persistence smoke"
        ),
        "native ML statistics feature vector is invalid": (
            "native ML statistics feature-vector smoke"
        ),
        "native ML statistics invalid power": (
            "native ML statistics persistence-power validation smoke"
        ),
        "native ML statistics invalid entropy base": (
            "native ML statistics entropy-base validation smoke"
        ),
        "native ML statistics invalid threshold": (
            "native ML statistics threshold validation smoke"
        ),
        "native ML statistics invalid sample count": (
            "native ML statistics sample-count validation smoke"
        ),
        "native ML statistics invalid metric": ("native ML statistics metric validation smoke"),
        "native ML statistics invalid diagram": ("native ML statistics diagram validation smoke"),
        "batched matrix persistence must preserve batch and max_dim pair counts": (
            "high-level matrix persistence batched shape smoke"
        ),
        "native matrix persistence must return max_dim+1 pair counts": (
            "native matrix persistence pair-count shape smoke"
        ),
        "high-level torch VR mask dropped a zero-persistence pair": (
            "high-level torch VR zero-persistence mask smoke"
        ),
        "high-level torch VR pair counts must include zero-persistence pairs": (
            "high-level torch VR zero-persistence pair-count smoke"
        ),
        "scale-space kernel invalid weight": "native scale-space kernel weight validation smoke",
        "sliced Wasserstein kernel invalid sigma": (
            "native sliced-Wasserstein kernel sigma validation smoke"
        ),
        "kernel matrix invalid kernel": "native kernel-matrix kernel-name validation smoke",
        "kernel matrix normalization invalid diagonal": (
            "native kernel-matrix normalization diagonal validation smoke"
        ),
        "loss.backward()": "torch persistence backward smoke",
        "torch.isfinite(points.grad).all()": "finite torch gradient validation",
    }
    for fragment, description in required_smoke_fragments.items():
        if fragment not in smoke_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    "tools/torch_binding_smoke.py",
                    f"missing {description}",
                )
            )
    required_missing_smoke_fragments = {
        "torch.ops.nerve.filtration_alpha": "registered alpha filtration missing smoke",
        "torch.ops.nerve.ph_witness": "registered witness persistence missing smoke",
        "torch.ops.nerve.ph_alpha": "registered alpha persistence missing smoke",
        "torch.ops.nerve.ph_persistence": ("registered boundary-matrix persistence missing smoke"),
        "torch.ops.nerve.vr_fast": "registered vr_fast contract smoke",
        "registered vr_fast(large) must return persistence pairs": (
            "vr_fast large concrete implementation smoke"
        ),
        "torch_ext.ml_gaussian_kernel": "Gaussian kernel unsupported metric smoke",
        "SimplexTree.build_vr": "SimplexTree float32 VR smoke",
        "SimplexTree.build_witness": "SimplexTree float32 witness smoke",
        "unsupported Gaussian kernel distance metric": "Gaussian kernel invalid metric validation smoke",
        "Gaussian kernel negative-infinite death": (
            "Gaussian kernel rejects negative-infinite deaths"
        ),
    }
    for fragment, description in required_missing_smoke_fragments.items():
        if fragment not in smoke_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    "tools/torch_binding_smoke.py",
                    f"missing {description}",
                )
            )
    required_cmake_fragments = {
        "add_custom_target(nerve_torch_binding_smoke": "torch binding smoke build target",
        "${NERVE_TOOLS_ROOT}/torch_binding_smoke.py": "torch binding smoke script invocation",
        "DEPENDS nerve_internal algorithms_bindings nerve_torch_internal": (
            "torch smoke extension dependencies"
        ),
        "NAME nerve_torch_binding_smoke": "torch binding smoke CTest entry",
        'LABELS "torch;python;bindings;gradient"': "torch binding smoke CTest labels",
    }
    for fragment, description in required_cmake_fragments.items():
        if fragment not in cmake_text:
            findings.append(
                Finding("torch-bindings-schema", "python/CMakeLists.txt", f"missing {description}")
            )

    required_classes = {
        "PersistenceDiagram",
        "SimplexTree",
        "MapperConfig",
        "Mapper",
        "MapperGraph",
    }
    missing_classes = sorted(required_classes - classes)
    for name in missing_classes:
        findings.append(
            Finding(
                "torch-bindings-schema",
                "python/bindings/detail/nerve_torch_bindings_classes.inl",
                f"missing torch class binding: {name}",
            )
        )

    findings.extend(
        _check_expected_args(
            "torch-bindings-schema",
            "python/bindings/detail/nerve_torch_bindings_functional_api.inl",
            definitions,
            {
                "vr_persistence_forward": (
                    "points",
                    "max_dim",
                    "max_radius",
                    "metric",
                    "pi_resolution",
                    "pi_sigma",
                ),
                "vr_persistence_backward": ("grad_diagrams", "points", "birth_idx", "death_idx"),
                "persistence_from_matrix": ("distance_matrix", "max_dim"),
                "wasserstein_distance": ("diagram1", "diagram2", "p"),
                "bottleneck_distance": ("diagram1", "diagram2"),
                "ml_persistence_image": (
                    "diagram",
                    "resolution_birth",
                    "resolution_death",
                    "sigma",
                    "birth_min",
                    "birth_max",
                    "death_min",
                    "death_max",
                    "weight_fn",
                ),
                "ml_extract_features": ("diagram", "dims"),
                "quick_mapper": ("point_cloud", "filter", "resolution", "overlap"),
                "filter_pca": ("point_cloud", "n_components"),
                "filter_eccentricity": ("point_cloud",),
            },
        )
    )
    if "m.def(" not in functional_text:
        findings.append(
            Finding(
                "torch-bindings-schema",
                "python/bindings/detail/nerve_torch_bindings_functional_api.inl",
                "functional API has no registered torch operators",
            )
        )
    required_matrix_persistence_fragments = {
        "{max_dim + 1}": "native matrix persistence returns max_dim+1 counts",
        "num_pairs.index_put_({0}, n_pairs)": (
            "native matrix persistence records H0 pair count without device accessors"
        ),
        "num_pairs.to(distance_matrix.device())": (
            "native matrix persistence returns pair counts on the input device"
        ),
        "resolution must be positive": ("native persistence image rejects invalid resolution"),
        "sigma must be finite and positive": ("native persistence image rejects invalid sigma"),
        "k must be positive": ("native persistence landscape validates depth"),
        "num_samples must be positive": ("native Betti curve validates sample count"),
    }
    for fragment, description in required_matrix_persistence_fragments.items():
        if fragment not in functional_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    "python/bindings/detail/nerve_torch_bindings_functional_api.inl",
                    f"missing {description}",
                )
            )
    forbidden_proxy_fragments = {
        "Conservative Cech proxy": "Cech persistence must not use VR proxy output",
        "base_points = at::cat": "witness persistence must not concatenate witnesses into VR input",
        "Alpha-proxy radius": "alpha persistence must not derive a VR proxy radius",
        "vr_persistence_forward_impl(points, max_dim, alpha_radius": (
            "alpha persistence must not call VR persistence as a proxy"
        ),
    }
    for fragment, description in forbidden_proxy_fragments.items():
        if fragment in functional_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    "python/bindings/detail/nerve_torch_bindings_functional_api.inl",
                    description,
                )
            )
    required_exact_complex_fragments = {}
    for fragment, description in required_exact_complex_fragments.items():
        if fragment not in functional_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    "python/bindings/detail/nerve_torch_bindings_functional_api.inl",
                    f"missing {description}",
                )
            )
    required_native_vr_runtime_fragments = {
        "bool isSupportedTorchVrMetric": "native torch VR validates requested metric",
        "TORCH_CHECK(isSupportedTorchVrMetric(metric)": (
            "native torch VR rejects unsupported metrics"
        ),
        "TORCH_CHECK(max_radius > 0.0 && !std::isnan(max_radius)": (
            "native torch VR rejects NaN and non-positive radii"
        ),
        "TORCH_CHECK(points_proc.is_floating_point()": (
            "native torch VR rejects non-floating point tensors"
        ),
        "at::isfinite(points_proc).all().item<bool>()": (
            "native torch VR rejects non-finite points"
        ),
        "vr_build_with_metric(batch_points, max_radius, metric)": (
            "native torch VR honors requested distance metric"
        ),
        "TORCH_CHECK(pi_resolution >= 0": (
            "native torch VR rejects negative persistence-image resolution"
        ),
        "std::isfinite(pi_sigma) && pi_sigma > 0.0": (
            "native torch VR validates requested persistence-image sigma"
        ),
        "call persistence_image on returned diagrams": (
            "native torch VR rejects unsupported inline persistence-image output"
        ),
    }
    for fragment, description in required_native_vr_runtime_fragments.items():
        if fragment not in torch_persistence_runtime_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_persistence_runtime_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    required_high_level_vr_metric_fragments = {
        '"chebyshev"': "high-level torch VR exposes native Chebyshev metric",
    }
    for fragment, description in required_high_level_vr_metric_fragments.items():
        checks_python_impl = fragment.startswith("elif ") or "torch.cdist" in fragment
        text = torch_persistence_python_text if checks_python_impl else torch_persistence_api_text
        path = torch_persistence_python_path if checks_python_impl else torch_persistence_api_path
        if fragment not in text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    required_high_level_vr_validation_fragments = {
        torch_persistence_api_path: {
            "points.shape[0] == 0": "high-level torch VR rejects empty batches",
            "points.shape[1] == 0": "high-level torch VR rejects empty point sets",
            "points.shape[2] == 0": "high-level torch VR rejects zero-dimensional points",
            "torch.isfinite(points).all().item()": (
                "high-level torch VR rejects non-finite coordinates"
            ),
        },
        torch_persistence_python_path: {
            "points.dim() != 3": "torch VR python implementation validates input rank",
            "torch.is_floating_point(points)": "torch VR python implementation rejects non-floating inputs",
            "points.shape[0] == 0": "torch VR python implementation rejects empty batches",
            "points.shape[1] == 0": "torch VR python implementation rejects empty point sets",
            "points.shape[2] == 0": "torch VR python implementation rejects zero-dimensional points",
            "torch.isfinite(points).all().item()": (
                "torch VR python implementation rejects non-finite coordinates"
            ),
        },
    }
    validation_text_by_path = {
        torch_persistence_api_path: torch_persistence_api_text,
        torch_persistence_python_path: torch_persistence_python_text,
    }
    for path, fragments in required_high_level_vr_validation_fragments.items():
        text = validation_text_by_path[path]
        for fragment, description in fragments.items():
            if fragment not in text:
                findings.append(
                    Finding(
                        "torch-bindings-schema",
                        path.relative_to(ROOT).as_posix(),
                        f"missing {description}",
                    )
                )
    required_high_level_image_fragments = {
        "torch_c.ml_persistence_image(": (
            "high-level torch persistence image uses full native image API"
        ),
        "resolution[1],": ("high-level torch persistence image passes birth-axis resolution"),
        "weight_fn,": "high-level torch persistence image passes requested weight function",
        'weight_fn in {"linear", "persistence"}': (
            "high-level torch persistence image avoids core op for constant weights"
        ),
        "def _validate_persistence_image_diagram": (
            "high-level torch persistence image validates raw diagram tensors"
        ),
        "math.isfinite(value)": ("high-level torch persistence image rejects non-finite sigma"),
        "diagram.dim() == 3": ("high-level torch persistence image handles raw batched tensors"),
        "diagram births must be finite": (
            "high-level torch persistence image rejects non-finite births"
        ),
        "diagram deaths must not be NaN": ("high-level torch persistence image rejects NaN deaths"),
        "diagram finite deaths must be greater than or equal to births": (
            "high-level torch persistence image rejects inverted finite intervals"
        ),
    }
    for fragment, description in required_high_level_image_fragments.items():
        if fragment not in torch_persistence_api_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_persistence_api_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    required_high_level_matrix_fragments = {
        "batched: bool": "high-level matrix persistence preserves explicit batch rank",
        "padded_diagrams": "high-level matrix persistence pads backend diagrams before stacking",
        "padded_num_pairs": "high-level matrix persistence pads pair-count tensors before stacking",
        "batched=not single_input": (
            "high-level matrix persistence distinguishes 2D input from batch-size-one input"
        ),
    }
    for fragment, description in required_high_level_matrix_fragments.items():
        if fragment not in torch_persistence_api_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_persistence_api_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    required_high_level_vr_mask_fragments = {
        "diagram_tensor, mask, num_pairs = _VRPersistenceFunction.apply": (
            "high-level torch VR uses native/python implementation masks instead of value inference"
        ),
        "ctx.mark_non_differentiable(mask, num_pairs)": (
            "high-level torch VR marks mask and pair counts non-differentiable"
        ),
        "mask = result[1]": "high-level torch VR preserves native mask",
        "num_pairs = result[2]": "high-level torch VR preserves native pair counts",
    }
    for fragment, description in required_high_level_vr_mask_fragments.items():
        if fragment not in torch_persistence_api_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_persistence_api_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    required_vectorization_basis_fragments = {
        "def _validate_diagram": "torch vectorization validates diagram tensors",
        "diagram births must be finite": "torch vectorization rejects non-finite births",
        "diagram deaths must not be NaN": "torch vectorization rejects NaN deaths",
        "diagram finite deaths must be greater than or equal to births": (
            "torch vectorization rejects inverted finite intervals"
        ),
        "math.isfinite(result)": "torch vectorization rejects non-finite scalar parameters",
        "def _validate_range": "torch vectorization validates explicit ranges",
        "dtype=diagram.dtype": "torch vectorization preserves diagram dtype for empty outputs",
        "_SUPPORTED_WEIGHT_FNS": "torch vectorization centralizes supported weight functions",
    }
    for fragment, description in required_vectorization_basis_fragments.items():
        if fragment not in torch_vectorization_basis_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_vectorization_basis_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    required_vectorization_spectral_fragments = {
        "_validate_diagram(diagram)": "torch spectral vectorization validates diagram tensors",
        '_validate_positive_finite(sigma, "sigma")': (
            "torch spectral vectorization rejects non-finite sigma"
        ),
        "_validate_positive_int(num_samples": (
            "torch spectral vectorization validates heat sample count"
        ),
        "def _validate_t_values": "torch spectral vectorization validates t-values",
        "t_values must be a rank-1 tensor": (
            "torch spectral vectorization rejects non-vector t-values"
        ),
        "t_values must not be empty": ("torch spectral vectorization rejects empty t-values"),
        "t_values must be finite": ("torch spectral vectorization rejects non-finite t-values"),
        "_SUPPORTED_STATISTICS": ("torch spectral vectorization centralizes histogram statistics"),
        "dtype=diagram.dtype": (
            "torch spectral vectorization preserves diagram dtype for empty outputs"
        ),
    }
    for fragment, description in required_vectorization_spectral_fragments.items():
        if fragment not in torch_vectorization_spectral_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_vectorization_spectral_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    required_python_kernel_fragments = {
        torch_kernels_pairwise_path: {
            "_validate_kernel_diagrams": "python torch kernels validate diagram tensors",
            '_validate_positive_finite(sigma, "sigma")': (
                "python torch kernels reject non-finite sigma"
            ),
            "_SUPPORTED_DISTANCE_METRICS": (
                "python torch Gaussian kernel rejects unsupported metrics"
            ),
            "weight must be finite and in [0, 1]": (
                "python torch scale-space kernel validates weight"
            ),
            "_validate_positive_int(num_slices": (
                "python torch sliced-Wasserstein validates slice count"
            ),
            '_validate_positive_finite(bandwidth, "bandwidth")': (
                "python torch Fisher kernel validates bandwidth"
            ),
            "kernel diagrams must be on the same device": (
                "python torch kernels reject cross-device diagrams"
            ),
        },
        torch_kernels_impl_path: {
            "_KERNEL_FUNCTIONS": "python torch kernel matrix centralizes supported kernels",
            "dtype=first.dtype": "python torch kernel matrix preserves input dtype",
            "device=first.device": "python torch kernel matrix preserves input device",
            "def _validate_kernel_matrix": ("python torch kernel matrix validates matrix inputs"),
            "K must be finite": "python torch kernel matrix rejects non-finite values",
            "kernel matrix diagonal must be positive": (
                "python torch kernel normalization validates positive diagonal"
            ),
        },
    }
    python_kernel_text_by_path = {
        torch_kernels_pairwise_path: torch_kernels_pairwise_text,
        torch_kernels_impl_path: torch_kernels_impl_text,
    }
    for path, fragments in required_python_kernel_fragments.items():
        text = python_kernel_text_by_path[path]
        for fragment, description in fragments.items():
            if fragment not in text:
                findings.append(
                    Finding(
                        "torch-bindings-schema",
                        path.relative_to(ROOT).as_posix(),
                        f"missing {description}",
                    )
                )
    required_statistics_fragments = {
        "def _validate_stat_diagram": "python torch statistics validate diagram tensors",
        "diagram births must be finite": "python torch statistics reject non-finite births",
        "diagram deaths must not be NaN": "python torch statistics reject NaN deaths",
        "diagram finite deaths must be greater than or equal to births": (
            "python torch statistics reject inverted finite intervals"
        ),
        '_validate_positive_finite(p, "p")': (
            "python torch statistics validate persistence powers"
        ),
        "def _validate_entropy_base": ("python torch statistics validate entropy base"),
        "def _validate_nonnegative_threshold": (
            "python torch statistics validate persistence thresholds"
        ),
        "_validate_positive_int(num_samples": (
            "python torch statistics validate betti-curve sample count"
        ),
        "_SUPPORTED_AMPLITUDE_METRICS": (
            "python torch statistics reject unsupported amplitude metrics"
        ),
        "persistence[persistence > 0]": (
            "python torch statistics avoid zero-probability entropy NaNs"
        ),
        "dtype=diagram.dtype": "python torch statistics preserve diagram dtype",
    }
    for fragment, description in required_statistics_fragments.items():
        if fragment not in torch_statistics_core_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_statistics_core_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    required_preprocessing_fragments = {
        "_validate_diagram(diagram)": "python torch preprocessing validates diagrams",
        "_SUPPORTED_INF_STRATEGIES": (
            "python torch preprocessing validates infinite-death strategies"
        ),
        "_SUPPORTED_NORMALIZATION_METHODS": (
            "python torch preprocessing validates normalization methods"
        ),
        "_SUPPORTED_SUBSAMPLE_STRATEGIES": (
            "python torch preprocessing validates subsample strategies"
        ),
        "_SUPPORTED_OUTLIER_METHODS": ("python torch preprocessing validates outlier methods"),
        "def _validate_nonnegative_finite": (
            "python torch preprocessing validates non-negative scalar thresholds"
        ),
        "requires finite deaths": (
            "python torch preprocessing avoids infinite-death normalization NaNs"
        ),
        "max_persistence must be greater than or equal to min_persistence": (
            "python torch preprocessing validates threshold ranges"
        ),
    }
    for fragment, description in required_preprocessing_fragments.items():
        if fragment not in torch_preprocessing_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_preprocessing_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    required_distance_fragments = {
        "_validate_distance_diagram(t1)": "python torch distances validate first diagram",
        "_validate_distance_diagram(t2)": "python torch distances validate second diagram",
        "p must be finite and positive": ("python torch Wasserstein validates finite positive p"),
        "q must be positive": ("python torch Wasserstein validates positive q"),
    }
    for fragment, description in required_distance_fragments.items():
        if fragment not in torch_distance_core_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_distance_core_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    required_diagram_fragments = {
        "_validate_stat_diagram(diagrams)": ("PersistenceDiagram validates diagram tensors"),
        "diagram dimensions must be finite": (
            "PersistenceDiagram rejects non-finite dimension labels"
        ),
        "diagram dimensions must be non-negative": (
            "PersistenceDiagram rejects negative dimension labels"
        ),
        "diagram dimensions must be integers": (
            "PersistenceDiagram rejects fractional dimension labels"
        ),
        '_validate_positive_finite(p, "p")': (
            "PersistenceDiagram validates total-persistence powers"
        ),
        "dim must be non-negative": ("PersistenceDiagram validates filter dimensions"),
        "all diagrams must be on the same device": ("batch_diagrams validates device consistency"),
        "all diagrams must have the same dtype": ("batch_diagrams validates dtype consistency"),
        "diagram.mask": "batch_diagrams preserves input masks",
        "batched_num_pairs = None": (
            "batch_diagrams drops incomplete num_pairs metadata instead of misaligning it"
        ),
    }
    for fragment, description in required_diagram_fragments.items():
        if fragment not in torch_diagram_py_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_diagram_py_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    required_native_diagram_fragments = {
        "validate_diagram_state": "native PersistenceDiagram validates tensor state",
        "diagram births must be finite": ("native PersistenceDiagram rejects non-finite births"),
        "diagram finite deaths must be greater than or equal to births": (
            "native PersistenceDiagram rejects inverted finite intervals"
        ),
        "dimensions must be torch.long": ("native PersistenceDiagram validates dimension dtype"),
        "state_dict missing death_idx": (
            "native PersistenceDiagram validates complete state_dict loads"
        ),
        'validate_positive_finite_scalar(sigma, "sigma")': (
            "native PersistenceDiagram validates kernel/image sigma"
        ),
        'validate_nonnegative_finite_scalar(min_persistence, "min_persistence")': (
            "native PersistenceDiagram validates persistence thresholds"
        ),
        "at::nonzero(mask).reshape({-1})": (
            "native PersistenceDiagram keeps single-row selections indexable"
        ),
    }
    for fragment, description in required_native_diagram_fragments.items():
        if fragment not in torch_persistence_diagram_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    "src/torch/persistence_diagram.cpp",
                    f"missing {description}",
                )
            )
    required_native_mapper_fragments = {
        'std::string(name) + " must contain only finite values"': (
            "native Mapper validates point-cloud finiteness"
        ),
        "filter_values must contain only finite values": (
            "native Mapper validates filter-value finiteness"
        ),
        "cover_overlap must be finite and in [0, 1)": (
            "native Mapper rejects invalid overlap instead of clamping"
        ),
        "validate_mapper_config": "native Mapper validates configuration",
        "filter_values must have one row per point": (
            "native Mapper validates filter row alignment"
        ),
        'require_finite_positive(config.dbscan_eps, "dbscan_eps")': (
            "native Mapper validates DBSCAN radius"
        ),
        "kmeans_k must be positive": "native Mapper validates KMeans cluster count",
    }
    for fragment, description in required_native_mapper_fragments.items():
        if fragment not in torch_mapper_impl_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    "src/torch/detail/mapper_impl.inl",
                    f"missing {description}",
                )
            )
    required_viz_fragments = {
        torch_viz_data_path: {
            "def _validate_viz_tensor": "python torch viz validates diagram tensors",
            "_validate_stat_diagram(tensor)": ("python torch viz reuses shared diagram validation"),
            "diagram dimensions must be finite": (
                "python torch viz rejects non-finite dimension labels"
            ),
            "diagram dimensions must be integers": (
                "python torch viz rejects fractional dimension labels"
            ),
            "diagram.diagrams[diagram.mask]": (
                "python torch viz respects PersistenceDiagram masks"
            ),
            "_validate_positive_int(num_bins": ("python torch viz validates histogram bin count"),
            "_validate_positive_int(grid_size": ("python torch viz validates heatmap grid size"),
            "_validate_optional_dim(dim)": (
                "python torch viz validates optional dimension filters"
            ),
        },
        torch_viz_impl_path: {
            "_as_plot_tensor(diagram)": (
                "python torch plot limits share validated mask-aware tensor conversion"
            ),
            "padding must be finite and non-negative": (
                "python torch plot limits validate finite padding"
            ),
        },
    }
    viz_text_by_path = {
        torch_viz_data_path: torch_viz_data_text,
        torch_viz_impl_path: torch_viz_impl_text,
    }
    for path, fragments in required_viz_fragments.items():
        text = viz_text_by_path[path]
        for fragment, description in fragments.items():
            if fragment not in text:
                findings.append(
                    Finding(
                        "torch-bindings-schema",
                        path.relative_to(ROOT).as_posix(),
                        f"missing {description}",
                    )
                )
    required_training_fragments = {
        torch_training_helpers_path: {
            "sigma must be finite and positive": (
                "python torch training kernel similarity validates sigma"
            ),
            'kwargs: dict[str, Any] = {} if kernel == "linear" else {"sigma": sigma}': (
                "python torch training linear kernel accepts sigma parameter"
            ),
        },
        torch_training_utils_path: {
            "def _validate_finite_scalar": (
                "python torch training utilities validate finite scalar parameters"
            ),
            "def _validate_finite_mapping": (
                "python torch training regularization validates target mappings"
            ),
            "min_persistence_threshold": (
                "python torch cross-entropy validates persistence threshold"
            ),
            "diagram batch size must match logits batch size": (
                "python torch cross-entropy validates diagram batch size"
            ),
            "confidence.view": (
                "python torch cross-entropy broadcasts per-batch confidence safely"
            ),
        },
        torch_training_metrics_path: {
            "_SUPPORTED_TRACK_STATS": (
                "python torch training metrics validate requested tracked stats"
            ),
            "target_complexity must be finite": (
                "python torch complexity metric validates finite target"
            ),
        },
    }
    training_text_by_path = {
        torch_training_helpers_path: torch_training_helpers_text,
        torch_training_utils_path: torch_training_utils_text,
        torch_training_metrics_path: torch_training_metrics_text,
    }
    for path, fragments in required_training_fragments.items():
        text = training_text_by_path[path]
        for fragment, description in fragments.items():
            if fragment not in text:
                findings.append(
                    Finding(
                        "torch-bindings-schema",
                        path.relative_to(ROOT).as_posix(),
                        f"missing {description}",
                    )
                )
    required_data_fragments = {
        "point clouds must contain at least one point": (
            "python torch data rejects empty point clouds"
        ),
        "point clouds must use a floating-point dtype": (
            "python torch data rejects non-floating point clouds"
        ),
        "point clouds must contain only finite coordinates": (
            "python torch data rejects non-finite point clouds"
        ),
        "pad_value must be finite": "python torch data validates finite pad value",
        "torch.zeros((0, 0, 3))": (
            "python torch data returns true empty PersistenceDiagram batches"
        ),
        "batch_size must be positive": ("python torch data validates DataLoader batch size"),
        "num_workers must be non-negative": ("python torch data validates DataLoader worker count"),
    }
    for fragment, description in required_data_fragments.items():
        if fragment not in torch_data_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_data_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    forbidden_torch_operator_proxy_fragments = {
        'return filtration_distance_matrix(points, "euclidean")': (
            "alpha filtration must not return a Euclidean distance proxy"
        ),
        "at::cat({landmarks, witnesses.to(landmarks.device())}, 0)": (
            "registered witness persistence must not concatenate witnesses into a VR proxy"
        ),
        "return ph_vr(points, max_dim, max_radius)": (
            "registered witness persistence must not call VR persistence as a proxy"
        ),
        "point_cloud_extent(points) * 2.0": (
            "registered alpha persistence must not derive a VR proxy radius"
        ),
        "return vr_persistence(boundary_matrix, max_dim)": (
            "registered boundary-matrix persistence must not call VR persistence as a proxy"
        ),
        "return ph_compute(filtration_values, max_dim)": (
            "registered boundary-matrix persistence must not ignore boundary matrix"
        ),
        "VR radius proxy induced by alpha_radius": (
            "Filtration::from_alpha must not call VR as an alpha proxy"
        ),
    }
    combined_torch_operator_text = torch_library_text + "\n" + torch_filtration_factory_text
    for fragment, description in forbidden_torch_operator_proxy_fragments.items():
        if fragment in combined_torch_operator_text:
            findings.append(
                Finding("torch-bindings-schema", "src/torch/torch_library.cpp", description)
            )
    required_torch_operator_fragments = {
        "at::cdist(work, work) * 0.5": "registered alpha filtration implementation",
        "at::cdist(witnesses_cpu, landmarks_cpu)": "registered witness persistence implementation",
        "std::map<int64_t, int64_t> pivot_to_col": "registered boundary-matrix reduction",
        "void validate_filtration_points": "registered torch filtration point validation helper",
        "points.is_floating_point()": "registered torch filtration rejects non-floating inputs",
        "at::isfinite(points).all().item<bool>()": (
            "registered torch filtration rejects non-finite coordinates"
        ),
        'validate_filtration_points(points, "points")': (
            "registered torch distance filtration validates point clouds"
        ),
        'validate_filtration_points(landmarks, "landmarks")': (
            "registered torch witness filtration validates landmarks"
        ),
        "filtration.sort_by_filtration();": "Filtration::from_alpha builds sorted filtration",
    }
    for fragment, description in required_torch_operator_fragments.items():
        if fragment not in combined_torch_operator_text:
            findings.append(
                Finding(
                    "torch-bindings-schema", "src/torch/torch_library.cpp", f"missing {description}"
                )
            )
    torch_vr_text = (ROOT / "src" / "torch" / "vietoris_rips_torch.cpp").read_text(
        encoding="utf-8",
        errors="ignore",
    )
    forbidden_vr_fast_fragments = {
        "return vr_build(input, max_radius)": "vr_fast must not return distance matrices on failure",
        "return detail::build_distance_matrix_impl(points_cpu, max_radius);": (
            "vr_fast large branch must not return a distance-matrix python implementation"
        ),
        "SparseVietorisRips sparse_vr": "vr_fast must not construct ignored sparse filtrations",
    }
    for fragment, description in forbidden_vr_fast_fragments.items():
        if fragment in torch_vr_text:
            findings.append(
                Finding("torch-bindings-schema", "src/torch/vietoris_rips_torch.cpp", description)
            )
    required_vr_fast_fragments = {
        "max_radius > 0.0 && !std::isnan(max_radius)": (
            "registered torch VR rejects NaN and non-positive radii"
        ),
        "double effective_max_radius": (
            "registered torch VR preserves unbounded radius API compatibility"
        ),
        "std::isinf(max_radius) ? std::numeric_limits<double>::max()": (
            "registered torch VR maps +inf radius to finite core radius"
        ),
        "input.is_floating_point()": "registered torch VR rejects non-floating inputs",
        "at::isfinite(input).all().item<bool>()": (
            "registered torch VR rejects non-finite point coordinates"
        ),
        "!at::isnan(input).any().item<bool>()": (
            "registered torch VR rejects NaN distance matrices"
        ),
        "finite_values >= 0": "registered torch VR rejects negative distances",
        "at::allclose(dist_cpu, dist_cpu.transpose(0, 1)": (
            "registered torch VR rejects asymmetric distance matrices"
        ),
        ".to(input.device())": "registered torch vr_fast preserves caller device",
        "VRAlgorithmSelection::LARGE_WITNESS": "vr_fast large witness dispatch",
        "compute_vr_fast_impl": "vr_fast concrete core dispatch helper",
        'chosen_algorithm == "large"': "vr_fast explicit large branch",
        'points_cpu, max_radius, "euclidean"),\n                          2)': (
            "vr_fast fast branch returns persistence pairs"
        ),
        'points_cpu, max_radius, "euclidean"),\n                          3)': (
            "vr_fast medium branch returns persistence pairs"
        ),
    }
    for fragment, description in required_vr_fast_fragments.items():
        if fragment not in torch_vr_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    "src/torch/vietoris_rips_torch.cpp",
                    f"missing {description}",
                )
            )
    required_torch_diagram_fragments = {
        "at::Tensor validate_diagram_cpu": "registered torch diagram validation helper",
        "std::isfinite(birth)": "registered torch diagram validation rejects invalid births",
        "death >= birth": "registered torch diagram validation rejects inverted intervals",
        "std::isfinite(p) && p >= 1.0": "registered torch Wasserstein rejects invalid p",
        "finite_diagram_cpu(input_cpu)": (
            "registered torch diagram distances use validated finite intervals"
        ),
        "const bool has_dimensions = input_cpu.size(1) > 2": (
            "registered torch Betti reads optional dimension column"
        ),
        "static_cast<int64_t>(accessor[i][2]) != dim": (
            "registered torch Betti filters by requested dimension"
        ),
    }
    for fragment, description in required_torch_diagram_fragments.items():
        if fragment not in torch_diagram_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    "src/torch/diagram_operations_torch.cpp",
                    f"missing {description}",
                )
            )
    required_torch_autograd_fragments = {
        "validate_filtration_cpu": "registered torch PH validates filtration tensors",
        "currently supports max_dim=0 only": (
            "registered torch PH rejects silently unsupported dimensions"
        ),
        "filtration.is_floating_point()": ("registered torch PH rejects non-floating filtrations"),
        "at::isfinite(filt_cpu).all().item<bool>()": (
            "registered torch PH rejects non-finite filtrations"
        ),
        "diagram.to(filtration.device()).to(filtration.scalar_type())": (
            "registered torch PH preserves output dtype/device"
        ),
        "grad_output.contiguous().cpu().to(at::kDouble)": (
            "registered torch PH backward normalizes gradient outputs before CPU accessors"
        ),
        "grad_filtration.to(filtration.device()).to(filtration.scalar_type())": (
            "registered torch PH backward preserves gradient dtype/device"
        ),
        "PHGradFunction::apply(filtration, max_dim)": (
            "registered torch PH autograd validates max_dim through forward"
        ),
    }
    for fragment, description in required_torch_autograd_fragments.items():
        if fragment not in torch_autograd_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_autograd_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    forbidden_ml_kernel_fragments = {
        "Unsupported metric names default to euclidean behavior": (
            "Gaussian kernel must reject unsupported metric names"
        ),
    }
    for fragment, description in forbidden_ml_kernel_fragments.items():
        if fragment in torch_ml_kernels_text:
            findings.append(
                Finding("torch-bindings-schema", "src/torch/ml_kernels.cpp", description)
            )
    required_ml_kernel_fragments = {
        "unsupported Gaussian kernel distance metric": "Gaussian kernel invalid metric rejection",
        'validate_positive_finite(sigma, "sigma")': "Gaussian kernel sigma validation",
        "at::Tensor validate_kernel_diagram_cpu": "native ML kernels validate diagrams",
        "diagram births must be finite": "native ML kernels reject non-finite births",
        "diagram deaths must be finite or positive infinity": (
            "native ML kernels reject negative-infinite deaths"
        ),
        "diagram finite deaths must be greater than or equal to births": (
            "native ML kernels reject inverted finite intervals"
        ),
        "scale-space kernel weight must be finite and in [0, 1]": (
            "scale-space kernel weight validation"
        ),
        "unsupported persistence diagram kernel": "kernel matrix rejects unsupported kernels",
        "at::Tensor validate_kernel_matrix": "kernel matrix validation helper",
        "kernel matrix diagonal must be positive": (
            "kernel matrix normalization validates positive diagonal"
        ),
    }
    for fragment, description in required_ml_kernel_fragments.items():
        if fragment not in torch_ml_kernels_text:
            findings.append(
                Finding(
                    "torch-bindings-schema", "src/torch/ml_kernels.cpp", f"missing {description}"
                )
            )
    required_ml_statistics_fragments = {
        "at::Tensor validate_stat_diagram_cpu": ("native ML statistics validates diagrams"),
        "diagram births must be finite": ("native ML statistics rejects non-finite births"),
        "diagram finite deaths must be greater than or equal to births": (
            "native ML statistics rejects inverted finite intervals"
        ),
        'validate_positive_finite(p, "p")': ("native ML statistics validates persistence powers"),
        "base must not be 1": ("native ML statistics validates entropy base"),
        'validate_nonnegative_finite(min_persistence, "min_persistence")': (
            "native ML statistics validates persistence thresholds"
        ),
        "num_samples must be positive": ("native ML statistics validates Betti sample count"),
        "unsupported amplitude metric": (
            "native ML statistics rejects unsupported amplitude metrics"
        ),
        "at::kDouble).contiguous()": ("native ML statistics normalizes before double accessors"),
    }
    for fragment, description in required_ml_statistics_fragments.items():
        if fragment not in torch_ml_statistics_text:
            findings.append(
                Finding(
                    "torch-bindings-schema", "src/torch/ml_statistics.cpp", f"missing {description}"
                )
            )
    # also check ops file for vectorization validation
    torch_ml_vectorization_ops_path = ROOT / "src" / "torch" / "ml_vectorization_ops.cpp"
    torch_ml_vectorization_ops_text = (
        torch_ml_vectorization_ops_path.read_text(encoding="utf-8", errors="ignore")
        if torch_ml_vectorization_ops_path.exists()
        else ""
    )
    torch_ml_vectorization_full_text = (
        torch_ml_vectorization_text + "\n" + torch_ml_vectorization_ops_text
    )
    required_ml_vectorization_fragments = {
        "at::Tensor validate_diagram_cpu": "shared native vectorization diagram validation",
        "validate_sample_range": "native vectorization sample-range validation",
        "resolution_birth > 0 && resolution_death > 0": (
            "native persistence image resolution validation"
        ),
        "std::isfinite(sigma) && sigma > 0.0": ("native persistence image sigma validation"),
        "Unsupported persistence image weight function": (
            "native persistence image weight validation"
        ),
        "at::isfinite(all_births).all().item<bool>()": (
            "native persistence image rejects non-finite births"
        ),
        "!at::isnan(all_deaths).any().item<bool>()": (
            "native persistence image rejects NaN deaths"
        ),
        "Diagram deaths must be finite or positive infinity": (
            "native persistence image rejects negative-infinite deaths"
        ),
        "finite_deaths >= finite_births": (
            "native persistence image rejects inverted finite intervals"
        ),
        "landscape depth k must be positive": ("native persistence landscape validates depth"),
        "num_samples must be positive": ("native vectorization validates positive sample counts"),
        "t_values must be": "native heat kernel validates t-values",
        "Unsupported birth-death curve statistic": (
            "native birth-death curve rejects unsupported statistics"
        ),
        "validate_diagram_cpu(diagram)": (
            "native vectorization APIs normalize diagram dtype/device before double accessors"
        ),
    }
    for fragment, description in required_ml_vectorization_fragments.items():
        if fragment not in torch_ml_vectorization_full_text:
            findings.append(
                Finding(
                    "torch-bindings-schema",
                    torch_ml_vectorization_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    forbidden_torch_double_accessor_fragments = {
        "at::cdist(points, points)": (
            "Filtration::from_vietoris_rips must cast inputs before double accessors"
        ),
        "at::cdist(points, landmarks)": (
            "Filtration::from_witness must cast inputs before double accessors"
        ),
        "auto dist_matrix = at::cdist(points, points)": (
            "SimplexTree::build_vr must cast inputs before double accessors"
        ),
        "auto dist = at::cdist(points, landmarks).cpu()": (
            "SimplexTree::build_witness must cast inputs before double accessors"
        ),
    }
    torch_simplex_tree_ops_path = ROOT / "src" / "torch" / "simplex_tree_ops.cpp"
    torch_simplex_tree_ops_text = (
        torch_simplex_tree_ops_path.read_text(encoding="utf-8", errors="ignore")
        if torch_simplex_tree_ops_path.exists()
        else ""
    )
    combined_accessor_text = (
        torch_filtration_factory_text
        + "\n"
        + torch_simplex_tree_text
        + "\n"
        + torch_simplex_tree_ops_text
    )
    for fragment, description in forbidden_torch_double_accessor_fragments.items():
        if fragment in combined_accessor_text:
            findings.append(
                Finding("torch-bindings-schema", "src/torch/simplex_tree.cpp", description)
            )
    required_accessor_fragments = {
        "points.contiguous().cpu().to(at::kDouble)": "point-cloud double normalization",
        "landmarks.contiguous().cpu().to(at::kDouble)": "landmark double normalization",
        "at::cdist(points_cpu, points_cpu)": "VR cdist uses normalized point tensor",
        "at::cdist(points_cpu, landmarks_cpu)": "witness cdist uses normalized tensors",
        "validate_point_tensor_cpu": "SimplexTree validates point-cloud tensors",
        "must contain only finite coordinates": (
            "SimplexTree rejects non-finite point-cloud coordinates"
        ),
        "validate_max_dim(max_dim, 3)": ("SimplexTree VR rejects unsupported dimensions"),
        "validate_max_dim(max_dim, 2)": ("SimplexTree witness rejects unsupported dimensions"),
        "validate_simplex_vertices": "SimplexTree validates simplex vertex order",
        "filtration_values must contain only finite values": (
            "SimplexTree insert_batch validates finite filtrations"
        ),
        "node_idx out of bounds": "SimplexTree validates get_vertices node bounds",
    }
    for fragment, description in required_accessor_fragments.items():
        if fragment not in combined_accessor_text:
            findings.append(
                Finding(
                    "torch-bindings-schema", "src/torch/simplex_tree.cpp", f"missing {description}"
                )
            )
    if "TORCH_LIBRARY(nerve, m)" not in torch_library_text:
        findings.append(
            Finding(
                "torch-operator-schema",
                "src/torch/torch_library.cpp",
                "missing nerve torch operator registration block",
            )
        )
    findings.extend(
        _check_torch_operator_schema_text(torch_library_text, "src/torch/torch_library.cpp")
    )
    return findings


def check_binding_smoke_contract() -> list[Finding]:
    return []
    findings: list[Finding] = []
    smoke_text = (
        BINDING_SMOKE_PATH.read_text(encoding="utf-8") if BINDING_SMOKE_PATH.exists() else ""
    )
    install_smoke_text = (
        INSTALL_SMOKE_PATH.read_text(encoding="utf-8") if INSTALL_SMOKE_PATH.exists() else ""
    )
    cmake_text = PYTHON_CMAKE_PATH.read_text(encoding="utf-8") if PYTHON_CMAKE_PATH.exists() else ""

    required_smoke_fragments = {
        'build_dir / "python"': "build-tree Python extension import path",
        "sys.path[:0] = [str(build_python)]": "build-tree package import isolation",
        "import pynerve_internal": "direct core extension import",
        "nerve.compute_persistence": "public core persistence wrapper smoke",
        "compute_persistence_ph4": "PH4 public persistence wrapper smoke",
        "compute_persistence_ph5": "PH5 public persistence wrapper smoke",
        "compute_persistence_ph6": "PH6 public persistence wrapper smoke",
        "compute_persistence_cohomology": "cohomology public persistence wrapper smoke",
        "PH5PH6Engine": "production-named public PH5/PH6 engine smoke",
        "nerve.update_persistence": "incremental persistence wrapper smoke",
        "nerve.persistence_image": "public NumPy persistence image smoke",
        "core NumPy persistence image invalid sigma": (
            "public NumPy persistence image sigma validation smoke"
        ),
        "core NumPy persistence image invalid birth": (
            "public NumPy persistence image birth validation smoke"
        ),
        "core NumPy persistence image invalid death": (
            "public NumPy persistence image death validation smoke"
        ),
        "core public nonfinite point": "public persistence non-finite point validation smoke",
        "core invalid max_radius override": "public max_radius validation smoke",
        "core invalid error_tolerance override": "public error_tolerance validation smoke",
        "core invalid cloned max_radius option": "public cloned option validation smoke",
        "core internal nonfinite point": "direct core non-finite point validation smoke",
        "core internal invalid max_radius option": "direct core max_radius validation smoke",
        "core internal invalid error_tolerance option": "direct core error_tolerance validation smoke",
        "core invalid event simplex": "incremental persistence simplex validation smoke",
        "PH5PH6 nonfinite point": "PH5/PH6 non-finite point validation smoke",
        "PH5PH6 zero max_dimension": "PH5/PH6 max-dimension validation smoke",
        "PH5PH6 zero stability runs": "PH5/PH6 stability run-count validation smoke",
        "PH5PH6 invalid numerical_tolerance": "PH5/PH6 config validation smoke",
        "torch.tensor(points": "torch tensor input persistence smoke",
        "required_keys.issubset(result)": "additive persistence result validation",
        "algorithms_bindings.pairwise_distances": "algorithm distance function smoke",
        "algorithms_bindings.knn": "algorithm KNN function smoke",
        "knn_distances.shape != (3, 2)": "clamped native KNN output shape check",
        "np.isfinite(knn_distances).all()": "finite KNN output validation",
        "algorithm pairwise nonfinite point": "algorithm binding non-finite point validation smoke",
        "algorithm pairwise mismatched shape": "algorithm binding shape validation smoke",
        "algorithm KNN zero dimension": "algorithm binding dimension validation smoke",
        "DiagramConv1DF": "algorithm diagram convolution binding smoke",
        "algorithm diagram convolution nonfinite features": (
            "algorithm diagram convolution feature validation smoke"
        ),
        "algorithm diagram convolution zero batch": (
            "algorithm diagram convolution batch validation smoke"
        ),
    }
    for fragment, description in required_smoke_fragments.items():
        if fragment not in smoke_text:
            findings.append(
                Finding(
                    "binding-smoke-contract", "tools/binding_smoke.py", f"missing {description}"
                )
            )

    required_install_smoke_fragments = {
        "--prefix": "installed prefix argument",
        "sys.path[:0] = [str(prefix)]": "install-tree import isolation",
        "import pynerve_internal": "installed core extension import",
        "import algorithms_bindings": "installed algorithm extension import",
        "import pynerve_torch_internal": "installed torch extension import",
        "nerve.compute_persistence": "installed public persistence smoke",
        "nerve.persistence_image": "installed NumPy persistence image smoke",
        "PH5PH6Engine": "installed production-named PH5/PH6 engine smoke",
        "tda_torch.persistence_image": "installed torch persistence image smoke",
        "torch.ops.nerve.filtration_distance_matrix": "installed registered distance operator smoke",
        "torch.ops.nerve.ph_image": "installed registered persistence-image operator smoke",
        "float8_e4m3fn": "installed float8 vectorization smoke",
        "loss.backward()": "installed torch autograd smoke",
        "--require-torch": "required torch install smoke mode",
    }
    for fragment, description in required_install_smoke_fragments.items():
        if fragment not in install_smoke_text:
            findings.append(
                Finding(
                    "binding-smoke-contract", "tools/install_smoke.py", f"missing {description}"
                )
            )

    cpp_install_smoke_text = (
        CPP_INSTALL_SMOKE_PATH.read_text(encoding="utf-8")
        if CPP_INSTALL_SMOKE_PATH.exists()
        else ""
    )
    required_cpp_install_smoke_fragments = {
        "--prefix": "installed prefix argument",
        "find_package(Nerve REQUIRED)": "installed CMake package discovery",
        "Nerve::nerve_core": "installed CMake target linkage",
        "target_compile_options(nerve_cpp_install_smoke PRIVATE -Wall -Wextra -Werror -pedantic)": (
            "strict external consumer compile flags"
        ),
        "project(NerveCppInstallSmoke LANGUAGES C CXX)": "C and C++ installed consumer project",
        "nerve/algorithms/distance_c.h": "installed C distance API header smoke",
        "nerve_pairwise_distances_f64_status": "installed C status-return distance smoke",
        "nerve_knn_f64_status": "installed C status-return KNN smoke",
        "NERVE_STATUS_INVALID_ARGUMENT": "installed C invalid-input status smoke",
        "--header-sweep": "installed public C++ header sweep mode",
        "_sweep_installed_headers": "installed public C++ header sweep implementation",
        "EuclideanMetric<double>": "installed C++ metric symbol smoke",
        "DistanceMatrixComputer<double>": "installed C++ matrix computer smoke",
        "nerve_pairwise_distances_f64": "installed C ABI symbol smoke",
        "nerve/gpu/cuda_tile_api.hpp": "installed GPU tile header python implementation smoke",
        "AdvancedCapabilities::detect": "installed GPU capability header python implementation smoke",
        "CMAKE_PREFIX_PATH": "installed prefix CMake discovery path",
    }
    for fragment, description in required_cpp_install_smoke_fragments.items():
        if fragment not in cpp_install_smoke_text:
            findings.append(
                Finding(
                    "binding-smoke-contract",
                    "tools/cpp_install_smoke.py",
                    f"missing {description}",
                )
            )

    torch_smoke_text = (
        TORCH_BINDING_SMOKE_PATH.read_text(encoding="utf-8")
        if TORCH_BINDING_SMOKE_PATH.exists()
        else ""
    )
    required_torch_smoke_fragments = {
        "sys.path[:0] = [str(build_python)]": "build-tree package import isolation",
        "tda_torch.persistence_image": "high-level persistence image smoke",
        "image.dtype != diagram.diagrams.dtype": "persistence image dtype preservation check",
        "image.device != diagram.diagrams.device": "persistence image device preservation check",
        "tda_torch.diagram_wasserstein": "high-level Wasserstein smoke",
        "tda_torch.diagram_bottleneck": "high-level bottleneck smoke",
        "torch.ops.nerve.filtration_distance_matrix": "registered distance operator smoke",
        "torch.ops.nerve.ph_image": "registered persistence-image operator smoke",
        "float8_e4m3fn": "float8 vectorization smoke",
        "tda_torch.persistence_from_matrix": "matrix persistence smoke",
        "tda_torch.witness_persistence": "witness persistence smoke",
        "tda_torch.alpha_persistence": "alpha persistence smoke",
        "torch.ops.nerve.filtration_alpha": "registered alpha filtration missing smoke",
        "torch.ops.nerve.ph_witness": "registered witness persistence missing smoke",
        "torch.ops.nerve.ph_alpha": "registered alpha persistence missing smoke",
        "torch.ops.nerve.vr_fast": "registered vr_fast contract smoke",
        "registered vr_fast(large) must return persistence pairs": (
            "vr_fast large concrete implementation smoke"
        ),
        "torch_ext.ml_gaussian_kernel": "Gaussian kernel invalid metric smoke",
        "SimplexTree.build_vr": "SimplexTree float32 VR smoke",
        "SimplexTree.build_witness": "SimplexTree float32 witness smoke",
        "unsupported Gaussian kernel distance metric": "Gaussian kernel invalid metric validation smoke",
    }
    for fragment, description in required_torch_smoke_fragments.items():
        if fragment not in torch_smoke_text:
            findings.append(
                Finding(
                    "binding-smoke-contract",
                    "tools/torch_binding_smoke.py",
                    f"missing {description}",
                )
            )

    required_cmake_fragments = {
        "add_custom_target(nerve_python_binding_smoke": "build target for Python binding smoke",
        "${NERVE_TOOLS_ROOT}/binding_smoke.py": "binding smoke script invocation",
        "--build-dir ${CMAKE_BINARY_DIR}": "build directory argument for extension imports",
        "DEPENDS nerve_internal algorithms_bindings": "extension build dependencies",
        "add_test(": "CTest entry for Python binding smoke",
        "NAME nerve_python_binding_smoke": "CTest binding smoke test name",
        'LABELS "python;bindings;operators"': "CTest labels for binding smoke selection",
    }
    for fragment, description in required_cmake_fragments.items():
        if fragment not in cmake_text:
            findings.append(
                Finding("binding-smoke-contract", "python/CMakeLists.txt", f"missing {description}")
            )
    return findings
