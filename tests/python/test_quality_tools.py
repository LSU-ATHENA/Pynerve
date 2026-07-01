from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _get_quality():
    sys.path.insert(0, str(ROOT / "tools"))
    import ast  # noqa: PLC0415
    from quality_checks import (  # noqa: PLC0415
        build_contracts,
        binding_contracts,
        common,
        import_api,
        static_text,
    )

    class _Q:
        pass

    q = _Q()
    q.Finding = common.Finding
    q.ROOT = common.ROOT
    q.ast = ast
    q._lazy_submodules = common._lazy_submodules
    q._load_tool_module = common._load_tool_module
    q._resolve_import_from = common._resolve_import_from
    q._detect_cycles = import_api._detect_cycles
    q._is_nerve_package_import = import_api._is_nerve_package_import
    q._check_torch_operator_schema_text = binding_contracts._check_torch_operator_schema_text
    q.check_operator_schema = binding_contracts.check_operator_schema
    q.check_public_api = import_api.check_public_api
    q.check_pybind_schema = binding_contracts.check_pybind_schema
    q.check_algorithm_bindings_schema = binding_contracts.check_algorithm_bindings_schema
    q.check_torch_bindings_schema = binding_contracts.check_torch_bindings_schema
    q.check_binding_smoke_contract = binding_contracts.check_binding_smoke_contract
    q.check_import_graph = import_api.check_import_graph
    q.check_static_text = static_text.check_static_text
    q.check_test_matrix_contract = build_contracts.check_test_matrix_contract
    q.check_performance_guard_contract = build_contracts.check_performance_guard_contract
    q.check_ctest_contract = build_contracts.check_ctest_contract
    q.check_build_install_contract = build_contracts.check_build_install_contract
    q.check_static_analysis_contract = build_contracts.check_static_analysis_contract
    q.check_ci_contract = build_contracts.check_ci_contract
    return q


@pytest.mark.quality
def test_operator_schema_has_core_surface() -> None:
    result = _get_quality().check_operator_schema()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_operator_schema_parser_handles_defaulted_cuda_signatures(tmp_path: Path) -> None:
    operator_schema = _get_quality()._load_tool_module("operator_schema")
    source = tmp_path / "cuda_api.cuh"
    source.write_text(
        """
errors::ErrorResult<void> launchDistanceMatrixKernel(
    const double* points,
    double* distances,
    Size n_points,
    Size point_dim,
    double max_radius,
    const CUDADistanceMatrixConfig& config,
    Size stream_offset = 0,
    Size stream_size = 0);
""",
        encoding="utf-8",
    )

    signatures = operator_schema.find_signatures(source, "launchDistanceMatrixKernel")
    assert len(signatures) == 1, f"expected 1 signature, got {len(signatures)}"
    assert signatures[0].return_type == "errors::ErrorResult<void>", (
        f"expected 'errors::ErrorResult<void>', got '{signatures[0].return_type}'"
    )
    param_names = tuple(parameter.name for parameter in signatures[0].params)
    assert param_names == (
        "points",
        "distances",
        "n_points",
        "point_dim",
        "max_radius",
        "config",
        "stream_offset",
        "stream_size",
    ), (
        f"expected parameter names {('points', 'distances', 'n_points', 'point_dim', 'max_radius', 'config', 'stream_offset', 'stream_size')}, got {param_names}"
    )
    assert signatures[0].params[5].type == "const CUDADistanceMatrixConfig&", (
        f"expected 'const CUDADistanceMatrixConfig&', got '{signatures[0].params[5].type}'"
    )


@pytest.mark.quality
def test_public_api_contract_is_enforced() -> None:
    result = _get_quality().check_public_api()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_pyi_generator_includes_package_init_public_functions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generate_pyi = _get_quality()._load_tool_module("generate_pyi")
    package = tmp_path / "python" / "nerve"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """
def public_api(value, optional=1, *, mode=None):
    return value

def _private_api(value):
    return value
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(generate_pyi, "ROOT", tmp_path)
    monkeypatch.setattr(generate_pyi, "PY_ROOT", package)

    pyi_result = generate_pyi.generate(check=False)
    assert pyi_result == [], f"expected no findings, got {pyi_result}"
    pyi_file = package / "__init__.pyi"
    pyi_content = pyi_file.read_text(encoding="utf-8")
    assert pyi_content == "def public_api(value, optional=..., *, mode=...) -> object: ...\n", (
        f"expected pyi content, got '{pyi_content}'"
    )
    check_result = generate_pyi.generate(check=True)
    assert check_result == [], f"expected no findings, got {check_result}"


@pytest.mark.quality
def test_pybind_schema_matches_public_python_api() -> None:
    result = _get_quality().check_pybind_schema()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_algorithm_bindings_schema_is_enforced() -> None:
    result = _get_quality().check_algorithm_bindings_schema()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_torch_bindings_schema_is_enforced() -> None:
    result = _get_quality().check_torch_bindings_schema()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_torch_operator_schema_parser_detects_arity_mismatch() -> None:
    source = """
namespace nerve::torch {
at::Tensor filtration_distance_matrix(const at::Tensor& points);
}
TORCH_LIBRARY(nerve, m) {
    m.def("filtration_distance_matrix(Tensor input, str metric) -> Tensor",
          &nerve::torch::filtration_distance_matrix);
}
"""

    findings = _get_quality()._check_torch_operator_schema_text(source, "synthetic.cpp")
    assert any("arity mismatch" in finding.message for finding in findings), (
        f"expected at least one finding with 'arity mismatch', got {[(f.path, f.message) for f in findings]}"
    )


@pytest.mark.quality
def test_python_binding_smoke_contract_is_enforced() -> None:
    result = _get_quality().check_binding_smoke_contract()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_import_graph_has_no_two_node_cycles() -> None:
    result = _get_quality().check_import_graph()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_import_graph_helpers_resolve_relative_imports_and_cycles() -> None:
    relative_import = _get_quality().ast.parse("from .._torch_diagrams import persistence").body[0]
    resolved = _get_quality()._resolve_import_from(
        "pynerve.training.curriculum",
        Path("curriculum.py"),
        relative_import,
    )
    assert resolved == "pynerve._torch_diagrams", (
        f"expected 'pynerve._torch_diagrams', got '{resolved}'"
    )
    assert _get_quality()._is_nerve_package_import("pynerve"), (
        "expected 'pynerve' to be a nerve package import"
    )
    assert _get_quality()._is_nerve_package_import("pynerve.torch"), (
        "expected 'pynerve.torch' to be a nerve package import"
    )
    assert not _get_quality()._is_nerve_package_import("nerve_internal"), (
        "expected 'nerve_internal' to NOT be a nerve package import"
    )

    cycles = _get_quality()._detect_cycles(
        {
            "pynerve.a": {"pynerve.b"},
            "pynerve.b": {"pynerve.c"},
            "pynerve.c": {"pynerve.a"},
        }
    )
    assert cycles == [["pynerve.a", "pynerve.b", "pynerve.c", "pynerve.a"]], (
        f"expected cycle [['pynerve.a', 'pynerve.b', 'pynerve.c', 'pynerve.a']], got {cycles}"
    )


@pytest.mark.quality
def test_import_graph_helper_extracts_lazy_submodule_names() -> None:
    tree = _get_quality().ast.parse(
        """
_LAZY_SUBMODULES = {"torch"}
def __getattr__(name):
    if name in {"nn"}:
        pass
"""
    )
    lazy = _get_quality()._lazy_submodules(tree)
    assert lazy == {"nn", "torch"}, f"expected lazy submodules {{'nn', 'torch'}}, got {lazy}"


@pytest.mark.quality
def test_static_text_has_no_banned_release_names() -> None:
    result = _get_quality().check_static_text()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_generated_matrix_contract_is_enforced() -> None:
    result = _get_quality().check_test_matrix_contract()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_performance_guard_contract_is_enforced() -> None:
    result = _get_quality().check_performance_guard_contract()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_ctest_contract_is_enforced() -> None:
    result = _get_quality().check_ctest_contract()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_build_install_contract_is_enforced() -> None:
    result = _get_quality().check_build_install_contract()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_static_analysis_contract_is_enforced() -> None:
    result = _get_quality().check_static_analysis_contract()
    assert result == [], f"expected no findings, got {result}"


@pytest.mark.quality
def test_static_analysis_selects_curated_compile_database_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static_analysis = _get_quality()._load_tool_module("static_analysis")
    monkeypatch.setattr(static_analysis, "ROOT", tmp_path)

    critical_source = tmp_path / "src" / "persistence" / "reduction" / "reduction_ops.cpp"
    distance_source = tmp_path / "src" / "algorithms" / "distance.cpp"
    extra_source = tmp_path / "src" / "metrics" / "matrix" / "matrix_distance_ops.cpp"
    binding_source = tmp_path / "python" / "bindings" / "nerve_torch_bindings.cpp"
    test_source = tmp_path / "tests" / "cpp" / "unit_test.cpp"
    for source in (critical_source, distance_source, extra_source, binding_source, test_source):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("int nerve_static_analysis_fixture() { return 0; }\n", encoding="utf-8")

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "compile_commands.json").write_text(
        json.dumps(
            [
                {"file": str(extra_source)},
                {"file": str(critical_source)},
                {"file": str(binding_source)},
                {"file": str(test_source)},
                {"file": "src/algorithms/distance.cpp"},
                {"file": str(critical_source)},
            ]
        ),
        encoding="utf-8",
    )

    compile_sources = static_analysis._compile_database_sources("build")
    compile_sources_list = [source.relative_to(tmp_path).as_posix() for source in compile_sources]
    assert compile_sources_list == [
        "src/metrics/matrix/matrix_distance_ops.cpp",
        "src/persistence/reduction/reduction_ops.cpp",
        "python/bindings/nerve_torch_bindings.cpp",
        "tests/cpp/unit_test.cpp",
        "src/algorithms/distance.cpp",
    ], f"expected compile sources list, got {compile_sources_list}"

    selected = static_analysis._select_clang_tidy_sources(
        compile_sources,
        "critical",
        critical_paths=(
            "src/persistence/reduction/reduction_ops.cpp",
            "src/algorithms/distance.cpp",
        ),
    )
    selected_list = [source.relative_to(tmp_path).as_posix() for source in selected]
    assert selected_list == [
        "src/persistence/reduction/reduction_ops.cpp",
        "src/algorithms/distance.cpp",
    ], f"expected critical sources, got {selected_list}"

    all_sources = static_analysis._select_clang_tidy_sources(compile_sources, "all")
    all_sources_list = [source.relative_to(tmp_path).as_posix() for source in all_sources]
    assert all_sources_list == [
        "python/bindings/nerve_torch_bindings.cpp",
        "src/algorithms/distance.cpp",
        "src/metrics/matrix/matrix_distance_ops.cpp",
        "src/persistence/reduction/reduction_ops.cpp",
    ], f"expected all sources list, got {all_sources_list}"

    command = static_analysis._clang_tidy_command("build", selected)
    assert command[-2:] == [
        "src/persistence/reduction/reduction_ops.cpp",
        "src/algorithms/distance.cpp",
    ], f"expected command suffix, got {command[-2:]}"
    cppcheck_command = static_analysis._cppcheck_command(selected)
    assert cppcheck_command[:3] == [
        "cppcheck",
        "--enable=warning,performance,portability",
        "--error-exitcode=1",
    ], f"expected cppcheck prefix, got {cppcheck_command[:3]}"
    assert cppcheck_command[-2:] == [
        "src/persistence/reduction/reduction_ops.cpp",
        "src/algorithms/distance.cpp",
    ], f"expected cppcheck suffix, got {cppcheck_command[-2:]}"

    monkeypatch.setattr(
        static_analysis,
        "_clang_tidy_extra_args",
        lambda: ["--extra-arg=-idirafter/toolchain/include"],
    )
    command_with_toolchain_include = static_analysis._clang_tidy_command("build", selected)
    assert "--extra-arg=-idirafter/toolchain/include" in command_with_toolchain_include, (
        f"expected toolchain include arg in command, got {command_with_toolchain_include}"
    )
    assert command_with_toolchain_include[-2:] == [
        "src/persistence/reduction/reduction_ops.cpp",
        "src/algorithms/distance.cpp",
    ], f"expected command suffix with toolchain, got {command_with_toolchain_include[-2:]}"


@pytest.mark.quality
def test_static_analysis_handles_absolute_build_dir_without_compile_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    static_analysis = _get_quality()._load_tool_module("static_analysis")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    build_dir = tmp_path / "external-build"
    build_dir.mkdir()

    monkeypatch.setattr(static_analysis, "ROOT", repo_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "static_analysis.py",
            "--language",
            "cpp",
            "--build-dir",
            str(build_dir),
        ],
    )

    ret = static_analysis.main()
    assert ret == 0, f"expected main() to return 0, got {ret}"
    output = capsys.readouterr().out
    assert f"missing {build_dir / 'compile_commands.json'}" in output, (
        f"expected 'missing {build_dir / 'compile_commands.json'}' in output, got {output}"
    )


@pytest.mark.quality
def test_static_analysis_required_module_rejects_missing_import(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    static_analysis = _get_quality()._load_tool_module("static_analysis")

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("missing Python module must fail before probing executables")

    monkeypatch.setattr(static_analysis.importlib.util, "find_spec", lambda _module: None)
    monkeypatch.setattr(static_analysis.shutil, "which", fail_if_called)
    monkeypatch.setattr(static_analysis.subprocess, "run", fail_if_called)
    monkeypatch.setattr(static_analysis.subprocess, "call", fail_if_called)

    ret = static_analysis._run([static_analysis.sys.executable, "-m", "mypy", "tools"], True)
    assert ret == 1, f"expected _run() to return 1, got {ret}"
    stderr = capsys.readouterr().err
    assert "missing static-analysis module: mypy" in stderr, (
        f"expected 'missing static-analysis module: mypy' in stderr, got '{stderr}'"
    )


@pytest.mark.quality
def test_static_analysis_required_cuda_rejects_wrong_nvcc_release(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    static_analysis = _get_quality()._load_tool_module("static_analysis")

    class VersionResult:
        returncode = 0
        stdout = "Cuda compilation tools, release 13.2, V13.2.78"
        stderr = ""

    def fail_if_run(_command: list[str], _required: bool) -> int:
        raise AssertionError("CUDA commands should not run after a failed required version check")

    monkeypatch.setattr(static_analysis.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        static_analysis.subprocess, "run", lambda *_args, **_kwargs: VersionResult()
    )
    monkeypatch.setattr(static_analysis, "_run", fail_if_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["static_analysis.py", "--language", "cuda", "--required"],
    )

    ret = static_analysis.main()
    assert ret == 1, f"expected main() to return 1, got {ret}"
    stderr = capsys.readouterr().err
    assert "CUDA static analysis requires at least CUDA 12" in stderr, (
        f"expected CUDA 12 message in stderr, got '{stderr}'"
    )


@pytest.mark.quality
def test_static_analysis_required_cuda_runs_nvcc_and_sanitizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static_analysis = _get_quality()._load_tool_module("static_analysis")
    commands: list[list[str]] = []

    class VersionResult:
        returncode = 0
        stdout = "Cuda compilation tools, release 12.4, V12.4.0"
        stderr = ""

    def record_run(command: list[str], required: bool) -> int:
        assert required, "expected required=True"
        commands.append(command)
        return 0

    monkeypatch.setattr(static_analysis.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        static_analysis.subprocess, "run", lambda *_args, **_kwargs: VersionResult()
    )
    monkeypatch.setattr(static_analysis, "_run", record_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["static_analysis.py", "--language", "cuda", "--required"],
    )

    ret = static_analysis.main()
    assert ret == 0, f"expected main() to return 0, got {ret}"
    assert commands == [
        ["nvcc", "--version"],
        ["compute-sanitizer", "--version"],
    ], f"expected commands list (no cuda_launch_audit), got {commands}"


@pytest.mark.quality
def test_ci_contract_is_enforced() -> None:
    result = _get_quality().check_ci_contract()
    assert result == [], f"expected no findings, got {result}"
