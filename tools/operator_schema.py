#!/usr/bin/env python3
"""Validate required operator signatures across C++, CUDA, and pybind layers."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISTANCE_HEADER = ROOT / "src" / "include" / "nerve" / "algorithms" / "distance.hpp"
DISTANCE_C_API_IMPL = ROOT / "src" / "algorithms" / "detail" / "distance_c_api_instantiations.inl"
ALGORITHM_BINDINGS = ROOT / "python" / "bindings" / "nerve_algorithms_bindings.cpp"
CORE_BINDINGS = ROOT / "python" / "bindings" / "nerve_api_bindings.cpp"
CUDA_DISTANCE_HEADER = (
    ROOT / "src" / "include" / "nerve" / "persistence" / "cuda" / "cuda_distance_matrix.hpp"
)
CUDA_DISTANCE_IMPL = ROOT / "src" / "persistence" / "cuda" / "matrix_distance_cuda.cu"
CUDA_TENSOR_CORE_HEADER = (
    ROOT / "src" / "include" / "nerve" / "persistence" / "cuda" / "cuda_tensor_core.hpp"
)
CUDA_TENSOR_CORE_IMPL = ROOT / "src" / "persistence" / "cuda" / "tensor_core_cuda.cu"
CUDA_TENSOR_CORE_WRAPPER = ROOT / "src" / "persistence" / "cuda" / "tensor_core_wrappers_cuda.cu"
CUDA_BLACKWELL_HEADER = (
    ROOT / "src" / "include" / "nerve" / "persistence" / "cuda" / "cuda_blackwell_tma.hpp"
)
CUDA_BLACKWELL_IMPL = ROOT / "src" / "persistence" / "cuda" / "cuda_blackwell_tma.cu"


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    message: str


@dataclass(frozen=True)
class Parameter:
    type: str
    name: str


@dataclass(frozen=True)
class Signature:
    return_type: str
    params: tuple[Parameter, ...]
    line: int


@dataclass(frozen=True)
class SignatureExpectation:
    check: str
    path: Path
    symbol: str
    return_type: str
    params: tuple[Parameter, ...]
    description: str


@dataclass(frozen=True)
class BindingExpectation:
    check: str
    path: Path
    symbol: str
    args: tuple[str, ...]
    description: str


def _normalize_type(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s*([*&])\s*", r"\1", text)
    text = re.sub(r"\b(?:inline|static)\b\s*", "", text)
    text = text.replace("[[nodiscard]]", "")
    return re.sub(r"\s+", " ", text.strip())


def param(type_: str, name: str) -> Parameter:
    return Parameter(_normalize_type(type_), name)


DISTANCE_MATRIX_PARAMS = (
    param("const double*", "points"),
    param("double*", "distances"),
    param("Size", "n_points"),
    param("Size", "point_dim"),
    param("double", "max_radius"),
)
CUDA_STREAM_PARAM = param("cudaStream_t", "stream")

SIGNATURE_EXPECTATIONS: tuple[SignatureExpectation, ...] = (
    SignatureExpectation(
        "operator-schema",
        DISTANCE_HEADER,
        "nerve_pairwise_distances_f32",
        "void",
        (
            param("const float*", "points"),
            param("size_t", "n"),
            param("size_t", "dim"),
            param("float*", "output"),
        ),
        "float32 C distance-matrix API declaration",
    ),
    SignatureExpectation(
        "operator-schema",
        DISTANCE_C_API_IMPL,
        "nerve_pairwise_distances_f32",
        "void",
        (
            param("const float*", "points"),
            param("size_t", "n"),
            param("size_t", "dim"),
            param("float*", "output"),
        ),
        "float32 C distance-matrix API implementation",
    ),
    SignatureExpectation(
        "operator-schema",
        DISTANCE_HEADER,
        "nerve_pairwise_distances_f64",
        "void",
        (
            param("const double*", "points"),
            param("size_t", "n"),
            param("size_t", "dim"),
            param("double*", "output"),
        ),
        "float64 C distance-matrix API declaration",
    ),
    SignatureExpectation(
        "operator-schema",
        DISTANCE_C_API_IMPL,
        "nerve_pairwise_distances_f64",
        "void",
        (
            param("const double*", "points"),
            param("size_t", "n"),
            param("size_t", "dim"),
            param("double*", "output"),
        ),
        "float64 C distance-matrix API implementation",
    ),
    SignatureExpectation(
        "operator-schema",
        DISTANCE_HEADER,
        "nerve_knn_f32",
        "void",
        (
            param("const float*", "points"),
            param("size_t", "n"),
            param("size_t", "dim"),
            param("size_t", "k"),
            param("float*", "out_distances"),
            param("size_t*", "out_indices"),
        ),
        "float32 C KNN API declaration",
    ),
    SignatureExpectation(
        "operator-schema",
        DISTANCE_C_API_IMPL,
        "nerve_knn_f32",
        "void",
        (
            param("const float*", "points"),
            param("size_t", "n"),
            param("size_t", "dim"),
            param("size_t", "k"),
            param("float*", "out_distances"),
            param("size_t*", "out_indices"),
        ),
        "float32 C KNN API implementation",
    ),
    SignatureExpectation(
        "operator-schema",
        DISTANCE_HEADER,
        "nerve_knn_f64",
        "void",
        (
            param("const double*", "points"),
            param("size_t", "n"),
            param("size_t", "dim"),
            param("size_t", "k"),
            param("double*", "out_distances"),
            param("size_t*", "out_indices"),
        ),
        "float64 C KNN API declaration",
    ),
    SignatureExpectation(
        "operator-schema",
        DISTANCE_C_API_IMPL,
        "nerve_knn_f64",
        "void",
        (
            param("const double*", "points"),
            param("size_t", "n"),
            param("size_t", "dim"),
            param("size_t", "k"),
            param("double*", "out_distances"),
            param("size_t*", "out_indices"),
        ),
        "float64 C KNN API implementation",
    ),
    SignatureExpectation(
        "operator-schema",
        CUDA_DISTANCE_HEADER,
        "launchDistanceMatrixKernel",
        "errors::ErrorResult<void>",
        (
            *DISTANCE_MATRIX_PARAMS,
            param("const CUDADistanceMatrixConfig&", "config"),
            param("Size", "stream_offset"),
            param("Size", "stream_size"),
        ),
        "configured CUDA distance-matrix launcher declaration",
    ),
    SignatureExpectation(
        "operator-schema",
        CUDA_DISTANCE_IMPL,
        "launchDistanceMatrixKernel",
        "errors::ErrorResult<void>",
        (
            *DISTANCE_MATRIX_PARAMS,
            param("const CUDADistanceMatrixConfig&", "config"),
            param("Size", "stream_offset"),
            param("Size", "stream_size"),
        ),
        "configured CUDA distance-matrix launcher implementation",
    ),
    SignatureExpectation(
        "operator-schema",
        CUDA_DISTANCE_HEADER,
        "launchDistanceMatrixKernel",
        "errors::ErrorResult<void>",
        (*DISTANCE_MATRIX_PARAMS, param("Size", "stream_offset"), param("Size", "stream_size")),
        "auto-configured CUDA distance-matrix launcher declaration",
    ),
    SignatureExpectation(
        "operator-schema",
        CUDA_DISTANCE_IMPL,
        "launchDistanceMatrixKernel",
        "errors::ErrorResult<void>",
        (*DISTANCE_MATRIX_PARAMS, param("Size", "stream_offset"), param("Size", "stream_size")),
        "auto-configured CUDA distance-matrix launcher implementation",
    ),
    SignatureExpectation(
        "operator-schema",
        CUDA_TENSOR_CORE_HEADER,
        "launchTensorCoreDistanceMatrix",
        "void",
        (
            param("const double*", "d_points_double"),
            param("float*", "d_distance_matrix"),
            param("int", "n_points"),
            param("int", "point_dim"),
            param("double", "max_radius"),
            CUDA_STREAM_PARAM,
        ),
        "Tensor Core CUDA launcher declaration",
    ),
    SignatureExpectation(
        "operator-schema",
        CUDA_TENSOR_CORE_IMPL,
        "launchTensorCoreDistanceMatrix",
        "void",
        (
            param("const double*", "d_points_double"),
            param("float*", "d_distance_matrix"),
            param("int", "n_points"),
            param("int", "point_dim"),
            param("double", "max_radius"),
            CUDA_STREAM_PARAM,
        ),
        "Tensor Core CUDA launcher implementation",
    ),
    SignatureExpectation(
        "operator-schema",
        CUDA_TENSOR_CORE_WRAPPER,
        "launchTensorCoreDistanceMatrix",
        "void",
        (
            param("const double*", "d_points"),
            param("float*", "d_distance_matrix"),
            param("int", "n_points"),
            param("int", "point_dim"),
            param("double", "max_radius"),
            CUDA_STREAM_PARAM,
        ),
        "Tensor Core C-linkage launcher wrapper",
    ),
    SignatureExpectation(
        "operator-schema",
        CUDA_BLACKWELL_HEADER,
        "launchBlackwellDistanceMatrix",
        "void",
        (
            param("const double*", "d_points"),
            param("float*", "d_distances"),
            param("int", "n_points"),
            param("int", "point_dim"),
            param("double", "max_radius"),
            CUDA_STREAM_PARAM,
        ),
        "Blackwell CUDA launcher declaration",
    ),
    SignatureExpectation(
        "operator-schema",
        CUDA_BLACKWELL_IMPL,
        "launchBlackwellDistanceMatrix",
        "void",
        (
            param("const double*", "d_points"),
            param("float*", "d_distances"),
            param("int", "n_points"),
            param("int", "point_dim"),
            param("double", "max_radius"),
            CUDA_STREAM_PARAM,
        ),
        "Blackwell CUDA launcher implementation",
    ),
)

BINDING_EXPECTATIONS: tuple[BindingExpectation, ...] = (
    BindingExpectation(
        "operator-schema",
        ALGORITHM_BINDINGS,
        "pairwise_distances",
        ("points", "n_points", "dim"),
        "algorithm distance pybind registration",
    ),
    BindingExpectation(
        "operator-schema",
        ALGORITHM_BINDINGS,
        "knn",
        ("points", "n_points", "dim", "k"),
        "algorithm KNN pybind registration",
    ),
    BindingExpectation(
        "operator-schema",
        CORE_BINDINGS,
        "compute_persistence",
        ("points", "options"),
        "core persistence pybind registration",
    ),
    BindingExpectation(
        "operator-schema",
        CORE_BINDINGS,
        "compute_persistence_up_to_dim_4",
        ("points", "options"),
        "PH4 persistence pybind registration",
    ),
    BindingExpectation(
        "operator-schema",
        CORE_BINDINGS,
        "compute_persistence_up_to_dim_5",
        ("points", "options"),
        "PH5 persistence pybind registration",
    ),
    BindingExpectation(
        "operator-schema",
        CORE_BINDINGS,
        "compute_persistence_up_to_dim_6",
        ("points", "options"),
        "PH6 persistence pybind registration",
    ),
    BindingExpectation(
        "operator-schema",
        CORE_BINDINGS,
        "compute_persistence_cohomology",
        ("points", "options"),
        "cohomology persistence pybind registration",
    ),
)


def _strip_comments(text: str) -> str:
    text = re.sub(
        r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), text, flags=re.DOTALL
    )
    return re.sub(r"//.*", "", text)


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    angle_depth = 0
    paren_depth = 0
    brace_depth = 0
    for index, char in enumerate(text):
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
        elif char == delimiter and angle_depth == 0 and paren_depth == 0 and brace_depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_default(text: str) -> str:
    return _split_top_level(text, "=")[0].strip()


def parse_params(text: str) -> tuple[Parameter, ...]:
    if not text.strip() or text.strip() == "void":
        return ()
    params: list[Parameter] = []
    for raw_param in _split_top_level(text):
        cleaned = _strip_default(raw_param).replace("__restrict__", "").strip()
        match = re.search(r"(?P<name>[A-Za-z_]\w*)\s*(?:\[[^\]]*\])?$", cleaned)
        if match is None:
            continue
        name = match.group("name")
        type_text = cleaned[: match.start()].strip()
        params.append(Parameter(_normalize_type(type_text), name))
    return tuple(params)


def _matching_paren(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def find_signatures(path: Path, symbol: str) -> tuple[Signature, ...]:
    if not path.exists():
        return ()
    text = _strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
    pattern = re.compile(
        rf"^[ \t]*(?P<return>(?:\[\[nodiscard\]\]\s+)?(?:inline\s+)?(?:static\s+)?"
        rf"[A-Za-z_][\w:<>,\s*&]*?)\s+{re.escape(symbol)}\s*\(",
        re.MULTILINE,
    )
    signatures: list[Signature] = []
    for match in pattern.finditer(text):
        open_index = text.find("(", match.end() - 1)
        close_index = _matching_paren(text, open_index)
        if close_index < 0:
            continue
        after = text[close_index + 1 : close_index + 32].lstrip()
        if after.startswith("const"):
            after = after[5:].lstrip()
        if not after.startswith(("{", ";")):
            continue
        params = parse_params(text[open_index + 1 : close_index])
        signatures.append(
            Signature(
                _normalize_type(match.group("return")),
                params,
                text.count("\n", 0, match.start()) + 1,
            )
        )
    return tuple(signatures)


def _format_signature(return_type: str, params: Iterable[Parameter]) -> str:
    rendered_params = ", ".join(f"{param.type} {param.name}".strip() for param in params)
    return f"{return_type}({rendered_params})"


def _cpp_call_end(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)


def pybind_defs(path: Path) -> dict[str, tuple[str, ...]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    definitions: dict[str, tuple[str, ...]] = {}
    for match in re.finditer(r"\bm\.def\(\s*\"([A-Za-z_]\w*)\"", text):
        block = text[match.start() : _cpp_call_end(text, match.start())]
        definitions[match.group(1)] = tuple(re.findall(r"py::arg\(\"([A-Za-z_]\w*)\"\)", block))
    return definitions


def check_signatures() -> list[Finding]:
    findings: list[Finding] = []
    for expectation in SIGNATURE_EXPECTATIONS:
        rel = expectation.path.relative_to(ROOT).as_posix()
        signatures = find_signatures(expectation.path, expectation.symbol)
        expected_return = _normalize_type(expectation.return_type)
        expected_params = expectation.params
        if any(
            signature.return_type == expected_return and signature.params == expected_params
            for signature in signatures
        ):
            continue
        observed = "; ".join(
            f"line {signature.line}: {_format_signature(signature.return_type, signature.params)}"
            for signature in signatures
        )
        if not observed:
            observed = "not found"
        findings.append(
            Finding(
                expectation.check,
                rel,
                f"{expectation.description} mismatch for {expectation.symbol}: "
                f"expected {_format_signature(expected_return, expected_params)}, observed {observed}",
            )
        )
    return findings


def check_bindings() -> list[Finding]:
    findings: list[Finding] = []
    cache: dict[Path, dict[str, tuple[str, ...]]] = {}
    for expectation in BINDING_EXPECTATIONS:
        rel = expectation.path.relative_to(ROOT).as_posix()
        definitions = cache.setdefault(expectation.path, pybind_defs(expectation.path))
        actual = definitions.get(expectation.symbol)
        if actual == expectation.args:
            continue
        findings.append(
            Finding(
                expectation.check,
                rel,
                f"{expectation.description} mismatch for {expectation.symbol}: "
                f"expected args {expectation.args}, observed {actual}",
            )
        )
    return findings


def check() -> list[Finding]:
    return [*check_signatures(), *check_bindings()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    findings = check()
    if args.json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2))
    else:
        for finding in findings:
            print(f"{finding.check}: {finding.path}: {finding.message}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
