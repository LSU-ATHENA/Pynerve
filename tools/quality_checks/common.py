"""Shared utilities for repository quality checks."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
PY_ROOT = ROOT / "python" / "nerve"
INCLUDE_ROOT = ROOT / "src" / "include" / "nerve"
TOOLS_ROOT = ROOT / "tools"
CMAKE_ROOT_PATH = ROOT / "CMakeLists.txt"
SRC_CMAKE_PATH = ROOT / "src" / "CMakeLists.txt"
SRC_TARGETS_PATH = ROOT / "src" / "cmake" / "targets.cmake"
SRC_SIMD_PATH = ROOT / "src" / "cmake" / "simd.cmake"
CMAKE_CONFIG_PATH = ROOT / "cmake" / "NerveConfig.cmake.in"
PYPROJECT_PATH = ROOT / "python" / "pyproject.toml"
PYTHON_CMAKE_PATH = ROOT / "python" / "CMakeLists.txt"
CI_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"

CMAKE_TESTS_PATH = ROOT / "tests" / "CMakeLists.txt"
STATIC_ANALYSIS_PATH = TOOLS_ROOT / "static_analysis.py"
BACKEND_CHECKS_PATH = TOOLS_ROOT / "backend_checks.py"
RUN_TESTS_PATH = TOOLS_ROOT / "run_tests.py"
PERFORMANCE_GUARDS_PATH = TOOLS_ROOT / "performance_guards.py"
OPERATOR_SCHEMA_PATH = TOOLS_ROOT / "operator_schema.py"
PYTHON_API_PATH = PY_ROOT / "__init__.py"
PYBIND_API_PATH = ROOT / "python" / "bindings" / "nerve_api_bindings.cpp"
ALGORITHMS_BINDINGS_PATH = ROOT / "python" / "bindings" / "nerve_algorithms_bindings.cpp"
TORCH_BINDINGS_PATH = ROOT / "python" / "bindings" / "nerve_torch_bindings.cpp"
TORCH_LIBRARY_PATH = ROOT / "src" / "torch" / "torch_library.cpp"
TORCH_ML_KERNELS_PATH = ROOT / "src" / "torch" / "ml_kernels.cpp"
TORCH_ML_STATISTICS_PATH = ROOT / "src" / "torch" / "ml_statistics.cpp"
TORCH_FILTRATION_FACTORY_PATH = ROOT / "src" / "torch" / "detail" / "filtration_factory_ops.inl"
TORCH_SIMPLEX_TREE_PATH = ROOT / "src" / "torch" / "simplex_tree.cpp"
TORCH_PERSISTENCE_DIAGRAM_PATH = ROOT / "src" / "torch" / "persistence_diagram.cpp"
TORCH_PERSISTENCE_DIAGRAM_OPS_PATH = (
    ROOT / "src" / "torch" / "detail" / "persistence_diagram_ops.inl"
)
TORCH_MAPPER_IMPL_PATH = ROOT / "src" / "torch" / "detail" / "mapper_impl.inl"
BINDING_SMOKE_PATH = TOOLS_ROOT / "binding_smoke.py"
TORCH_BINDING_SMOKE_PATH = TOOLS_ROOT / "torch_binding_smoke.py"
INSTALL_SMOKE_PATH = TOOLS_ROOT / "install_smoke.py"
CPP_INSTALL_SMOKE_PATH = TOOLS_ROOT / "cpp_install_smoke.py"
TORCH_BACKEND_PATH = PY_ROOT / "torch" / "_backend.py"
TORCH_FUNCTIONAL_BINDINGS_PATH = (
    ROOT / "python" / "bindings" / "detail" / "nerve_torch_bindings_functional_api.inl"
)
TORCH_CLASSES_BINDINGS_PATH = (
    ROOT / "python" / "bindings" / "detail" / "nerve_torch_bindings_classes.inl"
)


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    message: str


def _iter_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    ignored = {".venv", "build", "__pycache__", ".pytest_cache", ".ruff_cache"}
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and not any(part in ignored for part in path.parts)
    ]


def _load_tool_module(name: str) -> ModuleType:
    path = TOOLS_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"nerve_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _public_python_functions() -> dict[str, list[str]]:
    functions: dict[str, list[str]] = {}
    for path in _iter_files(PY_ROOT, (".py",)):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename = rel)
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not node.name.startswith("_")
        ]
        if names:
            functions[rel] = sorted(names)
    return functions


def _cpp_declarations() -> dict[str, list[str]]:
    declarations: dict[str, list[str]] = {}
    pattern = re.compile(
        r"^\s*(?:template\s*<[^;]+>\s*)?"
        r"(?:inline\s+|constexpr\s+|static\s+)*"
        r"[\w:<>*&,\s]+\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?[;{]",
        re.MULTILINE,
    )
    for path in _iter_files(INCLUDE_ROOT, (".hpp", ".h", ".cuh")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        names = sorted({name for name in pattern.findall(text) if not name.startswith("_")})
        if names:
            declarations[path.relative_to(ROOT).as_posix()] = names
    return declarations


def _public_python_all(path: Path = PYTHON_API_PATH) -> set[str]:
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename = str(path))
    exports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if not any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                continue
            if isinstance(node.value, (ast.List, ast.Tuple)):
                for item in node.value.elts:
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        exports.add(item.value)
    return exports


def _module_defined_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename = str(path))
    names: set[str] = set()

    def visit_statements(statements: list[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".", 1)[0])
            elif isinstance(node, ast.Try):
                visit_statements(node.body)
                for handler in node.handlers:
                    visit_statements(handler.body)
                visit_statements(node.orelse)
                visit_statements(node.finalbody)
            elif isinstance(node, ast.If):
                visit_statements(node.body)
                visit_statements(node.orelse)

    visit_statements(tree.body)
    return names


def _core_attribute_refs(path: Path = PYTHON_API_PATH) -> set[str]:
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename = str(path))
    refs: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "_core"
        ):
            refs.add(node.attr)
    return refs


def _pybind_module_defs(path: Path = PYBIND_API_PATH) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = list(re.finditer(r"\bm\.def\(\s*\"([A-Za-z_]\w*)\"", text))
    definitions: dict[str, list[str]] = {}
    for match in matches:
        block = text[match.start() : _cpp_call_end(text, match.start())]
        definitions[match.group(1)] = re.findall(r"py::arg\(\"([A-Za-z_]\w*)\"\)", block)
    return definitions


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


def _pybind_registered_types(path: Path = PYBIND_API_PATH) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8", errors="ignore")
    return set(re.findall(r"py::(?:class_|enum_)<[^>]+>\(\s*m\s*,\s*\"([A-Za-z_]\w*)\"", text))


def _split_cpp_params(params: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(params):
        if char in "(<[{":
            depth += 1
        elif char in ")>]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            piece = params[start:index].strip()
            if piece:
                pieces.append(piece)
            start = index + 1
    tail = params[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _torch_schema_args(schema_args: str) -> list[tuple[str, str]]:
    args: list[tuple[str, str]] = []
    for param in _split_cpp_params(schema_args):
        tokens = param.split()
        if len(tokens) < 2:
            args.append((param, ""))
        else:
            args.append((" ".join(tokens[:-1]), tokens[-1]))
    return args


def _torch_operator_defs(text: str) -> list[tuple[str, list[tuple[str, str]], str, str]]:
    defs: list[tuple[str, list[tuple[str, str]], str, str]] = []
    pattern = re.compile(
        r"\bm\.def\(\s*\"([^\"]+)\"\s*,\s*&nerve::torch::([A-Za-z_]\w*)",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        schema, target = match.groups()
        schema_match = re.fullmatch(r"\s*([A-Za-z_]\w*)\((.*)\)\s*->\s*([A-Za-z_]\w*)\s*", schema)
        if schema_match is None:
            defs.append((schema, [], "", target))
            continue
        name, raw_args, return_type = schema_match.groups()
        defs.append((name, _torch_schema_args(raw_args), return_type, target))
    return defs


def _cpp_function_signatures(text: str) -> dict[str, list[tuple[str, list[str]]]]:
    pattern = re.compile(
        r"^\s*(?:\[\[nodiscard\]\]\s*)?"
        r"([A-Za-z_:][\w:<>,\s*&]+?)\s+([A-Za-z_]\w*)\s*\(([^;{}]*)\)\s*(?:;|\{)",
        re.MULTILINE,
    )
    signatures: dict[str, list[tuple[str, list[str]]]] = {}
    for match in pattern.finditer(text):
        return_type = re.sub(r"\s+", " ", match.group(1).strip())
        name = match.group(2)
        params = _split_cpp_params(match.group(3))
        signatures.setdefault(name, []).append((return_type, params))
    return signatures


def _torch_return_type_matches(schema_type: str, cpp_return_type: str) -> bool:
    expected = {
        "Tensor": {"at::Tensor"},
        "float": {"double", "float"},
        "int": {"int64_t", "long"},
        "bool": {"bool"},
    }.get(schema_type)
    if expected is None:
        return True
    return cpp_return_type in expected


def _check_torch_operator_schema_text(text: str, path: str) -> list[Finding]:
    findings: list[Finding] = []
    operator_defs = _torch_operator_defs(text)
    signatures = _cpp_function_signatures(text)
    seen_ops: set[str] = set()
    required_ops = {
        "vr_build",
        "vr_fast",
        "vr_persistence",
        "diagram_wasserstein",
        "diagram_bottleneck",
        "diagram_landscape",
        "diagram_betti",
        "filtration_distance_matrix",
        "filtration_alpha",
        "filtration_witness",
        "ph_compute",
        "ph_grad",
        "ph_vr",
        "ph_witness",
        "ph_alpha",
        "ph_persistence",
        "ph_wasserstein",
        "ph_bottleneck",
        "ph_image",
        "differentiable_persistence",
    }
    for op_name, schema_args, schema_return, target in operator_defs:
        if not schema_return:
            findings.append(
                Finding("torch-operator-schema", path, f"cannot parse schema: {op_name}")
            )
            continue
        if op_name in seen_ops:
            findings.append(
                Finding("torch-operator-schema", path, f"duplicate operator schema: {op_name}")
            )
        seen_ops.add(op_name)
        target_signatures = signatures.get(target, [])
        if not target_signatures:
            findings.append(
                Finding(
                    "torch-operator-schema",
                    path,
                    f"operator {op_name} target {target} is undeclared",
                )
            )
            continue
        schema_arity = len(schema_args)
        arity_matches = [
            (return_type, params)
            for return_type, params in target_signatures
            if len(params) == schema_arity
        ]
        if not arity_matches:
            available = sorted({len(params) for _, params in target_signatures})
            findings.append(
                Finding(
                    "torch-operator-schema",
                    path,
                    f"operator {op_name} arity mismatch for {target}: schema has {schema_arity}, C++ has {available}",
                )
            )
            continue
        if not any(
            _torch_return_type_matches(schema_return, return_type)
            for return_type, _ in arity_matches
        ):
            returns = sorted({return_type for return_type, _ in arity_matches})
            findings.append(
                Finding(
                    "torch-operator-schema",
                    path,
                    f"operator {op_name} return mismatch for {target}: schema has {schema_return}, C++ has {returns}",
                )
            )
    missing = sorted(required_ops - seen_ops)
    for op_name in missing:
        findings.append(
            Finding("torch-operator-schema", path, f"missing registered operator: {op_name}")
        )
    return findings


def _pybind_readwrite_fields(class_name: str, path: Path = PYBIND_API_PATH) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8", errors="ignore")
    start_match = re.search(
        rf"py::class_<[^>]+>\(\s*m\s*,\s*\"{re.escape(class_name)}\"\)",
        text,
    )
    if start_match is None:
        return set()
    next_class = re.search(r"\n\s*py::(?:class_|enum_)<", text[start_match.end() :])
    end = start_match.end() + next_class.start() if next_class else len(text)
    block = text[start_match.start() : end]
    return set(re.findall(r"\.def_readwrite\(\s*\"([A-Za-z_]\w*)\"", block))


def _python_module_name(path: Path) -> str:
    parts = path.relative_to(PY_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return "nerve" if not parts else "nerve." + ".".join(parts)


def _python_module_inventory() -> tuple[dict[str, Path], set[str]]:
    modules: dict[str, Path] = {}
    packages = {"nerve"}
    for path in _iter_files(PY_ROOT, (".py",)):
        module = _python_module_name(path)
        modules[module] = path
        parts = module.split(".")
        for depth in range(1, len(parts)):
            packages.add(".".join(parts[:depth]))
    return modules, packages | set(modules)


def _resolve_import_from(module: str, path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    package_parts = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    trim = node.level - 1
    if trim:
        if trim > len(package_parts):
            return None
        package_parts = package_parts[:-trim]
    if node.module:
        package_parts = [*package_parts, *node.module.split(".")]
    return ".".join(package_parts)


def _optional_import_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(
            handler.type is None
            or (
                isinstance(handler.type, ast.Name)
                and handler.type.id in {"ImportError", "ModuleNotFoundError", "OSError"}
            )
            or (
                isinstance(handler.type, ast.Tuple)
                and any(
                    isinstance(item, ast.Name)
                    and item.id in {"ImportError", "ModuleNotFoundError", "OSError"}
                    for item in handler.type.elts
                )
            )
            for handler in node.handlers
        ):
            continue
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                lines.add(child.lineno)
    return lines


def _top_level_import_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()

    def visit_statements(statements: list[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                lines.add(statement.lineno)
            elif isinstance(statement, ast.Try):
                visit_statements(statement.body)
                for handler in statement.handlers:
                    visit_statements(handler.body)
                visit_statements(statement.orelse)
                visit_statements(statement.finalbody)
            elif isinstance(statement, ast.If):
                visit_statements(statement.body)
                visit_statements(statement.orelse)

    visit_statements(tree.body)
    return lines


def _is_nerve_package_import(name: str) -> bool:
    return name in ("nerve", "pynerve") or name.startswith(("nerve.", "pynerve."))


def _string_constants(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return {
            item.value
            for item in node.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
    return set()


def _attr_from_module_map(tree: ast.AST) -> set[str]:
    """Extract names from ``_ATTR_FROM_MODULE`` dicts.

    These names are lazy attributes resolved via a different module,
    so they should not be checked as standalone submodule targets.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target_name = None
            value = None
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_ATTR_FROM_MODULE":
                        target_name = target.id
                        value = node.value
                        break
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "_ATTR_FROM_MODULE":
                    target_name = node.target.id
                    value = node.value
            if target_name == "_ATTR_FROM_MODULE" and isinstance(value, ast.Dict):
                for key in value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        names.add(key.value)
    return names


def _lazy_submodules(tree: ast.AST) -> set[str]:
    names: set[str] = set()

    # First pass: collect assignments to named sets/frozensets
    named_sets: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = None
            target_ids: list[str] = []
            if isinstance(node, ast.Assign):
                value = node.value
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        target_ids.append(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_ids.append(node.target.id)
                value = node.value
            for tid in target_ids:
                if isinstance(value, (ast.Set, ast.Tuple, ast.List)):
                    named_sets[tid] = value
                elif (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id in ("frozenset", "set")
                ) and value.args:
                    named_sets[tid] = value.args[0]

    # Second pass: collect names from _LAZY_SUBMODULES and name-in-set patterns
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = None
            target_ids: list[str] = []
            if isinstance(node, ast.Assign):
                value = node.value
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        target_ids.append(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_ids.append(node.target.id)
                value = node.value
            for tid in target_ids:
                if tid in ("_LAZY_SUBMODULES",):
                    names.update(_string_constants(value))
                elif tid in named_sets:
                    pass  # Already collected
        elif isinstance(node, ast.Compare):
            if not (isinstance(node.left, ast.Name) and node.left.id == "name"):
                continue
            if not any(isinstance(operator, ast.In) for operator in node.ops):
                continue
            for comparator in node.comparators:
                if isinstance(comparator, ast.Name) and comparator.id in named_sets:
                    names.update(_string_constants(named_sets[comparator.id]))
                else:
                    names.update(_string_constants(comparator))
    return names


def _detect_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    visiting: list[str] = []
    visited: set[str] = set()
    emitted: set[tuple[str, ...]] = set()

    def visit(module: str) -> None:
        if module in visiting:
            cycle = visiting[visiting.index(module) :]
            rotated = min(
                (tuple(cycle[index:] + cycle[:index]) for index in range(len(cycle))),
                default = tuple(cycle),
            )
            if rotated not in emitted:
                emitted.add(rotated)
                cycles.append([*rotated, rotated[0]])
            return
        if module in visited:
            return
        visiting.append(module)
        for dep in sorted(graph.get(module, ())):
            if dep in graph:
                visit(dep)
        visiting.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)
    return cycles
