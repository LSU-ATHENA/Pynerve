"""Import graph and public API quality checks."""

from __future__ import annotations

import ast

from .common import (
    PY_ROOT,
    PYTHON_API_PATH,
    ROOT,
    Finding,
    _attr_from_module_map,
    _detect_cycles,
    _is_nerve_package_import,
    _lazy_submodules,
    _module_defined_names,
    _optional_import_lines,
    _public_python_all,
    _python_module_inventory,
    _resolve_import_from,
    _top_level_import_lines,
)


def check_import_graph() -> list[Finding]:
    findings: list[Finding] = []
    modules, available_modules = _python_module_inventory()
    graph: dict[str, set[str]] = {}
    for module, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        optional_lines = _optional_import_lines(tree)
        top_level_imports = _top_level_import_lines(tree)
        deps: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _is_nerve_package_import(alias.name):
                        continue
                    if alias.name not in available_modules and node.lineno not in optional_lines:
                        findings.append(
                            Finding(
                                "import-graph",
                                path.relative_to(ROOT).as_posix(),
                                f"broken import: {alias.name}",
                            )
                        )
                    if (
                        alias.name in available_modules
                        and alias.name != module
                        and node.lineno in top_level_imports
                    ):
                        deps.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imported = _resolve_import_from(module, path, node)
                if imported is None or not _is_nerve_package_import(imported):
                    continue
                if imported not in available_modules and node.lineno not in optional_lines:
                    findings.append(
                        Finding(
                            "import-graph",
                            path.relative_to(ROOT).as_posix(),
                            f"broken import: {imported}",
                        )
                    )
                    continue
                if (
                    node.module is not None
                    and imported in available_modules
                    and imported != module
                    and node.lineno in top_level_imports
                ):
                    deps.add(imported)
                for alias in node.names:
                    candidate = f"{imported}.{alias.name}"
                    if (
                        candidate in available_modules
                        and candidate != module
                        and node.lineno in top_level_imports
                    ):
                        deps.add(candidate)
        graph[module] = deps
        exports = _public_python_all(path)
        _attr_redirected = _attr_from_module_map(tree)
        for name in sorted(_lazy_submodules(tree)):
            if name in _attr_redirected:
                continue
            candidate = f"{module}.{name}"
            if candidate not in available_modules:
                findings.append(
                    Finding(
                        "import-graph",
                        path.relative_to(ROOT).as_posix(),
                        f"lazy submodule target does not exist: {candidate}",
                    )
                )
            if exports and name not in exports:
                findings.append(
                    Finding(
                        "import-graph",
                        path.relative_to(ROOT).as_posix(),
                        f"lazy submodule is missing from __all__: {name}",
                    )
                )
    for cycle in _detect_cycles(graph):
        findings.append(Finding("import-graph", cycle[0], "import cycle: " + " -> ".join(cycle)))
    return findings


def check_public_api() -> list[Finding]:
    findings: list[Finding] = []
    # .pyi type files are intentionally omitted; inline type annotations are sufficient.
    for path in (PYTHON_API_PATH, PY_ROOT / "torch" / "__init__.py"):
        exports = _public_python_all(path)
        if not exports:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        defined = _module_defined_names(path) | _lazy_submodules(tree) | _attr_from_module_map(tree)
        for name in sorted(exports - defined):
            findings.append(
                Finding(
                    "public-api",
                    path.relative_to(ROOT).as_posix(),
                    f"__all__ exports undefined name: {name}",
                )
            )
        imported_names = {
            alias.asname or alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        local_names = {n for n in defined if n not in imported_names}
        for name in sorted(local_names - exports):
            if name.startswith("_"):
                continue
            findings.append(
                Finding(
                    "public-api",
                    path.relative_to(ROOT).as_posix(),
                    f"public name {name} defined but missing from __all__",
                )
            )
    return findings
