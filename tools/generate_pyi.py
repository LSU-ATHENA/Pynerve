#!/usr/bin/env python3
"""Generate conservative .pyi type files from public Python signatures."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = ROOT / "python" / "nerve"


def _annotation(node: ast.expr | None) -> str:
    """Extract the annotation as a string, or return 'object'."""
    if node is None:
        return "object"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_annotation(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        value = _annotation(node.value)
        slice_ = _annotation(node.slice) if node.slice else ""
        return f"{value}[{slice_}]"
    if isinstance(node, ast.Tuple):
        elts = ", ".join(_annotation(e) for e in node.elts)
        return f"tuple[{elts}]"
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return f"'{node.value}'"
        return str(node.value)
    if isinstance(node, ast.List):
        elts = ", ".join(_annotation(e) for e in node.elts)
        return f"list[{elts}]"
    if isinstance(node, ast.BinOp):
        left = _annotation(node.left)
        right = _annotation(node.right)
        if isinstance(node.op, ast.BitOr):
            return f"{left} | {right}"
        return f"{left} | {right}"
    if isinstance(node, ast.Index):
        return _annotation(node.value)
    return "object"


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args: list[str] = []
    positional = [*node.args.posonlyargs, *node.args.args]
    default_offset = len(positional) - len(node.args.defaults)
    for index, arg in enumerate(positional):
        rendered = arg.arg
        ann = _annotation(arg.annotation)
        if ann != "object":
            rendered += f": {ann}"
        if index >= default_offset:
            rendered += "=..."
        args.append(rendered)
    if node.args.posonlyargs:
        args.insert(len(node.args.posonlyargs), "/")
    if node.args.vararg:
        v = f"*{node.args.vararg.arg}"
        ann = _annotation(node.args.vararg.annotation)
        if ann != "object":
            v += f": {ann}"
        args.append(v)
    elif node.args.kwonlyargs:
        args.append("*")
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=False):
        rendered = arg.arg
        ann = _annotation(arg.annotation)
        if ann != "object":
            rendered += f": {ann}"
        if default is not None:
            rendered += "=..."
        args.append(rendered)
    if node.args.kwarg:
        v = f"**{node.args.kwarg.arg}"
        ann = _annotation(node.args.kwarg.annotation)
        if ann != "object":
            v += f": {ann}"
        args.append(v)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return_ann = _annotation(node.returns)
    return f"{prefix} {node.name}({', '.join(args)}) -> {return_ann}: ..."


def _collect_public_names(tree: ast.Module) -> tuple[set[str], set[str], dict[str, str]]:
    """Collect locally-defined names, all_exported names, and re-export targets.

    Returns:
        (local_names, all_exported, reexports) where reexports maps
        name -> original module path (e.g. "PersistenceResult" -> "._compute_core").
    """
    local: set[str] = set()
    reexports: dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local.add(node.name)
        elif isinstance(node, ast.Assign) and hasattr(node, "targets"):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    local.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            local.add(node.target.id)

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name.startswith("_"):
                    continue
                name = alias.asname or alias.name
                if name not in local and name.startswith("_"):
                    continue
                reexports[name] = module

    # Parse __all__ to find export list
    all_exported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                all_exported.add(elt.value)

    return local, all_exported, reexports


def _format_reexport(name: str, module: str) -> str:
    """Format a re-export line for a type file."""
    if module and not module.startswith("."):
        module = "." + module
    from_path = f"from {module}" if module else "from ."
    return f"{from_path} import {name}"


def generate(check: bool) -> list[Path]:
    changed: list[Path] = []
    for source in PY_ROOT.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        local_names, all_exported, reexports = _collect_public_names(tree)

        lines: list[str] = []
        generated_exports: set[str] = set()

        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith("_"):
                lines.append(_signature(node))
                generated_exports.add(node.name)
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                lines.append(f"class {node.name}: ...")
                generated_exports.add(node.name)

        # Add re-exports for names in __all__ that aren't locally defined
        exported_not_local = all_exported - generated_exports
        if exported_not_local and reexports:
            for name in sorted(exported_not_local):
                if name in reexports:
                    lines.append(_format_reexport(name, reexports[name]))
                else:
                    lines.append(f"{name}: object  # re-export")

        if not lines:
            target = source.with_suffix(".pyi")
            if check:
                if target.exists():
                    changed.append(target)
            elif target.exists():
                target.unlink()
            continue

        target = source.with_suffix(".pyi")
        content = "\n".join(lines) + "\n"
        if check:
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                changed.append(target)
        else:
            target.write_text(content, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = generate(args.check)
    for path in changed:
        print(path.relative_to(ROOT).as_posix())
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
