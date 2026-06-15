#!/usr/bin/env python3
"""Run available static-analysis checks without hiding missing required CI tools."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_REQUIRED_RELEASE = "release 12"
CPP_CLANG_TIDY_ERRORS = (
    "bugprone-branch-clone,"
    "bugprone-implicit-widening-of-multiplication-result,"
    "bugprone-narrowing-conversions,"
    "performance-*,"
    "-performance-enum-size,"
    "-performance-unnecessary-copy-initialization,"
    "-performance-unnecessary-value-param"
)
CPP_CLANG_TIDY_CHECKS = CPP_CLANG_TIDY_ERRORS
CPP_CPPCHECK_CHECKS = "warning,performance,portability"
CPP_CLANG_TIDY_CRITICAL_PATHS = (
    "src/algorithms/distance.cpp",
    "src/algebra/boundary/boundary_matrix_reduction_ops.cpp",
    "src/core/memory/memory_pool_ops.cpp",
    "src/runtime/hardware_probe.cpp",
    "src/persistence/reduction/reduction_ops.cpp",
    "src/persistence/vr/vr_fast_ops.cpp",
    "src/persistence/accelerated/gpu_reduction_engine.cpp",
    "src/metrics/matrix/matrix_distance_ops.cpp",
    "src/serialization/serialization_manager_base.cpp",
    "src/torch/autograd_torch.cpp",
    "src/torch/diagram_operations_torch.cpp",
    "src/torch/torch_library.cpp",
    "src/torch/ml_vectorization.cpp",
    "src/torch/persistence_diagram.cpp",
    "src/torch/vietoris_rips_torch.cpp",
    "python/bindings/nerve_api_bindings.cpp",
    "python/bindings/nerve_torch_bindings.cpp",
)
CPP_CLANG_TIDY_ALL_PREFIXES = (
    "src/",
    "python/bindings/",
)
CPP_SOURCE_SUFFIXES = (".cc", ".cpp", ".cxx")


def _compile_database_path(build_dir: str) -> Path:
    return (ROOT / build_dir / "compile_commands.json").resolve()


def _display_path(path: Path) -> str:
    resolved_root = ROOT.resolve()
    resolved_path = path.resolve()
    if resolved_path.is_relative_to(resolved_root):
        return resolved_path.relative_to(resolved_root).as_posix()
    return str(resolved_path)


def _run(command: list[str], required: bool) -> int:
    executable = command[0]
    if len(command) >= 3 and executable == sys.executable and command[1] == "-m":
        module = command[2]
        if importlib.util.find_spec(module) is None:
            message = f"missing static-analysis module: {module}"
            if required:
                print(message, file=sys.stderr)
                return 1
            print(f"skipping optional {message}")
            return 0
    if shutil.which(executable) is None:
        message = f"missing static-analysis tool: {executable}"
        if required:
            print(message, file=sys.stderr)
            return 1
        print(f"skipping optional {message}")
        return 0
    print("+", " ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT)


def _check_required_cuda_version(required: bool) -> int:
    if not required:
        return 0
    if shutil.which("nvcc") is None:
        print("missing static-analysis tool: nvcc", file=sys.stderr)
        return 1
    version = subprocess.run(
        ["nvcc", "--version"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if version.returncode != 0:
        print("nvcc --version failed during required CUDA static analysis", file=sys.stderr)
        return version.returncode
    if CUDA_REQUIRED_RELEASE not in version.stdout + version.stderr:
        print(
            f"CUDA static analysis requires at least CUDA 12 (found {version.stdout.strip()})",
            file=sys.stderr,
        )
        return 1
    return 0


def _compile_database_sources(build_dir: str) -> list[Path]:
    compile_db = _compile_database_path(build_dir)
    raw_entries = json.loads(compile_db.read_text(encoding="utf-8"))
    if not isinstance(raw_entries, list):
        raise ValueError(f"{_display_path(compile_db)} must contain a JSON list")

    sources: list[Path] = []
    seen: set[Path] = set()
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        file_value = entry.get("file")
        if not isinstance(file_value, str):
            continue
        source = Path(file_value)
        if not source.is_absolute():
            source = ROOT / source
        source = source.resolve()
        if source.suffix not in CPP_SOURCE_SUFFIXES or not source.exists():
            continue
        if not source.is_relative_to(ROOT) or source in seen:
            continue
        seen.add(source)
        sources.append(source)
    return sources


def _select_clang_tidy_sources(
    compile_sources: list[Path],
    scope: str,
    critical_paths: tuple[str, ...] = CPP_CLANG_TIDY_CRITICAL_PATHS,
) -> list[Path]:
    root_sources = {
        source.relative_to(ROOT).as_posix(): source
        for source in compile_sources
        if source.is_relative_to(ROOT)
    }
    if scope == "all":
        return [
            source
            for relative, source in sorted(root_sources.items())
            if relative.startswith(CPP_CLANG_TIDY_ALL_PREFIXES)
        ]
    return [root_sources[relative] for relative in critical_paths if relative in root_sources]


def _compiler_resource_include(header: str, compiler: str = "c++") -> Path | None:
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        return None
    probe = subprocess.run(
        [compiler_path, f"-print-file-name=include/{header}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    header_path = Path(probe.stdout.strip())
    if not header_path.is_file():
        return None
    return header_path.parent


def _clang_tidy_extra_args() -> list[str]:
    extra_args: list[str] = []
    omp_include = _compiler_resource_include("omp.h")
    if omp_include is not None:
        extra_args.append(f"--extra-arg=-idirafter{omp_include}")
    return extra_args


def _clang_tidy_command(build_dir: str, sources: list[Path]) -> list[str]:
    return [
        "clang-tidy",
        "-p",
        build_dir,
        f"--checks={CPP_CLANG_TIDY_CHECKS}",
        f"--warnings-as-errors={CPP_CLANG_TIDY_ERRORS}",
        *_clang_tidy_extra_args(),
        *[source.relative_to(ROOT).as_posix() for source in sources],
    ]


def _cppcheck_command(sources: list[Path]) -> list[str]:
    return [
        "cppcheck",
        f"--enable={CPP_CPPCHECK_CHECKS}",
        "--error-exitcode=1",
        *[source.relative_to(ROOT).as_posix() for source in sources],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--required", action="store_true")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--language", choices=("all", "python", "cpp", "cuda"), default="all")
    parser.add_argument("--clang-tidy-scope", choices=("critical", "all"), default="critical")
    parser.add_argument("--cuda-launch-scope", choices=("configured", "all"), default="configured")
    args = parser.parse_args()

    commands: list[list[str]] = []
    if args.language in {"all", "python"}:
        commands.extend(
            [
                [sys.executable, "-m", "ruff", "check", "python", "tools", "tests"],
            ]
        )
        if not args.required:
            commands.extend(
                [
                    [sys.executable, "-m", "mypy", "python/pynerve", "--python-version", "3.10"],
                    [sys.executable, "-m", "mypy", "tools", "--python-version", "3.10"],
                ]
            )
    if args.language in {"all", "cpp"}:
        compile_db = _compile_database_path(args.build_dir)
        if compile_db.exists() or args.required:
            try:
                compile_sources = _compile_database_sources(args.build_dir)
                native_sources = _select_clang_tidy_sources(
                    compile_sources,
                    args.clang_tidy_scope,
                )
            except (OSError, json.JSONDecodeError, ValueError) as error:
                print(f"invalid native static-analysis compile database: {error}", file=sys.stderr)
                return 1
            if native_sources:
                commands.extend(
                    [
                        _cppcheck_command(native_sources),
                        _clang_tidy_command(args.build_dir, native_sources),
                    ]
                )
            elif args.required:
                print(
                    f"no native C++ sources selected from {_display_path(compile_db)}",
                    file=sys.stderr,
                )
                return 1
            else:
                print(
                    "skipping optional native static analysis: "
                    f"no selected C++ sources in {_display_path(compile_db)}"
                )
        else:
            print(f"skipping optional native static analysis: missing {_display_path(compile_db)}")
    if args.language in {"all", "cuda"}:
        cuda_version_result = _check_required_cuda_version(args.required)
        if cuda_version_result != 0:
            return cuda_version_result
        commands.extend(
            [
                [sys.executable, "tools/cuda_launch_audit.py", "--scope", args.cuda_launch_scope],
                ["nvcc", "--version"],
                ["compute-sanitizer", "--version"],
            ]
        )

    for command in commands:
        result = _run(command, args.required)
        if result != 0:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
