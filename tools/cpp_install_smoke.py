#!/usr/bin/env python3
"""Smoke-test an installed Nerve C++ CMake package."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def _write_consumer_project(source_dir: Path) -> None:
    (source_dir / "CMakeLists.txt").write_text(
        """
cmake_minimum_required(VERSION 3.20)
project(NerveCppInstallSmoke LANGUAGES C CXX)

find_package(Nerve REQUIRED)

add_executable(nerve_cpp_install_smoke main.cpp)
target_compile_features(nerve_cpp_install_smoke PRIVATE cxx_std_20)
target_link_libraries(nerve_cpp_install_smoke PRIVATE Nerve::nerve_core)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(nerve_cpp_install_smoke PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

add_executable(nerve_c_install_smoke c_api.c)
target_link_libraries(nerve_c_install_smoke PRIVATE Nerve::nerve_core)
set_target_properties(nerve_c_install_smoke PROPERTIES LINKER_LANGUAGE CXX)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(nerve_c_install_smoke PRIVATE -Wall -Wextra -Werror -pedantic)
endif()
""".lstrip(),
        encoding="utf-8",
    )
    (source_dir / "main.cpp").write_text(
        """
#include <array>
#include <cmath>
#include <span>

#include "nerve/algorithms/distance.hpp"
#ifdef __has_include
#if __has_include("nerve/gpu/cuda_tile_api.hpp")
#include "nerve/gpu/cuda_tile_api.hpp"
#endif
#if __has_include("nerve/gpu/gpu_capability_core.hpp")
#include "nerve/gpu/gpu_capability_core.hpp"
#endif
#endif

int main() {
    const std::array<double, 4> points{0.0, 0.0, 3.0, 4.0};

    nerve::algorithms::EuclideanMetric<double> metric;
    const double metric_distance =
        metric.compute(std::span<const double>(points.data(), 2),
                       std::span<const double>(points.data() + 2, 2));
    if (std::abs(metric_distance - 5.0) > 1e-12) {
        return 1;
    }

    nerve::algorithms::DistanceMatrixComputer<double> computer;
    const auto distances = computer.compute(std::span<const double>(points.data(), points.size()), 2, 2);
    if (distances.size() != 4 || std::abs(distances[1] - 5.0) > 1e-12 ||
        std::abs(distances[2] - 5.0) > 1e-12) {
        return 2;
    }

    std::array<double, 4> c_distances{};
    nerve::algorithms::nerve_pairwise_distances_f64(points.data(), 2, 2, c_distances.data());
    if (std::abs(c_distances[1] - 5.0) > 1e-12 || std::abs(c_distances[2] - 5.0) > 1e-12) {
        return 3;
    }

#ifdef __has_include
#if __has_include("nerve/gpu/cuda_tile_api.hpp") && __has_include("nerve/gpu/gpu_capability_core.hpp")
    const bool tile_available = nerve::gpu::tile::tileApiAvailable();
    const auto capabilities = nerve::gpu::advanced::AdvancedCapabilities::detect();
    if (tile_available && !capabilities.cuda_available) {
        return 4;
    }
#endif
#endif

    return 0;
}
""".lstrip(),
        encoding="utf-8",
    )
    (source_dir / "c_api.c").write_text(
        """
#include <stddef.h>

#include "nerve/algorithms/distance_c.h"

int main(void) {
    const double points[4] = {0.0, 0.0, 3.0, 4.0};
    double distances[4] = {0.0, 0.0, 0.0, 0.0};
    double knn_distances[2] = {0.0, 0.0};
    size_t knn_indices[2] = {0, 0};

    if (nerve_pairwise_distances_f64_status(points, 2, 2, distances) != NERVE_STATUS_SUCCESS) {
        return 1;
    }
    if (distances[1] < 4.999 || distances[1] > 5.001 || distances[2] < 4.999 ||
        distances[2] > 5.001) {
        return 2;
    }
    if (nerve_knn_f64_status(points, 2, 2, 1, knn_distances, knn_indices) !=
        NERVE_STATUS_SUCCESS) {
        return 3;
    }
    if (nerve_pairwise_distances_f64_status(NULL, 1, 2, distances) !=
        NERVE_STATUS_INVALID_ARGUMENT) {
        return 4;
    }

    return 0;
}
""".lstrip(),
        encoding="utf-8",
    )


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _compile_installed_header(prefix: Path, header: Path, compiler: str) -> None:
    relative = header.relative_to(prefix / "include")
    with tempfile.TemporaryDirectory(prefix="nerve-cpp-header-smoke-") as tmp:
        source = Path(tmp) / "header_smoke.cpp"
        source.write_text(
            f'#include "{relative.as_posix()}"\nint main() {{ return 0; }}\n',
            encoding="utf-8",
        )
        _run(
            [
                compiler,
                "-std=c++20",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                "-fsyntax-only",
                f"-I{prefix / 'include'}",
                str(source),
            ],
            Path(tmp),
        )


def _sweep_installed_headers(prefix: Path) -> None:
    include_root = prefix / "include" / "nerve"
    compiler = os.environ.get("CXX", "c++")
    headers = sorted(
        path
        for path in include_root.rglob("*")
        if path.suffix in {".h", ".hpp"} and "nerve/torch/" not in path.as_posix()
    )
    if not headers:
        raise FileNotFoundError(f"missing installed public C++ headers under {include_root}")
    for header in headers:
        _compile_installed_header(prefix, header, compiler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--generator", default=os.environ.get("CMAKE_GENERATOR"))
    parser.add_argument("--header-sweep", action="store_true")
    args = parser.parse_args()

    prefix = args.prefix.resolve()
    if not (prefix / "include" / "nerve").is_dir():
        raise FileNotFoundError(f"missing installed Nerve headers under {prefix}")

    with tempfile.TemporaryDirectory(prefix="nerve-cpp-install-smoke-") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        build_dir = root / "build"
        source_dir.mkdir()
        _write_consumer_project(source_dir)

        configure = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={prefix}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        if args.generator:
            configure[1:1] = ["-G", args.generator]
        _run(configure, root)
        _run(["cmake", "--build", str(build_dir), "--parallel"], root)
        _run([str(build_dir / "nerve_cpp_install_smoke")], root)
        _run([str(build_dir / "nerve_c_install_smoke")], root)
    if args.header_sweep:
        _sweep_installed_headers(prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
