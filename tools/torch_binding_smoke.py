#!/usr/bin/env python3
"""Smoke-test built PyTorch-native extension modules."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _prepend_import_paths(build_dir: Path) -> None:
    build_python = build_dir / "python"
    if not build_python.exists():
        raise FileNotFoundError(f"missing Python build directory: {build_python}")
    sys.path[:0] = [str(build_python)]


def _check_torch_bindings() -> None:
    import pynerve.torch as tda_torch  # type: ignore[import-not-found]
    import pynerve_torch_internal as torch_ext  # type: ignore[import-not-found]
    from pynerve.torch._backend import get_backend_info  # type: ignore[import-not-found]
    from pynerve.torch._persistence_python import (
        compute_vr_python,  # type: ignore[import-not-found]
    )

    import torch  # type: ignore[import-not-found]

    points = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    raw_result = torch_ext.vr_persistence_forward(points, 1, 2.0, "euclidean", 0, 0.1)
    if not isinstance(raw_result, tuple) or len(raw_result) < 3:
        raise RuntimeError("torch extension returned an invalid VR persistence result")
    diagrams, mask, num_pairs = raw_result[:3]
    if diagrams.shape[-1] != 3:
        raise RuntimeError(f"unexpected torch extension diagram shape: {tuple(diagrams.shape)}")
    if mask.dtype != torch.bool:
        raise RuntimeError(f"unexpected torch extension mask dtype: {mask.dtype}")
    if num_pairs.numel() == 0:
        raise RuntimeError("torch extension returned an empty pair-count tensor")
    metric_points = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)
    manhattan_result = torch_ext.vr_persistence_forward(metric_points, 1, 3.0, "manhattan", 0, 0.1)[
        0
    ]
    euclidean_result = torch_ext.vr_persistence_forward(metric_points, 1, 3.0, "euclidean", 0, 0.1)[
        0
    ]
    if not torch.any(torch.isclose(manhattan_result[..., 1], torch.tensor(2.0))):
        raise RuntimeError("native torch VR did not honor the manhattan metric")
    if manhattan_result.shape != euclidean_result.shape:
        raise RuntimeError("native torch VR metric selection changed output shape")

    def _expect_validation(name: str, call: Callable[[], object], fragment: str) -> None:
        try:
            call()
        except (RuntimeError, ValueError) as exc:
            if fragment not in str(exc):
                raise RuntimeError(f"{name} validation error lost expected context: {exc}") from exc
        else:
            raise RuntimeError(f"{name} accepted invalid input")

    def _invalid_mapper_config(module: object, **updates: object) -> object:
        config = module.MapperConfig()
        for key, value in updates.items():
            setattr(config, key, value)
        return config

    _expect_validation(
        "native vr invalid metric",
        lambda: torch_ext.vr_persistence_forward(points, 1, 2.0, "invalid", 0, 0.1),
        "Unsupported metric",
    )
    _expect_validation(
        "native vr nonfinite points",
        lambda: torch_ext.vr_persistence_forward(
            torch.tensor([[[0.0, 0.0], [float("nan"), 0.0]]], dtype=torch.float32),
            1,
            2.0,
            "euclidean",
            0,
            0.1,
        ),
        "finite coordinates",
    )
    _expect_validation(
        "native vr negative persistence image resolution",
        lambda: torch_ext.vr_persistence_forward(points, 1, 2.0, "euclidean", -1, 0.1),
        "pi_resolution",
    )
    _expect_validation(
        "native vr unsupported persistence image output",
        lambda: torch_ext.vr_persistence_forward(points, 1, 2.0, "euclidean", 8, 0.1),
        "does not return persistence images",
    )

    backend_info = get_backend_info()
    if not backend_info["torch_c_available"]:
        raise RuntimeError(
            f"torch backend dispatcher did not find nerve_torch_internal: {backend_info}"
        )

    diagram = tda_torch.vr_persistence(points, max_dim=1, max_radius=2.0)
    if diagram.diagrams.shape[-1] != 3:
        raise RuntimeError(
            f"unexpected high-level torch diagram shape: {tuple(diagram.diagrams.shape)}"
        )
    duplicate_points = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    duplicate_diagram = tda_torch.vr_persistence(duplicate_points, max_dim=1, max_radius=2.0)
    valid_zero_pairs = (
        (duplicate_diagram.diagrams[..., 0] == 0)
        & (duplicate_diagram.diagrams[..., 1] == 0)
        & duplicate_diagram.mask
    )
    if not torch.any(valid_zero_pairs):
        raise RuntimeError("high-level torch VR mask dropped a zero-persistence pair")
    if duplicate_diagram.num_pairs is None or int(duplicate_diagram.num_pairs[0, 0]) != 3:
        raise RuntimeError(
            "high-level torch VR pair counts must include zero-persistence pairs: "
            f"{duplicate_diagram.num_pairs}"
        )
    default_radius_diagram = tda_torch.vr_persistence(points, max_dim=1)
    if default_radius_diagram.diagrams.shape[-1] != 3:
        raise RuntimeError(
            "high-level torch VR default max_radius must remain supported: "
            f"{tuple(default_radius_diagram.diagrams.shape)}"
        )
    _expect_validation(
        "high-level vr nonfinite points",
        lambda: tda_torch.vr_persistence(
            torch.tensor([[[0.0, 0.0], [float("inf"), 0.0]]], dtype=torch.float32),
            max_dim=1,
            max_radius=2.0,
        ),
        "finite coordinates",
    )
    chebyshev_diagram = tda_torch.vr_persistence(
        metric_points, max_dim=1, max_radius=3.0, metric="chebyshev"
    )
    chebyshev_death = torch.tensor(
        1.0, dtype=chebyshev_diagram.diagrams.dtype, device=chebyshev_diagram.diagrams.device
    )
    if not torch.any(torch.isclose(chebyshev_diagram.diagrams[..., 1], chebyshev_death)):
        raise RuntimeError("high-level torch VR rejected or ignored the chebyshev metric")
    python_chebyshev = compute_vr_python(metric_points, 1, "chebyshev", 3.0)
    python_death = torch.tensor(1.0, dtype=python_chebyshev.dtype, device=python_chebyshev.device)
    if not torch.any(torch.isclose(python_chebyshev[..., 1], python_death)):
        raise RuntimeError(
            "torch VR python implementation rejected or ignored the chebyshev metric"
        )
    image = tda_torch.persistence_image(diagram, resolution=(8, 8), sigma=0.1)
    if image.shape != (1, 8, 8):
        raise RuntimeError(f"unexpected high-level persistence image shape: {tuple(image.shape)}")
    if image.dtype != diagram.diagrams.dtype or image.device != diagram.diagrams.device:
        raise RuntimeError(
            "persistence image must preserve high-level diagram dtype/device; "
            f"got {image.dtype} on {image.device}"
        )
    image_diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
    rectangular_constant_image = tda_torch.persistence_image(
        image_diagram, resolution=(6, 10), sigma=0.2, weight_fn="constant"
    )
    rectangular_persistence_image = tda_torch.persistence_image(
        image_diagram, resolution=(6, 10), sigma=0.2, weight_fn="persistence"
    )
    if rectangular_constant_image.shape != (6, 10):
        raise RuntimeError(
            "high-level persistence image must honor non-square resolution: "
            f"{tuple(rectangular_constant_image.shape)}"
        )
    if torch.allclose(rectangular_constant_image, rectangular_persistence_image):
        raise RuntimeError("high-level persistence image ignored weight_fn")
    raw_batched_image = tda_torch.persistence_image(
        image_diagram.unsqueeze(0), resolution=(6, 10), sigma=0.2, weight_fn="constant"
    )
    if raw_batched_image.shape != (1, 6, 10):
        raise RuntimeError(
            "high-level persistence image must handle raw batched diagrams: "
            f"{tuple(raw_batched_image.shape)}"
        )
    _expect_validation(
        "high-level persistence image NaN sigma",
        lambda: tda_torch.persistence_image(image_diagram, resolution=(6, 10), sigma=float("nan")),
        "finite and positive",
    )
    _expect_validation(
        "high-level persistence image invalid birth",
        lambda: tda_torch.persistence_image(
            torch.tensor([[float("nan"), 1.0]], dtype=torch.float32),
            resolution=(6, 10),
            sigma=0.2,
        ),
        "births",
    )
    _expect_validation(
        "high-level persistence image invalid interval",
        lambda: tda_torch.persistence_image(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            resolution=(6, 10),
            sigma=0.2,
        ),
        "deaths",
    )
    vector_landscape = tda_torch.vectorization.persistence_landscape(
        image_diagram.to(torch.float64), k=2, num_samples=7
    )
    if vector_landscape.shape != (2, 7) or vector_landscape.dtype != torch.float64:
        raise RuntimeError(
            "torch vectorization landscape must preserve requested shape and dtype: "
            f"{vector_landscape.shape}, {vector_landscape.dtype}"
        )
    _expect_validation(
        "torch vectorization image invalid sigma",
        lambda: tda_torch.vectorization.persistence_image(image_diagram, sigma=float("nan")),
        "finite and positive",
    )
    _expect_validation(
        "torch vectorization invalid diagram birth",
        lambda: tda_torch.vectorization.persistence_silhouette(
            torch.tensor([[float("inf"), float("inf")]], dtype=torch.float32),
            num_samples=7,
        ),
        "births",
    )
    _expect_validation(
        "torch vectorization invalid interval",
        lambda: tda_torch.vectorization.persistence_landscape(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            k=2,
            num_samples=7,
        ),
        "deaths",
    )
    heat_vector = tda_torch.vectorization.heat_kernel_signature(
        image_diagram.to(torch.float64),
        num_samples=6,
        sigma=0.2,
        t_values=torch.tensor([0.1, 1.0], dtype=torch.float32),
    )
    if heat_vector.shape != (2, 6) or heat_vector.dtype != torch.float64:
        raise RuntimeError(
            f"unexpected torch vectorization heat output: {heat_vector.shape}, {heat_vector.dtype}"
        )
    _expect_validation(
        "torch vectorization heat invalid sigma",
        lambda: tda_torch.vectorization.heat_kernel_signature(image_diagram, sigma=float("nan")),
        "finite and positive",
    )
    _expect_validation(
        "torch vectorization heat invalid t values",
        lambda: tda_torch.vectorization.heat_kernel_signature(
            image_diagram,
            t_values=torch.tensor([float("nan")], dtype=torch.float32),
        ),
        "t_values",
    )
    _expect_validation(
        "torch vectorization spectral invalid birth",
        lambda: tda_torch.vectorization.heat_kernel_signature(
            torch.tensor([[float("inf"), float("inf")]], dtype=torch.float32),
            num_samples=6,
        ),
        "births",
    )
    _expect_validation(
        "torch vectorization histogram invalid statistic",
        lambda: tda_torch.vectorization.birth_death_curve(
            image_diagram, num_bins=5, statistic="median"
        ),
        "statistic",
    )
    native_constant_image = torch_ext.ml_persistence_image(
        image_diagram, 10, 6, 0.2, 0.0, 0.0, 0.0, 0.0, "constant"
    )
    if native_constant_image.shape != (6, 10):
        raise RuntimeError(
            "native persistence image must honor birth/death resolutions: "
            f"{tuple(native_constant_image.shape)}"
        )
    _expect_validation(
        "native persistence image invalid resolution",
        lambda: torch_ext.ml_persistence_image(
            image_diagram, 0, 6, 0.2, 0.0, 0.0, 0.0, 0.0, "constant"
        ),
        "resolution",
    )
    _expect_validation(
        "native persistence image invalid sigma",
        lambda: torch_ext.ml_persistence_image(
            image_diagram, 10, 6, 0.0, 0.0, 0.0, 0.0, 0.0, "constant"
        ),
        "sigma",
    )
    _expect_validation(
        "native persistence image invalid weight",
        lambda: torch_ext.ml_persistence_image(
            image_diagram, 10, 6, 0.2, 0.0, 0.0, 0.0, 0.0, "invalid"
        ),
        "weight function",
    )
    _expect_validation(
        "native persistence image invalid diagram",
        lambda: torch_ext.ml_persistence_image(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            10,
            6,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            "constant",
        ),
        "deaths",
    )
    _expect_validation(
        "native persistence image negative-infinite death",
        lambda: torch_ext.ml_persistence_image(
            torch.tensor([[0.0, float("-inf")]], dtype=torch.float32),
            10,
            6,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            "constant",
        ),
        "positive infinity",
    )
    image = torch_ext.persistence_image(image_diagram, 4, 0.1)
    if image.shape != (4, 4):
        raise RuntimeError(f"unexpected image persistence output: {image.shape}")
    _expect_validation(
        "image persistence invalid resolution",
        lambda: torch_ext.persistence_image(image_diagram, 0, 0.1),
        "resolution",
    )
    _expect_validation(
        "image persistence invalid sigma",
        lambda: torch_ext.persistence_image(image_diagram, 4, float("nan")),
        "sigma",
    )
    _expect_validation(
        "persistence landscape invalid depth",
        lambda: torch_ext.persistence_landscape(image_diagram, 0, 8),
        "k",
    )
    _expect_validation(
        "betti curve invalid sample count",
        lambda: torch_ext.betti_curve(image_diagram, 0, 0),
        "num_samples",
    )
    landscape = torch_ext.ml_persistence_landscape(image_diagram, 2, 8, 0.0, 0.0)
    if landscape.shape != (2, 8) or landscape.dtype != image_diagram.dtype:
        raise RuntimeError(f"unexpected native persistence landscape output: {landscape.shape}")
    silhouette = torch_ext.ml_persistence_silhouette(image_diagram, 8, 0.0, 0.0, "constant")
    if silhouette.shape != (8,) or silhouette.dtype != image_diagram.dtype:
        raise RuntimeError(f"unexpected native persistence silhouette output: {silhouette.shape}")
    heat_signature = torch_ext.ml_heat_kernel_signature(
        image_diagram, 8, 0.2, torch.tensor([0.1, 1.0], dtype=torch.float32)
    )
    if heat_signature.shape != (2, 8) or heat_signature.dtype != image_diagram.dtype:
        raise RuntimeError(
            f"unexpected native heat kernel signature output: {heat_signature.shape}"
        )
    birth_death_curve = torch_ext.ml_birth_death_curve(image_diagram, 5, "count")
    if birth_death_curve.shape != (5,) or birth_death_curve.dtype != image_diagram.dtype:
        raise RuntimeError(f"unexpected native birth-death curve output: {birth_death_curve.shape}")
    _expect_validation(
        "native persistence landscape invalid depth",
        lambda: torch_ext.ml_persistence_landscape(image_diagram, 0, 8, 0.0, 0.0),
        "k must be positive",
    )
    _expect_validation(
        "native heat kernel invalid t values",
        lambda: torch_ext.ml_heat_kernel_signature(
            image_diagram, 8, 0.2, torch.tensor([-1.0], dtype=torch.float32)
        ),
        "t_values",
    )
    _expect_validation(
        "native birth-death curve invalid statistic",
        lambda: torch_ext.ml_birth_death_curve(image_diagram, 5, "sum"),
        "statistic",
    )
    if torch_ext.ml_total_persistence(image_diagram, -1, 1.0) <= 0.0:
        raise RuntimeError("native ML statistics total persistence must be positive")
    native_features = torch_ext.ml_extract_features(image_diagram, [0, 1])
    if native_features.ndim != 1 or not torch.isfinite(native_features).all():
        raise RuntimeError("native ML statistics feature vector is invalid")
    _expect_validation(
        "native ML statistics invalid power",
        lambda: torch_ext.ml_total_persistence(image_diagram, -1, float("nan")),
        "p",
    )
    _expect_validation(
        "native ML statistics invalid entropy base",
        lambda: torch_ext.ml_persistence_entropy(image_diagram, -1, 1.0),
        "base",
    )
    _expect_validation(
        "native ML statistics invalid threshold",
        lambda: torch_ext.ml_number_of_features(image_diagram, -1, float("nan")),
        "min_persistence",
    )
    _expect_validation(
        "native ML statistics invalid sample count",
        lambda: torch_ext.ml_betti_curve(image_diagram, 0, -1),
        "num_samples",
    )
    _expect_validation(
        "native ML statistics invalid metric",
        lambda: torch_ext.ml_amplitude(image_diagram, "invalid", 1.0, -1),
        "amplitude metric",
    )
    _expect_validation(
        "native ML statistics invalid diagram",
        lambda: torch_ext.ml_total_persistence(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32), -1, 1.0
        ),
        "finite deaths",
    )
    op_distances = torch.ops.nerve.filtration_distance_matrix(points[0], "euclidean")
    if op_distances.shape != (3, 3) or op_distances.dtype != torch.float64:
        raise RuntimeError(f"unexpected registered distance operator output: {op_distances}")
    unbounded_distances = torch.ops.nerve.vr_build(points[0], float("inf"))
    if unbounded_distances.shape != (3, 3) or not torch.isfinite(unbounded_distances).all():
        raise RuntimeError(
            "registered vr_build must support +inf max_radius as an all-edges radius"
        )
    fast_diagram = torch.ops.nerve.vr_fast(points[0], 2.0, "fast")
    if fast_diagram.ndim != 2 or fast_diagram.shape[-1] != 2:
        raise RuntimeError(
            f"registered vr_fast must return persistence pairs: {fast_diagram.shape}"
        )
    auto_diagram = torch.ops.nerve.vr_fast(points[0], 2.0, "auto")
    if auto_diagram.ndim != 2 or auto_diagram.shape[-1] != 2:
        raise RuntimeError(
            f"registered vr_fast(auto) must return persistence pairs: {auto_diagram.shape}"
        )

    def _expect_registered_validation(name: str, call: Callable[[], object], fragment: str) -> None:
        try:
            call()
        except RuntimeError as exc:
            if fragment not in str(exc):
                raise RuntimeError(f"{name} validation error lost expected context: {exc}") from exc
        else:
            raise RuntimeError(f"{name} accepted invalid input")

    _expect_registered_validation(
        "filtration_distance_matrix nonfinite points",
        lambda: torch.ops.nerve.filtration_distance_matrix(
            torch.tensor([[0.0, 0.0], [float("inf"), 0.0]], dtype=torch.float64),
            "euclidean",
        ),
        "finite coordinates",
    )
    _expect_registered_validation(
        "filtration_witness nonfinite landmarks",
        lambda: torch.ops.nerve.filtration_witness(
            torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
            torch.tensor([[float("nan"), 0.0]], dtype=torch.float64),
        ),
        "finite coordinates",
    )
    _expect_registered_validation(
        "vr_fast nonfinite points",
        lambda: torch.ops.nerve.vr_fast(
            torch.tensor([[0.0, 0.0], [float("nan"), 0.0]], dtype=torch.float64),
            2.0,
            "fast",
        ),
        "finite coordinates",
    )
    _expect_registered_validation(
        "vr_build NaN radius",
        lambda: torch.ops.nerve.vr_build(points[0], float("nan")),
        "positive and not NaN",
    )
    _expect_registered_validation(
        "vr_persistence negative distance",
        lambda: torch.ops.nerve.vr_persistence(
            torch.tensor([[0.0, -1.0], [-1.0, 0.0]], dtype=torch.float64),
            1,
        ),
        "non-negative",
    )
    _expect_registered_validation(
        "vr_persistence asymmetric distance",
        lambda: torch.ops.nerve.vr_persistence(
            torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float64),
            1,
        ),
        "symmetric",
    )
    large_diagram = torch.ops.nerve.vr_fast(points[0], 2.0, "large")
    if large_diagram.ndim != 2 or large_diagram.shape[-1] != 2:
        raise RuntimeError(
            f"registered vr_fast(large) must return persistence pairs: {large_diagram.shape}"
        )
    op_image = torch.ops.nerve.ph_image(diagram.diagrams[0, :, :2], 4, 4, 0.1)
    if op_image.shape != (4, 4):
        raise RuntimeError(
            f"unexpected registered persistence-image operator shape: {op_image.shape}"
        )
    filtration = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, requires_grad=True)
    ph_diagram = torch.ops.nerve.ph_grad(filtration, 0)
    if ph_diagram.dtype != filtration.dtype or ph_diagram.device != filtration.device:
        raise RuntimeError(
            "registered ph_grad must preserve filtration dtype/device: "
            f"{ph_diagram.dtype} on {ph_diagram.device}"
        )
    ph_diagram[:, 0].sum().backward()
    if filtration.grad is None or filtration.grad.dtype != filtration.dtype:
        raise RuntimeError("registered ph_grad backward did not preserve gradient dtype")
    _expect_registered_validation(
        "ph_compute unsupported dimension",
        lambda: torch.ops.nerve.ph_compute(filtration.detach(), 1),
        "max_dim=0",
    )
    _expect_registered_validation(
        "ph_grad nonfinite filtration",
        lambda: torch.ops.nerve.ph_grad(
            torch.tensor([0.0, float("nan")], dtype=torch.float32),
            0,
        ),
        "finite values",
    )

    alpha_filtration = torch.ops.nerve.filtration_alpha(points[0])
    if alpha_filtration.shape != (3, 3):
        raise RuntimeError(f"registered alpha filtration returned {alpha_filtration.shape}")
    witness_diagram = torch.ops.nerve.ph_witness(points[0, :2, :], points[0, 2:, :], 1, 2.0)
    if witness_diagram.dim() != 2 or witness_diagram.shape[-1] != 2:
        raise RuntimeError(f"registered witness persistence returned {witness_diagram.shape}")
    alpha_diagram = torch.ops.nerve.ph_alpha(points[0], 1)
    if alpha_diagram.dim() != 2 or alpha_diagram.shape[-1] != 2:
        raise RuntimeError(f"registered alpha persistence returned {alpha_diagram.shape}")
    boundary_diagram = torch.ops.nerve.ph_persistence(
        torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        torch.tensor([0.0, 0.0], dtype=torch.float32),
        0,
    )
    if boundary_diagram.shape != (1, 2):
        raise RuntimeError(f"registered boundary persistence returned {boundary_diagram.shape}")

    python_kernel_matrix = tda_torch.kernels.compute_kernel_matrix(
        [image_diagram.to(torch.float64), image_diagram.to(torch.float64)],
        kernel="gaussian",
        sigma=0.5,
    )
    if python_kernel_matrix.shape != (2, 2) or python_kernel_matrix.dtype != torch.float64:
        raise RuntimeError(
            "unexpected python torch kernel matrix output: "
            f"{python_kernel_matrix.shape}, {python_kernel_matrix.dtype}"
        )
    _expect_validation(
        "python torch gaussian kernel invalid metric",
        lambda: tda_torch.kernels.gaussian_kernel(
            image_diagram, image_diagram, distance_metric="invalid"
        ),
        "distance_metric",
    )
    _expect_validation(
        "python torch kernel invalid diagram birth",
        lambda: tda_torch.kernels.persistence_scale_space_kernel(
            torch.tensor([[float("nan"), 1.0]], dtype=torch.float32),
            image_diagram,
        ),
        "births",
    )
    _expect_validation(
        "python torch scale-space kernel invalid weight",
        lambda: tda_torch.kernels.persistence_scale_space_kernel(
            image_diagram, image_diagram, weight=float("nan")
        ),
        "weight",
    )
    _expect_validation(
        "python torch kernel matrix invalid diagonal",
        lambda: tda_torch.kernels.normalize_kernel_matrix(torch.zeros((2, 2), dtype=torch.float32)),
        "diagonal",
    )
    _expect_validation(
        "python torch kernel matrix centering nonfinite",
        lambda: tda_torch.kernels.center_kernel_matrix(
            torch.tensor([[1.0, float("nan")], [0.0, 1.0]], dtype=torch.float32)
        ),
        "finite",
    )
    stats_diagram = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
    stats_entropy = tda_torch.statistics.persistence_entropy(stats_diagram)
    stats_betti = tda_torch.statistics.betti_curve(stats_diagram, num_samples=5)
    if (
        stats_entropy.dtype != torch.float64
        or not torch.isfinite(stats_entropy)
        or stats_betti.shape != (5,)
        or stats_betti.dtype != torch.float64
    ):
        raise RuntimeError(
            "unexpected python torch statistics output: "
            f"{stats_entropy}, {stats_betti.shape}, {stats_betti.dtype}"
        )
    _expect_validation(
        "python torch statistics invalid persistence power",
        lambda: tda_torch.statistics.total_persistence(stats_diagram, p=float("nan")),
        "finite and positive",
    )
    _expect_validation(
        "python torch statistics invalid entropy base",
        lambda: tda_torch.statistics.persistence_entropy(stats_diagram, base=1.0),
        "base",
    )
    _expect_validation(
        "python torch statistics invalid diagram interval",
        lambda: tda_torch.statistics.mean_persistence(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        ),
        "deaths",
    )
    preprocessing_diagram = torch.tensor([[0.0, 1.0], [0.5, float("inf")]], dtype=torch.float64)
    finite_preprocessed = tda_torch.preprocessing.handle_infinite_deaths(
        preprocessing_diagram, strategy="max"
    )
    if (
        finite_preprocessed.dtype != torch.float64
        or not torch.isfinite(finite_preprocessed[:, 1]).all()
        or tda_torch.preprocessing.clean_diagram(preprocessing_diagram.unsqueeze(0)).dim() != 3
    ):
        raise RuntimeError("unexpected python torch preprocessing output")
    _expect_validation(
        "python torch preprocessing invalid normalization method",
        lambda: tda_torch.preprocessing.normalize_diagram(finite_preprocessed, method="invalid"),
        "method",
    )
    _expect_validation(
        "python torch preprocessing infinite normalization",
        lambda: tda_torch.preprocessing.normalize_diagram(preprocessing_diagram),
        "finite deaths",
    )
    _expect_validation(
        "python torch preprocessing invalid threshold range",
        lambda: tda_torch.preprocessing.threshold_diagram(
            finite_preprocessed, min_persistence=2.0, max_persistence=1.0
        ),
        "max_persistence",
    )
    _expect_validation(
        "python torch preprocessing invalid diagram interval",
        lambda: tda_torch.preprocessing.threshold_diagram(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        ),
        "deaths",
    )
    distance_value = tda_torch.diagram_wasserstein(
        image_diagram.to(torch.float64), image_diagram.to(torch.float64)
    )
    if distance_value.dtype != torch.float64 or distance_value.device != image_diagram.device:
        raise RuntimeError(f"unexpected python torch distance output: {distance_value}")
    _expect_validation(
        "python torch distance invalid persistence power",
        lambda: tda_torch.diagram_wasserstein(image_diagram, image_diagram, p=float("nan")),
        "finite and positive",
    )
    _expect_validation(
        "python torch distance invalid diagram birth",
        lambda: tda_torch.diagram_wasserstein(
            torch.tensor([[float("nan"), 1.0]], dtype=torch.float32),
            image_diagram,
        ),
        "births",
    )
    _expect_validation(
        "python torch distance invalid diagram interval",
        lambda: tda_torch.diagram_bottleneck(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            image_diagram,
        ),
        "deaths",
    )
    container_tensor = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]], dtype=torch.float64)
    masked_container = tda_torch.PersistenceDiagram(
        container_tensor,
        mask=torch.tensor([[False, True]]),
        num_pairs=torch.tensor([[0, 1]]),
    )
    partial_metadata_container = tda_torch.PersistenceDiagram(
        torch.tensor([[[0.0, 2.0, 0.0]]], dtype=torch.float64),
        mask=torch.tensor([[True]]),
    )
    batched_container = tda_torch.batch_diagrams([masked_container, partial_metadata_container])
    if (
        batched_container.mask.shape != (2, 2)
        or bool(batched_container.mask[0, 0])
        or bool(batched_container.mask[1, 1])
        or batched_container.num_pairs is not None
    ):
        raise RuntimeError("PersistenceDiagram batching corrupted mask metadata")
    _expect_validation(
        "python torch diagram invalid dimension value",
        lambda: tda_torch.PersistenceDiagram(
            torch.tensor([[[0.0, 1.0, 0.5]]], dtype=torch.float32)
        ),
        "dimensions",
    )
    _expect_validation(
        "python torch diagram invalid persistence power",
        lambda: masked_container.total_persistence(p=float("nan")),
        "finite and positive",
    )
    scatter_data = tda_torch.viz.diagram_to_scatter_data(masked_container)
    heatmap_data = tda_torch.viz.diagram_to_heatmap_data(masked_container, grid_size=4)
    if scatter_data["births"].shape != (1,) or heatmap_data["grid"].shape != (4, 4):
        raise RuntimeError("unexpected python torch viz data output")
    _expect_validation(
        "python torch viz invalid diagram birth",
        lambda: tda_torch.viz.diagram_to_scatter_data(
            torch.tensor([[float("nan"), 1.0]], dtype=torch.float32)
        ),
        "births",
    )
    _expect_validation(
        "python torch viz invalid dimension value",
        lambda: tda_torch.viz.diagram_to_scatter_data(
            torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float32)
        ),
        "dimensions",
    )
    _expect_validation(
        "python torch viz invalid padding",
        lambda: tda_torch.viz.get_plot_limits(container_tensor[0], padding=float("nan")),
        "padding",
    )
    linear_similarity = tda_torch.training_utils.compute_kernel_similarity(
        [image_diagram], [image_diagram], kernel="linear"
    )
    if linear_similarity.shape != (1, 1):
        raise RuntimeError(
            f"unexpected python torch training linear-kernel output: {linear_similarity}"
        )
    _expect_validation(
        "python torch training invalid kernel sigma",
        lambda: tda_torch.training_utils.compute_kernel_similarity(
            [image_diagram], [image_diagram], kernel="gaussian", sigma=float("nan")
        ),
        "sigma",
    )
    _expect_validation(
        "python torch training invalid distance loss p",
        lambda: tda_torch.training_utils.DiagramDistanceLoss(p=float("nan")),
        "finite",
    )
    _expect_validation(
        "python torch training invalid regularization target",
        lambda: tda_torch.training_utils.TopologicalRegularization({"h0_count": float("nan")}),
        "target_complexity",
    )
    _expect_validation(
        "python torch training invalid cross-entropy threshold",
        lambda: tda_torch.training_utils.PersistenceCrossEntropy(
            min_persistence_threshold=float("nan")
        ),
        "min_persistence_threshold",
    )
    collated_points = tda_torch.data.collate_point_clouds(
        [
            torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        ]
    )
    empty_diagram_batch = tda_torch.data.collate_diagrams([])
    if collated_points.shape != (2, 2, 2) or empty_diagram_batch.batch_size != 0:
        raise RuntimeError("unexpected python torch data collation output")
    _expect_validation(
        "python torch data invalid point cloud",
        lambda: tda_torch.data.collate_point_clouds(
            [torch.tensor([[float("nan"), 0.0]], dtype=torch.float32)]
        ),
        "finite coordinates",
    )
    _expect_validation(
        "python torch data invalid pad value",
        lambda: tda_torch.data.collate_point_clouds(
            [torch.tensor([[0.0, 0.0]], dtype=torch.float32)],
            pad_value=float("nan"),
        ),
        "pad_value",
    )

    try:
        torch_ext.ml_gaussian_kernel(
            diagram.diagrams[0], diagram.diagrams[0], 0.1, "invalid_metric"
        )
    except ValueError as exc:
        if "unsupported Gaussian kernel distance metric" not in str(exc):
            raise RuntimeError(f"unexpected Gaussian kernel metric error: {exc}") from exc
    else:
        raise RuntimeError("Gaussian kernel accepted an unsupported metric")
    _expect_validation(
        "Gaussian kernel negative-infinite death",
        lambda: torch_ext.ml_gaussian_kernel(
            torch.tensor([[0.0, float("-inf")]], dtype=torch.float32),
            diagram.diagrams[0],
            0.1,
            "euclidean",
        ),
        "positive infinity",
    )
    _expect_validation(
        "scale-space kernel invalid weight",
        lambda: torch_ext.ml_persistence_scale_space_kernel(
            diagram.diagrams[0], diagram.diagrams[0], 0.1, 2.0
        ),
        "weight",
    )
    _expect_validation(
        "sliced Wasserstein kernel invalid sigma",
        lambda: torch_ext.ml_sliced_wasserstein_kernel(
            diagram.diagrams[0], diagram.diagrams[0], 8, 0.0
        ),
        "sigma",
    )
    _expect_validation(
        "kernel matrix invalid kernel",
        lambda: torch_ext.ml_compute_kernel_matrix(
            [diagram.diagrams[0], diagram.diagrams[0]], "invalid", 0.1, 8
        ),
        "unsupported persistence diagram kernel",
    )
    _expect_validation(
        "kernel matrix normalization invalid diagonal",
        lambda: torch_ext.ml_normalize_kernel_matrix(
            torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        ),
        "diagonal",
    )
    _expect_validation(
        "kernel matrix centering nonfinite",
        lambda: torch_ext.ml_center_kernel_matrix(
            torch.tensor([[1.0, float("nan")], [0.0, 1.0]], dtype=torch.float32)
        ),
        "finite",
    )

    if hasattr(torch, "float8_e4m3fn"):
        float8_diagram = torch.tensor(
            [[0.0, 1.0], [0.25, 0.75]],
            dtype=torch.float8_e4m3fn,
        )
        float8_image = tda_torch.persistence_image(float8_diagram, resolution=(4, 4), sigma=0.1)
        if float8_image.shape != (4, 4) or float8_image.dtype != float8_diagram.dtype:
            raise RuntimeError(
                "float8 persistence image must preserve dtype; "
                f"got {float8_image.shape}, {float8_image.dtype}"
            )
    wasserstein = tda_torch.diagram_wasserstein(diagram, diagram)
    bottleneck = tda_torch.diagram_bottleneck(diagram, diagram)
    if wasserstein.ndim != 0 or bottleneck.ndim != 0:
        raise RuntimeError("diagram distances must return scalar tensors")
    if not torch.isfinite(wasserstein) or not torch.isfinite(bottleneck):
        raise RuntimeError("diagram distances must be finite for identical diagrams")
    _expect_registered_validation(
        "diagram_wasserstein infinite p",
        lambda: torch.ops.nerve.diagram_wasserstein(
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            float("inf"),
        ),
        "finite and >= 1",
    )
    _expect_registered_validation(
        "diagram_bottleneck invalid interval",
        lambda: torch.ops.nerve.diagram_bottleneck(
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        ),
        "death values must be >= birth",
    )
    landscape = torch.ops.nerve.diagram_landscape(
        torch.tensor([[0.0, 1.0], [0.0, float("inf")]], dtype=torch.float64),
        8,
    )
    if not torch.isfinite(landscape).all():
        raise RuntimeError("registered diagram_landscape must ignore infinite intervals")
    betti = torch.ops.nerve.diagram_betti(
        torch.tensor(
            [[0.0, float("inf"), 0.0], [0.0, float("inf"), 1.0]],
            dtype=torch.float64,
        ),
        1,
    )
    if int(betti.item()) != 1:
        raise RuntimeError(f"registered diagram_betti must honor dimension column: {betti}")
    matrix_diagram = tda_torch.persistence_from_matrix(torch.cdist(points, points), max_dim=1)
    if (
        matrix_diagram.diagrams.dim() != 3
        or matrix_diagram.diagrams.shape[0] != 1
        or matrix_diagram.diagrams.shape[-1] != 3
    ):
        raise RuntimeError(
            f"unexpected matrix persistence diagram shape: {tuple(matrix_diagram.diagrams.shape)}"
        )
    if matrix_diagram.num_pairs is None or matrix_diagram.num_pairs.shape != (1, 2):
        raise RuntimeError(
            "batched matrix persistence must preserve batch and max_dim pair counts: "
            f"{None if matrix_diagram.num_pairs is None else tuple(matrix_diagram.num_pairs.shape)}"
        )
    native_matrix_result = torch_ext.persistence_from_matrix(torch.cdist(points, points)[0], 1)
    if native_matrix_result[2].shape != (2,):
        raise RuntimeError(
            "native matrix persistence must return max_dim+1 pair counts: "
            f"{tuple(native_matrix_result[2].shape)}"
        )

    witness_result = tda_torch.witness_persistence(
        points[:, :2, :],
        points[:, 2:, :],
        max_dim=1,
        max_radius=2.0,
    )
    if witness_result.diagrams.shape[-1] != 3:
        raise RuntimeError(
            f"high-level witness persistence returned {witness_result.diagrams.shape}"
        )
    alpha_result = tda_torch.alpha_persistence(points, max_dim=1)
    if alpha_result.diagrams.shape[-1] != 3:
        raise RuntimeError(f"high-level alpha persistence returned {alpha_result.diagrams.shape}")

    native_pd = torch_ext.PersistenceDiagram(
        torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    )
    if native_pd.diagram().dtype != torch.float32 or native_pd.threshold(0.5).num_pairs() != 1:
        raise RuntimeError("native PersistenceDiagram lost dtype or threshold behavior")
    _expect_validation(
        "PersistenceDiagram invalid interval",
        lambda: torch_ext.PersistenceDiagram(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        ),
        "finite deaths",
    )
    _expect_validation(
        "PersistenceDiagram invalid dimensions dtype",
        lambda: torch_ext.PersistenceDiagram(
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        ),
        "dimensions must be torch.long",
    )
    _expect_validation(
        "PersistenceDiagram invalid image sigma",
        lambda: native_pd.to_persistence_image(8, float("nan")),
        "sigma",
    )
    _expect_validation(
        "PersistenceDiagram invalid threshold",
        lambda: native_pd.threshold(float("nan")),
        "min_persistence",
    )
    _expect_validation(
        "PersistenceDiagram incomplete state_dict",
        lambda: native_pd.load_state_dict(native_pd.state_dict()[:-1]),
        "state_dict missing death_idx",
    )

    mapper_graph = torch_ext.quick_mapper(points[0].detach(), "pca_2d", 2, 0.25)
    if mapper_graph.to_adjacency_matrix().shape != (
        len(mapper_graph.nodes),
        len(mapper_graph.nodes),
    ):
        raise RuntimeError("native Mapper adjacency shape does not match node count")
    _expect_validation(
        "Mapper nonfinite point cloud",
        lambda: torch_ext.quick_mapper(
            torch.tensor([[0.0, 0.0], [float("nan"), 0.0]], dtype=torch.float32),
            "pca_2d",
            2,
            0.25,
        ),
        "finite values",
    )
    _expect_validation(
        "Mapper invalid overlap",
        lambda: torch_ext.quick_mapper(points[0].detach(), "pca_2d", 2, float("nan")),
        "cover_overlap",
    )
    _expect_validation(
        "Mapper invalid dbscan eps",
        lambda: torch_ext.Mapper(_invalid_mapper_config(torch_ext, dbscan_eps=float("nan"))),
        "dbscan_eps",
    )
    _expect_validation(
        "Mapper empty filter input",
        lambda: torch_ext.filter_pca(torch.empty((0, 2), dtype=torch.float32), 1),
        "non-empty",
    )

    simplex_tree = torch_ext.SimplexTree()
    simplex_tree.build_vr(points[0], 2.0, 2)
    if simplex_tree.num_simplices() <= 1:
        raise RuntimeError("SimplexTree.build_vr produced no non-root simplices for float32 input")
    witness_tree = torch_ext.SimplexTree()
    witness_tree.build_witness(points[0], points[0, :2, :], 2.0, 2)
    if witness_tree.num_simplices() <= 1:
        raise RuntimeError(
            "SimplexTree.build_witness produced no non-root simplices for float32 input"
        )
    _expect_validation(
        "SimplexTree.build_vr nonfinite points",
        lambda: torch_ext.SimplexTree().build_vr(
            torch.tensor([[0.0, 0.0], [float("nan"), 0.0]], dtype=torch.float32),
            2.0,
            2,
        ),
        "finite coordinates",
    )
    _expect_validation(
        "SimplexTree.build_witness unsupported dimension",
        lambda: torch_ext.SimplexTree().build_witness(points[0], points[0, :2, :], 2.0, 3),
        "max_dim",
    )
    _expect_validation(
        "SimplexTree.insert_batch invalid filtration",
        lambda: torch_ext.SimplexTree().insert_batch(
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([float("nan")], dtype=torch.float32),
        ),
        "finite values",
    )
    _expect_validation(
        "SimplexTree.insert invalid vertices",
        lambda: torch_ext.SimplexTree().insert([1, 0], 0.0),
        "strictly increasing",
    )
    loss = torch.nan_to_num(diagram.diagrams[..., :2]).sum()
    loss.backward()
    if points.grad is None or not torch.isfinite(points.grad).all():
        raise RuntimeError("torch persistence backward path did not produce finite gradients")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()

    _prepend_import_paths(args.build_dir.resolve())
    _check_torch_bindings()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
