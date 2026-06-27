from __future__ import annotations

import pytest
from pynerve.exceptions import InvalidArgumentError, ValidationError

torch = pytest.importorskip("torch")


def _skip_if_cuda_unusable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device missing")
    major, minor = torch.cuda.get_device_capability()
    device_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    if supported_arches and device_arch not in supported_arches:
        pytest.skip(f"CUDA device capability {device_arch} is not supported by this PyTorch build")
    try:
        torch.ones(1, device="cuda").sum().item()
    except Exception as exc:
        pytest.skip(f"CUDA device unusable by this PyTorch build: {exc}")


@pytest.mark.generated
@pytest.mark.gradient
@pytest.mark.torch
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_torch_distance_gradient_cross_product(device: str, dtype: torch.dtype) -> None:
    if device == "cuda":
        _skip_if_cuda_unusable()

    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        pytest.skip("CPU low-precision backward support is backend dependent")

    x = torch.randn(8, 4, device=device, dtype=dtype).float().requires_grad_(True)
    distances = torch.cdist(x, x)
    loss = distances.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.torch
def test_exact_complex_torch_apis_return_diagrams() -> None:
    pytest.importorskip("pynerve_internal")
    from pynerve.torch import alpha_persistence, witness_persistence

    points = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )

    witness_diagram = witness_persistence(
        points[:, :2, :],
        points[:, 2:, :],
        max_dim=1,
        max_radius=2.0,
    )
    alpha_diagram = alpha_persistence(points, max_dim=1)

    assert witness_diagram.diagrams.shape[-1] == 3
    assert alpha_diagram.diagrams.shape[-1] == 3


@pytest.mark.torch
def test_torch_vr_chebyshev_metric_is_supported_by_python_impl() -> None:
    from pynerve.torch._persistence_api import _validate_metric
    from pynerve.torch._persistence_python import compute_vr_python

    assert _validate_metric("chebyshev") == "chebyshev"
    points = torch.tensor([[[0.0, 0.0], [1.0, 2.0]]], dtype=torch.float32)
    diagram = compute_vr_python(points, max_dim=1, metric="chebyshev", max_radius=3.0)
    expected_death = torch.tensor(2.0, dtype=diagram.dtype, device=diagram.device)

    assert torch.any(torch.isclose(diagram[..., 1], expected_death))
    with pytest.raises(ValueError, match="finite coordinates"):
        compute_vr_python(
            torch.tensor([[[0.0, 0.0], [float("nan"), 0.0]]], dtype=torch.float32),
            max_dim=1,
            metric="chebyshev",
            max_radius=3.0,
        )


@pytest.mark.torch
def test_torch_persistence_image_honors_resolution_and_weight() -> None:
    from pynerve.torch import persistence_image

    diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
    constant_image = persistence_image(diagram, resolution=(5, 9), sigma=0.2, weight_fn="constant")
    persistence_weighted_image = persistence_image(
        diagram, resolution=(5, 9), sigma=0.2, weight_fn="persistence"
    )

    assert constant_image.shape == (5, 9)
    assert not torch.allclose(constant_image, persistence_weighted_image)
    assert persistence_image(diagram.unsqueeze(0), resolution=(5, 9)).shape == (1, 5, 9)
    with pytest.raises(ValueError, match="finite and positive"):
        persistence_image(diagram, resolution=(5, 9), sigma=float("nan"))
    with pytest.raises(ValueError, match="births"):
        persistence_image(
            torch.tensor([[float("nan"), 1.0]], dtype=torch.float32),
            resolution=(5, 9),
        )
    with pytest.raises(ValueError, match="deaths"):
        persistence_image(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            resolution=(5, 9),
        )


@pytest.mark.torch
def test_torch_matrix_persistence_preserves_batched_shape_and_counts() -> None:
    pytest.importorskip("nerve_torch_internal")
    from pynerve.torch import persistence_from_matrix

    points = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    diagram = persistence_from_matrix(torch.cdist(points, points), max_dim=2)

    assert diagram.diagrams.dim() == 3
    assert diagram.diagrams.shape[0] == 1
    assert diagram.diagrams.shape[-1] == 3
    assert diagram.mask is not None and diagram.mask.shape[0] == 1
    assert diagram.num_pairs is not None
    assert diagram.num_pairs.shape == (1, 3)


@pytest.mark.torch
def test_torch_vr_mask_preserves_zero_persistence_pairs() -> None:
    pytest.importorskip("nerve_torch_internal")
    from pynerve.torch import vr_persistence

    points = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    diagram = vr_persistence(points, max_dim=1, max_radius=2.0)
    zero_pairs = (diagram.diagrams[..., 0] == 0) & (diagram.diagrams[..., 1] == 0) & diagram.mask

    assert torch.any(zero_pairs)
    assert diagram.num_pairs is not None
    assert int(diagram.num_pairs[0, 0]) == 3


@pytest.mark.torch
def test_nn_persistent_homology_uses_validated_public_core_api() -> None:
    pytest.importorskip("pynerve_internal")
    from pynerve import compute_persistence  # noqa: PLC0415

    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    result = compute_persistence(points, max_dim=1, max_radius=float("inf"))
    assert len(result.pairs) > 0
    assert len(result.betti_numbers) > 0
    with pytest.raises((ValidationError, InvalidArgumentError), match="finite"):
        compute_persistence(
            torch.tensor([[0.0, 0.0], [float("nan"), 0.0]], dtype=torch.float32),
            max_dim=1,
            max_radius=2.0,
        )


@pytest.mark.torch
def test_nn_building_blocks_use_validated_public_core_api() -> None:
    pytest.importorskip("pynerve_internal")
    from pynerve.nn.building_blocks import SparseRipsPersistence

    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    diagram = SparseRipsPersistence(max_dim=1, sparse_parameter=0.1)(points)

    assert diagram.births.ndim == 1
    assert diagram.deaths.shape == diagram.births.shape
    assert diagram.dimensions.shape == diagram.births.shape
    with pytest.raises((ValueError, ValidationError), match="finite"):
        SparseRipsPersistence(max_dim=1)(
            torch.tensor([[0.0, 0.0], [float("nan"), 0.0]], dtype=torch.float32)
        )


@pytest.mark.torch
def test_diff_persistent_homology_resolves_infinite_radius() -> None:
    pytest.importorskip("pynerve_internal")
    from pynerve.diff.ph_layer_module import DifferentiablePersistentHomology

    points = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    diagrams = DifferentiablePersistentHomology(max_dim=1)(points)

    assert len(diagrams) == 2
    assert all(diagram.shape[0] == 1 and diagram.shape[-1] == 2 for diagram in diagrams)
    with pytest.raises((ValueError, ValidationError), match="max_radius"):
        DifferentiablePersistentHomology(max_radius=float("nan"))


@pytest.mark.torch
def test_diff_representations_reject_invalid_numeric_inputs() -> None:
    from pynerve.diff._ph_representations import (
        compute_persistence_landscape,
        persistence_image,
    )

    diagram = torch.tensor([[0.0, 1.0], [0.25, 0.75]], dtype=torch.float32)

    assert persistence_image(diagram, resolution=4, sigma=0.2).shape == (4, 4)
    assert compute_persistence_landscape(diagram, n_layers=2, resolution=5).shape == (2, 5)
    with pytest.raises(ValueError, match="sigma"):
        persistence_image(diagram, sigma=float("nan"))
    with pytest.raises(ValidationError, match="births"):
        persistence_image(torch.tensor([[float("nan"), 1.0]], dtype=torch.float32))
    with pytest.raises(ValidationError, match="deaths"):
        compute_persistence_landscape(torch.tensor([[1.0, 0.0]], dtype=torch.float32))


@pytest.mark.torch
def test_ssl_topology_augmentations_reject_nonfinite_inputs() -> None:
    from pynerve.ssl.contrastive_learning import (
        BYOLTopology,
        PersistencePredictionTask,
        SimCLRTopology,
        TopologyAugmentation,
    )

    diagram = torch.tensor([[0.0, 1.0, 0.0], [0.25, 0.75, 1.0]], dtype=torch.float32)
    augmentation = TopologyAugmentation()

    assert torch.isfinite(augmentation.birth_death_perturbation(diagram, sigma=0.0)).all()
    with pytest.raises(ValidationError, match="sigma"):
        augmentation.birth_death_perturbation(diagram, sigma=float("nan"))
    with pytest.raises(ValidationError, match="scale_range"):
        augmentation.persistence_scaling(diagram, scale_range=(0.9, float("nan")))
    with pytest.raises(ValidationError, match="max_shift"):
        augmentation.small_birth_death_shift(diagram, max_shift=float("inf"))
    with pytest.raises(ValidationError, match="drop_prob"):
        augmentation.feature_dropout(diagram, drop_prob=float("nan"))
    with pytest.raises(ValidationError, match="shuffle_prob"):
        augmentation.dimension_shuffle(diagram, shuffle_prob=float("nan"))
    with pytest.raises(ValueError, match="birth/death"):
        augmentation.birth_death_perturbation(
            torch.tensor([[float("nan"), 1.0]], dtype=torch.float32)
        )
    with pytest.raises(ValueError, match="deaths"):
        augmentation.persistence_scaling(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    with pytest.raises(ValidationError, match="temperature"):
        SimCLRTopology(torch.nn.Identity(), temperature=float("nan"))
    with pytest.raises(ValidationError, match="tau"):
        BYOLTopology(torch.nn.Identity(), tau=float("nan"))
    with pytest.raises(ValidationError, match="mask_ratio"):
        PersistencePredictionTask(torch.nn.Identity(), mask_ratio=float("nan"))


@pytest.mark.torch
def test_ssl_topology_completion_rejects_nonfinite_inputs() -> None:
    from pynerve.ssl.topology_completion import (
        BettiNumberPrediction,
        FiltrationOrderingTask,
        MultiTaskTopologySSL,
        TopologyCompletionModel,
        TopologyDenoising,
    )

    diagram = torch.tensor([[0.0, 1.0, 0.0], [0.25, 0.75, 1.0]], dtype=torch.float32)
    mask = torch.tensor([True, False])
    invalid_birth = torch.tensor([[float("nan"), 1.0, 0.0]], dtype=torch.float32)
    invalid_death = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    completion = TopologyCompletionModel(torch.nn.Identity(), torch.nn.Identity())
    completed, rows = completion(diagram, mask)
    assert completed.shape == diagram.shape
    assert rows.shape == diagram.shape
    with pytest.raises(ValidationError, match="completion_threshold"):
        TopologyCompletionModel(
            torch.nn.Identity(),
            torch.nn.Identity(),
            completion_threshold=float("nan"),
        )
    with pytest.raises(ValueError, match="birth/death"):
        completion(invalid_birth, torch.tensor([True]))
    with pytest.raises(ValueError, match="deaths"):
        completion(invalid_death, torch.tensor([True]))
    with pytest.raises(ValueError, match="birth/death"):
        BettiNumberPrediction(torch.nn.Identity()).forward(invalid_birth)
    with pytest.raises(ValidationError, match="noise_threshold"):
        TopologyDenoising(torch.nn.Identity(), noise_threshold=float("nan"))

    ordering = FiltrationOrderingTask(torch.nn.Identity())
    with pytest.raises(ValueError, match="finite"):
        ordering(
            torch.tensor([[0.0, float("nan")]], dtype=torch.float32),
            torch.tensor([[1.0]], dtype=torch.float32),
        )
    with pytest.raises(ValidationError, match="task_weights"):
        MultiTaskTopologySSL(
            torch.nn.Identity(),
            task_weights={"completion": float("nan")},
        )


@pytest.mark.torch
def test_torch_vectorization_basis_validates_diagrams_and_parameters() -> None:
    from pynerve.torch import vectorization

    diagram = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)

    assert vectorization.persistence_image(diagram, resolution=(4, 6)).shape == (4, 6)
    assert vectorization.persistence_landscape(diagram, k=2, num_samples=7).dtype == diagram.dtype
    assert vectorization.persistence_silhouette(diagram, num_samples=7).shape == (7,)
    with pytest.raises(ValueError, match="finite and positive"):
        vectorization.persistence_image(diagram, sigma=float("nan"))
    with pytest.raises(ValidationError, match="births"):
        vectorization.persistence_landscape(
            torch.tensor([[float("nan"), 1.0]], dtype=torch.float32),
            k=2,
            num_samples=7,
        )
    with pytest.raises(ValidationError, match="deaths"):
        vectorization.persistence_silhouette(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            num_samples=7,
        )
    assert isinstance(
        vectorization.persistence_image(
            torch.tensor([[0.0, float("inf")]], dtype=torch.float32), resolution=(4, 4)
        ),
        torch.Tensor,
    )
    with pytest.raises(ValueError, match="x_range"):
        vectorization.persistence_landscape(diagram, k=2, num_samples=7, x_range=(1.0, 0.0))
    with pytest.raises(ValueError, match="min_sigma <= max_sigma"):
        vectorization.adaptive_persistence_image(diagram, min_sigma=1.0, max_sigma=0.5)


@pytest.mark.torch
def test_torch_vectorization_spectral_validates_diagrams_and_parameters() -> None:
    from pynerve.torch import vectorization

    diagram = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
    t_values = torch.tensor([0.1, 1.0], dtype=torch.float32)

    heat = vectorization.heat_kernel_signature(diagram, num_samples=5, sigma=0.2, t_values=t_values)
    assert heat.shape == (2, 5)
    assert heat.dtype == diagram.dtype
    assert vectorization.birth_death_curve(diagram, num_bins=4, statistic="mean").shape == (4,)
    with pytest.raises(ValueError, match="finite and positive"):
        vectorization.heat_kernel_signature(diagram, sigma=float("nan"))
    with pytest.raises(ValueError, match="t_values"):
        vectorization.heat_kernel_signature(
            diagram, t_values=torch.tensor([float("nan")], dtype=torch.float32)
        )
    with pytest.raises(ValidationError, match="births"):
        vectorization.heat_kernel_signature(
            torch.tensor([[float("inf"), float("inf")]], dtype=torch.float32),
            num_samples=5,
        )
    with pytest.raises(ValidationError, match="deaths"):
        vectorization.birth_death_curve(torch.tensor([[1.0, 0.0]], dtype=torch.float32), num_bins=4)
    assert isinstance(
        vectorization.heat_kernel_signature(
            torch.tensor([[0.0, float("inf")]], dtype=torch.float32),
            num_samples=5,
        ),
        torch.Tensor,
    )
    with pytest.raises(ValueError, match="statistic"):
        vectorization.birth_death_curve(diagram, num_bins=4, statistic="median")


@pytest.mark.torch
def test_torch_python_kernels_validate_diagrams_and_parameters() -> None:
    from pynerve.torch import kernels

    diagram = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)

    kernel_matrix = kernels.compute_kernel_matrix([diagram, diagram], kernel="gaussian", sigma=0.5)
    assert kernel_matrix.shape == (2, 2)
    assert kernel_matrix.dtype == diagram.dtype
    assert kernels.normalize_kernel_matrix(torch.eye(2, dtype=torch.float64)).dtype == torch.float64
    with pytest.raises(ValueError, match="finite and positive"):
        kernels.gaussian_kernel(diagram, diagram, sigma=float("nan"))
    with pytest.raises(ValueError, match="distance_metric"):
        kernels.gaussian_kernel(diagram, diagram, distance_metric="invalid")
    with pytest.raises(ValidationError, match="births"):
        kernels.persistence_scale_space_kernel(
            torch.tensor([[float("nan"), 1.0]], dtype=torch.float32),
            diagram.to(torch.float32),
        )
    with pytest.raises(ValidationError, match="deaths"):
        kernels.sliced_wasserstein_kernel(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            diagram.to(torch.float32),
        )
    with pytest.raises(ValueError, match="finite deaths"):
        kernels.gaussian_kernel(
            torch.tensor([[0.0, float("inf")]], dtype=torch.float32),
            diagram.to(torch.float32),
        )
    with pytest.raises(ValueError, match="weight"):
        kernels.persistence_scale_space_kernel(diagram, diagram, weight=float("nan"))
    with pytest.raises(ValidationError, match="num_slices"):
        kernels.sliced_wasserstein_kernel(diagram, diagram, num_slices=0)
    with pytest.raises(ValueError, match="bandwidth"):
        kernels.persistence_fisher_kernel(diagram, diagram, bandwidth=float("inf"))
    with pytest.raises(ValueError, match="Unknown kernel"):
        kernels.compute_kernel_matrix([], kernel="invalid")
    with pytest.raises(ValueError, match="diagonal"):
        kernels.normalize_kernel_matrix(torch.zeros((2, 2), dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        kernels.center_kernel_matrix(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]))


@pytest.mark.torch
def test_torch_statistics_validate_diagrams_and_parameters() -> None:
    from pynerve.torch import statistics

    diagram = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)

    assert statistics.total_persistence(diagram).dtype == diagram.dtype
    entropy = statistics.persistence_entropy(diagram)
    assert entropy.dtype == diagram.dtype
    assert torch.isfinite(entropy)
    betti = statistics.betti_curve(diagram, num_samples=5)
    assert betti.shape == (5,)
    assert betti.dtype == diagram.dtype
    with pytest.raises(ValueError, match="finite and positive"):
        statistics.total_persistence(diagram, p=float("nan"))
    with pytest.raises(ValueError, match="base"):
        statistics.persistence_entropy(diagram, base=1.0)
    with pytest.raises(ValueError, match="min_persistence"):
        statistics.number_of_features(diagram, min_persistence=float("nan"))
    with pytest.raises(ValidationError, match="num_samples"):
        statistics.betti_curve(diagram, num_samples=0)
    with pytest.raises(ValueError, match="metric"):
        statistics.amplitude(diagram, metric="invalid")
    with pytest.raises(ValueError, match="births"):
        statistics.mean_persistence(torch.tensor([[float("nan"), 1.0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="deaths"):
        statistics.max_persistence(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="finite deaths"):
        statistics.total_persistence(torch.tensor([[0.0, float("inf")]], dtype=torch.float32))


@pytest.mark.torch
def test_torch_preprocessing_validates_diagrams_and_parameters() -> None:
    from pynerve.torch import preprocessing

    diagram = torch.tensor([[0.0, 1.0], [0.5, float("inf")]], dtype=torch.float64)
    handled = preprocessing.handle_infinite_deaths(diagram, strategy="max")

    assert handled.dtype == diagram.dtype
    assert torch.isfinite(handled[:, 1]).all()
    assert preprocessing.normalize_diagram(handled).dtype == diagram.dtype
    assert preprocessing.clean_diagram(diagram.unsqueeze(0)).dim() == 3
    with pytest.raises(ValueError, match="large_value_factor"):
        preprocessing.handle_infinite_deaths(
            torch.tensor([[2.0, float("inf")]], dtype=torch.float32),
            strategy="large_value",
            large_value_factor=1.0,
        )
    with pytest.raises(ValueError, match="finite deaths"):
        preprocessing.normalize_diagram(diagram)
    with pytest.raises(ValueError, match="method"):
        preprocessing.normalize_diagram(handled, method="invalid")
    with pytest.raises(ValueError, match="max_persistence"):
        preprocessing.threshold_diagram(handled, min_persistence=2.0, max_persistence=1.0)
    with pytest.raises(ValueError, match="strategy"):
        preprocessing.subsample_diagram(handled, max_features=1, strategy="invalid")
    with pytest.raises((ValueError, ValidationError), match="threshold"):
        preprocessing.remove_outliers(handled, threshold=float("nan"))
    with pytest.raises((ValueError, ValidationError), match="min_persistence"):
        preprocessing.threshold_diagram(handled, min_persistence=float("nan"))


@pytest.mark.torch
def test_torch_distances_validate_diagrams_and_parameters() -> None:
    from pynerve.torch import diagram_bottleneck, diagram_wasserstein

    diagram = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)

    w_dist = diagram_wasserstein(diagram, diagram)
    assert isinstance(w_dist, (torch.Tensor, float))
    b_dist = diagram_bottleneck(diagram, diagram)
    assert isinstance(b_dist, (torch.Tensor, float))
    with pytest.raises(ValueError, match="finite and positive"):
        diagram_wasserstein(diagram, diagram, p=float("nan"))
    with pytest.raises(ValueError, match="q must be positive"):
        diagram_wasserstein(diagram, diagram, q=float("nan"))
    with pytest.raises(ValueError, match="births"):
        diagram_wasserstein(
            torch.tensor([[float("nan"), 1.0]], dtype=torch.float32),
            diagram.to(torch.float32),
        )
    with pytest.raises(ValueError, match="deaths"):
        diagram_bottleneck(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32), diagram.to(torch.float32)
        )
    diagram_wasserstein(
        torch.tensor([[0.0, float("inf")]], dtype=torch.float32),
        diagram.to(torch.float32),
    )
    with pytest.raises(TypeError, match="floating-point"):
        diagram_bottleneck(torch.tensor([[0, 1]], dtype=torch.int64), diagram.to(torch.float32))


@pytest.mark.torch
def test_torch_persistence_diagram_validates_and_batches_metadata() -> None:
    from pynerve.torch import PersistenceDiagram, batch_diagrams

    tensor = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]], dtype=torch.float64)
    mask = torch.tensor([[False, True]])
    diagram = PersistenceDiagram(tensor, mask=mask, num_pairs=torch.tensor([[0, 1]]))
    other = PersistenceDiagram(
        torch.tensor([[[0.0, 2.0, 0.0]]], dtype=torch.float64),
        mask=torch.tensor([[True]]),
    )
    batched = batch_diagrams([diagram, other])

    assert batched.mask.shape == (2, 2)
    assert not bool(batched.mask[0, 0])
    assert not bool(batched.mask[1, 1])
    assert batched.num_pairs is None
    assert diagram.total_persistence().dtype == tensor.dtype
    with pytest.raises(ValueError, match="births"):
        PersistenceDiagram(torch.tensor([[[float("nan"), 1.0, 0.0]]], dtype=torch.float32))
    with pytest.raises(ValueError, match="deaths"):
        PersistenceDiagram(torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32))
    with pytest.raises(ValidationError, match="dimensions"):
        PersistenceDiagram(torch.tensor([[[0.0, 1.0, 0.5]]], dtype=torch.float32))
    with pytest.raises(ValueError, match="finite and positive"):
        diagram.total_persistence(p=float("nan"))
    with pytest.raises((ValueError, ValidationError), match="dim"):
        diagram.filter_by_dimension(-1)


@pytest.mark.torch
def test_torch_viz_data_validates_and_respects_masks() -> None:
    from pynerve.torch import PersistenceDiagram, viz

    tensor = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]], dtype=torch.float64)
    diagram = PersistenceDiagram(tensor, mask=torch.tensor([[False, True]]))

    scatter = viz.diagram_to_scatter_data(diagram)
    assert scatter["births"].shape == (1,)
    assert viz.diagram_to_histogram_data(diagram, num_bins=4)["bins"].shape == (5,)
    assert viz.diagram_to_heatmap_data(diagram, grid_size=4)["grid"].shape == (4, 4)
    assert viz.get_plot_limits(diagram, padding=0.1)[0] <= 0.0
    with pytest.raises(ValueError, match="births"):
        viz.diagram_to_scatter_data(torch.tensor([[float("nan"), 1.0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="finite deaths"):
        viz.diagram_to_heatmap_data(torch.tensor([[0.0, float("inf")]], dtype=torch.float32))
    with pytest.raises(ValueError, match="dimensions"):
        viz.diagram_to_scatter_data(torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float32))
    with pytest.raises(ValidationError, match="num_bins"):
        viz.diagram_to_histogram_data(tensor[0], num_bins=0)
    with pytest.raises(ValidationError, match="grid_size"):
        viz.diagram_to_heatmap_data(tensor[0], grid_size=0)
    with pytest.raises(ValueError, match="padding"):
        viz.get_plot_limits(tensor[0], padding=float("nan"))


@pytest.mark.torch
def test_torch_training_helpers_validate_scalars_and_linear_kernel() -> None:
    from pynerve.torch import PersistenceDiagram, training_utils
    from pynerve.torch._training_helpers import compute_kernel_similarity

    diagram = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float32)
    similarity = compute_kernel_similarity([diagram], [diagram], kernel="linear")
    assert similarity.shape == (1, 1)
    with pytest.raises(ValueError, match="sigma"):
        compute_kernel_similarity([diagram], [diagram], kernel="gaussian", sigma=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        training_utils.DiagramDistanceLoss(p=float("nan"))
    with pytest.raises(ValueError, match="target_complexity"):
        training_utils.TopologicalRegularization({"h0_count": float("nan")})
    with pytest.raises(ValueError, match="min_persistence_threshold"):
        training_utils.PersistenceCrossEntropy(min_persistence_threshold=float("nan"))
    with pytest.raises(ValueError, match="track_stats"):
        training_utils.DiagramMetric(track_stats=["unknown"])
    with pytest.raises(ValueError, match="target_complexity"):
        training_utils.TopologicalComplexityMetric(target_complexity=float("nan"))
    with pytest.raises(ValueError, match="target_complexity"):
        training_utils.TopologicalComplexityMetric(target_complexity=-1)
    with pytest.raises(ValidationError, match="dim"):
        training_utils.DiagramMetric(dim=1.5)
    with pytest.raises(TypeError, match="track_stats"):
        training_utils.DiagramMetric(track_stats="total")
    with pytest.raises(ValueError, match="min_delta"):
        training_utils.TopologicalEarlyStopping(min_delta=float("nan"))
    with pytest.raises(ValueError, match="target_complexity"):
        training_utils.TopologicalEarlyStopping(target_complexity=float("inf"))
    with pytest.raises(ValidationError, match="patience"):
        training_utils.TopologicalEarlyStopping(patience=True)
    with pytest.raises(ValidationError, match="log_every"):
        training_utils.DiagramVisualizationCallback(log_every=1.5)

    logits = torch.randn(2, 3)
    targets = torch.tensor([0, 1])
    diagrams = PersistenceDiagram(
        torch.tensor(
            [
                [[0.0, 1.0, 0.0]],
                [[0.0, 2.0, 0.0]],
            ],
            dtype=torch.float32,
        )
    )
    loss = training_utils.PersistenceCrossEntropy(reduction="none")(logits, targets, diagrams)
    assert loss.shape == (2,)


@pytest.mark.torch
def test_torch_data_helpers_validate_point_clouds_and_empty_diagram_batches() -> None:
    from pynerve.torch import data

    points_a = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    points_b = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    collated = data.collate_point_clouds([points_a, points_b], pad_value=0.0)
    assert collated.shape == (2, 2, 2)
    empty_diagrams = data.collate_diagrams([])
    assert empty_diagrams.batch_size == 0
    with pytest.raises(ValueError, match="finite coordinates"):
        data.collate_point_clouds([torch.tensor([[float("nan"), 0.0]], dtype=torch.float32)])
    with pytest.raises(ValueError, match="pad_value"):
        data.collate_point_clouds([points_a], pad_value=float("nan"))
    with pytest.raises(TypeError, match="point clouds"):
        data.collate_point_clouds([[0.0, 1.0]])
    with pytest.raises(TypeError, match="floating-point"):
        data.PointCloudDataset([torch.tensor([[0, 1]], dtype=torch.int64)])
    with pytest.raises(ValidationError, match="batch_size"):
        data.create_dataloader(data.PointCloudDataset([points_a]), batch_size=0)
    with pytest.raises(TypeError, match="num_workers"):
        data.create_dataloader(data.PointCloudDataset([points_a]), num_workers=1.5)


@pytest.mark.torch
def test_registered_ph_autograd_validates_filtration_and_preserves_dtype() -> None:
    pytest.importorskip("nerve_torch_internal")

    filtration = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, requires_grad=True)
    diagram = torch.ops.pynerve.ph_grad(filtration, 0)

    assert diagram.dtype == filtration.dtype
    assert diagram.device == filtration.device
    diagram[:, 0].sum().backward()
    assert filtration.grad is not None
    assert filtration.grad.dtype == filtration.dtype
    with pytest.raises(RuntimeError, match="max_dim=0"):
        torch.ops.pynerve.ph_compute(filtration.detach(), 1)
    with pytest.raises(RuntimeError, match="finite values"):
        torch.ops.pynerve.ph_grad(torch.tensor([0.0, float("nan")], dtype=torch.float32), 0)
    with pytest.raises(RuntimeError, match="floating-point"):
        torch.ops.pynerve.ph_compute(torch.tensor([0, 1], dtype=torch.int64), 0)


@pytest.mark.torch
def test_registered_boundary_matrix_persistence_reduces_input_matrix() -> None:
    pytest.importorskip("nerve_torch_internal")

    diagram = torch.ops.pynerve.ph_persistence(
        torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        torch.tensor([0.0, 0.0], dtype=torch.float32),
        0,
    )
    assert diagram.shape == (1, 2)


@pytest.mark.torch
def test_native_simplex_tree_validates_tensor_inputs() -> None:
    torch_ext = pytest.importorskip("nerve_torch_internal")

    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    tree = torch_ext.SimplexTree()
    tree.build_vr(points, 2.0, 2)
    assert tree.num_simplices() > 1
    with pytest.raises(RuntimeError, match="finite coordinates"):
        torch_ext.SimplexTree().build_vr(
            torch.tensor([[0.0, 0.0], [float("nan"), 0.0]], dtype=torch.float32),
            2.0,
            2,
        )
    with pytest.raises(RuntimeError, match="max_dim"):
        torch_ext.SimplexTree().build_witness(points, points[:2], 2.0, 3)
    with pytest.raises(RuntimeError, match="finite values"):
        torch_ext.SimplexTree().insert_batch(
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([float("nan")], dtype=torch.float32),
        )
    with pytest.raises(RuntimeError, match="strictly increasing"):
        torch_ext.SimplexTree().insert([1, 0], 0.0)
    with pytest.raises(RuntimeError, match="node_idx"):
        tree.get_vertices(10_000)


@pytest.mark.torch
def test_native_persistence_diagram_validates_state_and_scalars() -> None:
    torch_ext = pytest.importorskip("nerve_torch_internal")

    diagram = torch_ext.PersistenceDiagram(
        torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    )
    assert diagram.diagram().dtype == torch.float32
    assert diagram.threshold(0.5).num_pairs() == 1

    with pytest.raises(RuntimeError, match="finite deaths"):
        torch_ext.PersistenceDiagram(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        )
    with pytest.raises(RuntimeError, match="dimensions must be torch.long"):
        torch_ext.PersistenceDiagram(
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        )
    with pytest.raises(RuntimeError, match="sigma"):
        diagram.to_persistence_image(8, float("nan"))
    with pytest.raises(RuntimeError, match="min_persistence"):
        diagram.threshold(float("nan"))
    with pytest.raises(RuntimeError, match="state_dict missing death_idx"):
        diagram.load_state_dict(diagram.state_dict()[:-1])


@pytest.mark.torch
def test_native_mapper_validates_inputs_and_config() -> None:
    torch_ext = pytest.importorskip("nerve_torch_internal")

    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    graph = torch_ext.quick_mapper(points, "pca_2d", 2, 0.25)
    assert graph.to_adjacency_matrix().shape == (len(graph.nodes), len(graph.nodes))

    with pytest.raises((ValueError, RuntimeError, ValidationError), match="finite values"):
        torch_ext.quick_mapper(
            torch.tensor([[0.0, 0.0], [float("nan"), 0.0]], dtype=torch.float32),
            "pca_2d",
            2,
            0.25,
        )
    with pytest.raises((ValueError, RuntimeError, ValidationError), match="cover_overlap"):
        torch_ext.quick_mapper(points, "pca_2d", 2, float("nan"))
    with pytest.raises((ValueError, RuntimeError, ValidationError), match="dbscan_eps"):
        config = torch_ext.MapperConfig()
        config.dbscan_eps = float("nan")
        torch_ext.Mapper(config)
    with pytest.raises((ValueError, RuntimeError, ValidationError), match="non-empty"):
        torch_ext.filter_pca(torch.empty((0, 2), dtype=torch.float32), 1)


@pytest.mark.torch
def test_native_functional_helpers_validate_scalars() -> None:
    torch_ext = pytest.importorskip("nerve_torch_internal")

    diagram = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    assert torch_ext.ml_persistence_image(diagram, 4, 4, 0.1).shape == (4, 4)

    with pytest.raises(RuntimeError, match="positive"):
        torch_ext.ml_persistence_image(diagram, 0, 0, 0.1)
    with pytest.raises(RuntimeError, match="finite"):
        torch_ext.ml_persistence_image(diagram, 4, 4, float("nan"))
    with pytest.raises(RuntimeError, match="k"):
        torch_ext.ml_persistence_landscape(diagram, 0, 8)
    with pytest.raises((RuntimeError, ValueError), match="num_samples"):
        torch_ext.ml_betti_curve(diagram, 0, -1)


@pytest.mark.torch
def test_native_ml_statistics_validate_diagrams_and_scalars() -> None:
    torch_ext = pytest.importorskip("nerve_torch_internal")

    diagram = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    assert torch_ext.ml_total_persistence(diagram, -1, 1.0) == pytest.approx(1.0)
    assert torch_ext.ml_betti_curve(diagram, 4, -1).device == diagram.device
    assert torch.isfinite(torch_ext.ml_extract_features(diagram, [0, 1])).all()

    with pytest.raises(ValueError, match="p"):
        torch_ext.ml_total_persistence(diagram, -1, float("nan"))
    with pytest.raises(ValueError, match="base"):
        torch_ext.ml_persistence_entropy(diagram, -1, 1.0)
    with pytest.raises(ValueError, match="min_persistence"):
        torch_ext.ml_number_of_features(diagram, -1, float("nan"))
    with pytest.raises(ValueError, match="num_samples"):
        torch_ext.ml_betti_curve(diagram, 0, -1)
    with pytest.raises(ValueError, match="amplitude metric"):
        torch_ext.ml_amplitude(diagram, "invalid", 1.0, -1)
    with pytest.raises(ValueError, match="finite deaths"):
        torch_ext.ml_total_persistence(torch.tensor([[1.0, 0.0]], dtype=torch.float32), -1, 1.0)


@pytest.mark.torch
def test_native_vectorization_and_kernels_reject_negative_infinite_deaths() -> None:
    torch_ext = pytest.importorskip("nerve_torch_internal")

    bad = torch.tensor([[0.0, float("-inf")]], dtype=torch.float32)
    good = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    with pytest.raises((RuntimeError, ValueError), match="positive infinity"):
        torch_ext.ml_persistence_image(bad, 4, 4, 0.1, 0.0, 0.0, 0.0, 0.0, "constant")
    with pytest.raises((RuntimeError, ValueError), match="positive infinity"):
        torch_ext.ml_gaussian_kernel(bad, good, 0.1, "euclidean")
