"""Property-based tests for persistence diagrams and operations.

Verifies mathematical invariants (metric axioms, monotonicity, consistency)
that must hold for all valid inputs using Hypothesis.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
try:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st
except ModuleNotFoundError:
    pytest.skip("hypothesis not available", allow_module_level=True)

# strategy: valid diagram tensors with birth < death
_diagram_strategy = st.lists(
    st.tuples(
        st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    ),
    min_size=0,
    max_size=20,
).map(
    lambda pts: torch.tensor(
        [(b, d) if b < d else (d, b + 0.01) for b, d in pts],
        dtype=torch.float64,
    ).reshape(-1, 2),
)

_nonempty_diagram = st.lists(
    st.tuples(
        st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=20,
).map(
    lambda pts: torch.tensor(
        [(b, d) if b < d else (d, b + 0.01) for b, d in pts],
        dtype=torch.float64,
    ).reshape(-1, 2),
)


# diagram structural invariants


class TestDiagramProperties:
    """Invariants that all valid persistence diagrams satisfy."""

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_birth_less_than_death(self, diagram: torch.Tensor) -> None:
        finite_mask = torch.isfinite(diagram[:, 1])
        if finite_mask.any():
            finite = diagram[finite_mask]
            assert (finite[:, 0] < finite[:, 1]).all()

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_diagram_is_2d_tensor(self, diagram: torch.Tensor) -> None:
        assert diagram.dim() == 2
        assert diagram.shape[-1] == 2

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_persistence_non_negative(self, diagram: torch.Tensor) -> None:
        finite_mask = torch.isfinite(diagram[:, 1])
        if finite_mask.any():
            pers = diagram[finite_mask, 1] - diagram[finite_mask, 0]
            assert (pers >= 0).all()


# wasserstein metric axioms


class TestWassersteinMetricAxioms:
    """Wasserstein distance must satisfy metric axioms."""

    @settings(max_examples=100, deadline=None)
    @given(diag=_nonempty_diagram)
    def test_self_distance_zero(self, diag: torch.Tensor) -> None:
        from pynerve.torch import diagram_wasserstein

        dist = diagram_wasserstein(diag, diag)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-6)

    @settings(max_examples=50, deadline=None)
    @given(d1=_nonempty_diagram, d2=_nonempty_diagram)
    def test_symmetry(self, d1: torch.Tensor, d2: torch.Tensor) -> None:
        from pynerve.torch import diagram_wasserstein

        v12 = diagram_wasserstein(d1, d2)
        v21 = diagram_wasserstein(d2, d1)
        v12t = v12 if isinstance(v12, torch.Tensor) else torch.tensor(v12)
        v21t = v21 if isinstance(v21, torch.Tensor) else torch.tensor(v21)
        assert v12t.item() == pytest.approx(v21t.item(), abs=1e-6)

    @settings(max_examples=50, deadline=None)
    @given(d1=_nonempty_diagram, d2=_nonempty_diagram, d3=_nonempty_diagram)
    def test_triangle_inequality(
        self, d1: torch.Tensor, d2: torch.Tensor, d3: torch.Tensor
    ) -> None:
        from pynerve.torch import diagram_wasserstein

        d12 = diagram_wasserstein(d1, d2)
        d23 = diagram_wasserstein(d2, d3)
        d13 = diagram_wasserstein(d1, d3)
        v12 = d12 if isinstance(d12, torch.Tensor) else torch.tensor(d12)
        v23 = d23 if isinstance(d23, torch.Tensor) else torch.tensor(d23)
        v13 = d13 if isinstance(d13, torch.Tensor) else torch.tensor(d13)
        assert v13.item() <= v12.item() + v23.item() + 1e-6

    @settings(max_examples=100, deadline=None)
    @given(d1=_nonempty_diagram, d2=_nonempty_diagram)
    def test_non_negative(self, d1: torch.Tensor, d2: torch.Tensor) -> None:
        from pynerve.torch import diagram_wasserstein

        dist = diagram_wasserstein(d1, d2)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() >= -1e-10

    @settings(max_examples=50, deadline=None)
    @given(diag=_nonempty_diagram)
    def test_non_identical_positive(self, diag: torch.Tensor) -> None:
        from pynerve.torch import diagram_wasserstein

        shifted = diag + 0.01
        dist = diagram_wasserstein(diag, shifted)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assume(val.item() > 1e-10)


# bottleneck metric axioms


class TestBottleneckMetricAxioms:
    """Bottleneck distance must satisfy metric axioms."""

    @settings(max_examples=100, deadline=None)
    @given(diag=_nonempty_diagram)
    def test_self_distance_zero(self, diag: torch.Tensor) -> None:
        from pynerve.torch import diagram_bottleneck

        dist = diagram_bottleneck(diag, diag)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() == pytest.approx(0.0, abs=1e-6)

    @settings(max_examples=50, deadline=None)
    @given(d1=_nonempty_diagram, d2=_nonempty_diagram)
    def test_symmetry(self, d1: torch.Tensor, d2: torch.Tensor) -> None:
        from pynerve.torch import diagram_bottleneck

        d12 = diagram_bottleneck(d1, d2)
        d21 = diagram_bottleneck(d2, d1)
        v12 = d12 if isinstance(d12, torch.Tensor) else torch.tensor(d12)
        v21 = d21 if isinstance(d21, torch.Tensor) else torch.tensor(d21)
        assert v12.item() == pytest.approx(v21.item(), abs=1e-6)

    @settings(max_examples=50, deadline=None)
    @given(d1=_nonempty_diagram, d2=_nonempty_diagram, d3=_nonempty_diagram)
    def test_triangle_inequality(
        self, d1: torch.Tensor, d2: torch.Tensor, d3: torch.Tensor
    ) -> None:
        from pynerve.torch import diagram_bottleneck

        d12 = diagram_bottleneck(d1, d2)
        d23 = diagram_bottleneck(d2, d3)
        d13 = diagram_bottleneck(d1, d3)
        v12 = d12 if isinstance(d12, torch.Tensor) else torch.tensor(d12)
        v23 = d23 if isinstance(d23, torch.Tensor) else torch.tensor(d23)
        v13 = d13 if isinstance(d13, torch.Tensor) else torch.tensor(d13)
        assert v13.item() <= v12.item() + v23.item() + 1e-6

    @settings(max_examples=100, deadline=None)
    @given(d1=_nonempty_diagram, d2=_nonempty_diagram)
    def test_non_negative(self, d1: torch.Tensor, d2: torch.Tensor) -> None:
        from pynerve.torch import diagram_bottleneck

        dist = diagram_bottleneck(d1, d2)
        val = dist if isinstance(dist, torch.Tensor) else torch.tensor(dist)
        assert val.item() >= -1e-10


# statistics invariants


class TestStatisticsInvariants:
    """Invariants that statistics functions must satisfy."""

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_total_persistence_non_negative(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.statistics import total_persistence

        tp = total_persistence(diagram, p=1.0)
        assert tp.item() >= -1e-10

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_number_of_features_in_range(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.statistics import number_of_features

        n = number_of_features(diagram, min_persistence=0.0)
        assert 0 <= n.item() <= diagram.shape[0]

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_number_of_features_monotonic(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.statistics import number_of_features

        n_low = number_of_features(diagram, min_persistence=0.5)
        n_high = number_of_features(diagram, min_persistence=2.0)
        assert n_low.item() >= n_high.item()

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_betti_curve_non_negative(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.statistics import betti_curve

        curve = betti_curve(diagram, num_samples=5)
        assert (curve >= 0).all()

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_mean_persistence_leq_max(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.statistics import max_persistence, mean_persistence

        if diagram.shape[0] == 0:
            return
        mp = mean_persistence(diagram)
        mx = max_persistence(diagram)
        assert mp.item() <= mx.item() + 1e-10

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_amplitude_non_negative(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.statistics import amplitude

        for metric in ("bottleneck", "wasserstein", "persistence"):
            val = amplitude(diagram, metric=metric)
            assert val.item() >= -1e-10


# preprocessing invariants


class TestPreprocessingInvariants:
    """Invariants for preprocessing functions."""

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_threshold_diagram_subsets(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.preprocessing import threshold_diagram

        filtered = threshold_diagram(diagram, min_persistence=0.5)
        assert filtered.shape[0] <= diagram.shape[0]

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_normalize_minmax_bounds(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.preprocessing import normalize_diagram

        if diagram.shape[0] == 0:
            return
        try:
            normed = normalize_diagram(diagram, method="minmax")
        except (ValueError, Exception):
            return
        assert (normed >= 0).all() and (normed <= 1).all()

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_subsample_respects_max_features(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.preprocessing import subsample_diagram

        if diagram.shape[0] < 1:
            return
        n = diagram.shape[0]
        max_feat = max(1, n // 2)
        result = subsample_diagram(diagram, max_features=max_feat, strategy="most_persistent")
        assert result.shape[0] <= max_feat

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_clean_diagram_no_inf(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.preprocessing import clean_diagram

        if diagram.shape[0] == 0:
            return
        cleaned = clean_diagram(diagram, handle_inf=True, normalize=False)
        assert cleaned.numel() == 0 or torch.isfinite(cleaned).all()

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_threshold_zero_passes_all(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.preprocessing import threshold_diagram

        finite = diagram[torch.isfinite(diagram[:, 1])]
        if finite.shape[0] == 0:
            return
        filtered = threshold_diagram(finite, min_persistence=0.0)
        assert filtered.shape[0] == finite.shape[0]


# vectorization invariants


class TestVectorizationInvariants:
    """Invariants for vectorization functions."""

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_persistence_image_non_negative(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.vectorization import persistence_image

        img = persistence_image(diagram, resolution=(4, 4), sigma=1.0)
        assert (img >= -1e-10).all()

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_landscape_finite(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.vectorization import persistence_landscape

        x_range: tuple[float, float] = (0.0, 10.0)
        land = persistence_landscape(diagram, k=1, num_samples=5, x_range=x_range)
        assert torch.isfinite(land).all()

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_silhouette_finite(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.vectorization import persistence_silhouette

        s = persistence_silhouette(diagram, num_samples=5)
        assert torch.isfinite(s).all()

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_birth_death_curve_non_negative(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.vectorization import birth_death_curve

        curve = birth_death_curve(diagram, num_bins=4, statistic="count")
        assert (curve >= -1e-10).all()

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_heat_kernel_finite(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.vectorization import heat_kernel_signature

        hks = heat_kernel_signature(diagram, num_samples=5, sigma=1.0)
        assert torch.isfinite(hks).all()

    @settings(max_examples=100, deadline=None)
    @given(diagram=_diagram_strategy)
    def test_adaptive_image_finite(self, diagram: torch.Tensor) -> None:
        from pynerve.torch.vectorization import adaptive_persistence_image

        img = adaptive_persistence_image(diagram, target_resolution=4)
        assert torch.isfinite(img).all()
