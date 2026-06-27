from __future__ import annotations

import itertools

import pytest
import test_matrix

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


if hypothesis is not None and strategies is not None:

    @hypothesis.settings(max_examples=128, deadline=None)
    @hypothesis.given(
        shard_count=strategies.integers(min_value=1, max_value=32),
        shard_index=strategies.integers(min_value=0, max_value=31),
    )
    @pytest.mark.generated
    def test_shard_selection_property(shard_count: int, shard_index: int) -> None:
        shard_index %= shard_count
        sample_cases = list(itertools.islice(test_matrix.iter_cases(True), 2048))
        selected = test_matrix.select_cases(sample_cases, shard_index, shard_count)
        for case in selected:
            assert int(case.id[:8], 16) % shard_count == shard_index, (
                f"expected {shard_index}, got {int(case.id[:8], 16) % shard_count}"
            )
else:

    @pytest.mark.generated
    def test_shard_selection_property() -> None:
        pytest.skip("hypothesis is missing")


@pytest.mark.generated
def test_generated_matrix_regression_signature() -> None:
    cases = list(test_matrix.iter_cases(True))
    assert len(cases) == 92160, f"expected 92160, got {len(cases)}"
    assert cases[0].id == "5d695feb0d2d", f'expected "5d695feb0d2d", got {cases[0].id!r}'
    assert cases[-1].id == "b1b83a0017d8", f'expected "b1b83a0017d8", got {cases[-1].id!r}'
    assert sum(1 for case in cases if case.autograd == "backward") == 46080, (
        f"expected 46080, got {sum(1 for case in cases if case.autograd == 'backward')}"
    )
