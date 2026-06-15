from __future__ import annotations

import pytest
from pynerve.exceptions import DtypeError, ShapeError, ValidationError

try:
    import hypothesis
    from hypothesis import strategies
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent.
    hypothesis = None
    strategies = None


def test_error_hierarchies_validate_metadata() -> None:
    from pynerve._error_codes import (
        E21_NUM_NO_CONVERGE,
        E41_RESOURCE_LIMIT,
        E81_MATRIX_EMPTY,
        UNKNOWN,
    )
    from pynerve._error_translation import translate_cpp_exception
    from pynerve.exceptions import (
        BackendRequiredError,
        ConvergenceError,
        DeviceError,
        NerveError,
        NerveMemoryError,
        PersistenceError,
        ShapeMismatchError,
    )

    err1 = NerveError("ok", details={"key": "value"})
    assert err1.details["key"] == "value", f"expected 'value', got {err1.details['key']}"
    err2 = PersistenceError("failed", backend="cpu", operation="vr")
    assert err2.backend == "cpu", f"expected 'cpu', got {err2.backend}"
    err3 = DeviceError("missing", requested_device="cuda", available_devices=["cpu"])
    assert err3.available_devices == ["cpu"], f"expected ['cpu'], got {err3.available_devices}"
    err4 = ShapeError("bad", expected_shape=(2, 3), actual_shape=(2,), expected_ndim=2)
    assert err4.expected_ndim == 2, f"expected 2, got {err4.expected_ndim}"
    err5 = DtypeError("bad", expected_dtypes=["float32"], actual_dtype="int64")
    assert err5.actual_dtype == "int64", f"expected 'int64', got {err5.actual_dtype}"
    assert ConvergenceError("bad").error_code == E21_NUM_NO_CONVERGE, (
        f"expected {E21_NUM_NO_CONVERGE}, got {ConvergenceError('bad').error_code}"
    )
    assert NerveMemoryError("bad").error_code == E41_RESOURCE_LIMIT, (
        f"expected {E41_RESOURCE_LIMIT}, got {NerveMemoryError('bad').error_code}"
    )
    assert BackendRequiredError("missing", backend="torch_c").backend == "torch_c", (
        f"expected 'torch_c', got {BackendRequiredError('missing', backend='torch_c').backend}"
    )

    class _CppError(Exception):
        error_code = E81_MATRIX_EMPTY

    translated = translate_cpp_exception(_CppError("shape"))
    assert isinstance(translated, ShapeMismatchError), (
        f"expected ShapeMismatchError, got {type(translated).__name__}"
    )
    assert translated.error_code == ShapeMismatchError.error_code, (
        f"expected {ShapeMismatchError.error_code}, got {translated.error_code}"
    )

    class _BadCodeError(Exception):
        error_code = "bad"

    result = translate_cpp_exception(_BadCodeError("bad"))
    assert result.error_code == UNKNOWN, f"expected {UNKNOWN}, got {result.error_code}"

    with pytest.raises(TypeError, match="details"):
        NerveError("bad", details=object())
    with pytest.raises(ValidationError, match="available_devices"):
        DeviceError("bad", available_devices=[""])
    with pytest.raises((TypeError, ShapeError), match="expected_shape"):
        ShapeError("bad", expected_shape=(1.5,))
    with pytest.raises(ValidationError, match="expected_ndim"):
        ShapeError("bad", expected_ndim=-1)
    with pytest.raises((TypeError, ValidationError), match="expected_dtypes"):
        DtypeError("bad", expected_dtypes="float32")
    with pytest.raises(ValidationError, match="backend"):
        PersistenceError("bad", backend="")
        with pytest.raises((TypeError, ValidationError), match="requested_device"):
            DeviceError("bad", requested_device="")
    with pytest.raises(TypeError, match="cpp_exception"):
        translate_cpp_exception(object())
