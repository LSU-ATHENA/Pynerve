"""Python API unit tests -- no C++ build required.

These tests cover the parts of Nerve that don't depend on ``nerve_internal``
(the C++ extension). Run with: ``pytest tests/python/test_python_api.py -v``
"""

from __future__ import annotations

import pytest
from pynerve.exceptions import (
    AllocationError,
    BackendRequiredError,
    BettiError,
    BudgetExceededError,
    ConvergenceError,
    DeterminismError,
    DeviceError,
    DimensionError,
    DtypeError,
    GPUError,
    GPULaunchError,
    GPUMemoryError,
    InvalidArgumentError,
    InvalidSimplexError,
    NerveError,
    NerveIOError,
    NerveMemoryError,
    NUMAError,
    NumericalError,
    NumericalInstabilityError,
    OutOfMemoryError,
    PersistenceError,
    PrecisionError,
    ShapeError,
    ShapeMismatchError,
    TypeMismatchError,
    ValidationError,
)


class TestExceptionHierarchy:
    def test_all_exceptions_are_nerve_errors(self):
        """Every Nerve exception should be catchable as NerveError."""
        exceptions = [
            NerveError("test"),
            ValidationError("test"),
            ShapeError("test"),
            DtypeError("test"),
            DeviceError("test"),
            BackendRequiredError("test"),
            PersistenceError("test"),
            ShapeMismatchError("test"),
            DimensionError("test"),
            TypeMismatchError("test"),
            InvalidSimplexError("test"),
            GPUError("test"),
            GPUMemoryError("test"),
            GPULaunchError("test"),
            NerveMemoryError("test"),
            OutOfMemoryError("test"),
            AllocationError("test"),
            NumericalError("test"),
            ConvergenceError("test"),
            PrecisionError("test"),
            NumericalInstabilityError("test"),
            InvalidArgumentError("test"),
            BudgetExceededError("test"),
            NerveIOError("test"),
            DeterminismError("test"),
            NUMAError("test"),
            BettiError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, NerveError), f"{type(exc).__name__} not a NerveError"

    def test_nerve_error_formats_message_with_code(self):
        exc = NerveError("something went wrong")
        assert "[NerveError]" in str(exc), f"expected '[NerveError]' in {str(exc)!r}"
        assert "something went wrong" in str(exc), (
            f"expected 'something went wrong' in {str(exc)!r}"
        )
        assert "code=0x" in str(exc), f"expected 'code=0x' in {str(exc)!r}"
        assert "category=" in str(exc), f"expected 'category=' in {str(exc)!r}"

    def test_validation_error_has_structured_metadata(self):
        exc = ValidationError("bad input", parameter="x", expected="int", actual="str")
        assert exc.parameter == "x", f"expected 'x', got {exc.parameter!r}"
        assert exc.expected == "int", f"expected 'int', got {exc.expected!r}"
        assert exc.actual == "str", f"expected 'str', got {exc.actual!r}"

    def test_shape_error_has_shape_metadata(self):
        exc = ShapeError(
            "bad shape", expected_shape=(10, 3), actual_shape=(5, 3), expected_ndim=2, actual_ndim=2
        )
        assert exc.expected_shape == (10, 3), f"expected (10, 3), got {exc.expected_shape}"
        assert exc.actual_shape == (5, 3), f"expected (5, 3), got {exc.actual_shape}"

    def test_dtype_error_has_dtype_metadata(self):
        exc = DtypeError("bad dtype", expected_dtypes=["float32", "float64"], actual_dtype="int32")
        assert "float32" in exc.expected_dtypes, f"expected 'float32' in {exc.expected_dtypes}"
        assert exc.actual_dtype == "int32", f"expected 'int32', got {exc.actual_dtype!r}"

    def test_device_error_has_device_metadata(self):
        exc = DeviceError("no gpu", requested_device="cuda:0", available_devices=["cpu"])
        assert exc.requested_device == "cuda:0", f"expected 'cuda:0', got {exc.requested_device!r}"
        assert exc.available_devices == ["cpu"], f"expected ['cpu'], got {exc.available_devices}"

    def test_backend_required_error_has_hint(self):
        exc = BackendRequiredError(
            "need torch", backend="torch", installation_hint="pip install torch"
        )
        assert exc.backend == "torch", f"expected 'torch', got {exc.backend!r}"
        assert exc.installation_hint is not None, "installation_hint should not be None"

    def test_persistence_error_has_backend_and_operation(self):
        exc = PersistenceError("aborted", backend="ph4", operation="reduce")
        assert exc.backend == "ph4", f"expected 'ph4', got {exc.backend!r}"
        assert exc.operation == "reduce", f"expected 'reduce', got {exc.operation!r}"

    def test_cpp_facing_classes_have_error_code_class_attr(self):
        assert ShapeMismatchError.error_code is not None, "ShapeMismatchError.error_code is None"
        assert GPUError.error_code is not None, "GPUError.error_code is None"
        assert NerveMemoryError.error_code is not None, "NerveMemoryError.error_code is None"
        assert NumericalError.error_code is not None, "NumericalError.error_code is None"
        assert NerveIOError.error_code is not None, "NerveIOError.error_code is None"

    def test_error_code_differentiation(self):
        assert GPUError.error_code != NerveMemoryError.error_code, (
            "GPUError and NerveMemoryError share the same error_code"
        )
        assert GPUMemoryError.error_code == GPUError.error_code, (
            "GPUMemoryError.error_code should equal GPUError.error_code"
        )
        assert GPULaunchError.error_code != GPUError.error_code, (
            "GPULaunchError and GPUError share the same error_code"
        )

    def test_nerves_error_repr(self):
        exc = NerveError("test")
        assert "NerveError(" in repr(exc), f"expected 'NerveError(' in {repr(exc)!r}"
        assert "code=0x" in repr(exc), f"expected 'code=0x' in {repr(exc)!r}"
        assert "category=" in repr(exc), f"expected 'category=' in {repr(exc)!r}"


class TestErrorCodes:
    def test_error_codes_importable_from_nerve(self):
        import pynerve

        assert pynerve.UNKNOWN == 0xFFFFFFFF, f"expected 0xFFFFFFFF, got {pynerve.UNKNOWN}"
        assert pynerve.SUCCESS == 0x00000000, f"expected 0x00000000, got {pynerve.SUCCESS}"

    def test_error_category_is_enum(self):
        from pynerve import ErrorCategory

        assert ErrorCategory.ALGORITHMIC == 6, f"expected 6, got {ErrorCategory.ALGORITHMIC}"
        assert int(ErrorCategory.GPU_COMPUTE) == 2, (
            f"expected 2, got {int(ErrorCategory.GPU_COMPUTE)}"
        )
        assert isinstance(ErrorCategory.SUCCESS, int), "ErrorCategory.SUCCESS is not an int"

    def test_error_severity_is_enum(self):
        from pynerve import ErrorSeverity

        assert ErrorSeverity.ERROR == 2, f"expected 2, got {ErrorSeverity.ERROR}"
        assert ErrorSeverity.CRITICAL == 3, f"expected 3, got {ErrorSeverity.CRITICAL}"


class TestPersistenceOptions:
    def test_default_values(self):
        from pynerve import PersistenceBackend, PersistenceMode, PersistenceOptions

        opts = PersistenceOptions()
        assert opts.mode == PersistenceMode.EXACT, f"expected EXACT, got {opts.mode}"
        assert opts.backend == PersistenceBackend.CPU_ADAPTIVE_ACCELERATION, (
            f"expected CPU_ADAPTIVE_ACCELERATION, got {opts.backend}"
        )
        assert opts.max_dim == 2, f"expected 2, got {opts.max_dim}"
        assert opts.max_radius is None, f"expected None, got {opts.max_radius}"
        assert opts.threads == 0, f"expected 0, got {opts.threads}"
        assert opts.error_tolerance == 0.0, f"expected 0.0, got {opts.error_tolerance}"

    def test_override_at_construction(self):
        from pynerve import PersistenceMode, PersistenceOptions

        opts = PersistenceOptions(max_dim=3, mode=PersistenceMode.APPROX)
        assert opts.max_dim == 3, f"expected 3, got {opts.max_dim}"
        assert opts.mode == PersistenceMode.APPROX, f"expected APPROX, got {opts.mode}"

    def test_frozen_prevents_mutation(self):
        from dataclasses import FrozenInstanceError

        from pynerve import PersistenceOptions

        opts = PersistenceOptions(max_dim=2)
        with pytest.raises(FrozenInstanceError):
            opts.max_dim = 5

    def test_replace_creates_new_instance(self):
        from dataclasses import replace

        from pynerve import PersistenceOptions

        opts = PersistenceOptions(max_dim=2, max_radius=1.0)
        modified = replace(opts, max_dim=3)
        assert modified.max_dim == 3, f"expected 3, got {modified.max_dim}"
        assert modified.max_radius == 1.0, f"expected 1.0, got {modified.max_radius}"
        assert opts.max_dim == 2, f"expected original max_dim 2, got {opts.max_dim}"


class TestPH5PH6Metrics:
    def test_default_values(self):
        from pynerve import PH5PH6Metrics

        m = PH5PH6Metrics()
        assert m.computation_time_ms == 0.0, f"expected 0.0, got {m.computation_time_ms}"
        assert m.peak_memory_bytes == 0, f"expected 0, got {m.peak_memory_bytes}"
        assert m.passed_stability_checks is False, (
            f"expected False, got {m.passed_stability_checks}"
        )
        assert m.checksum_validation_passed is False, (
            f"expected False, got {m.checksum_validation_passed}"
        )

    def test_instance_independence(self):
        from pynerve import PH5PH6Metrics

        a = PH5PH6Metrics(computation_time_ms=1.0)
        b = PH5PH6Metrics(computation_time_ms=2.0)
        assert a.computation_time_ms == 1.0, f"expected 1.0, got {a.computation_time_ms}"
        assert b.computation_time_ms == 2.0, f"expected 2.0, got {b.computation_time_ms}"


class TestPH5PH6Config:
    def test_defaults(self):
        from pynerve import PH5PH6Config

        cfg = PH5PH6Config()
        assert cfg.numerical_tolerance == 1e-9, f"expected 1e-9, got {cfg.numerical_tolerance}"
        assert cfg.max_iterations == 1000, f"expected 1000, got {cfg.max_iterations}"

    def test_frozen(self):
        from dataclasses import FrozenInstanceError

        from pynerve import PH5PH6Config

        cfg = PH5PH6Config()
        with pytest.raises(FrozenInstanceError):
            cfg.numerical_tolerance = 1e-6


class TestValidation:
    def test_nonempty_string_validates(self):
        from pynerve._validation import validate_nonempty_string

        assert validate_nonempty_string("hello", "test") == "hello", (
            "validate_nonempty_string should return the input"
        )
        with pytest.raises(ValidationError):
            validate_nonempty_string("", "test")

    def test_positive_finite_validates(self):
        from pynerve._validation import validate_positive_finite

        assert validate_positive_finite(1.0, "test") == 1.0, (
            "validate_positive_finite should return the input"
        )
        with pytest.raises(ValidationError):
            validate_positive_finite(-1.0, "test")
        with pytest.raises(ValidationError):
            validate_positive_finite(float("nan"), "test")


class TestAsyncAPI:
    def test_public_exports(self):
        from pynerve.async_api import (
            compute_persistence_async,
            load_diagrams_async,
            stream_persistence,
        )

        assert callable(compute_persistence_async), "compute_persistence_async is not callable"
        assert callable(load_diagrams_async), "load_diagrams_async is not callable"
        assert callable(stream_persistence), "stream_persistence is not callable"


class TestMakefileTargets:
    def test_required_makefile_targets(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        makefile = root / "Makefile"
        if not makefile.exists():
            pytest.skip("Makefile not found (running from installed wheel)")
        content = makefile.read_text()
        for target in (
            "help",
            "install",
            "build",
            "test",
            "test-quick",
            "test-coverage",
            "lint",
            "format",
            "format-fix",
            "typecheck",
            "quality",
            "pyi",
            "ci-local",
            "clean",
            "pre-commit",
        ):
            assert f"{target}:" in content, f"Makefile must have a '{target}' target"
