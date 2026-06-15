# Input Validation Rules

All public API functions validate inputs against these rules:

### Point cloud validation

The point cloud must have exactly 2 dimensions (`points.ndim == 2`), otherwise an `InvalidArgumentError` is raised. At least 2 points are required (`points.shape[0] >= 2`), or a `ShapeMismatchError` is raised. Each dimension must have at least 1 element (`points.shape[1] >= 1`), or an `InvalidArgumentError` is raised. All values must be finite -- NaN or infinity triggers a `NumericalError`. The dtype must be float16, float32, or float64; integer or complex types raise an `InvalidArgumentError`.

### Diagram validation

The diagram must have exactly 2 dimensions (`diagram.ndim == 2`), otherwise an `InvalidArgumentError` is raised. At least 2 columns are required (`diagram.shape[1] >= 2`), or a `ShapeMismatchError` is raised. All birth values must be finite -- NaN in the birth column triggers a `NumericalError`. For finite deaths, the death value must be greater than or equal to the birth value; if death is less than birth, an `InvalidArgumentError` is raised.

### Options validation

The `max_dim` must be at least 0 (negative values raise `InvalidArgumentError`) and at most 5 (exceeding raises `DimensionError`, though this limit is configurable). The `max_radius` must be positive; a non-positive value raises `InvalidArgumentError`. The thread count must be at least 0; negative values raise `InvalidArgumentError`. The `error_tolerance` must be at least 0; a negative tolerance raises `InvalidArgumentError`. For persistence images, `sigma` must be positive; a non-positive value raises `InvalidArgumentError`. The `resolution` must be at least 1; zero or negative values raise `InvalidArgumentError`.


[Back to Correctness Index](index.md)
