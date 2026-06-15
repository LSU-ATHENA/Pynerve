"""Shared validation helpers to reduce duplication across modules.

This module is a re-export hub. The actual implementations live in the
``pynerve._validation`` subpackage:
  - ``_helpers``   --  error-raising helper functions
  - ``_scalars``   --  scalar parameter validators
  - ``_geometric``  --  tensor and geometric validators
"""

from pynerve._validation._geometric import (
    validate_device_spec,
    validate_diagram,
    validate_diagram_array,
    validate_finite_deaths,
    validate_finite_tensor,
    validate_floating_tensor,
    validate_points,
    validate_shape,
    validate_shape_tuple,
)
from pynerve._validation._scalars import (
    parse_nonnegative_int,
    validate_bool,
    validate_device_id,
    validate_finite_scalar,
    validate_max_dist,
    validate_max_radius,
    validate_nonempty_string,
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_optional_finite,
    validate_optional_nonnegative_int,
    validate_optional_positive_int,
    validate_optional_string,
    validate_positive_finite,
    validate_positive_int,
    validate_probability,
    validate_seed,
    validate_string_list,
)

__all__ = [
    "validate_positive_int",
    "validate_nonnegative_int",
    "validate_positive_finite",
    "validate_nonnegative_finite",
    "validate_nonempty_string",
    "validate_bool",
    "validate_finite_deaths",
    "validate_finite_tensor",
    "validate_diagram",
    "validate_probability",
    "validate_points",
    "validate_finite_scalar",
    "validate_floating_tensor",
    "validate_device_id",
    "validate_optional_positive_int",
    "validate_optional_nonnegative_int",
    "validate_optional_finite",
    "validate_seed",
    "validate_shape",
    "validate_optional_string",
    "validate_string_list",
    "validate_shape_tuple",
    "validate_max_dist",
    "validate_max_radius",
    "validate_diagram_array",
    "parse_nonnegative_int",
    "validate_device_spec",
]
