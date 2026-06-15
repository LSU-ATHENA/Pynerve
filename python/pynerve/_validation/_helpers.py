"""Internal validation error helper functions."""

from __future__ import annotations

from typing import Any


def _validation_error(msg: str, param: str | None = None) -> None:
    from pynerve.exceptions import ValidationError  # noqa: PLC0415

    raise ValidationError(msg, parameter=param)


def _shape_error(msg: str, param: str | None = None, **kw: Any) -> None:
    from pynerve.exceptions import ShapeError  # noqa: PLC0415

    raise ShapeError(msg, parameter=param, **kw)


def _dtype_error(msg: str, param: str | None = None) -> None:
    from pynerve.exceptions import DtypeError  # noqa: PLC0415

    raise DtypeError(msg, parameter=param)
