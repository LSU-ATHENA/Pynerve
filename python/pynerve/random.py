"""Explicit pseudo-random key management."""

from __future__ import annotations

import hashlib
import secrets
import sys
from collections.abc import Iterator
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from ._validation import (
    validate_finite_scalar,
    validate_nonnegative_int,
    validate_positive_int,
)
from ._validation import (
    validate_shape as _validate_shape,
)
from .exceptions import InvalidArgumentError

_NUMPY_SEED_MODULUS = 2**128


def _validate_integral(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise InvalidArgumentError(f"{name} must be an integer", parameter=name)
    return int(value)


def _validate_shape_or_none(
    shape: int | tuple[int, ...] | None, name: str = "size"
) -> tuple[int, ...] | None:
    if shape is None:
        return None
    return _validate_shape(shape, name=name)


def _normalize_seed(seed: int) -> int:
    return _validate_integral(seed, "seed") % _NUMPY_SEED_MODULUS


def _unseeded_seed() -> int:
    return secrets.randbits(64)


@dataclass(frozen=True)
class PRNGKey:
    """A pseudo-random number generator key (JAX-style).

    Wraps a seed and an auto-incrementing counter to produce deterministic
    pseudo-random draws. Each call to a sampling method advances the
    counter transparently so repeated calls yield different results.
    """

    seed: int
    counter: int = 0

    def __post_init__(self) -> None:
        """Validate fields after frozen dataclass initialisation.

        :raises InvalidArgumentError: If *seed* or *counter* are not
            integers, or if *counter* is negative.
        """
        object.__setattr__(self, "seed", _validate_integral(self.seed, "seed"))
        object.__setattr__(
            self,
            "counter",
            _validate_integral(self.counter, "counter"),
        )
        if self.counter < 0:
            raise InvalidArgumentError("counter must be non-negative", parameter="counter")

    def __repr__(self) -> str:
        return f"PRNGKey(seed={self.seed}, counter={self.counter})"

    def split(self, n: int = 2) -> tuple[PRNGKey, ...]:
        """Split this key into *n* independent :class:`PRNGKey` instances.

        Each new key is derived deterministically from the parent seed,
        counter, and index via SHA-256 hashing.

        :param n: Number of child keys to produce (default 2).
        :returns: A tuple of ``(n)`` new :class:`PRNGKey` instances.
        :raises InvalidArgumentError: If *n* is not a positive integer.
        """
        n = validate_positive_int(n, "n")
        keys = []
        for i in range(n):
            new_seed = hashlib.sha256(f"{self.seed}:{self.counter}:{i}".encode()).hexdigest()
            seed_int = int(new_seed[:16], 16)
            keys.append(PRNGKey(seed_int, 0))

        return tuple(keys)

    def __iter__(self) -> Iterator[PRNGKey]:
        """Allow unpacking: key, subkey = key.split()"""
        return iter(self.split())

    def _as_output(self, data: np.ndarray, as_tensor: bool = False) -> np.ndarray | Any:
        if as_tensor and torch is not None:
            return torch.from_numpy(data)
        return data

    def normal(
        self,
        shape: int | tuple[int, ...] = (1,),
        dtype: np.dtype = np.float64,  # type: ignore[assignment]
        as_tensor: bool = False,
    ) -> np.ndarray | Any:
        """Sample from the standard normal distribution.

        :param shape: Output shape. Accepts an integer or tuple of
            dimensions.
        :param dtype: NumPy dtype of the output (default ``float64``).
        :param as_tensor: If ``True`` and PyTorch is installed, return a
            PyTorch tensor instead of a NumPy array.
        :returns: An array of samples drawn from ``Normal(0, 1)``.
        :raises InvalidArgumentError: If *shape* is invalid.
        """
        shape = _validate_shape(shape)
        rng = np.random.default_rng(_normalize_seed(self.seed + self.counter))
        return self._as_output(rng.normal(size=shape).astype(dtype), as_tensor)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: int | tuple[int, ...] = (1,),
        dtype: np.dtype = np.float64,  # type: ignore[assignment]
        as_tensor: bool = False,
    ) -> np.ndarray | Any:
        """Sample from a uniform distribution on ``[low, high)``.

        :param low: Lower bound (default 0.0).
        :param high: Upper bound (default 1.0).
        :param shape: Output shape.
        :param dtype: NumPy dtype of the output (default ``float64``).
        :param as_tensor: If ``True`` and PyTorch is installed, return a
            PyTorch tensor instead of a NumPy array.
        :returns: An array of uniform samples.
        :raises InvalidArgumentError: If bounds are invalid or *shape* is
            invalid.
        """
        low = validate_finite_scalar(low, "low")
        high = validate_finite_scalar(high, "high")
        if high < low:
            raise InvalidArgumentError(
                "high must be greater than or equal to low", parameter="high"
            )
        shape = _validate_shape(shape)
        rng = np.random.default_rng(_normalize_seed(self.seed + self.counter))
        return self._as_output(rng.uniform(low, high, size=shape).astype(dtype), as_tensor)

    def randint(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] = 1,
        as_tensor: bool = False,
    ) -> int | np.ndarray | Any:
        """Sample random integers from a discrete uniform distribution.

        When *high* is ``None``, samples are drawn from ``[0, low)``.

        :param low: Lowest integer to draw (inclusive). If *high* is
            ``None``, this is the exclusive upper bound.
        :param high: Exclusive upper bound. If ``None``, samples are drawn
            from ``[0, low)``.
        :param size: Output shape. An integer or tuple of dimensions.
        :param as_tensor: If ``True`` and PyTorch is installed, return a
            PyTorch tensor.
        :returns: An integer or array of random integers.
        :raises InvalidArgumentError: If bounds or *size* are invalid.
        """
        low = _validate_integral(low, "low")
        if high is None:
            if low <= 0:
                raise InvalidArgumentError(
                    "low must be positive when high is omitted", parameter="low"
                )
        else:
            high = _validate_integral(high, "high")
            if high <= low:
                raise InvalidArgumentError("high must be greater than low", parameter="high")
        resolved_size: tuple[int, ...] | None = _validate_shape_or_none(size, "size")
        rng = np.random.default_rng(_normalize_seed(self.seed + self.counter))
        result = rng.integers(low, high, size=resolved_size)
        if isinstance(result, (int, np.integer)):
            return int(result)
        return self._as_output(result, as_tensor)

    def choice(
        self,
        a: int | SequenceABC[Any],
        size: int | tuple[int, ...] = 1,
        replace: bool = True,
        p: np.ndarray | None = None,
        as_tensor: bool = False,
    ) -> np.ndarray | Any:
        """Draw a random sample from an array-like or integer range.

        :param a: A positive integer (treated as ``range(a)``) or a
            non-empty sequence to sample from.
        :param size: Output shape.
        :param replace: Whether to sample with replacement (default
            ``True``).
        :param p: Optional probability weights of shape ``(len(a),)``.
            Must be non-negative, finite, and sum to 1.
        :param as_tensor: If ``True`` and PyTorch is installed, return a
            PyTorch tensor.
        :returns: An array of sampled values.
        :raises InvalidArgumentError: If *a*, *replace*, or *p* are
            invalid.
        """
        if isinstance(a, Integral) and not isinstance(a, bool):
            a_int: int = validate_positive_int(int(a), "a")
            population_size = a_int
        elif isinstance(a, np.ndarray) or (
            isinstance(a, SequenceABC) and not isinstance(a, (str, bytes))
        ):
            if np.asarray(a).ndim == 0 or len(a) == 0:
                raise InvalidArgumentError("a must be non-empty", parameter="a")
            population_size = len(a)
        else:
            raise InvalidArgumentError(
                "a must be a positive integer or non-empty sequence", parameter="a"
            )
        if not isinstance(replace, bool):
            raise InvalidArgumentError("replace must be a boolean", parameter="replace")
        if p is not None:
            p = np.asarray(p, dtype=np.float64)
            if p.ndim != 1 or p.shape[0] != population_size:
                raise InvalidArgumentError("p must have shape (len(a),)", parameter="p")
            if not np.isfinite(p).all():
                raise InvalidArgumentError(
                    "p must contain only finite probabilities", parameter="p"
                )
            if (p < 0).any():
                raise InvalidArgumentError(
                    "p must contain non-negative probabilities", parameter="p"
                )
            if not np.isclose(p.sum(), 1.0):
                raise InvalidArgumentError("p probabilities must sum to 1", parameter="p")
        rng = np.random.default_rng(_normalize_seed(self.seed + self.counter))
        return self._as_output(rng.choice(a, size=size, replace=replace, p=p), as_tensor)

    def permutation(self, x: int | np.ndarray, as_tensor: bool = False) -> np.ndarray | Any:
        """Return a random permutation of an array or range.

        :param x: A non-negative integer (treated as ``range(x)``) or an
            array to permute.
        :param as_tensor: If ``True`` and PyTorch is installed, return a
            PyTorch tensor.
        :returns: A permuted array.
        :raises InvalidArgumentError: If *x* is a negative integer.
        """
        if isinstance(x, Integral) and not isinstance(x, bool):
            x = validate_nonnegative_int(x, "x")  # pyright: ignore[reportArgumentType]
        rng = np.random.default_rng(_normalize_seed(self.seed + self.counter))
        return self._as_output(rng.permutation(x), as_tensor)


_global_key: PRNGKey | None = None


def _seed_nerve_core(seed_value: int) -> None:
    pynerve_mod = sys.modules.get("pynerve")
    if pynerve_mod is not None:
        core = getattr(pynerve_mod, "_core", None)
        if core is not None and hasattr(core, "determinism_seed"):
            core.determinism_seed(seed_value)


def seed(seed_value: int | None = None) -> PRNGKey:
    """Set the global PRNG key (JAX-style primary seeding API).

    Returns the new global key so it can be stored for later use.

    See also :func:`manual_seed`, an alias with the same behavior
    provided for PyTorch compatibility.

    :param seed_value: Seed value. If ``None``, a cryptographically random
        seed is generated.
    :returns: The new global :class:`PRNGKey`.
    """
    resolved = _unseeded_seed() if seed_value is None else _validate_integral(seed_value, "seed")
    globals()["_global_key"] = PRNGKey(resolved)
    _seed_nerve_core(resolved)
    assert _global_key is not None
    return _global_key


def manual_seed(seed_value: int) -> PRNGKey:
    """Set the global PRNG key (PyTorch-compatible alias).

    Identical to :func:`seed`. Exists for users who prefer the
    PyTorch-style ``manual_seed`` naming.

    :param seed_value: Seed value.
    :returns: The new global :class:`PRNGKey`.
    """
    return seed(seed_value)


def key() -> PRNGKey:
    """Return the current global :class:`PRNGKey`.

    Automatically initialises a random key if none has been set yet.

    :returns: The global :class:`PRNGKey`.
    """
    if _global_key is None:
        globals()["_global_key"] = PRNGKey(_unseeded_seed())
    assert _global_key is not None
    return _global_key


def split(n: int = 2) -> tuple[PRNGKey, ...]:
    """Split the global :class:`PRNGKey` into *n* independent keys.

    The first child key becomes the new global key; all *n* keys are
    returned.

    :param n: Number of child keys to produce (default 2).
    :returns: A tuple of ``(n)`` :class:`PRNGKey` instances.
    """
    n = validate_positive_int(n, "n")
    k = key()
    keys = k.split(n)
    globals()["_global_key"] = keys[0]
    return keys


def next_key() -> PRNGKey:
    """Advance the global :class:`PRNGKey` and return the next key.

    Equivalent to ``split(2)[1]``. Discards the current global key and
    returns one of the two derived child keys.

    :returns: A new :class:`PRNGKey` derived from the previous global key.
    """
    return split(2)[1]


class ReproducibleContext:
    """Context manager that temporarily sets a fixed global PRNG key.

    On entry the current global key is saved and replaced with a
    ``PRNGKey(seed)``. On exit the original key is restored, even if an
    exception occurred.
    """

    def __init__(self, seed: int = 0) -> None:
        """Temporarily fix the global PRNG key.

        :param seed: Seed value to use inside the context (default 0).
        """
        self.seed: int = seed
        self._old_key: PRNGKey | None = None

    def __repr__(self) -> str:
        return f"ReproducibleContext(seed={self.seed})"

    def __enter__(self) -> PRNGKey:
        """Set the global key to ``PRNGKey(self.seed)`` and return it.

        :returns: The fixed :class:`PRNGKey` active inside the context.
        """
        self._old_key = _global_key
        globals()["_global_key"] = PRNGKey(self.seed)
        assert _global_key is not None
        return _global_key

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        globals()["_global_key"] = self._old_key


#: Alias for :class:`ReproducibleContext`.
reproducible = ReproducibleContext

__all__ = [
    "PRNGKey",
    "seed",
    "manual_seed",
    "key",
    "split",
    "next_key",
    "ReproducibleContext",
    "reproducible",
]
