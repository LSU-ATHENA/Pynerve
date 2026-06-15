"""Pickle-file memoization with disk persistence."""

from __future__ import annotations

import functools
import hashlib
import inspect
import pickle
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from .._validation import validate_positive_int
from ._engine import _CACHE_RECOVERABLE_ERRORS, _THIRTY_DAYS
from ._smart import _validate_ignore_args


class MemoizePersistent:
    """Callable memoizer that persists results to disk as pickle files.

    Each unique call signature produces a ``.pkl`` file in the configured
    cache directory.  Expired entries are silently skipped on read and
    overwritten on write.

    :param func: The function to memoize
    :param cache_dir: Directory for pickle files; expands ``~``
    :param ttl: Entry time-to-live in seconds
    :param ignore_args: Argument names to exclude from cache-key computation
    :raises TypeError: If *func* is not callable
    """

    def __init__(
        self,
        func: Callable[..., Any],
        cache_dir: str = "~/.nerve_memo",
        ttl: int = _THIRTY_DAYS,
        ignore_args: list[str] | None = None,
    ) -> None:
        if not callable(func):
            raise TypeError("func must be callable")
        ttl = validate_positive_int(ttl, "ttl")
        self.func = func
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.ignore_args = _validate_ignore_args(ignore_args)
        self._signature: inspect.Signature | None = None
        with suppress(TypeError, ValueError):
            self._signature = inspect.signature(func)

        functools.update_wrapper(self, func)

    def __repr__(self) -> str:
        return (
            f"MemoizePersistent(func={self.func.__name__!r}, "
            f"cache_dir={str(self.cache_dir)!r}, ttl={self.ttl})"
        )

    def _make_key(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        key_items: Any
        if self._signature is None:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in self.ignore_args}
            key_items = (args, tuple(sorted(filtered_kwargs.items())))
        else:
            bound = self._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            key_items = tuple(
                (name, value)
                for name, value in bound.arguments.items()
                if name not in self.ignore_args
            )
        key_data = pickle.dumps(tuple(key_items))
        return hashlib.sha256(key_data).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the memoized function, retrieving or computing the result.

        :param args: Positional arguments forwarded to the wrapped function
        :param kwargs: Keyword arguments forwarded to the wrapped function
        :returns: The cached (or freshly computed) return value
        """
        key = self._make_key(args, kwargs)
        cache_path = self._cache_path(key)

        is_fresh = False
        with suppress(OSError):
            is_fresh = cache_path.exists() and time.time() - cache_path.stat().st_mtime < self.ttl
        if is_fresh:
            with suppress(*_CACHE_RECOVERABLE_ERRORS), cache_path.open("rb") as f:
                return pickle.load(f)

        result = self.func(*args, **kwargs)

        tmp_path = None
        with suppress(*_CACHE_RECOVERABLE_ERRORS):
            with tempfile.NamedTemporaryFile("wb", dir=self.cache_dir, delete=False) as f:
                tmp_path = Path(f.name)
                pickle.dump(result, f)
            tmp_path.replace(cache_path)
        if tmp_path is not None:
            with suppress(OSError):
                tmp_path.unlink()

        return result

    def clear(self) -> None:
        """Remove all cached pickle files from the cache directory."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            with suppress(OSError):
                cache_file.unlink()


def memoize_persistent(
    cache_dir: str = "~/.nerve_memo",
    ttl: int = _THIRTY_DAYS,
    ignore_args: list[str] | None = None,
) -> Callable[[Callable[..., Any]], MemoizePersistent]:
    """Decorator that wraps a function with :class:`MemoizePersistent` disk-backed memoization.

    :param cache_dir: Directory for pickle files; expands ``~``
    :param ttl: Entry time-to-live in seconds
    :param ignore_args: Argument names to exclude from the cache key
    :returns: A decorator that returns a :class:`MemoizePersistent` instance
    """
    ttl = validate_positive_int(ttl, "ttl")
    _validate_ignore_args(ignore_args)

    def decorator(func: Callable[..., Any]) -> MemoizePersistent:
        return MemoizePersistent(func, cache_dir=cache_dir, ttl=ttl, ignore_args=ignore_args)

    return decorator
