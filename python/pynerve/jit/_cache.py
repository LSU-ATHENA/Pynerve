"""JIT compilation cache to avoid recompilation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._setup import jit


class JITCache:
    """Cache compiled JIT callables keyed by function and compilation options.

    Avoids recompilation when the same function is requested with the same
    arguments.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[Any, ...], Callable[..., Any]] = {}

    def __repr__(self) -> str:
        return f"JITCache(entries={len(self._store)})"

    def get_or_compile(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Callable[..., Any]:
        """Return a cached compiled version of *func* or compile and cache it.

        :param func: The function to JIT-compile.
        :param args: Positional arguments forwarded to the JIT decorator.
        :param kwargs: Keyword arguments forwarded to the JIT decorator.
        :returns: The compiled callable.
        :raises TypeError: If *func* is not callable.
        """
        if not callable(func):
            raise TypeError("func must be callable")
        key = (func.__name__, args, tuple(kwargs.items()))

        if key not in self._store:
            self._store[key] = jit(func, *args, **kwargs)

        return self._store[key]

    def clear(self) -> None:
        """Remove all cached compiled functions."""
        self._store.clear()


_global_cache = JITCache()


def cached_jit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a function with cached JIT compilation.

    The decorated function is compiled once via Numba JIT with
    ``nopython=True, parallel=True, cache=True``, and subsequent
    calls reuse the compiled version.

    :param func: The function to JIT-compile.
    :returns: Wrapped function that uses the cached compiled version.
    :raises TypeError: If *func* is not callable.
    """
    if not callable(func):
        raise TypeError("func must be callable")

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        compiled = _global_cache.get_or_compile(func, nopython=True, parallel=True, cache=True)
        return compiled(*args, **kwargs)

    return wrapper
