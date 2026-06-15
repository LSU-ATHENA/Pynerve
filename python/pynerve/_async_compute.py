"""Async persistence computation pipeline."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from collections.abc import AsyncIterator as AsyncIteratorABC
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

from ._compute_api import compute_persistence
from ._compute_core import PersistenceResult
from ._validation import validate_positive_int


async def _shutdown_executor(executor: ThreadPoolExecutor) -> None:
    """Offload blocking executor shutdown to a thread to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, executor.shutdown, True)


class AsyncPersistenceComputer:
    """Run persistence computations from an async batch source."""

    def __init__(self, max_workers: int = 4, buffer_size: int = 3):
        max_workers = validate_positive_int(max_workers, "max_workers")
        buffer_size = validate_positive_int(buffer_size, "buffer_size")
        self.max_workers = max_workers
        self.buffer_size = buffer_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._closed = False

    async def compute_batch_async(
        self,
        data_batches: AsyncIterator[np.ndarray],
        compute_fn: Callable[..., Any] | None = None,
        **compute_fn_kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield computed results while the next batch is being loaded."""
        if self._closed:
            raise RuntimeError("AsyncPersistenceComputer is closed")
        if not isinstance(data_batches, AsyncIteratorABC):
            raise TypeError("data_batches must be an async iterator")
        compute_fn = compute_fn or self._default_compute
        if not callable(compute_fn):
            raise TypeError("compute_fn must be callable")

        pending: deque[Future[Any]] = deque()
        try:
            async for batch in data_batches:
                pending.append(self.executor.submit(compute_fn, batch, **compute_fn_kwargs))  # type: ignore[arg-type]
                if len(pending) >= self.buffer_size:
                    yield await self._await_future(pending.popleft())

            while pending:
                yield await self._await_future(pending.popleft())
        finally:
            for future in pending:
                future.cancel()

    async def _await_future(self, future: Future[Any]) -> Any:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, future.result)
        except asyncio.CancelledError:
            future.cancel()
            raise

    def _default_compute(self, batch: np.ndarray) -> PersistenceResult:
        return compute_persistence(batch)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await _shutdown_executor(self.executor)

    async def __aenter__(self) -> AsyncPersistenceComputer:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        await self.close()


__all__ = [
    "AsyncIteratorABC",
    "AsyncPersistenceComputer",
]
