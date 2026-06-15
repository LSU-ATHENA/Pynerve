"""Core pipeline primitives."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from numbers import Integral
from typing import Any

from .exceptions import InvalidArgumentError, ValidationError


def _validate_representations(representations: list[str]) -> list[str]:
    if isinstance(representations, (str, bytes)) or not isinstance(representations, Sequence):
        raise ValidationError(
            "representations must be a sequence of strings",
            parameter="representations",
            expected="sequence of strings",
            actual=f"type={type(representations).__name__}",
        )
    result = list(representations)
    if not result:
        raise InvalidArgumentError(
            "representations must be non-empty",
            parameter="representations",
            expected="non-empty sequence",
        )
    if not all(isinstance(rep, str) and rep for rep in result):
        raise InvalidArgumentError(
            "representations must contain non-empty strings",
            parameter="representations",
        )
    if len(set(result)) != len(result):
        raise InvalidArgumentError(
            "representations must be unique",
            parameter="representations",
        )
    return result


class Pipeline:
    """Composable sequential transformation pipeline.

    Steps are executed in order, each receiving the output of the previous
    step. Steps can be named or auto-named, and can be accessed, inserted,
    removed, or sliced by name or index.
    """

    def __init__(self, *steps: tuple[str, Callable[..., Any]] | Callable[..., Any]):
        """Build a pipeline from named or unnamed steps.

        :param steps: Positional steps. Each can be a ``(name, callable)``
            tuple or a bare callable (auto-named with an underscore prefix).
        :raises InvalidArgumentError: If a step tuple has invalid length or
            a step is not callable.
        """
        self._steps: OrderedDict[str, Callable[..., Any]] = OrderedDict()
        for i, step in enumerate(steps):
            if isinstance(step, tuple):
                if len(step) != 2:
                    raise InvalidArgumentError(
                        f"Invalid step tuple with length {len(step)}. "
                        "Steps must be (name, callable) tuples or bare callables."
                    )
                name, func = step
            elif callable(step):
                name = f"_step_{i}"
                func = step
            else:
                raise InvalidArgumentError(
                    f"Invalid step type: {type(step).__name__}. "
                    "Steps must be (name, callable) tuples or bare callables."
                )

            self.add_step(name, func)

    def __call__(self, input_data: Any) -> Any:
        """Execute the pipeline by passing *input_data* through each step.

        :param input_data: Input data for the first step.
        :returns: Result after all steps have been applied sequentially.
        """
        result: Any = input_data
        for func in self._steps.values():
            result = func(result)
        return result

    def __getitem__(self, index: int | str | slice) -> Callable[..., Any] | Pipeline:
        """Access a step by integer index, string name, or slice.

        :param index: Integer index, string name, or slice object.
        :returns: The callable at the given index or name, or a new
            ``Pipeline`` for slices.
        :raises IndexError: If *index* is an integer out of range.
        :raises KeyError: If *index* is a name that does not exist.
        """
        if isinstance(index, slice):
            items = list(self._steps.items())[index]
            return Pipeline(*items)
        if isinstance(index, int):
            return list(self._steps.values())[index]
        return self._steps[index]

    def __len__(self) -> int:
        """Return the number of steps.

        :returns: Step count.
        """
        return len(self._steps)

    def __iter__(self) -> Iterator[tuple[str, Callable[..., Any]]]:
        """Iterate over (name, callable) pairs in order.

        :returns: An iterator over ``(step_name, step_callable)`` tuples.
        """
        return iter(self._steps.items())

    def add_step(self, name: str, func: Callable[..., Any], *, replace: bool = False) -> Pipeline:
        """Append a new step to the end of the pipeline.

        :param name: Step name. Must be a non-empty string.
        :param func: Callable implementing the step.
        :param replace: If ``True``, overwrite an existing step with the
            same name instead of raising.
        :returns: ``self`` for method chaining.
        :raises InvalidArgumentError: If *name* already exists and
            ``replace`` is ``False``.
        """
        self._validate_step(name, func)
        if name in self._steps and not replace:
            raise InvalidArgumentError(f"Pipeline step already exists: {name}")
        self._steps[name] = func
        return self

    def insert_step(self, index: int, name: str, func: Callable[..., Any]) -> Pipeline:
        """Insert a step at a specific position.

        :param index: Insertion position (0-indexed).
        :param name: Step name. Must be a non-empty string.
        :param func: Callable implementing the step.
        :returns: ``self`` for method chaining.
        :raises InvalidArgumentError: If *index* is not an integer or
            *name* already exists.
        """
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise InvalidArgumentError(
                "index must be an integer",
                parameter="index",
                expected="int",
                actual=f"type={type(index).__name__}",
            )
        self._validate_step(name, func)
        if name in self._steps:
            raise InvalidArgumentError(f"Pipeline step already exists: {name}")
        items = list(self._steps.items())
        items.insert(index, (name, func))
        self._steps = OrderedDict(items)
        return self

    def remove_step(self, name: str) -> Pipeline:
        """Remove a step by name.

        :param name: Name of the step to remove.
        :returns: ``self`` for method chaining.
        :raises KeyError: If *name* does not exist.
        """
        del self._steps[name]
        return self

    def pop_step(self, name: str) -> tuple[str, Callable[..., Any]]:
        """Remove and return a step by name.

        :param name: Name of the step to remove.
        :returns: The ``(name, callable)`` pair.
        :raises KeyError: If *name* does not exist.
        """
        func = self._steps.pop(name)
        return name, func

    def names(self) -> list[str]:
        """Return step names in insertion order.

        :returns: List of step names.
        """
        return list(self._steps.keys())

    def to_list(self) -> list[tuple[str, Callable[..., Any]]]:
        """Return all steps as a list of (name, callable) pairs.

        :returns: List of ``(name, callable)`` tuples.
        """
        return list(self._steps.items())

    def copy(self) -> Pipeline:
        """Create a shallow copy of this pipeline.

        :returns: A new ``Pipeline`` with the same steps.
        """
        return Pipeline(*self.to_list())

    def __repr__(self) -> str:
        lines = ["Pipeline("]
        for name, func in self._steps.items():
            func_name = getattr(func, "__name__", func.__class__.__name__)
            lines.append(f"  ({name!r}, {func_name}),")
        lines.append(")")
        return "\n".join(lines)

    @staticmethod
    def _validate_step(name: str, func: Callable[..., Any]) -> None:
        if not isinstance(name, str) or not name:
            raise InvalidArgumentError(
                "step name must be a non-empty string",
                parameter="name",
            )
        if not callable(func):
            raise InvalidArgumentError(
                "pipeline step must be callable",
                parameter="func",
                expected="callable",
                actual=f"type={type(func).__name__}",
            )
