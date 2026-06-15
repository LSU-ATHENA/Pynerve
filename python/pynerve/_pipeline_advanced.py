"""Conditional and parallel pipeline variants."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._pipeline_core import Pipeline
from .exceptions import InvalidArgumentError


class ConditionalPipeline:
    """Pipeline with unconditional and conditional branching steps.

    Supports both simple steps (always executed) and conditional steps
    that select between two callables based on a predicate.
    """

    def __init__(self, *steps: tuple[str, Callable[..., Any]] | Callable[..., Any]):
        """Build a conditional pipeline from named or unnamed steps.

        :param steps: Positional steps as ``(name, callable)`` tuples or
            bare callables (auto-named with an underscore prefix).
        """
        self._steps: list[dict[str, Any]] = []
        for step in steps:
            if isinstance(step, tuple):
                name, func = step
                self.add_step(name, func)
            elif callable(step):
                self.add_step(f"_step_{len(self._steps)}", step)

    def __repr__(self) -> str:
        lines = ["ConditionalPipeline("]
        for step in self._steps:
            name = step["name"]
            stype = step["type"]
            func = step.get("func") or step.get("if_true")
            func_name = getattr(func, "__name__", func.__class__.__name__) if func else "?"
            lines.append(f"  ({name!r}, type={stype}, fn={func_name}),")
        lines.append(")")
        return "\n".join(lines)

    def add_step(
        self, name: str, func: Callable[..., Any], *, replace: bool = False
    ) -> ConditionalPipeline:
        """Append a simple (unconditional) step.

        :param name: Step name. Must be a non-empty string.
        :param func: Callable to execute on the current data.
        :param replace: If ``True``, overwrite an existing step with the
            same name instead of raising.
        :returns: ``self`` for method chaining.
        :raises InvalidArgumentError: If *name* already exists and
            ``replace`` is ``False``.
        """
        Pipeline._validate_step(name, func)
        if not replace:
            self._validate_new_name(name)
        if replace:
            for s in self._steps:
                if s["name"] == name:
                    s["func"] = func
                    return self
        self._steps.append({"type": "simple", "name": name, "func": func})
        return self

    def add_conditional(
        self,
        name: str,
        condition: Callable[[Any], bool],
        if_true: Callable[..., Any],
        if_false: Callable[..., Any] | None = None,
    ) -> ConditionalPipeline:
        """Append a conditional branching step.

        :param name: Step name. Must be unique and non-empty.
        :param condition: A callable ``condition(data) -> bool`` that
            determines which branch to execute.
        :param if_true: Callable invoked when *condition* returns a truthy
            value.
        :param if_false: Optional callable invoked when *condition* returns
            a falsy value. ``None`` means nothing is executed on the false
            branch.
        :returns: ``self`` for method chaining.
        :raises TypeError: If *condition* or *if_false* are not callable.
        :raises InvalidArgumentError: If *name* already exists.
        """
        Pipeline._validate_step(name, if_true)
        self._validate_new_name(name)
        if not callable(condition):
            raise TypeError("condition must be callable")
        if if_false is not None and not callable(if_false):
            raise TypeError("if_false must be callable")
        self._steps.append(
            {
                "type": "conditional",
                "name": name,
                "condition": condition,
                "if_true": if_true,
                "if_false": if_false,
            }
        )
        return self

    def remove_step(self, name: str) -> ConditionalPipeline:
        """Remove a step by name.

        :param name: Name of the step to remove.
        :returns: ``self`` for method chaining.
        """
        self._steps = [s for s in self._steps if s["name"] != name]
        return self

    def names(self) -> list[str]:
        """Return step names in insertion order.

        :returns: List of step names.
        """
        return [s["name"] for s in self._steps]

    def __call__(self, input_data: Any) -> Any:
        """Execute the pipeline, branching on conditional steps.

        :param input_data: Input data for the first step.
        :returns: Result after all steps have been executed.
        :raises RuntimeError: If an unknown step type is encountered.
        """
        for step in self._steps:
            if step["type"] == "simple":
                input_data = step["func"](input_data)
            elif step["type"] == "conditional":
                condition_result = step["condition"](input_data)
                if bool(condition_result):
                    input_data = step["if_true"](input_data)
                elif step["if_false"] is not None:
                    input_data = step["if_false"](input_data)
            else:
                raise RuntimeError(f"Unknown step type: {step['type']}")
        return input_data

    def _validate_new_name(self, name: str) -> None:
        if any(step["name"] == name for step in self._steps):
            raise InvalidArgumentError(f"Pipeline step already exists: {name}")


class ParallelPipeline:
    """Pipeline that executes multiple sub-pipelines on the same input and combines results.

    Each sub-pipeline receives the same input data. Their outputs are
    collected into a ``dict[str, Any]`` and passed to a user-supplied
    combination function.
    """

    def __init__(
        self,
        pipelines: dict[str, Pipeline | Callable[..., Any]],
        combine_fn: Callable[[dict[str, Any]], Any],
    ):
        """Build a parallel pipeline.

        :param pipelines: Mapping of unique names to ``Pipeline`` instances
            or callables.
        :param combine_fn: Callable that receives a ``{name: output}`` dict
            and returns the combined result.
        :raises InvalidArgumentError: If *pipelines* is empty, *combine_fn*
            is not callable, or a name is invalid.
        """
        if not pipelines:
            raise InvalidArgumentError("pipelines must be non-empty")
        if not callable(combine_fn):
            raise InvalidArgumentError(
                "combine_fn must be callable",
                parameter="combine_fn",
                expected="callable",
                actual=f"type={type(combine_fn).__name__}",
            )
        for name, pipeline in pipelines.items():
            if not isinstance(name, str) or not name:
                raise InvalidArgumentError("pipeline names must be non-empty strings")
            Pipeline._validate_step(name, pipeline)
        self.pipelines = pipelines
        self.combine_fn = combine_fn

    def __repr__(self) -> str:
        names = list(self.pipelines)
        return f"ParallelPipeline(pipelines={names}, combine_fn={self.combine_fn.__name__!r})"

    def __call__(self, input_data: Any) -> Any:
        """Broadcast *input_data* to every sub-pipeline and combine results.

        :param input_data: Input passed to each sub-pipeline.
        :returns: The result of ``combine_fn(result_dict)``.
        """
        results = {}

        for name, pipeline in self.pipelines.items():
            results[name] = pipeline(input_data)

        return self.combine_fn(results)
