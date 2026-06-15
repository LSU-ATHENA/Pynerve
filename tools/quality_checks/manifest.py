"""Quality manifest generation."""

from __future__ import annotations

from .common import *  # noqa: F403
from .common import _cpp_declarations, _public_python_functions  # noqa: F401


def generate_manifest(output: Path) -> None:
    payload = {
        "python_api": _public_python_functions(),
        "cpp_declarations": _cpp_declarations(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
