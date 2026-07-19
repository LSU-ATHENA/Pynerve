"""CLI entry point for Pynerve.

Usage:
    python -m pynerve --help
    python -m pynerve info
    python -m pynerve diagram load <file>
    python -m pynerve diagram save <input> <output>
    python -m pynerve compute <file> [--max-dim N] [--max-radius R]
"""

from __future__ import annotations

import argparse
import sys


def _cmd_info() -> int:
    import pynerve  # noqa: PLC0415

    print(f"Pynerve version: {pynerve.__version__}")
    print(f"Core extension: {'available' if pynerve._core is not None else 'not found'}")
    print(f"PyTorch: {'available' if pynerve._pytorch is not None else 'not found'}")
    print(f"Engines: {[e.value for e in pynerve.PersistenceEngine]}")
    print(f"Backends: {[b.value for b in pynerve.PersistenceBackend]}")
    return 0


def _cmd_diagram_load(args: argparse.Namespace) -> int:
    import numpy as np  # noqa: PLC0415

    from pynerve.formats import load_diagrams  # noqa: PLC0415

    try:
        result = load_diagrams(args.file)
    except Exception as exc:
        print(f"Error loading diagram: {exc}", file=sys.stderr)
        return 1
    if isinstance(result, np.ndarray):
        arr: np.ndarray = result  # narrow type for pyright
        print(f"Loaded {arr.shape[0]} points with {arr.shape[1]} dimensions")
    else:
        print(f"Loaded {len(result)} persistence pairs")
    return 0


def _cmd_diagram_save(args: argparse.Namespace) -> int:
    from pynerve.formats import load_diagrams, save_diagrams  # noqa: PLC0415

    try:
        diagram = load_diagrams(args.input)
        save_diagrams(diagram, args.output)
    except Exception as exc:
        print(f"Error converting diagram: {exc}", file=sys.stderr)
        return 1
    print(f"Saved diagram to {args.output}")
    return 0


def _cmd_compute(args: argparse.Namespace) -> int:
    import numpy as np  # noqa: PLC0415

    from pynerve import compute_persistence  # noqa: PLC0415
    from pynerve.formats import save_diagrams  # noqa: PLC0415

    try:
        if args.file is None:
            print(
                "Error: no input file specified. Usage: python -m pynerve compute <file> [options]",
                file=sys.stderr,
            )
            return 1
        data = np.loadtxt(args.file)
    except Exception as exc:
        print(f"Error loading data: {exc}", file=sys.stderr)
        return 1

    try:
        result = compute_persistence(data, max_dim=args.max_dim, max_radius=args.max_radius)
    except Exception as exc:
        print(f"Computation error: {exc}", file=sys.stderr)
        return 1

    print(f"Betti numbers: {result.betti_numbers}")
    print(f"Pairs: {len(result.pairs)}")

    if args.output:
        save_diagrams(result.pairs, args.output)
        print(f"Saved diagram to {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    import pynerve  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Pynerve - Fast persistent homology")
    parser.add_argument(
        "--version",
        action="version",
        help="Show version",
        version=f"%(prog)s {pynerve.__version__}",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")
    sub.add_parser("info", help="Show system information")

    diagram = sub.add_parser("diagram", help="Diagram operations")
    diagram_sub = diagram.add_subparsers(dest="diagram_command", help="Diagram sub-commands")
    load_cmd = diagram_sub.add_parser("load", help="Load a diagram file")
    load_cmd.add_argument("file", help="Path to diagram file")
    save_cmd = diagram_sub.add_parser("save", help="Convert diagram format")
    save_cmd.add_argument("input", help="Input diagram file")
    save_cmd.add_argument("output", help="Output diagram file")

    compute = sub.add_parser("compute", help="Compute persistent homology")
    compute.add_argument(
        "file", nargs="?", default=None, help="Point cloud file (numpy txt format)"
    )
    compute.add_argument("--max-dim", type=int, default=2, help="Maximum homology dimension")
    compute.add_argument("--max-radius", type=float, default=None, help="Filtration radius cutoff")
    compute.add_argument("--output", "-o", default=None, help="Save diagram to file")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        return _cmd_info()
    if args.command == "diagram":
        if args.diagram_command == "load":
            return _cmd_diagram_load(args)
        if args.diagram_command == "save":
            return _cmd_diagram_save(args)
        parser.print_help()
        return 1
    if args.command == "compute":
        return _cmd_compute(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
