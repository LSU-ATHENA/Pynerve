"""Tests for pynerve/__main__.py — CLI entry point, argument parsing, command dispatch."""

from __future__ import annotations

import pytest

from pynerve.__main__ import _build_parser, main


class TestArgumentParser:
    def test_builds_without_error(self):
        parser = _build_parser()
        assert parser is not None

    def test_info_command(self):
        parser = _build_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"

    def test_diagram_load(self):
        parser = _build_parser()
        args = parser.parse_args(["diagram", "load", "test.csv"])
        assert args.command == "diagram"
        assert args.diagram_command == "load"
        assert args.file == "test.csv"

    def test_diagram_save(self):
        parser = _build_parser()
        args = parser.parse_args(["diagram", "save", "in.csv", "out.json"])
        assert args.diagram_command == "save"
        assert args.input == "in.csv"
        assert args.output == "out.json"

    def test_compute_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["compute", "data.txt"])
        assert args.command == "compute"
        assert args.file == "data.txt"
        assert args.max_dim == 2
        assert args.max_radius is None
        assert args.output is None

    def test_compute_with_options(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["compute", "data.txt", "--max-dim", "3", "--max-radius", "10.0", "-o", "out.csv"]
        )
        assert args.max_dim == 3
        assert args.max_radius == 10.0
        assert args.output == "out.csv"

    def test_no_command_shows_help(self):
        result = main([])
        assert result == 1

    def test_version_flag(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--version"])


class TestMainCommands:
    def test_info_returns_zero(self):
        result = main(["info"])
        assert result == 0

    def test_compute_no_file(self):
        result = main(["compute"])
        assert result == 1

    def test_diagram_no_subcommand(self):
        result = main(["diagram"])
        assert result == 1

    def test_unknown_diagram_subcommand(self):
        with pytest.raises(SystemExit):
            main(["diagram", "unknown", "file.csv"])
