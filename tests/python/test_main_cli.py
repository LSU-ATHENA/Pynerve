"""Tests for __main__.py — CLI entry point."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("mock_gpu_deps")


class TestMainCLI:
    """Covers __main__.py main() and command handlers."""

    def test_no_command_prints_help(self, capsys):
        from pynerve.__main__ import main
        result = main([])
        assert result == 1
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "command" in captured.out.lower()

    def test_info_command(self, capsys):
        from pynerve.__main__ import main
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Pynerve version" in captured.out

    def test_diagram_no_subcommand(self, capsys):
        from pynerve.__main__ import main
        result = main(["diagram"])
        assert result == 1

    def test_compute_no_file(self, capsys):
        from pynerve.__main__ import main
        result = main(["compute"])
        assert result == 1
        captured = capsys.readouterr()
        assert "no input file" in captured.err.lower() or "error" in captured.err.lower()

    def test_compute_nonexistent_file(self, capsys):
        from pynerve.__main__ import main
        result = main(["compute", "/nonexistent/file.txt"])
        assert result == 1
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    def test_build_parser(self):
        from pynerve.__main__ import _build_parser
        parser = _build_parser()
        assert parser is not None

    def test_cmd_info(self, capsys):
        from pynerve.__main__ import _cmd_info
        result = _cmd_info()
        assert result == 0

    def test_cmd_diagram_load_nonexistent(self, capsys):
        from pynerve.__main__ import _cmd_diagram_load
        import argparse
        args = argparse.Namespace(file="/nonexistent/file.npy")
        result = _cmd_diagram_load(args)
        assert result == 1
