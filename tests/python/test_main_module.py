"""Tests for pynerve/__main__.py -- CLI entry point.

Tests all command branches through main() with necessary mocking.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestBuildParser:
    def test_parser_has_subparsers(self):
        from pynerve.__main__ import _build_parser

        parser = _build_parser()
        actions = [a.dest for a in parser._actions]
        assert "command" in actions
        # Verify subparsers exist via the help text
        help_text = parser.format_help()
        assert "info" in help_text
        assert "diagram" in help_text
        assert "compute" in help_text


class TestMainHelp:
    def test_main_empty_shows_help(self):
        """main([]) returns 1 and prints help (no command)."""
        from pynerve.__main__ import main

        with patch("sys.stdout", new_callable=io.StringIO) as buf:
            rc = main([])
            output = buf.getvalue()
        assert rc == 1
        # Help text should mention commands
        assert "info" in output or "diagram" in output or "compute" in output

    def test_main_help_flag(self):
        """--help should show version and usage."""
        from pynerve.__main__ import main

        with patch("sys.stdout", new_callable=io.StringIO) as buf:
            with pytest.raises(SystemExit):
                main(["--help"])
            output = buf.getvalue()
        assert "usage" in output.lower() or "positional" in output.lower()


class TestMainInfo:
    def test_main_info(self):
        """main(['info']) returns 0."""
        from pynerve.__main__ import main

        with patch("sys.stdout", new_callable=io.StringIO) as buf:
            rc = main(["info"])
            output = buf.getvalue()
        assert rc == 0
        assert "Pynerve version:" in output
        assert "Core extension:" in output
        assert "PyTorch:" in output


class TestMainDiagramLoad:
    def test_diagram_load_success_ndarray(self):
        """diagram load prints shape info for ndarray result."""
        from pynerve.__main__ import main

        fake_diagram = np.zeros((10, 3))

        with patch("pynerve.formats.load_diagrams", return_value=fake_diagram):
            with patch("sys.stdout", new_callable=io.StringIO) as buf:
                rc = main(["diagram", "load", "test.csv"])
                output = buf.getvalue()
        assert rc == 0
        assert "10 points" in output
        assert "3 dimensions" in output

    def test_diagram_load_success_pairs(self):
        """diagram load prints count for list-of-pairs result."""
        from pynerve.__main__ import main

        fake_pairs = [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]

        with patch("pynerve.formats.load_diagrams", return_value=fake_pairs):
            with patch("sys.stdout", new_callable=io.StringIO) as buf:
                rc = main(["diagram", "load", "test.csv"])
                output = buf.getvalue()
        assert rc == 0
        assert "3 persistence pairs" in output

    def test_diagram_load_error(self):
        """diagram load returns 1 on exception."""
        from pynerve.__main__ import main

        with patch("pynerve.formats.load_diagrams", side_effect=IOError("file missing")):
            with patch("sys.stderr", new_callable=io.StringIO) as buf:
                rc = main(["diagram", "load", "nonexistent.csv"])
                output = buf.getvalue()
        assert rc == 1
        assert "file missing" in output


class TestMainDiagramSave:
    def test_diagram_save_success(self):
        """diagram save converts and saves, returns 0."""
        from pynerve.__main__ import main

        fake_diagram = np.zeros((5, 2))

        with patch("pynerve.formats.load_diagrams", return_value=fake_diagram):
            with patch("pynerve.formats.save_diagrams") as mock_save:
                with patch("sys.stdout", new_callable=io.StringIO) as buf:
                    rc = main(["diagram", "save", "in.json", "out.npy"])
                    output = buf.getvalue()
        assert rc == 0
        assert "out.npy" in output
        mock_save.assert_called_once()

    def test_diagram_save_error(self):
        """diagram save returns 1 on exception."""
        from pynerve.__main__ import main

        with patch("pynerve.formats.load_diagrams", side_effect=ValueError("bad format")):
            with patch("sys.stderr", new_callable=io.StringIO) as buf:
                rc = main(["diagram", "save", "bad.txt", "out.npy"])
                output = buf.getvalue()
        assert rc == 1
        assert "bad format" in output


class TestMainDiagramNoSubcommand:
    def test_diagram_no_subcommand(self):
        """diagram with no subcommand prints help and returns 1."""
        from pynerve.__main__ import main

        with patch("sys.stdout", new_callable=io.StringIO) as buf:
            rc = main(["diagram"])
            output = buf.getvalue()
        assert rc == 1
        assert "positional arguments" in output or "diagram" in output


class TestMainCompute:
    def test_compute_no_file(self):
        """compute with no file argument shows error."""
        from pynerve.__main__ import main

        with patch("sys.stderr", new_callable=io.StringIO) as buf:
            rc = main(["compute"])
            output = buf.getvalue()
        assert rc == 1
        assert "no input file" in output.lower()

    def test_compute_load_error(self):
        """compute returns 1 when loadtxt fails."""
        from pynerve.__main__ import main

        with patch("numpy.loadtxt", side_effect=OSError("permission denied")):
            with patch("sys.stderr", new_callable=io.StringIO) as buf:
                rc = main(["compute", "unreadable.txt"])
                output = buf.getvalue()
        assert rc == 1
        assert "permission denied" in output

    def test_compute_success(self):
        """compute runs persistence and prints betti numbers + pairs."""
        from pynerve.__main__ import main

        fake_data = np.array([[0.0, 0.0], [1.0, 1.0]])
        fake_result = MagicMock()
        fake_result.betti_numbers = [1, 0, 0]
        fake_result.pairs = [(0.0, 1.0)]

        with patch("numpy.loadtxt", return_value=fake_data):
            with patch("pynerve.compute_persistence", return_value=fake_result):
                with patch("sys.stdout", new_callable=io.StringIO) as buf:
                    rc = main(["compute", "data.txt"])
                    output = buf.getvalue()
        assert rc == 0
        assert "Betti numbers" in output
        assert "Pairs: 1" in output

    def test_compute_success_with_output(self):
        """compute saves diagram when --output is specified."""
        from pynerve.__main__ import main

        fake_data = np.array([[0.0, 0.0]])
        fake_result = MagicMock()
        fake_result.betti_numbers = [1, 0]
        fake_result.pairs = [(0.0, 1.0)]

        with patch("numpy.loadtxt", return_value=fake_data):
            with patch("pynerve.compute_persistence", return_value=fake_result):
                with patch("pynerve.formats.save_diagrams") as mock_save:
                    with patch("sys.stdout", new_callable=io.StringIO) as buf:
                        rc = main(["compute", "data.txt", "-o", "out.npy"])
                        output = buf.getvalue()
        assert rc == 0
        assert "Saved diagram" in output
        mock_save.assert_called_once_with([(0.0, 1.0)], "out.npy")

    def test_compute_persistence_error(self):
        """compute returns 1 when persistence computation fails."""
        from pynerve.__main__ import main

        fake_data = np.array([[0.0, 0.0]])

        with patch("numpy.loadtxt", return_value=fake_data):
            with patch(
                "pynerve.compute_persistence",
                side_effect=RuntimeError("VR computation failed"),
            ):
                with patch("sys.stderr", new_callable=io.StringIO) as buf:
                    rc = main(["compute", "data.txt"])
                    output = buf.getvalue()
        assert rc == 1
        assert "VR computation failed" in output
