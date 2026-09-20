import pytest
from unittest.mock import patch
import sys
from vaxguard.cli import main
from vaxguard import __version__


def test_cli_version(capsys):
    with patch.object(sys, "argv", ["vaxguard", "--version"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert f"vaxguard v{__version__}" in captured.out or f"vaxguard v{__version__}" in captured.err


def test_cli_help(capsys):
    with patch.object(sys, "argv", ["vaxguard", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "VaxGuard" in captured.out
        assert "serve" in captured.out
        assert "scan" in captured.out


def test_cli_serve_subcommand():
    with patch.object(sys, "argv", ["vaxguard", "serve", "--port", "9000", "--host", "127.0.0.1"]):
        with patch("uvicorn.run") as mock_uvicorn:
            main()
            mock_uvicorn.assert_called_once_with(
                "vaxguard.api.app:app", host="127.0.0.1", port=9000, reload=False
            )


def test_cli_scan_subcommand():
    with patch.object(sys, "argv", ["vaxguard", "scan", "--concurrency", "2"]):
        with patch("asyncio.run") as mock_asyncio_run:
            main()
            mock_asyncio_run.assert_called_once()
