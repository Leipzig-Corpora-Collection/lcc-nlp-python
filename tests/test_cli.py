import typing
from pathlib import Path

import pytest

import lcc.cli

if typing.TYPE_CHECKING:  # pragma: no cover
    from pytest import CaptureFixture
    from pytest_mock import MockerFixture

    # from pytest import LogCaptureFixture


# ---------------------------------------------------------------------------


def test_cli_main_noop(tmp_path: Path, mocker: "MockerFixture"):
    mocker.patch("lcc.cli.main")
    mocker.patch("lcc.cli.LOGFILE")

    log_file = tmp_path / "file.log"
    lcc.cli.LOGFILE = str(log_file)

    lcc.cli.cli_main()

    # was created but nothing logged yet
    assert log_file.exists()
    assert log_file.stat().st_size == 0

    lcc.cli.main.assert_called_once_with(args=None)


def test_main_nomode(capsys: "CaptureFixture"):
    with pytest.raises(SystemExit) as exc_info:
        lcc.cli.main(args=[])  # not None, that will default to sys.argv
    # NOTE: explicitely provide arguments, otherwise in tox some pytest cli arguments will found in sys.argv
    assert exc_info.value.code == 0

    captured = capsys.readouterr()

    assert captured.out.endswith("\n  -h, --help  show this help message and exit\n")
    assert captured.err.endswith("Please specify a MODE!\n")


# ---------------------------------------------------------------------------
