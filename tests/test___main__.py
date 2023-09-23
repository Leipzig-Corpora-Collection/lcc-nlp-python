import runpy
import sys
import typing

import lcc.cli

if typing.TYPE_CHECKING:  # pragma: no cover
    from pytest_mock import MockerFixture


# ---------------------------------------------------------------------------


def test_module_call(mocker: "MockerFixture"):
    # NOTE: seems to be required to avoid warning:
    # tests/test___main__.py::test_module_call
    #   /usr/lib/python3.10/runpy.py:126: RuntimeWarning: 'lcc.__main__' found in sys.modules after import of package 'lcc', but prior to execution of 'lcc.__main__'; this may result in unpredictable behaviour
    #     warn(RuntimeWarning(msg))
    if "lcc.__main__" in sys.modules:  # pragma: no cover
        del sys.modules["lcc.__main__"]

    mocker.patch("lcc.cli.cli_main")

    # simulate: python -m lcc
    runpy.run_module("lcc", run_name="__main__")

    lcc.cli.cli_main.assert_called_once_with()


def test_module_call_import(mocker: "MockerFixture"):
    mocker.patch("lcc.cli.cli_main")

    # simulate import
    # runpy.run_module("lcc", run_name="lcc.__main__")
    import lcc.__main__

    lcc.cli.cli_main.assert_not_called()


# ---------------------------------------------------------------------------
