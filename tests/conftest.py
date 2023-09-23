import io
import typing
from pathlib import Path
from typing import Callable
from typing import Optional

import pytest

if typing.TYPE_CHECKING:  # pragma: no cover
    from pytest import MonkeyPatch


# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def resource_path() -> Path:
    root = Path(__file__).parent.parent
    return root / "resources"


# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def set_stdin_content(monkeypatch: "MonkeyPatch") -> Callable[[bytes], None]:
    class FakeStdin(io.TextIOBase):
        """Fake Stdin object to simulate `sys.stdin` and `sys.stdin.buffer` access."""

        # see: https://github.com/pytest-dev/pytest/issues/1407#issuecomment-187460730
        def __init__(self, content: bytes):
            self.buffer = io.BytesIO(content)
            self._buffer_text = io.StringIO(self.buffer.getvalue().decode())

        def readline(self, __size: int = -1) -> str:
            return self._buffer_text.readline(__size)

        def readlines(self, __hint: int = -1) -> list[str]:
            return self._buffer_text.readlines(__hint)

        def read(self, __size: Optional[int] = None) -> str:
            return self._buffer_text.read(__size)

        def readable(self) -> bool:
            return self._buffer_text.readable()

    def set_stdin_content(content: bytes):
        fake_stdin = FakeStdin(content)
        monkeypatch.setattr("sys.stdin", fake_stdin)

    return set_stdin_content


# ---------------------------------------------------------------------------
