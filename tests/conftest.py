import os
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from ten_texter.config import Settings


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.upper().startswith("TENTEXTER_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("external network access is forbidden in the test suite")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "ten_texter_test.db"


@pytest.fixture
def settings(database_path: Path) -> Iterator[Settings]:
    yield Settings(database_url=f"sqlite+pysqlite:///{database_path}")

