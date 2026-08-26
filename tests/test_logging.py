import logging

from ten_texter.config import LogLevel
from ten_texter.logging import configure_logging


def test_logging_honors_level_and_writes_to_stderr(capsys) -> None:
    configure_logging(LogLevel.WARNING)
    logger = logging.getLogger("ten_texter.test")

    logger.info("hidden")
    logger.warning("visible")

    captured = capsys.readouterr()
    assert "hidden" not in captured.err
    assert "WARNING ten_texter.test visible" in captured.err


def test_logging_setup_is_idempotent() -> None:
    configure_logging(LogLevel.INFO)
    configure_logging(LogLevel.DEBUG)

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1

