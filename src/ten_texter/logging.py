import logging
import logging.config

from ten_texter.config import LogLevel


def configure_logging(level: LogLevel) -> None:
    """Configure one deterministic stderr handler for the process root logger."""

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                }
            },
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stderr",
                }
            },
            "root": {
                "handlers": ["stderr"],
                "level": level.value,
            },
        }
    )

