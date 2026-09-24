from __future__ import annotations

from datetime import UTC

from ten_texter.models import now


def test_model_timestamp_default_is_utc() -> None:
    timestamp = now()

    assert timestamp.tzinfo is UTC
    assert timestamp.utcoffset().total_seconds() == 0
