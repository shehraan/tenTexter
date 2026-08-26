from sqlalchemy import inspect

from ten_texter.config import Settings
from ten_texter.db.engine import create_database_resources
from ten_texter.db.migrations import current_database_revision, upgrade_database

BASELINE_REVISION = "0001_phase1_baseline"


def test_empty_database_migrates_from_zero_to_head(settings: Settings) -> None:
    assert current_database_revision(settings) is None

    assert upgrade_database(settings) == BASELINE_REVISION
    assert current_database_revision(settings) == BASELINE_REVISION

    resources = create_database_resources(settings)
    try:
        assert inspect(resources.engine).get_table_names() == ["alembic_version"]
    finally:
        resources.dispose()


def test_upgrade_to_head_is_idempotent(settings: Settings) -> None:
    assert upgrade_database(settings) == BASELINE_REVISION
    assert upgrade_database(settings) == BASELINE_REVISION
    assert current_database_revision(settings) == BASELINE_REVISION
