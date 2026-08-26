from pathlib import Path

from sqlalchemy import create_engine, inspect

from ten_texter.config import Settings
from ten_texter.migrations import upgrade_to_head


def test_empty_database_migrates_from_zero_to_head(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "app.db"
    settings = Settings(database_path=database_path)

    upgrade_to_head(settings)

    engine = create_engine(settings.database_url)
    assert inspect(engine).get_table_names() == ["alembic_version"]
    with engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one() == "0001_phase_1"

