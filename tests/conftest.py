from collections.abc import Generator
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    # Give every test its own temporary SQLite DB and artifact folder so tests never clash
    db_path = tmp_path / "test_registry.db"
    artifacts_dir = tmp_path / "artifacts"
    return Settings(
        database_url=f"sqlite:///{db_path.as_posix()}",
        artifact_storage_dir=artifacts_dir,
        log_level="WARNING",
    )


@pytest.fixture()
def client(test_settings: Settings) -> Generator[TestClient, None, None]:
    # Start a fresh FastAPI app (with lifespan startup & cleanup) for each test
    app = create_app(test_settings)
    with TestClient(app) as test_client:
        yield test_client
