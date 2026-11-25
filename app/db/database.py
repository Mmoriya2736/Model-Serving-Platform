from collections.abc import Generator
from pathlib import Path
from typing import Any

from fastapi import Request
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# Base class that all our SQLAlchemy database tables inherit from
class Base(DeclarativeBase):
    pass


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    # Create the parent folder for the SQLite file if it doesn't exist yet
    if not database_url.startswith("sqlite:///"):
        return
    raw_path = database_url.removeprefix("sqlite:///")
    if raw_path in ("", ":memory:"):
        return
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)


def create_db_engine(database_url: str) -> Engine:
    # Make sure the DB folder exists and allow SQLite connections across FastAPI threads
    _ensure_sqlite_parent_dir(database_url)
    is_sqlite = database_url.startswith("sqlite")
    connect_args: dict[str, Any] = {"check_same_thread": False} if is_sqlite else {}
    engine = create_engine(database_url, connect_args=connect_args, future=True)

    # Whenever a new SQLite connection opens, turn on WAL mode and foreign keys
    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            # WAL lets readers and writers work at the same time without locking each other out
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            # Wait up to 5 seconds if another thread is briefly writing
            cursor.execute("PRAGMA busy_timeout=5000;")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    # Build a factory that hands out fresh database sessions per request
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    # Import all table classes first so SQLAlchemy knows about them before creating tables
    from app.db import models as _models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db(request: Request) -> Generator[Session, None, None]:
    # FastAPI dependency: open a DB session for one request and always close it when done
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
