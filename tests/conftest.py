from __future__ import annotations

import os
import time
from typing import Any, Generator
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session as SASession, sessionmaker
from autorag.db import schema as m


def test_postgres_healthcheck():
	"""PostgreSQL docker-compose container healthcheck."""
	timeout = 10.0
	interval = 0.5

	# PostgreSQL connection URL
	db_url = _db_url_from_env()

	deadline = time.time() + timeout
	last_err: Exception | None = None

	while time.time() < deadline:
		try:
			# Create engine with connection timeout
			engine = create_engine(db_url, connect_args={"connect_timeout": 1})
			# Test connection with a simple query
			with engine.connect() as conn:
				conn.execute(text("SELECT 1"))
				conn.close()
			engine.dispose()
			return
		except OperationalError as e:
			last_err = e
			time.sleep(interval)

	pytest.fail(f"PostgreSQL is not reachable at {db_url} after {timeout}s: {last_err}")


def pytest_sessionstart(session):
	os.environ["BM25"] = "bm25"
	test_postgres_healthcheck()


def _db_url_from_env() -> str:
	host = os.getenv("POSTGRES_HOST", "localhost")
	port = int(os.getenv("POSTGRES_PORT", "5432"))
	user = os.getenv("POSTGRES_USER", "postgres")
	password = os.getenv("POSTGRES_PASSWORD", "")
	database = os.getenv("POSTGRES_DB", "postgres")
	return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


@pytest.fixture()
def session() -> Generator[SASession, Any, Any]:
	"""Synchronous SQLAlchemy Session bound to Postgres for tests.

	- Creates schema once if missing.
	- Wraps each test in a transaction; rolls back afterwards.
	"""
	url = _db_url_from_env()
	engine = create_engine(url, pool_pre_ping=True)
	# Ensure tables exist
	with engine.begin() as conn:
		m.Base.metadata.create_all(conn)

	connection = engine.connect()
	trans = connection.begin()
	SessionLocal = sessionmaker(
		bind=connection, autoflush=False, autocommit=False, future=True
	)
	sess = SessionLocal()
	try:
		yield sess
	finally:
		sess.close()
		trans.rollback()
		connection.close()
		engine.dispose()
