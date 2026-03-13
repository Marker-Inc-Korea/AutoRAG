from __future__ import annotations

import os
import time
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError


def test_postgres_healthcheck():
	"""PostgreSQL docker-compose container healthcheck."""
	host = os.getenv("POSTGRES_HOST", "localhost")
	port = int(os.getenv("POSTGRES_PORT", "5432"))
	user = os.getenv("POSTGRES_USER", "postgres")
	password = os.getenv("POSTGRES_PASSWORD", "")
	database = os.getenv("POSTGRES_DB", "postgres")

	timeout = 10.0
	interval = 0.5

	# PostgreSQL connection URL
	db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

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

	pytest.fail(
		f"PostgreSQL is not reachable at {host}:{port} after {timeout}s: {last_err}"
	)


def pytest_sessionstart(session):
	os.environ["BM25"] = "bm25"
	test_postgres_healthcheck()
