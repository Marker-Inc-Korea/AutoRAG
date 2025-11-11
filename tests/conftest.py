from __future__ import annotations

import os
import socket
import time
import pytest


def test_postgres_healthcheck():
	"""PostgreSQL docker-compose container healthcheck."""
	host = os.getenv("POSTGRES_HOST", "localhost")
	port = int(os.getenv("POSTGRES_PORT", "5432"))
	timeout = 10.0
	interval = 0.5

	deadline = time.time() + timeout
	last_err: Exception | None = None

	while time.time() < deadline:
		try:
			with socket.create_connection((host, port), timeout=1.0):
				return
		except OSError as e:
			last_err = e
			time.sleep(interval)

	pytest.fail(
		f"PostgreSQL is not reachable at {host}:{port} after {timeout}s: {last_err}"
	)


def pytest_sessionstart(session):
	os.environ["BM25"] = "bm25"
	test_postgres_healthcheck()
