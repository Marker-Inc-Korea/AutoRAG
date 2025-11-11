from __future__ import annotations

import os
import socket
import time
import pytest


def pytest_sessionstart(session):
	os.environ["BM25"] = "bm25"


def _get_pg_host_port() -> tuple[str, int]:
	"""Return default host:port based on docker-compose in autorag/postgresql.

	The compose file exposes `${PG_PORT:-5432}:5432` with container_name `postgres`.
	For local testing outside Docker-in-Docker, the service is reachable at
	`localhost:PG_PORT` (defaults to 5432). We use this deterministic default
	so no env variables are required by default.
	"""
	port_env = os.getenv("PG_PORT")
	port = int(port_env) if port_env else 5432
	return ("localhost", port)


@pytest.fixture(scope="session", autouse=True)
def _postgres_healthcheck():
	"""Simple TCP healthcheck for Dockerized PostgreSQL.

	If a postgres host:port is discoverable from env, verify the socket is
	reachable. Fail fast if not reachable within a short timeout.
	"""
	host, port = _get_pg_host_port()

	deadline_seconds = float(os.getenv("POSTGRES_HEALTHCHECK_TIMEOUT", "10"))
	interval = 0.5
	deadline = time.time() + deadline_seconds
	last_err: Exception | None = None
	while time.time() < deadline:
		try:
			with socket.create_connection((host, port), timeout=1.0):
				return
		except OSError as e:
			last_err = e
			time.sleep(interval)

	pytest.fail(
		f"PostgreSQL is not reachable at {host}:{port} after {deadline_seconds}s: {last_err}"
	)
