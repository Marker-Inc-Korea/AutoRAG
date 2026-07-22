import os


def pytest_sessionstart(session):
    os.environ["BM25"] = "bm25"
