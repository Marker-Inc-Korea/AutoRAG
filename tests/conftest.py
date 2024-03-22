import os

import nest_asyncio


def pytest_sessionstart(session):
    nest_asyncio.apply()
    os.environ["BM25"] = "bm25"
