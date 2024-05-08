import os

import nest_asyncio


def pytest_sessionstart(session):
    nest_asyncio.apply()
    os.environ["BM25"] = "bm25"
    os.environ["JINAAI_API_KEY"] = "mock_jinaai_api_key"
    os.environ["COHERE_API_KEY"] = "mock_cohere_api_key"
