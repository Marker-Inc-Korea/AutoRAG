import os


def pytest_sessionstart(session):
	os.environ["BM25"] = "bm25"
	os.environ["JINAAI_API_KEY"] = "mock_jinaai_api_key"
	os.environ["COHERE_API_KEY"] = "mock_cohere_api_key"
