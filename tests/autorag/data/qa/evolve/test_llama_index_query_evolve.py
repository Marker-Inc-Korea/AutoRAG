from llama_index.core.llms import MockLLM

from autorag.data.qa.evolve.llama_index_query_evolve import (
	conditional_evolve_ragas,
	reasoning_evolve_ragas,
	compress_ragas,
)
from autorag.data.qa.schema import QA
from tests.autorag.data.qa.evolve.base_test_query_evolve import qa_df

llm = MockLLM()
qa = QA(qa_df)


def test_conditional_evolve_ragas():
	new_qa = qa.batch_apply(conditional_evolve_ragas, llm=llm)
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)
	assert all(
		x != y for x, y in zip(new_qa.data["query"].tolist(), qa_df["query"].tolist())
	)


def test_reasoning_evolve_ragas():
	new_qa = qa.batch_apply(reasoning_evolve_ragas, llm=llm)
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)
	assert all(
		x != y for x, y in zip(new_qa.data["query"].tolist(), qa_df["query"].tolist())
	)


def test_compress_ragas():
	new_qa = qa.batch_apply(compress_ragas, llm=llm)
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)
	assert all(
		x != y for x, y in zip(new_qa.data["query"].tolist(), qa_df["query"].tolist())
	)
