from llama_index.core.llms import MockLLM

from autorag.data.beta.query.llama_gen_query import factoid_query_gen
from autorag.data.beta.schema.data import QA
from tests.autorag.data.beta.query.base_test_query_gen import qa_df

llm = MockLLM()


def test_make_factoid_query_gen():
	qa = QA(qa_df)
	new_qa = qa.batch_apply(factoid_query_gen, llm=llm)
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)


def test_make_factoid_query_gen_ko():
	qa = QA(qa_df)
	new_qa = qa.batch_apply(factoid_query_gen, llm=llm, lang="ko")
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)
