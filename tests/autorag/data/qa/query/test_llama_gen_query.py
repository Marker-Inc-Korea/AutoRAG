from unittest.mock import patch

from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole
from llama_index.core.llms import MockLLM

from autorag.data.qa.query.llama_gen_query import (
	factoid_query_gen,
	concept_completion_query_gen,
	two_hop_incremental,
)
from autorag.data.qa.schema import QA
from tests.autorag.data.qa.query.base_test_query_gen import qa_df, multi_hop_qa_df

llm = MockLLM()
qa = QA(qa_df)


def test_make_factoid_query_gen():
	new_qa = qa.batch_apply(factoid_query_gen, llm=llm)
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)


def test_make_factoid_query_gen_ko():
	new_qa = qa.batch_apply(factoid_query_gen, llm=llm, lang="ko")
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)

def test_make_factoid_query_gen_ja():
    new_qa = qa.batch_apply(factoid_query_gen, llm=llm, lang="ja")
    assert "query" in new_qa.data.columns
    assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
    assert len(new_qa.data) == len(qa_df)

def test_concept_completion_query_gen():
	new_qa = qa.batch_apply(concept_completion_query_gen, llm=llm, lang="ko")
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)

def test_concept_completion_query_gen_ja():
    new_qa = qa.batch_apply(concept_completion_query_gen, llm=llm, lang="ja")
    assert "query" in new_qa.data.columns
    assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
    assert len(new_qa.data) == len(qa_df)



async def two_hop_incremental_mock_achat(*args, **kwargs) -> ChatResponse:
	return ChatResponse(
		message=ChatMessage(
			role=MessageRole.ASSISTANT,
			content="""Answer: Tamaulipas
One-hop question (using Document 1): In which Mexican state is Nuevo Laredo located?
Two-hop question (using Document 2):  In which Mexican state can one find the Ciudad Deportiva, home to the Tecolotes de Nuevo Laredo?""",
		)
	)


@patch.object(MockLLM, "achat", two_hop_incremental_mock_achat)
def test_two_hop_incremental_query_gen():
	multi_hop_qa = QA(multi_hop_qa_df)
	new_qa = multi_hop_qa.batch_apply(two_hop_incremental, llm=llm, lang="ko")
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(multi_hop_qa_df)
	assert (
		new_qa.data["query"].iloc[0]
		== "In which Mexican state can one find the Ciudad Deportiva, home to the Tecolotes de Nuevo Laredo?"
	)
