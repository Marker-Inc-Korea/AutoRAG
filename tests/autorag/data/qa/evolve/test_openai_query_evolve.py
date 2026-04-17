from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from openai import AsyncOpenAI
import openai.resources.responses
from autorag.data.qa.evolve.openai_query_evolve import (
	conditional_evolve_ragas,
	Response,
	reasoning_evolve_ragas,
	compress_ragas,
)
from autorag.data.qa.schema import QA
from tests.autorag.data.qa.evolve.base_test_query_evolve import qa_df


@dataclass
class _MockParsedResponse:
	output_parsed: Any


async def mock_gen_gt_response(*args, **kwargs) -> _MockParsedResponse:
	return _MockParsedResponse(output_parsed=Response(evolved_query="mock answer"))


client = AsyncOpenAI()


@patch.object(
	openai.resources.responses.AsyncResponses,
	"parse",
	mock_gen_gt_response,
)
def test_conditional_evolve_ragas():
	qa = QA(qa_df)
	new_qa = qa.batch_apply(conditional_evolve_ragas, client=client)
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)
	assert all(
		x != y for x, y in zip(new_qa.data["query"].tolist(), qa_df["query"].tolist())
	)


@patch.object(
	openai.resources.responses.AsyncResponses,
	"parse",
	mock_gen_gt_response,
)
def test_reasoning_evolve_ragas():
	qa = QA(qa_df)
	new_qa = qa.batch_apply(reasoning_evolve_ragas, client=client)
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)
	assert all(
		x != y for x, y in zip(new_qa.data["query"].tolist(), qa_df["query"].tolist())
	)


@patch.object(
	openai.resources.responses.AsyncResponses,
	"parse",
	mock_gen_gt_response,
)
def test_compress_ragas():
	qa = QA(qa_df)
	new_qa = qa.batch_apply(compress_ragas, client=client)
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)
	assert all(
		x != y for x, y in zip(new_qa.data["query"].tolist(), qa_df["query"].tolist())
	)
