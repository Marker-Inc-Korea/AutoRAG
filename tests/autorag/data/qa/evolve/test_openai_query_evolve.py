import time
from unittest.mock import patch

from openai import AsyncOpenAI
import openai.resources.beta.chat
from autorag.data.qa.evolve.openai_query_evolve import (
	conditional_evolve_ragas,
	Response,
	reasoning_evolve_ragas,
	compress_ragas,
)
from autorag.data.qa.schema import QA
from tests.autorag.data.qa.evolve.base_test_query_evolve import qa_df
from openai.types.chat import (
	ParsedChatCompletion,
	ParsedChatCompletionMessage,
	ParsedChoice,
)


async def mock_gen_gt_response(*args, **kwargs) -> ParsedChatCompletion[Response]:
	return ParsedChatCompletion(
		id="test_id",
		choices=[
			ParsedChoice(
				finish_reason="stop",
				index=0,
				message=ParsedChatCompletionMessage(
					parsed=Response(evolved_query="mock answer"),
					role="assistant",
				),
			)
		],
		created=int(time.time()),
		model="gpt-4o-mini-2024-07-18",
		object="chat.completion",
	)


client = AsyncOpenAI()


@patch.object(
	openai.resources.beta.chat.completions.AsyncCompletions,
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
	openai.resources.beta.chat.completions.AsyncCompletions,
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
	openai.resources.beta.chat.completions.AsyncCompletions,
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
