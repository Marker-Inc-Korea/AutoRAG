import time
from unittest.mock import patch

import openai.resources.beta.chat
from openai import AsyncOpenAI
from openai.types.chat import (
	ParsedChatCompletion,
	ParsedChatCompletionMessage,
	ParsedChoice,
)

from autorag.data.beta.query.openai_gen_query import Response, factoid_query_gen
from autorag.data.beta.schema.data import QA
from tests.autorag.data.beta.query.base_test_query_gen import qa_df


client = AsyncOpenAI()


async def mock_gen_gt_response(*args, **kwargs) -> ParsedChatCompletion[Response]:
	return ParsedChatCompletion(
		id="test_id",
		choices=[
			ParsedChoice(
				finish_reason="stop",
				index=0,
				message=ParsedChatCompletionMessage(
					parsed=Response(query="mock answer"),
					role="assistant",
				),
			)
		],
		created=int(time.time()),
		model="gpt-4o-mini-2024-07-18",
		object="chat.completion",
	)


@patch.object(
	openai.resources.beta.chat.completions.AsyncCompletions,
	"parse",
	mock_gen_gt_response,
)
def test_make_factoid_query_gen():
	qa = QA(qa_df)
	new_qa = qa.batch_apply(factoid_query_gen, client=client)
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)


@patch.object(
	openai.resources.beta.chat.completions.AsyncCompletions,
	"parse",
	mock_gen_gt_response,
)
def test_make_factoid_query_gen_ko():
	qa = QA(qa_df)
	new_qa = qa.batch_apply(factoid_query_gen, client=client, lang="ko")
	assert "query" in new_qa.data.columns
	assert all(isinstance(query, str) for query in new_qa.data["query"].tolist())
	assert len(new_qa.data) == len(qa_df)
