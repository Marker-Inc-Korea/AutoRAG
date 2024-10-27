import time
from unittest.mock import patch

import pandas as pd
from llama_index.llms.openai import OpenAI
from openai import AsyncOpenAI
from openai.types.chat import (
	ParsedChatCompletion,
	ParsedChoice,
	ParsedChatCompletionMessage,
)
import openai.resources.beta.chat

from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole
from autorag.data.qa.filter.passage_dependency import (
	passage_dependency_filter_openai,
	passage_dependency_filter_llama_index,
	Response,
)
from autorag.data.qa.schema import QA

en_qa_df = pd.DataFrame(
	{
		"query": [
			"What is the most significant discovery mentioned in the research paper?",
			"What was the ruling in the case described in this legal brief?",
			"What is the passage reranker role in the Advanced RAG system?",
			"Who is the latest 30 homerun 30 steal record owner in the KBO league?",
		]
	}
)

ko_qa_df = pd.DataFrame(
	{
		"query": [
			"연구 논문에서 언급된 가장 중요한 발견은 무엇입니까?",
			"이 판결문에 기술된 사건의 판결은 무엇이었습니까?",
			"Advanced RAG 시스템에서 리랭커 역할은 무엇입니까?",
			"KBO 리그에서 가장 최근 30홈런 30도루를 기록한 선수는 누구입니까?",
		]
	}
)

ja_qa_df = pd.DataFrame(
	{
		"query": [
			"研究論文で言及された最も重要な発見は何ですか？",
			"この判例に記述された判決は何ですか？",
			"Advanced RAGシステムにおけるリランカーの役割は何ですか?",
			"KBOリーグで最新の30本塁打30盗塁の記録保持者は誰ですか?",
		]
	}
)

expected_df_en = pd.DataFrame(
	{
		"query": [
			"What is the passage reranker role in the Advanced RAG system?",
			"Who is the latest 30 homerun 30 steal record owner in the KBO league?",
		]
	}
)

expected_df_ko = pd.DataFrame(
	{
		"query": [
			"Advanced RAG 시스템에서 리랭커 역할은 무엇입니까?",
			"KBO 리그에서 가장 최근 30홈런 30도루를 기록한 선수는 누구입니까?",
		]
	}
)

expected_df_ja = pd.DataFrame(
	{
		"query": [
			"Advanced RAGシステムにおけるリランカーの役割は何ですか?",
			"KBOリーグで最新の30本塁打30盗塁の記録保持者は誰ですか?",
		]
	}
)


passage_dependent_response = [
	"What is the most significant discovery mentioned in the research paper?",
	"What was the ruling in the case described in this legal brief?",
	"연구 논문에서 언급된 가장 중요한 발견은 무엇입니까?",
	"이 판결문에 기술된 사건의 판결은 무엇이었습니까?",
	"研究論文で言及された最も重要な発見は何ですか？",
	"この判例に記述された判決は何ですか？",
]


async def mock_openai_response(*args, **kwargs) -> ParsedChatCompletion[Response]:
	user_prompt = kwargs["messages"][1]["content"]
	return ParsedChatCompletion(
		id="test_id",
		choices=[
			ParsedChoice(
				finish_reason="stop",
				index=0,
				message=ParsedChatCompletionMessage(
					parsed=Response(
						is_passage_dependent=user_prompt.split("\n")[0]
						.split(":")[1]
						.strip()
						in passage_dependent_response
					),
					role="assistant",
				),
			)
		],
		created=int(time.time()),
		model="gpt-4o-mini-2024-07-18",
		object="chat.completion",
	)


async def mock_llama_index_response(*args, **kwargs) -> ChatResponse:
	user_prompt = kwargs["messages"][1].content
	return ChatResponse(
		message=ChatMessage(
			role=MessageRole.ASSISTANT,
			content=str(
				user_prompt.split("\n")[0].split(":")[1].strip()
				in passage_dependent_response
			),
		)
	)


@patch.object(
	openai.resources.beta.chat.completions.AsyncCompletions,
	"parse",
	mock_openai_response,
)
def test_passage_dependency_filter_openai():
	client = AsyncOpenAI()
	en_qa = QA(en_qa_df)
	result_en_qa = en_qa.batch_filter(
		passage_dependency_filter_openai, client=client, lang="en"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_en_qa.data, expected_df_en)

	ko_qa = QA(ko_qa_df)
	result_ko_qa = ko_qa.batch_filter(
		passage_dependency_filter_openai, client=client, lang="ko"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_ko_qa.data, expected_df_ko)

	ja_qa = QA(ja_qa_df)
	result_ja_qa = ja_qa.batch_filter(
		passage_dependency_filter_openai, client=client, lang="ja"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_ja_qa.data, expected_df_ja)


@patch.object(
	OpenAI,
	"achat",
	mock_llama_index_response,
)
def test_passage_dependency_filter_llama_index():
	llm = OpenAI(temperature=0, model="gpt-4o-mini")
	en_qa = QA(en_qa_df)
	result_en_qa = en_qa.batch_filter(
		passage_dependency_filter_llama_index, llm=llm, lang="en"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_en_qa.data, expected_df_en)

	ko_qa = QA(ko_qa_df)
	result_ko_qa = ko_qa.batch_filter(
		passage_dependency_filter_llama_index, llm=llm, lang="ko"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_ko_qa.data, expected_df_ko)

	ja_qa = QA(ja_qa_df)
	result_ja_qa = ja_qa.batch_filter(
		passage_dependency_filter_llama_index, llm=llm, lang="ja"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_ja_qa.data, expected_df_ja)
