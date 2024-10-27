import time
from unittest.mock import patch

import pandas as pd
from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from openai import AsyncOpenAI
from openai.types.chat import (
	ParsedChatCompletion,
	ParsedChoice,
	ParsedChatCompletionMessage,
)
import openai.resources.beta.chat

from autorag.data.qa.filter.dontknow import (
	dontknow_filter_rule_based,
	dontknow_filter_openai,
	Response,
	dontknow_filter_llama_index,
)
from autorag.data.qa.schema import QA

en_qa_df = pd.DataFrame(
	{
		"generation_gt": [
			["I don't know what to say", "This is a test"],
			["This is another test", "I do not know the answer"],
			["This is fine", "All good"],
		]
	}
)

ko_qa_df = pd.DataFrame(
	{
		"generation_gt": [
			["몰라요", "테스트입니다"],
			["모르겠습니다", "이것은 테스트입니다"],
			["모르겠어요", "이것은 또 다른 테스트입니다"],
			["이것은 괜찮습니다", "모든 것이 좋습니다"],
		]
	}
)

ja_qa_df = pd.DataFrame(
	{
		"generation_gt": [
			["わかりません", "これはテストです"],
			["答えがわかりません", "これは別のテストです"],
			["これは大丈夫です", "すべて問題ありません"],
		]
	}
)

dont_know_lang = [
	"I don't know what to say",
	"I do not know the answer",
	"몰라요",
	"모르겠습니다",
	"모르겠어요",
	"わかりません",
	"答えがわかりません",
]

# Expected data after filtering
expected_df_en = pd.DataFrame({"generation_gt": [["This is fine", "All good"]]})

expected_df_ko = pd.DataFrame(
	{"generation_gt": [["이것은 괜찮습니다", "모든 것이 좋습니다"]]}
)
expected_df_ja = pd.DataFrame(
	{"generation_gt": [["これは大丈夫です", "すべて問題ありません"]]}
)


async def mock_openai_response(*args, **kwargs) -> ParsedChatCompletion[Response]:
	user_prompt = kwargs["messages"][1]["content"]
	return ParsedChatCompletion(
		id="test_id",
		choices=[
			ParsedChoice(
				finish_reason="stop",
				index=0,
				message=ParsedChatCompletionMessage(
					parsed=Response(is_dont_know=user_prompt in dont_know_lang),
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
			role=MessageRole.ASSISTANT, content=str(user_prompt in dont_know_lang)
		)
	)


def test_dontknow_filter_rule_based():
	# Test for English
	en_qa = QA(en_qa_df)
	result_en_qa = en_qa.filter(dontknow_filter_rule_based, lang="en").map(
		lambda df: df.reset_index(drop=True)
	)
	pd.testing.assert_frame_equal(result_en_qa.data, expected_df_en)

	# Test for Korean
	ko_qa = QA(ko_qa_df)
	result_ko_qa = ko_qa.filter(dontknow_filter_rule_based, lang="ko").map(
		lambda df: df.reset_index(drop=True)
	)
	pd.testing.assert_frame_equal(result_ko_qa.data, expected_df_ko)
	# Test for Japanese
	ja_qa = QA(ja_qa_df)
	result_ja_qa = ja_qa.filter(dontknow_filter_rule_based, lang="ja").map(
		lambda df: df.reset_index(drop=True)
	)
	pd.testing.assert_frame_equal(result_ja_qa.data, expected_df_ja)


@patch.object(
	openai.resources.beta.chat.completions.AsyncCompletions,
	"parse",
	mock_openai_response,
)
def test_dontknow_filter_openai():
	client = AsyncOpenAI()
	en_qa = QA(en_qa_df)
	result_en_qa = en_qa.batch_filter(
		dontknow_filter_openai, client=client, lang="en"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_en_qa.data, expected_df_en)

	# Test for Korean
	ko_qa = QA(ko_qa_df)
	result_ko_qa = ko_qa.batch_filter(
		dontknow_filter_openai, client=client, lang="ko"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_ko_qa.data, expected_df_ko)

	# Test for Japanese
	ja_qa = QA(ja_qa_df)
	result_ja_qa = ja_qa.batch_filter(
		dontknow_filter_openai, client=client, lang="ja"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_ja_qa.data, expected_df_ja)


@patch.object(
	OpenAI,
	"achat",
	mock_llama_index_response,
)
def test_dontknow_filter_llama_index():
	llm = OpenAI()
	en_qa = QA(en_qa_df)
	result_en_qa = en_qa.batch_filter(
		dontknow_filter_llama_index, llm=llm, lang="en"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_en_qa.data, expected_df_en)

	ko_qa = QA(ko_qa_df)
	result_ko_qa = ko_qa.batch_filter(
		dontknow_filter_llama_index, llm=llm, lang="ko"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_ko_qa.data, expected_df_ko)

	ja_qa = QA(ja_qa_df)
	result_ja_qa = ja_qa.batch_filter(
		dontknow_filter_llama_index, llm=llm, lang="ja"
	).map(lambda df: df.reset_index(drop=True))
	pd.testing.assert_frame_equal(result_ja_qa.data, expected_df_ja)
