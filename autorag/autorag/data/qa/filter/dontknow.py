from typing import Dict, List

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse
from llama_index.llms.openai.utils import to_openai_message_dicts
from openai import AsyncClient
from pydantic import BaseModel

from autorag.data.qa.filter.prompt import FILTER_PROMPT

dont_know_phrases = {
	"en": [
		"I don't know",
		"I do not know",
		"Don't know",
		"Do not know",
	],
	"ko": [
		"몰라요",
		"모르겠습니다",
		"모르겠어요",
		"몰라",
		"내가 어떻게 알아?",
		"모르겠소",
		"몰라유",
		"모르것는디",
		"모르겠어유",
		"모르겠네유",
		"모르겠네요",
	],
	"ja": [
		"知りません",
		"わかりません",
		"分かりません",
		"知らないです",
		"よく分かってません",
		"わかりかねます",
		"存じません",
		"お答えいたしかねます",
	],
}


def dontknow_filter_rule_based(row: Dict, lang: str = "en") -> bool:
	assert (
		"generation_gt" in row.keys()
	), "generation_gt column is not in the DataFrame."
	dont_know_phrase = dont_know_phrases[lang]
	return not any(
		phrase in s for phrase in dont_know_phrase for s in row["generation_gt"]
	)


class Response(BaseModel):
	is_dont_know: bool


async def dontknow_filter_openai(
	row: Dict,
	client: AsyncClient,
	model_name: str = "gpt-4o-mini-2024-07-18",
	lang: str = "en",
) -> bool:
	"""
	This will drop rows that have a "don't know" answer.
	It will drop unanswerable questions from the QA dataset.
	You can use this filter with the ` batch_filter ` function at `QA` class.

	:param row: The row dict from QA dataset.
	:param client: The OpenAI client.
	:param model_name: The model name.
		You have to use gpt-4o-2024-08-06 or gpt-4o-mini-2024-07-18.
	:param lang: The supported language is en, ko or ja.
	:return: False if the row generation_gt is a "don't know" meaning.
	"""
	assert "generation_gt" in row.keys(), "generation_gt column is not in the row."
	system_prompt: List[ChatMessage] = FILTER_PROMPT["dontknow_filter"][lang]
	result = []
	for gen_gt in row["generation_gt"]:
		completion = await client.beta.chat.completions.parse(
			model=model_name,
			messages=to_openai_message_dicts(
				system_prompt + [ChatMessage(role=MessageRole.USER, content=gen_gt)]
			),
			response_format=Response,
		)
		result.append(completion.choices[0].message.parsed.is_dont_know)
	return not any(result)


async def dontknow_filter_llama_index(
	row: Dict,
	llm: BaseLLM,
	lang: str = "en",
) -> bool:
	"""
	This will drop rows that have a "don't know" answer.
	It will drop unanswerable questions from the QA dataset.
	You can use this filter with the ` batch_filter ` function at `QA` class.

	:param row: The row dict from QA dataset.
	:param llm: The Llama index llm instance.
		It will be good if you set max tokens to low for saving tokens.
	:param lang: The supported language is en, ko or ja.
	:return: False if the row generation_gt is a "don't know" meaning.
	"""
	assert "generation_gt" in row.keys(), "generation_gt column is not in the row."
	system_prompt: List[ChatMessage] = FILTER_PROMPT["dontknow_filter"][lang]
	results = []
	for gen_gt in row["generation_gt"]:
		response: ChatResponse = await llm.achat(
			messages=system_prompt
			+ [ChatMessage(role=MessageRole.USER, content=gen_gt)]
		)
		result_str = response.message.content
		results.append("true" in result_str.lower().strip())
	return not any(results)
