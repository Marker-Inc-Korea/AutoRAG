from typing import Dict, List

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse
from llama_index.llms.openai.utils import to_openai_message_dicts
from openai import AsyncClient
from pydantic import BaseModel

from autorag.data.qa.filter.prompt import FILTER_PROMPT


class Response(BaseModel):
	is_passage_dependent: bool


async def passage_dependency_filter_openai(
	row: Dict,
	client: AsyncClient,
	model_name: str = "gpt-4o-mini-2024-07-18",
	lang: str = "en",
) -> bool:
	"""
	This will drop passage-dependent question rows.
	Passage-dependent questions are questions that the answer will change depending on what passage you choose.
	The passage-dependent questions will not be good for RAG evaluation, because any retrieval system can't find the right passage with passage-dependent question.
	For example, when someone asks "What is the highest score according to the table?" the answer will be different depending on the table.
	And what is the table? The retrieval system can't find the right passage with this question.
	You can use this filter with the ` batch_filter ` function at `QA` class.

	:param row: The row dict from QA dataset.
	    :param client: The OpenAI client.
	    :param model_name: The model name.
	            You have to use gpt-4o-2024-08-06 or gpt-4o-mini-2024-07-18.
	    :param lang: The supported language is en, ko or ja.
	:return: False if the row question is a passage-dependent question (to be filtered).
	"""
	assert "query" in row.keys(), "query column is not in the row."
	system_prompt: List[ChatMessage] = FILTER_PROMPT["passage_dependency"][lang]
	query = row["query"]
	completion = await client.beta.chat.completions.parse(
		model=model_name,
		messages=to_openai_message_dicts(
			system_prompt
			+ [
				ChatMessage(
					role=MessageRole.USER,
					content=f"Question: {query}\nIs this the question passage dependent?",
				)
			]
		),
		response_format=Response,
	)
	return not completion.choices[0].message.parsed.is_passage_dependent


async def passage_dependency_filter_llama_index(
	row: Dict,
	llm: BaseLLM,
	lang: str = "en",
) -> bool:
	"""
	This will drop passage-dependent question rows.
	Passage-dependent questions are questions that the answer will change depending on what passage you choose.
	The passage-dependent questions will not be good for RAG evaluation, because any retrieval system can't find the right passage with passage-dependent question.
	For example, when someone asks "What is the highest score according to the table?" the answer will be different depending on the table.
	And what is the table? The retrieval system can't find the right passage with this question.
	You can use this filter with the ` batch_filter ` function at `QA` class.

	:param row: The row dict from QA dataset.
	:param llm: The Llama index llm instance.
	            It will be good if you set max tokens to low for saving tokens.
	    :param lang: The supported language is en, ko or ja.
	:return: False if the row question is a passage-dependent question (to be filtered).
	"""
	assert "query" in row.keys(), "query column is not in the row."
	system_prompt: List[ChatMessage] = FILTER_PROMPT["passage_dependency"][lang]
	query = row["query"]
	response: ChatResponse = await llm.achat(
		messages=system_prompt
		+ [
			ChatMessage(
				role=MessageRole.USER,
				content=f"Question: {query}\nIs this the question passage dependent?",
			)
		]
	)
	result_str = response.message.content
	return "true" not in result_str.lower().strip()
