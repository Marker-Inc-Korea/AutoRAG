import itertools
from typing import Dict

from openai import AsyncClient
from pydantic import BaseModel

from autorag.data.qa.generation_gt.base import add_gen_gt
from autorag.data.qa.generation_gt.prompt import GEN_GT_SYSTEM_PROMPT


class Response(BaseModel):
	answer: str


async def make_gen_gt_openai(
	row: Dict,
	client: AsyncClient,
	system_prompt: str,
	model_name: str = "gpt-4o-2024-08-06",
):
	retrieval_gt_contents = list(
		itertools.chain.from_iterable(row["retrieval_gt_contents"])
	)
	query = row["query"]
	passage_str = "\n".join(retrieval_gt_contents)
	user_prompt = f"Text:\n<|text_start|>\n{passage_str}\n<|text_end|>\n\nQuestion:\n{query}\n\nAnswer:"

	completion = await client.beta.chat.completions.parse(
		model=model_name,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		temperature=0.0,
		response_format=Response,
	)
	response: Response = completion.choices[0].message.parsed
	return add_gen_gt(row, response.answer)


async def make_concise_gen_gt(
	row: Dict,
	client: AsyncClient,
	model_name: str = "gpt-4o-2024-08-06",
	lang: str = "en",
):
	"""
	Generate concise generation_gt using OpenAI Structured Output for preventing errors.
	It generates a concise answer, so it is generally a word or just a phrase.

	:param row: The input row of the qa dataframe.
	:param client: The OpenAI async client.
	:param model_name: The model name that supports structured output.
	    It has to be "gpt-4o-2024-08-06" or "gpt-4o-mini-2024-07-18".
	:param lang: The language code of the prompt.
		Default is "en".
	:return: The output row of the qa dataframe with added "generation_gt" in it.
	"""
	return await make_gen_gt_openai(
		row, client, GEN_GT_SYSTEM_PROMPT["concise"][lang], model_name
	)


async def make_basic_gen_gt(
	row: Dict,
	client: AsyncClient,
	model_name: str = "gpt-4o-2024-08-06",
	lang: str = "en",
):
	"""
	Generate basic generation_gt using OpenAI Structured Output for preventing errors.
	It generates a "basic" answer, and its prompt is simple.

	:param row: The input row of the qa dataframe.
	:param client: The OpenAI async client.
	:param model_name: The model name that supports structured output.
	    It has to be "gpt-4o-2024-08-06" or "gpt-4o-mini-2024-07-18".
	:param lang: The language code of the prompt.
		Default is "en".
	:return: The output row of the qa dataframe with added "generation_gt" in it.
	"""
	return await make_gen_gt_openai(
		row, client, GEN_GT_SYSTEM_PROMPT["basic"][lang], model_name
	)
