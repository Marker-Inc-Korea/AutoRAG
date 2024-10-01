import itertools
from typing import Dict, List

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai.utils import to_openai_message_dicts
from openai import AsyncClient
from pydantic import BaseModel

from autorag.data.qa.evolve.prompt import QUERY_EVOLVE_PROMPT


class Response(BaseModel):
	evolved_query: str


async def query_evolve_openai_base(
	row: Dict,
	client: AsyncClient,
	messages: List[ChatMessage],
	model_name: str = "gpt-4o-2024-08-06",
):
	"""
	Evolve the original query to a new evolved query using OpenAI structured outputs.
	"""
	original_query = row["query"]
	context = list(itertools.chain.from_iterable(row["retrieval_gt_contents"]))
	context_str = "Text:\n" + "\n".join(
		[f"{i + 1}. {c}" for i, c in enumerate(context)]
	)
	user_prompt = f"Question: {original_query}\nContext: {context_str}\nOutput: "
	messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

	completion = await client.beta.chat.completions.parse(
		model=model_name,
		messages=to_openai_message_dicts(messages),
		response_format=Response,
	)
	row["query"] = completion.choices[0].message.parsed.evolved_query
	return row


async def conditional_evolve_ragas(
	row: Dict,
	client: AsyncClient,
	model_name: str = "gpt-4o-2024-08-06",
	lang: str = "en",
) -> Dict:
	return await query_evolve_openai_base(
		row, client, QUERY_EVOLVE_PROMPT["conditional_evolve_ragas"][lang], model_name
	)


async def reasoning_evolve_ragas(
	row: Dict,
	client: AsyncClient,
	model_name: str = "gpt-4o-2024-08-06",
	lang: str = "en",
) -> Dict:
	return await query_evolve_openai_base(
		row, client, QUERY_EVOLVE_PROMPT["reasoning_evolve_ragas"][lang], model_name
	)


async def compress_ragas(
	row: Dict,
	client: AsyncClient,
	model_name: str = "gpt-4o-2024-08-06",
	lang: str = "en",
) -> Dict:
	original_query = row["query"]
	messages = QUERY_EVOLVE_PROMPT["compress_ragas"][lang]
	user_prompt = f"Question: {original_query}\nOutput: "
	messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

	completion = await client.beta.chat.completions.parse(
		model=model_name,
		messages=to_openai_message_dicts(messages),
		response_format=Response,
	)
	row["query"] = completion.choices[0].message.parsed.evolved_query
	return row
