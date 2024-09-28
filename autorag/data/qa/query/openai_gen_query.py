import itertools
from typing import Dict, List

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai.utils import to_openai_message_dicts
from openai import AsyncClient
from pydantic import BaseModel

from autorag.data.qa.query.prompt import QUERY_GEN_PROMPT


class Response(BaseModel):
	query: str


# Single hop QA generation OpenAI
async def query_gen_openai_base(
	row: Dict,
	client: AsyncClient,
	messages: List[ChatMessage],
	model_name: str = "gpt-4o-2024-08-06",
):
	context = list(itertools.chain.from_iterable(row["retrieval_gt_contents"]))
	context_str = "Text:\n" + "\n".join(
		[f"{i + 1}. {c}" for i, c in enumerate(context)]
	)
	user_prompt = f"{context_str}\n\nGenerated Question from the Text:\n"
	messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

	completion = await client.beta.chat.completions.parse(
		model=model_name,
		messages=to_openai_message_dicts(messages),
		response_format=Response,
	)
	row["query"] = completion.choices[0].message.parsed.query
	return row


async def factoid_query_gen(
	row: Dict,
	client: AsyncClient,
	model_name: str = "gpt-4o-2024-08-06",
	lang: str = "en",
) -> Dict:
	return await query_gen_openai_base(
		row, client, QUERY_GEN_PROMPT["factoid_single_hop"][lang], model_name
	)


async def concept_completion_query_gen(
	row: Dict,
	client: AsyncClient,
	model_name: str = "gpt-4o-2024-08-06",
	lang: str = "en",
) -> Dict:
	return await query_gen_openai_base(
		row, client, QUERY_GEN_PROMPT["factoid_single_hop"][lang], model_name
	)


class TwoHopIncrementalResponse(BaseModel):
	answer: str
	one_hop_question: str
	two_hop_question: str


async def two_hop_incremental(
	row: Dict,
	client: AsyncClient,
	model_name: str = "gpt-4o-2024-08-06",
	lang: str = "en",
) -> Dict:
	"""
	Create a two-hop question using incremental prompt.
	Incremental prompt is more effective to create multi-hop question.
	The input retrieval_gt has to include more than one passage.

	:return: The two-hop question using openai incremental prompt
	"""
	messages = QUERY_GEN_PROMPT["two_hop_incremental"][lang]
	passages = row["retrieval_gt_contents"]
	assert (
		len(passages) >= 2
	), "You have to sample more than two passages for making two-hop questions."
	context_str = f"Document 1: {passages[0][0]}\nDocument 2: {passages[1][0]}"
	user_prompt = f"{context_str}\n\nGenerated two-hop Question from two Documents:\n"
	messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

	completion = await client.beta.chat.completions.parse(
		model=model_name,
		messages=to_openai_message_dicts(messages),
		response_format=TwoHopIncrementalResponse,
	)
	row["query"] = completion.choices[0].message.parsed.two_hop_question
	return row
