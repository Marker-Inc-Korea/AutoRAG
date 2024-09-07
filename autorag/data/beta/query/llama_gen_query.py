import itertools
from typing import Dict, List

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole

from autorag.data.beta.query.prompt import QUERY_GEN_PROMPT


async def llama_index_generate_base(
	row: Dict,
	llm: BaseLLM,
	messages: List[ChatMessage],
) -> Dict:
	context = list(itertools.chain.from_iterable(row["retrieval_gt_contents"]))
	context_str = "Text:\n" + "\n".join(
		[f"{i + 1}. {c}" for i, c in enumerate(context)]
	)
	user_prompt = f"Text:\n{context_str}\n\nGenerated Question from the Text:\n"
	messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

	chat_response: ChatResponse = await llm.achat(messages=messages)
	row["query"] = chat_response.message.content
	return row


async def factoid_query_gen(
	row: Dict,
	llm: BaseLLM,
	lang: str = "en",
) -> Dict:
	return await llama_index_generate_base(
		row, llm, QUERY_GEN_PROMPT["factoid_single_hop"][lang]
	)


async def concept_completion_query_gen(
	row: Dict,
	llm: BaseLLM,
	lang: str = "en",
) -> Dict:
	return await llama_index_generate_base(
		row, llm, QUERY_GEN_PROMPT["concept_completion"][lang]
	)


async def two_hop_incremental(
	row: Dict,
	llm: BaseLLM,
	lang: str = "en",
) -> Dict:
	messages = QUERY_GEN_PROMPT["two_hop_incremental"][lang]
	passages = row["retrieval_gt_contents"]
	assert (
		len(passages) >= 2
	), "You have to sample more than two passages for making two-hop questions."
	context_str = f"Document 1: {passages[0][0]}\nDocument 2: {passages[1][0]}"
	user_prompt = f"{context_str}\n\nGenerated two-hop Question from two Documents:\n"
	messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

	chat_response: ChatResponse = await llm.achat(messages=messages)
	response = chat_response.message.content
	row["query"] = response.split(":")[-1].strip()
	return row
