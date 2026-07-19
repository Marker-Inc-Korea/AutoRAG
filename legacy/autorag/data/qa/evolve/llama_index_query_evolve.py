import itertools
from typing import Dict, List

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole

from autorag.data.qa.evolve.prompt import QUERY_EVOLVE_PROMPT


async def llama_index_generate_base(
	row: Dict,
	llm: BaseLLM,
	messages: List[ChatMessage],
) -> Dict:
	original_query = row["query"]
	context = list(itertools.chain.from_iterable(row["retrieval_gt_contents"]))
	context_str = "Text:\n" + "\n".join(
		[f"{i + 1}. {c}" for i, c in enumerate(context)]
	)
	user_prompt = f"Question: {original_query}\nContext: {context_str}\nOutput: "
	messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

	chat_response: ChatResponse = await llm.achat(messages=messages)
	row["query"] = chat_response.message.content
	return row


async def conditional_evolve_ragas(
	row: Dict,
	llm: BaseLLM,
	lang: str = "en",
) -> Dict:
	return await llama_index_generate_base(
		row,
		llm,
		QUERY_EVOLVE_PROMPT["conditional_evolve_ragas"][lang],
	)


async def reasoning_evolve_ragas(
	row: Dict,
	llm: BaseLLM,
	lang: str = "en",
) -> Dict:
	return await llama_index_generate_base(
		row,
		llm,
		QUERY_EVOLVE_PROMPT["reasoning_evolve_ragas"][lang],
	)


async def compress_ragas(
	row: Dict,
	llm: BaseLLM,
	lang: str = "en",
) -> Dict:
	original_query = row["query"]
	user_prompt = f"Question: {original_query}\nOutput: "
	messages = QUERY_EVOLVE_PROMPT["compress_ragas"][lang]
	messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

	chat_response: ChatResponse = await llm.achat(messages=messages)
	row["query"] = chat_response.message.content
	return row
