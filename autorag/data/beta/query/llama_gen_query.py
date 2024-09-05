import itertools
from typing import Dict

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole

from autorag.data.beta.query.prompt import QUERY_GEN_SYSTEM_PROMPT


async def llama_index_generate_base(
	row: Dict,
	llm: BaseLLM,
	system_prompt: str,
) -> Dict:
	context = list(itertools.chain.from_iterable(row["retrieval_gt_contents"]))
	context_str = "Text:\n" + "\n".join(
		[f"{i + 1}. {c}" for i, c in enumerate(context)]
	)
	user_prompt = f"Text:\n{context_str}\n\nGenerated Question from the Text:\n"

	chat_response: ChatResponse = await llm.achat(
		messages=[
			ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
			ChatMessage(role=MessageRole.USER, content=user_prompt),
		]
	)
	row["query"] = chat_response.message.content
	return row


async def factoid_query_gen(
	row: Dict,
	llm: BaseLLM,
	lang: str = "en",
) -> Dict:
	return await llama_index_generate_base(
		row, llm, QUERY_GEN_SYSTEM_PROMPT["factoid_single_hop"][lang]
	)
