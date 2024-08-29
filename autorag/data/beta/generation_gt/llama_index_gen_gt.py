import itertools
from typing import Dict


from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import MessageRole, ChatMessage

from autorag.data.beta.generation_gt.prompt import (
	concise_answer_system_prompt,
	basic_answer_system_prompt,
)


async def make_gen_gt_llama_index(row: Dict, llm: BaseLLM, system_prompt: str):
	retrieval_gt_contents = list(
		itertools.chain.from_iterable(row["retrieval_gt_contents"])
	)
	query = row["query"]
	passage_str = "\n".join(retrieval_gt_contents)
	user_prompt = f"Text:\n<|text_start|>\n{passage_str}\n<|text_end|>\n\nQuestion:\n{query}\n\nAnswer:"

	response = await llm.achat(
		messages=[
			ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
			ChatMessage(role=MessageRole.USER, content=user_prompt),
		],
		temperature=0.0,
	)
	row["generation_gt"] = response.message.content
	return row


async def make_concise_gen_gt(row: Dict, llm: BaseLLM) -> Dict:
	return await make_gen_gt_llama_index(row, llm, concise_answer_system_prompt)


async def make_basic_gen_gt(row: Dict, llm: BaseLLM) -> Dict:
	return await make_gen_gt_llama_index(row, llm, basic_answer_system_prompt)
