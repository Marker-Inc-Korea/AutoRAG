import itertools
from typing import Dict


from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import MessageRole, ChatMessage

from autorag.data.qa.generation_gt.base import add_gen_gt
from autorag.data.qa.generation_gt.prompt import GEN_GT_SYSTEM_PROMPT


async def make_gen_gt_llama_index(row: Dict, llm: BaseLLM, system_prompt: str) -> Dict:
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
	return add_gen_gt(row, response.message.content)


async def make_concise_gen_gt(row: Dict, llm: BaseLLM, lang: str = "en") -> Dict:
	return await make_gen_gt_llama_index(
		row, llm, GEN_GT_SYSTEM_PROMPT["concise"][lang]
	)


async def make_basic_gen_gt(row: Dict, llm: BaseLLM, lang: str = "en") -> Dict:
	return await make_gen_gt_llama_index(row, llm, GEN_GT_SYSTEM_PROMPT["basic"][lang])


async def make_custom_gen_gt(row: Dict, llm: BaseLLM, system_prompt: str) -> Dict:
	return await make_gen_gt_llama_index(row, llm, system_prompt)
