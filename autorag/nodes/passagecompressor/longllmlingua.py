from typing import List, Optional

import torch
from llmlingua import PromptCompressor

from autorag.nodes.passagecompressor.base import passage_compressor_node


@passage_compressor_node
def longllmlingua(
	queries: List[str],
	contents: List[List[str]],
	scores,
	ids,
	model_name: str = "NousResearch/Llama-2-7b-hf",
	instructions: Optional[str] = None,
	target_token: int = 300,
	**kwargs,
) -> List[str]:
	"""
	Compresses the retrieved texts using LongLLMLingua.
	For more information, visit https://github.com/microsoft/LLMLingua.

	:param queries: The queries for retrieved passages.
	:param contents: The contents of retrieved passages.
	:param scores: The scores of retrieved passages.
	    Do not use in this function, so you can pass an empty list.
	:param ids: The ids of retrieved passages.
	    Do not use in this function, so you can pass an empty list.
	:param model_name: The model name to use for compression.
	    Default is "NousResearch/Llama-2-7b-hf".
	:param instructions: The instructions for compression.
	    Default is None. When it is None, it will use default instructions.
	:param target_token: The target token for compression.
	    Default is 300.
	:param kwargs: Additional keyword arguments.
	:return: The list of compressed texts.
	"""
	if instructions is None:
		instructions = "Given the context, please answer the final question"
	llm_lingua = PromptCompressor(
		model_name=model_name,
	)
	results = [
		llmlingua_pure(
			query, contents_, llm_lingua, instructions, target_token, **kwargs
		)
		for query, contents_ in zip(queries, contents)
	]

	del llm_lingua
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return results


def llmlingua_pure(
	query: str,
	contents: List[str],
	llm_lingua: PromptCompressor,
	instructions: str,
	target_token: int = 300,
	**kwargs,
) -> str:
	"""
	Return the compressed text.

	:param query: The query for retrieved passages.
	:param contents: The contents of retrieved passages.
	:param llm_lingua: The llm instance that will be used to compress.
	:param instructions: The instructions for compression.
	:param target_token: The target token for compression.
	    Default is 300.
	:param kwargs: Additional keyword arguments.
	:return: The compressed text.
	"""
	# split by "\n\n" (recommended by LongLLMLingua authors)
	new_context_texts = [c for context in contents for c in context.split("\n\n")]
	compressed_prompt = llm_lingua.compress_prompt(
		new_context_texts,
		question=query,
		instruction=instructions,
		rank_method="longllmlingua",
		target_token=target_token,
		**kwargs,
	)
	compressed_prompt_txt = compressed_prompt["compressed_prompt"]

	# separate out the question and instruction
	result = "\n\n".join(compressed_prompt_txt.split("\n\n")[1:-1])

	return result
