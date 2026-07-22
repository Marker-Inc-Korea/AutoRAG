from typing import List, Optional

import pandas as pd

from autorag.nodes.passagecompressor.base import BasePassageCompressor
from autorag.utils.util import pop_params, result_to_dataframe, empty_cuda_cache


# TODO: Parallel Processing Refactoring at #460


class LongLLMLingua(BasePassageCompressor):
	def __init__(
		self, project_dir: str, model_name: str = "NousResearch/Llama-2-7b-hf", **kwargs
	):
		try:
			from llmlingua import PromptCompressor
		except ImportError:
			raise ImportError(
				"LongLLMLingua is not installed. Please install it by running `pip install llmlingua`."
			)

		super().__init__(project_dir)
		model_init_params = pop_params(PromptCompressor.__init__, kwargs)
		self.llm_lingua = PromptCompressor(model_name=model_name, **model_init_params)

	def __del__(self):
		del self.llm_lingua
		empty_cuda_cache()
		super().__del__()

	@result_to_dataframe(["retrieved_contents"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, retrieved_contents = self.cast_to_run(previous_result)
		results = self._pure(queries, retrieved_contents, **kwargs)
		return list(map(lambda x: [x], results))

	def _pure(
		self,
		queries: List[str],
		contents: List[List[str]],
		instructions: Optional[str] = None,
		target_token: int = 300,
		**kwargs,
	) -> List[str]:
		"""
		Compresses the retrieved texts using LongLLMLingua.
		For more information, visit https://github.com/microsoft/LLMLingua.

		:param queries: The queries for retrieved passages.
		:param contents: The contents of retrieved passages.
		:param model_name: The model name to use for compression.
		    The default is "NousResearch/Llama-2-7b-hf".
		:param instructions: The instructions for compression.
		    Default is None. When it is None, it will use default instructions.
		:param target_token: The target token for compression.
		    Default is 300.
		:param kwargs: Additional keyword arguments.
		:return: The list of compressed texts.
		"""
		if instructions is None:
			instructions = "Given the context, please answer the final question"
		results = [
			llmlingua_pure(
				query, contents_, self.llm_lingua, instructions, target_token, **kwargs
			)
			for query, contents_ in zip(queries, contents)
		]

		return results


def llmlingua_pure(
	query: str,
	contents: List[str],
	llm_lingua,
	instructions: str,
	target_token: int = 300,
	**kwargs,
) -> str:
	"""
	Return the compressed text.

	:param query: The query for retrieved passages.
	:param contents: The contents of retrieved passages.
	:param llm_lingua: The llm instance, that will be used to compress.
	:param instructions: The instructions for compression.
	:param target_token: The target token for compression.
	    Default is 300.
	:param kwargs: Additional keyword arguments.
	:return: The compressed text.
	"""
	try:
		from llmlingua import PromptCompressor
	except ImportError:
		raise ImportError(
			"LongLLMLingua is not installed. Please install it by running `pip install llmlingua`."
		)
	# split by "\n\n" (recommended by LongLLMLingua authors)
	new_context_texts = [c for context in contents for c in context.split("\n\n")]
	compress_prompt_params = pop_params(PromptCompressor.compress_prompt, kwargs)
	compressed_prompt = llm_lingua.compress_prompt(
		new_context_texts,
		question=query,
		instruction=instructions,
		rank_method="longllmlingua",
		target_token=target_token,
		**compress_prompt_params,
	)
	compressed_prompt_txt = compressed_prompt["compressed_prompt"]

	# separate out the question and instruction
	result = "\n\n".join(compressed_prompt_txt.split("\n\n")[1:-1])

	return result
