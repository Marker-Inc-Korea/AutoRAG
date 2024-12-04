from typing import List, Optional

from llama_index.core import PromptTemplate
from llama_index.core.prompts import PromptType
from llama_index.core.prompts.utils import is_chat_model
from llama_index.core.response_synthesizers import TreeSummarize as ts

from autorag.nodes.passagecompressor.base import LlamaIndexCompressor
from autorag.utils.util import get_event_loop, process_batch


class TreeSummarize(LlamaIndexCompressor):
	def _pure(
		self,
		queries: List[str],
		contents: List[List[str]],
		prompt: Optional[str] = None,
		chat_prompt: Optional[str] = None,
		batch: int = 16,
	) -> List[str]:
		"""
		Recursively merge retrieved texts and summarizes them in a bottom-up fashion.
		This function is a wrapper for llama_index.response_synthesizers.TreeSummarize.
		For more information, visit https://docs.llamaindex.ai/en/latest/examples/response_synthesizers/tree_summarize.html.

		:param queries: The queries for retrieved passages.
		:param contents: The contents of retrieved passages.
		:param prompt: The prompt template for summarization.
		    If you want to use chat prompt, you should pass chat_prompt instead.
		    At prompt, you must specify where to put 'context_str' and 'query_str'.
		    Default is None. When it is None, it will use llama index default prompt.
		:param chat_prompt: The chat prompt template for summarization.
		    If you want to use normal prompt, you should pass prompt instead.
		    At prompt, you must specify where to put 'context_str' and 'query_str'.
		    Default is None. When it is None, it will use llama index default chat prompt.
		:param batch: The batch size for llm.
		    Set low if you face some errors.
		    Default is 16.
		:return: The list of compressed texts.
		"""
		if prompt is not None and not is_chat_model(self.llm):
			summary_template = PromptTemplate(prompt, prompt_type=PromptType.SUMMARY)
		elif chat_prompt is not None and is_chat_model(self.llm):
			summary_template = PromptTemplate(
				chat_prompt, prompt_type=PromptType.SUMMARY
			)
		else:
			summary_template = None
		summarizer = ts(llm=self.llm, summary_template=summary_template, use_async=True)
		tasks = [
			summarizer.aget_response(query, content)
			for query, content in zip(queries, contents)
		]
		loop = get_event_loop()
		results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
		return results
