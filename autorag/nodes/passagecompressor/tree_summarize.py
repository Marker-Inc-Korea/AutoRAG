import asyncio
from typing import List, Optional

from llama_index import PromptTemplate, ServiceContext
from llama_index.llms.base import BaseLLM
from llama_index.prompts import PromptType
from llama_index.prompts.utils import is_chat_model
from llama_index.response_synthesizers import TreeSummarize

from autorag.nodes.passagecompressor.base import passage_compressor_node
from autorag.utils.util import process_batch


@passage_compressor_node
def tree_summarize(queries: List[str],
                   contents: List[List[str]],
                   scores,
                   ids,
                   llm: BaseLLM,
                   prompt: Optional[str] = None,
                   chat_prompt: Optional[str] = None,
                   context_window: Optional[int] = None,
                   num_output: int = 1,
                   batch: int = 16,
                   ) -> List[str]:
    """
    Recursively merge retrieved texts and summarizes them in a bottom-up fashion.
    This function is a wrapper for llama_index.response_synthesizers.TreeSummarize.
    For more information, visit https://docs.llamaindex.ai/en/latest/examples/response_synthesizers/tree_summarize.html.

    :param queries: The queries for retrieved passages.
    :param contents: The contents of retrieved passages.
    :param scores: The scores of retrieved passages.
        Do not use in this function, so you can pass an empty list.
    :param ids: The ids of retrieved passages.
        Do not use in this function, so you can pass an empty list.
    :param llm: The llm instance that will be used to summarize.
    :param prompt: The prompt template for summarization.
        If you want to use chat prompt, you should pass chat_prompt instead.
        At prompt, you must specify where to put 'context_str' and 'query_str'.
        Default is None. When it is None, it will use llama index default prompt.
    :param chat_prompt: The chat prompt template for summarization.
        If you want to use normal prompt, you should pass prompt instead.
        At prompt, you must specify where to put 'context_str' and 'query_str'.
        Default is None. When it is None, it will use llama index default chat prompt.
    :param context_window: The context window size for summarization.
        Default is None. When it is None, it will use a llama index default context window.
    :param num_output: The amount of summarization output.
        Default is 1.
    :param batch: The batch size for llm.
        Set low if you face some errors.
        Default is 16.
    :return: The list of compressed texts.
    """
    if prompt is not None and not is_chat_model(llm):
        summary_template = PromptTemplate(prompt, prompt_type=PromptType.SUMMARY)
    elif chat_prompt is not None and is_chat_model(llm):
        summary_template = PromptTemplate(chat_prompt, prompt_type=PromptType.SUMMARY)
    else:
        summary_template = None
    llm_context = ServiceContext.from_defaults(llm=llm,
                                               context_window=context_window,
                                               num_output=num_output)
    summarizer = TreeSummarize(summary_template=summary_template,
                               service_context=llm_context,
                               use_async=True)
    tasks = [summarizer.aget_response(query, content) for query, content in zip(queries, contents)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
    return results
