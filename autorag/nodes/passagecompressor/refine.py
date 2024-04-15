import asyncio
from typing import List, Optional

from llama_index.core import PromptTemplate
from llama_index.core.prompts import PromptType
from llama_index.core.prompts.utils import is_chat_model
from llama_index.core.response_synthesizers import Refine
from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType

from autorag.nodes.passagecompressor.base import passage_compressor_node
from autorag.utils.util import process_batch


@passage_compressor_node
def refine(queries: List[str],
           contents: List[List[str]],
           scores,
           ids,
           llm: LLMPredictorType,
           prompt: Optional[str] = None,
           chat_prompt: Optional[str] = None,
           batch: int = 16,
           ) -> List[str]:
    """
    Refine a response to a query across text chunks.
    This function is a wrapper for llama_index.response_synthesizers.Refine.
    For more information, visit https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/refine/.

    :param queries: The queries for retrieved passages.
    :param contents: The contents of retrieved passages.
    :param scores: The scores of retrieved passages.
        Do not use in this function, so you can pass an empty list.
    :param ids: The ids of retrieved passages.
        Do not use in this function, so you can pass an empty list.
    :param llm: The llm instance that will be used to summarize.
    :param prompt: The prompt template for refine.
        If you want to use chat prompt, you should pass chat_prompt instead.
        At prompt, you must specify where to put 'context_msg' and 'query_str'.
        Default is None. When it is None, it will use llama index default prompt.
    :param chat_prompt: The chat prompt template for refine.
        If you want to use normal prompt, you should pass prompt instead.
        At prompt, you must specify where to put 'context_msg' and 'query_str'.
        Default is None. When it is None, it will use llama index default chat prompt.
    :param batch: The batch size for llm.
        Set low if you face some errors.
        Default is 16.
    :return: The list of compressed texts.
    """
    if prompt is not None and not is_chat_model(llm):
        refine_template = PromptTemplate(prompt, prompt_type=PromptType.REFINE)
    elif chat_prompt is not None and is_chat_model(llm):
        refine_template = PromptTemplate(chat_prompt, prompt_type=PromptType.REFINE)
    else:
        refine_template = None
    summarizer = Refine(llm=llm,
                        refine_template=refine_template,
                        verbose=True)
    tasks = [summarizer.aget_response(query, content) for query, content in zip(queries, contents)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
    return results
