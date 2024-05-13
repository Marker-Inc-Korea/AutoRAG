from typing import List

from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType
from llmlingua import PromptCompressor

from autorag.nodes.passagecompressor.base import passage_compressor_node


@passage_compressor_node
def llmlingua(queries: List[str],
              contents: List[List[str]],
              scores,
              ids,
              llm: LLMPredictorType,
              instructions: str,
              ) -> List[str]:
    llm_lingua = PromptCompressor()


def llmlingua_pure(query: str,
                   conents: List[str],
                   instructions: str,
                   ):
    pass
