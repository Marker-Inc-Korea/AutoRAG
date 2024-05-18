from typing import List, Optional

from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType
from llmlingua import PromptCompressor

from autorag.nodes.passagecompressor.base import passage_compressor_node


@passage_compressor_node
def longllmlingua(queries: List[str],
                  contents: List[List[str]],
                  scores,
                  ids,
                  llm: LLMPredictorType,
                  instructions: Optional[str] = None,
                  target_token: int = 300,
                  **kwargs,
                  ) -> List[str]:
    if instructions is None:
        instructions = "Given the context, please answer the final question"
    llm_lingua = PromptCompressor()
    results = [llmlingua_pure(query, contents_, llm_lingua, instructions, target_token, **kwargs)
               for query, contents_ in zip(queries, contents)]
    return results


def llmlingua_pure(query: str,
                   contents: List[str],
                   llm_lingua: PromptCompressor,
                   instructions: str,
                   target_token: int = 300,
                   **kwargs,
                   ) -> str:
    compressed_prompt = llm_lingua.compress_prompt(
        contents,
        question=query,
        instruction=instructions,
        rank_method="longllmlingua",
        target_token=target_token,
        **kwargs,
    )
    compressed_prompt_txt = compressed_prompt["compressed_prompt"]

    # separate out the question and instruction (appended to top and bottom)
    compressed_prompt_txt_list = compressed_prompt_txt.split("\n\n")
    compressed_prompt_txt_list = compressed_prompt_txt_list[1:-1]

    return compressed_prompt_txt_list
