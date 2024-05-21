from typing import List, Optional

from llmlingua import PromptCompressor

from autorag.nodes.passagecompressor.base import passage_compressor_node


@passage_compressor_node
def longllmlingua(queries: List[str],
                  contents: List[List[str]],
                  scores,
                  ids,
                  model_name: str = "NousResearch/Llama-2-7b-hf",
                  instructions: Optional[str] = None,
                  target_token: int = 300,
                  **kwargs,
                  ) -> List[str]:
    if instructions is None:
        instructions = "Given the context, please answer the final question"
    llm_lingua = PromptCompressor(
        model_name=model_name,
    )
    results = [llmlingua_pure(query, contents_, llm_lingua, instructions, target_token, **kwargs)
               for query, contents_ in zip(queries, contents)]
    del llm_lingua
    return results


def llmlingua_pure(query: str,
                   contents: List[str],
                   llm_lingua: PromptCompressor,
                   instructions: str,
                   target_token: int = 300,
                   **kwargs,
                   ) -> str:
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
    result = '\n\n'.join(compressed_prompt_txt.split("\n\n")[1:-1])

    return result
