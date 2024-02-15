from llama_index.core.base.llms.types import CompletionResponse


async def mock_openai_acomplete(prompt: str, **kwargs):
    return CompletionResponse(text=prompt)
