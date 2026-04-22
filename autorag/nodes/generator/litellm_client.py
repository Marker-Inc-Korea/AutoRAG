import logging
from typing import Dict, List, Tuple, Union

import pandas as pd
import tiktoken
from tiktoken import Encoding

from autorag.nodes.generator.base import BaseGenerator
from autorag.utils.util import (
	get_event_loop,
	process_batch,
	result_to_dataframe,
)

logger = logging.getLogger("AutoRAG")


class LiteLLMGenerator(BaseGenerator):
	def __init__(self, project_dir, llm: str, batch: int = 16, *args, **kwargs):
		"""
		LiteLLM generator module for AutoRAG.

		Routes to 100+ LLM providers (OpenAI, Anthropic, Azure, Bedrock,
		Vertex AI, Groq, Together, Ollama, etc.) through a single unified
		interface via litellm.acompletion().

		:param project_dir: The project directory.
		:param llm: A LiteLLM model string. For example, ``gpt-4o``,
		    ``anthropic/claude-sonnet-4-20250514``, ``azure/gpt-4o``,
		    ``bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0``.
		    See https://docs.litellm.ai/docs/providers for the full list.
		:param batch: Batch size for API calls. Default is 16.
		:param api_key: Provider API key. When not set, LiteLLM resolves
		    credentials from provider-specific env vars (OPENAI_API_KEY,
		    ANTHROPIC_API_KEY, etc.) based on the model prefix.
		:param api_base: Custom base URL for self-hosted endpoints or proxies.
		:param kwargs: Extra parameters forwarded to litellm.acompletion().
		"""
		try:
			import litellm  # noqa: F401
		except ImportError as err:
			raise ImportError(
				"litellm is not installed. Install with: pip install litellm"
			) from err

		super().__init__(project_dir, llm, *args, **kwargs)
		assert batch > 0, "batch size must be greater than 0."
		self.batch = batch

		self.api_key = kwargs.pop("api_key", None)
		self.api_base = kwargs.pop("api_base", None)

		try:
			self.tokenizer = tiktoken.encoding_for_model(self.llm)
		except KeyError:
			self.tokenizer = tiktoken.get_encoding("o200k_base")

		self.max_token_size = kwargs.pop("max_token_size", 128_000) - 7

	@result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		prompts = self.cast_to_run(previous_result)
		return self._pure(prompts, **kwargs)

	def _pure(
		self,
		prompts: Union[List[str], List[List[dict]]],
		truncate: bool = True,
		**kwargs,
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		if kwargs.get("logprobs") is not None:
			kwargs.pop("logprobs")
			logger.warning(
				"LiteLLM does not guarantee logprobs across all providers. "
				"This parameter is ignored; pseudo log probs will be returned."
			)
		if kwargs.get("n") is not None:
			kwargs.pop("n")
			logger.warning("parameter n does not effective. It always set to 1.")

		if truncate:
			prompts = list(
				map(
					lambda prompt: truncate_by_token(
						prompt, self.tokenizer, self.max_token_size
					),
					prompts,
				)
			)

		loop = get_event_loop()
		tasks = [self.get_result(prompt, **kwargs) for prompt in prompts]
		result = loop.run_until_complete(process_batch(tasks, self.batch))
		answer_result = list(map(lambda x: x[0], result))
		token_result = list(map(lambda x: x[1], result))
		logprob_result = list(map(lambda x: x[2], result))
		return answer_result, token_result, logprob_result

	async def astream(self, prompt: Union[str, List[Dict]], **kwargs):
		import litellm

		if kwargs.get("logprobs") is not None:
			kwargs.pop("logprobs")
		if kwargs.get("n") is not None:
			kwargs.pop("n")

		prompt = truncate_by_token(prompt, self.tokenizer, self.max_token_size)

		call_kwargs = {
			"model": self.llm,
			"messages": parse_prompt(prompt),
			"stream": True,
			"drop_params": True,
			**kwargs,
		}
		if self.api_key:
			call_kwargs["api_key"] = self.api_key
		if self.api_base:
			call_kwargs["api_base"] = self.api_base

		stream = await litellm.acompletion(**call_kwargs)
		result = ""
		async for chunk in stream:
			if chunk.choices[0].delta.content is not None:
				result += chunk.choices[0].delta.content
				yield result

	def stream(self, prompt: Union[str, List[Dict]], **kwargs):
		raise NotImplementedError("stream method is not implemented yet.")

	async def get_result(self, prompt: Union[str, List[dict]], **kwargs):
		import litellm

		messages = parse_prompt(prompt)

		call_kwargs = {
			"model": self.llm,
			"messages": messages,
			"n": 1,
			"drop_params": True,
			**kwargs,
		}
		if self.api_key:
			call_kwargs["api_key"] = self.api_key
		if self.api_base:
			call_kwargs["api_base"] = self.api_base

		response = await litellm.acompletion(**call_kwargs)
		answer = response.choices[0].message.content or ""

		tokens = self.tokenizer.encode(answer, allowed_special="all")
		pseudo_log_probs = [0.5] * len(tokens)
		return answer, tokens, pseudo_log_probs


def truncate_by_token(
	prompt: Union[str, List[Dict]], tokenizer: Encoding, max_token_size: int
):
	if isinstance(prompt, list):
		prompt = messages_to_string(prompt)
	tokens = tokenizer.encode(prompt, allowed_special="all")
	return tokenizer.decode(tokens[:max_token_size])


def messages_to_string(messages: List[Dict[str, str]]) -> str:
	formatted_parts = [
		f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
		for message in messages
	]
	formatted_parts.append("<|im_start|>assistant")
	return "\n".join(formatted_parts)


def parse_prompt(prompt: Union[str, List[Dict]]) -> List[Dict]:
	if isinstance(prompt, str):
		return [{"role": "user", "content": prompt}]
	elif isinstance(prompt, list):
		return prompt
	else:
		raise ValueError("prompt must be a string or a list of dicts.")
