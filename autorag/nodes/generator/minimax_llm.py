import logging
from typing import List, Tuple, Union, Dict

import pandas as pd
import tiktoken
from openai import AsyncOpenAI

from autorag.nodes.generator.base import BaseGenerator
from autorag.utils.util import (
	get_event_loop,
	process_batch,
	pop_params,
	result_to_dataframe,
)

logger = logging.getLogger("AutoRAG")

MINIMAX_API_BASE = "https://api.minimax.io/v1"

MAX_TOKEN_DICT = {
	"MiniMax-M2.7": 1_048_576,
	"MiniMax-M2.7-highspeed": 1_048_576,
	"MiniMax-M2.5": 1_048_576,
	"MiniMax-M2.5-highspeed": 204_800,
}


class MiniMaxLLM(BaseGenerator):
	def __init__(self, project_dir, llm: str, batch: int = 16, *args, **kwargs):
		"""
		MiniMax LLM generator module for AutoRAG.

		Uses the MiniMax API (OpenAI-compatible) for generating answers.
		It returns pseudo token ids and log probs since MiniMax does not support logprobs.

		:param project_dir: The project directory.
		:param llm: A model name for MiniMax. For example, ``MiniMax-M2.7`` or ``MiniMax-M2.5``.
		:param batch: Batch size for API calls. Default is 16.
		:param api_key: MiniMax API key. You can also set this to env variable ``MINIMAX_API_KEY``.
		:param kwargs: Extra parameters for the MiniMax chat completion API.
		"""
		super().__init__(project_dir, llm, *args, **kwargs)
		assert batch > 0, "batch size must be greater than 0."
		self.batch = batch

		api_key = kwargs.pop("api_key", None)
		base_url = kwargs.pop("base_url", MINIMAX_API_BASE)

		self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

		try:
			self.tokenizer = tiktoken.encoding_for_model(self.llm)
		except KeyError:
			self.tokenizer = tiktoken.get_encoding("o200k_base")

		self.max_token_size = MAX_TOKEN_DICT.get(self.llm)
		if self.max_token_size is None:
			logger.warning(
				f"Model {self.llm} is not in the known MiniMax models. "
				f"Using default max token size of 204800. "
				f"Known models: {list(MAX_TOKEN_DICT.keys())}"
			)
			self.max_token_size = 204_800
		self.max_token_size -= 7  # reserve for chat token usage

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
		"""
		MiniMax generator module.
		Uses the MiniMax API (OpenAI-compatible) for generating answers.
		It returns pseudo token ids and log probs.

		:param prompts: A list of prompts.
		:param truncate: Whether to truncate the input prompt. Default is True.
		:param kwargs: Optional parameters for the MiniMax chat completion API.
		:return: A tuple of three elements.
		    The first element is a list of generated text.
		    The second element is a list of generated text's pseudo token ids.
		    The third element is a list of generated text's pseudo log probs.
		"""
		if kwargs.get("logprobs") is not None:
			kwargs.pop("logprobs")
			logger.warning(
				"MiniMax does not support logprobs. This parameter is ignored."
			)
		if kwargs.get("n") is not None:
			kwargs.pop("n")
			logger.warning("parameter n does not effective. It always set to 1.")

		# Clamp temperature to MiniMax's valid range [0, 1]
		if "temperature" in kwargs:
			temp = kwargs["temperature"]
			if temp > 1.0:
				logger.warning(
					f"MiniMax temperature must be between 0 and 1. "
					f"Clamping {temp} to 1.0."
				)
				kwargs["temperature"] = 1.0

		if truncate:
			prompts = list(
				map(
					lambda prompt: truncate_by_token(
						prompt, self.tokenizer, self.max_token_size
					),
					prompts,
				)
			)

		openai_chat_params = pop_params(self.client.chat.completions.create, kwargs)
		loop = get_event_loop()
		tasks = [
			self.get_result(prompt, **openai_chat_params) for prompt in prompts
		]
		result = loop.run_until_complete(process_batch(tasks, self.batch))
		answer_result = list(map(lambda x: x[0], result))
		token_result = list(map(lambda x: x[1], result))
		logprob_result = list(map(lambda x: x[2], result))
		return answer_result, token_result, logprob_result

	async def astream(self, prompt: Union[str, List[Dict]], **kwargs):
		if kwargs.get("logprobs") is not None:
			kwargs.pop("logprobs")
		if kwargs.get("n") is not None:
			kwargs.pop("n")

		if "temperature" in kwargs:
			temp = kwargs["temperature"]
			if temp > 1.0:
				kwargs["temperature"] = 1.0

		prompt = truncate_by_token(prompt, self.tokenizer, self.max_token_size)

		openai_chat_params = pop_params(self.client.chat.completions.create, kwargs)

		stream = await self.client.chat.completions.create(
			model=self.llm,
			messages=parse_prompt(prompt),
			n=1,
			stream=True,
			**openai_chat_params,
		)
		result = ""
		async for chunk in stream:
			if chunk.choices[0].delta.content is not None:
				result += chunk.choices[0].delta.content
				yield result

	def stream(self, prompt: Union[str, List[Dict]], **kwargs):
		raise NotImplementedError("stream method is not implemented yet.")

	async def get_result(self, prompt: Union[str, List[dict]], **kwargs):
		messages = parse_prompt(prompt)

		response = await self.client.chat.completions.create(
			model=self.llm,
			messages=messages,
			n=1,
			**kwargs,
		)
		answer = response.choices[0].message.content

		# Strip thinking tags if present (MiniMax M2.5+ may include them)
		if answer and "<think>" in answer:
			import re

			answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)

		# MiniMax does not support logprobs, so return pseudo values
		tokens = self.tokenizer.encode(answer, allowed_special="all")
		pseudo_log_probs = [0.5] * len(tokens)
		return answer, tokens, pseudo_log_probs


def truncate_by_token(
	prompt: Union[str, List[Dict]], tokenizer, max_token_size: int
):
	if isinstance(prompt, list):
		prompt = messages_to_string(prompt)
	tokens = tokenizer.encode(prompt, allowed_special="all")
	return tokenizer.decode(tokens[:max_token_size])


def messages_to_string(messages: List[Dict[str, str]]) -> str:
	"""Convert chat messages to string format for token counting."""
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
