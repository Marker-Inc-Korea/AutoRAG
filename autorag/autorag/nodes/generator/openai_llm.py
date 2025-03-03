import logging
from typing import List, Tuple

import pandas as pd
import tiktoken
from openai import AsyncOpenAI
from tiktoken import Encoding

from autorag.nodes.generator.base import BaseGenerator
from autorag.utils.util import (
	get_event_loop,
	process_batch,
	pop_params,
	result_to_dataframe,
)

logger = logging.getLogger("AutoRAG")

MAX_TOKEN_DICT = {  # model name : token limit
	"gpt-4.5-preview": 128_000,
	"gpt-4.5-preview-2025-02-27": 128_000,
	"o1": 200_000,
	"o1-preview": 128_000,
	"o1-preview-2024-09-12": 128_000,
	"o1-mini": 128_000,
	"o1-mini-2024-09-12": 128_000,
	"o3-mini": 200_000,
	"gpt-4o-mini": 128_000,
	"gpt-4o-mini-2024-07-18": 128_000,
	"gpt-4o": 128_000,
	"gpt-4o-2024-08-06": 128_000,
	"gpt-4o-2024-05-13": 128_000,
	"chatgpt-4o-latest": 128_000,
	"gpt-4-turbo": 128_000,
	"gpt-4-turbo-2024-04-09": 128_000,
	"gpt-4-turbo-preview": 128_000,
	"gpt-4-0125-preview": 128_000,
	"gpt-4-1106-preview": 128_000,
	"gpt-4-vision-preview": 128_000,
	"gpt-4-1106-vision-preview": 128_000,
	"gpt-4": 8_192,
	"gpt-4-0613": 8_192,
	"gpt-4-32k": 32_768,
	"gpt-4-32k-0613": 32_768,
	"gpt-3.5-turbo-0125": 16_385,
	"gpt-3.5-turbo": 16_385,
	"gpt-3.5-turbo-1106": 16_385,
	"gpt-3.5-turbo-instruct": 4_096,
	"gpt-3.5-turbo-16k": 16_385,
	"gpt-3.5-turbo-0613": 4_096,
	"gpt-3.5-turbo-16k-0613": 16_385,
}


class OpenAILLM(BaseGenerator):
	def __init__(self, project_dir, llm: str, batch: int = 16, *args, **kwargs):
		super().__init__(project_dir, llm, *args, **kwargs)
		assert batch > 0, "batch size must be greater than 0."
		self.batch = batch

		client_init_params = pop_params(AsyncOpenAI.__init__, kwargs)
		self.client = AsyncOpenAI(**client_init_params)

		if self.llm.startswith("gpt-4.5"):
			self.tokenizer = tiktoken.get_encoding("o200k_base")
		else:
			self.tokenizer = tiktoken.encoding_for_model(self.llm)

		self.max_token_size = (
			MAX_TOKEN_DICT.get(self.llm) - 7
		)  # because of chat token usage
		if self.max_token_size is None:
			raise ValueError(
				f"Model {self.llm} does not supported. "
				f"Please select the model between {list(MAX_TOKEN_DICT.keys())}"
			)

	@result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		prompts = self.cast_to_run(previous_result)
		return self._pure(prompts, **kwargs)

	def _pure(
		self,
		prompts: List[str],
		truncate: bool = True,
		**kwargs,
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		"""
		OpenAI generator module.
		Uses an official openai library for generating answer from the given prompt.
		It returns real token ids and log probs, so you must use this for using token ids and log probs.

		:param prompts: A list of prompts.
		:param llm: A model name for openai.
		    Default is gpt-3.5-turbo.
		:param batch: Batch size for openai api call.
		    If you get API limit errors, you should lower the batch size.
		    Default is 16.
		:param truncate: Whether to truncate the input prompt.
		    Default is True.
		:param api_key: OpenAI API key. You can set this by passing env variable `OPENAI_API_KEY`
		:param kwargs: The optional parameter for openai api call `openai.chat.completion`
		    See https://platform.openai.com/docs/api-reference/chat/create for more details.
		:return: A tuple of three elements.
		    The first element is a list of generated text.
		    The second element is a list of generated text's token ids.
		    The third element is a list of generated text's log probs.
		"""
		if kwargs.get("logprobs") is not None:
			kwargs.pop("logprobs")
			logger.warning(
				"parameter logprob does not effective. It always set to True."
			)
		if kwargs.get("n") is not None:
			kwargs.pop("n")
			logger.warning("parameter n does not effective. It always set to 1.")

		# TODO: fix this after updating tiktoken for the gpt-4.5 model. It is not yet supported yet.
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
		if self.llm.startswith("o1") or self.llm.startswith("o3"):
			tasks = [
				self.get_result_o1(prompt, **openai_chat_params) for prompt in prompts
			]
		else:
			tasks = [
				self.get_result(prompt, **openai_chat_params) for prompt in prompts
			]
		result = loop.run_until_complete(process_batch(tasks, self.batch))
		answer_result = list(map(lambda x: x[0], result))
		token_result = list(map(lambda x: x[1], result))
		logprob_result = list(map(lambda x: x[2], result))
		return answer_result, token_result, logprob_result

	def structured_output(self, prompts: List[str], output_cls, **kwargs):
		supported_models = [
			"gpt-4o-mini-2024-07-18",
			"gpt-4o-2024-08-06",
		]
		if self.llm not in supported_models:
			raise ValueError(
				f"{self.llm} is not a valid model name for structured output. "
				f"Please select the model between {supported_models}"
			)

		if kwargs.get("logprobs") is not None:
			kwargs.pop("logprobs")
			logger.warning(
				"parameter logprob does not effective. It always set to False."
			)
		if kwargs.get("n") is not None:
			kwargs.pop("n")
			logger.warning("parameter n does not effective. It always set to 1.")

		# TODO: fix this after updating tiktoken for the gpt-4.5 model. It is not yet supported yet.
		prompts = list(
			map(
				lambda prompt: truncate_by_token(
					prompt, self.tokenizer, self.max_token_size
				),
				prompts,
			)
		)

		openai_chat_params = pop_params(self.client.beta.chat.completions.parse, kwargs)
		loop = get_event_loop()
		tasks = [
			self.get_structured_result(prompt, output_cls, **openai_chat_params)
			for prompt in prompts
		]
		result = loop.run_until_complete(process_batch(tasks, self.batch))
		return result

	async def astream(self, prompt: str, **kwargs):
		# TODO: gpt-4.5-preview does not support logprobs. It should be fixed after the openai update.
		if kwargs.get("logprobs") is not None:
			kwargs.pop("logprobs")
			logger.warning(
				"parameter logprob does not effective. It always set to False."
			)
		if kwargs.get("n") is not None:
			kwargs.pop("n")
			logger.warning("parameter n does not effective. It always set to 1.")

		prompt = truncate_by_token(prompt, self.tokenizer, self.max_token_size)

		openai_chat_params = pop_params(self.client.chat.completions.create, kwargs)

		stream = await self.client.chat.completions.create(
			model=self.llm,
			messages=[
				{"role": "user", "content": prompt},
			],
			logprobs=False,
			n=1,
			stream=True,
			**openai_chat_params,
		)
		result = ""
		async for chunk in stream:
			if chunk.choices[0].delta.content is not None:
				result += chunk.choices[0].delta.content
				yield result

	def stream(self, prompt: str, **kwargs):
		raise NotImplementedError("stream method is not implemented yet.")

	async def get_structured_result(self, prompt: str, output_cls, **kwargs):
		logprobs = True
		if self.llm.startswith("gpt-4.5"):
			logprobs = False
		response = await self.client.beta.chat.completions.parse(
			model=self.llm,
			messages=[
				{"role": "user", "content": prompt},
			],
			response_format=output_cls,
			logprobs=logprobs,
			n=1,
			**kwargs,
		)
		return response.choices[0].message.parsed

	async def get_result(self, prompt: str, **kwargs):
		# TODO: gpt-4.5-preview does not support logprobs. It should be fixed after the openai update.
		logprobs = True
		if self.llm.startswith("gpt-4.5"):
			logprobs = False
		response = await self.client.chat.completions.create(
			model=self.llm,
			messages=[
				{"role": "user", "content": prompt},
			],
			logprobs=logprobs,
			n=1,
			**kwargs,
		)
		choice = response.choices[0]
		answer = choice.message.content
		# TODO: gpt-4.5-preview does not support logprobs. It should be fixed after the openai update.
		if self.llm.startswith("gpt-4.5"):
			tokens = self.tokenizer.encode(answer, allowed_special="all")
			logprobs = [0.5] * len(tokens)
			logger.warning("gpt-4.5-preview does not support logprobs yet.")
		else:
			logprobs = list(map(lambda x: x.logprob, choice.logprobs.content))
			tokens = list(
				map(
					lambda x: self.tokenizer.encode(x.token, allowed_special="all")[0],
					choice.logprobs.content,
				)
			)
			assert len(tokens) == len(
				logprobs
			), "tokens and logprobs size is different."
		return answer, tokens, logprobs

	async def get_result_o1(self, prompt: str, **kwargs):
		assert self.llm.startswith("o1") or self.llm.startswith(
			"o3"
		), "This function only supports o1 or o3 model."
		# The default temperature for the o1 model is 1. 1 is only supported.
		# See https://platform.openai.com/docs/guides/reasoning about beta limitation of o1 models.
		kwargs["temperature"] = 1
		kwargs["top_p"] = 1
		kwargs["presence_penalty"] = 0
		kwargs["frequency_penalty"] = 0
		response = await self.client.chat.completions.create(
			model=self.llm,
			messages=[
				{"role": "user", "content": prompt},
			],
			logprobs=False,
			n=1,
			**kwargs,
		)
		answer = response.choices[0].message.content
		tokens = self.tokenizer.encode(answer, allowed_special="all")
		pseudo_log_probs = [0.5] * len(tokens)
		return answer, tokens, pseudo_log_probs


def truncate_by_token(prompt: str, tokenizer: Encoding, max_token_size: int):
	tokens = tokenizer.encode(prompt, allowed_special="all")
	return tokenizer.decode(tokens[:max_token_size])
