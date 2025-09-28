import logging
from typing import List, Tuple, Union, Dict

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
	"gpt-5": 272_000,
	"gpt-5-2025-08-07": 272_000,
	"gpt-5-chat-latest": 272_000,
	"gpt-5-mini-2025-08-07": 272_000,
	"gpt-5-mini": 272_000,
	"gpt-5-nano-2025-08-07": 272_000,
	"gpt-4.1": 1_000_000,
	"gpt-4.1-2025-04-14": 1_000_000,
	"gpt-4.1-mini": 1_047_576,
	"gpt-4.1-mini-2025-04-14": 1_047_576,
	"gpt-4.1-nano": 1_000_000,
	"gpt-4.1-nano-2025-04-14": 1_000_000,
	"o1": 200_000,
	"o1-preview": 128_000,
	"o1-preview-2024-09-12": 128_000,
	"o1-mini": 128_000,
	"o1-mini-2024-09-12": 128_000,
	"o1-pro": 200_000,
	"o1-pro-2025-03-19": 200_000,
	"o3": 200_000,
	"o3-mini": 200_000,
	"o3-mini-2025-01-31": 200_000,
	"o4-mini": 200_000,
	"o4-mini-2025-04-16": 200_000,
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
		try:
			self.tokenizer = tiktoken.encoding_for_model(self.llm)
		except KeyError:
			self.tokenizer = tiktoken.get_encoding("o200k_base")

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
		prompts: Union[List[str], List[List[dict]]],
		truncate: bool = True,
		**kwargs,
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		"""
		OpenAI generator module.
		Uses an official openai library for generating answer from the given prompt.
		It returns real token ids and log probs, so you must use this for using token ids and log probs.

		:param prompts: A list of prompts.
		:param llm: A model name for OpenAI.
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
		if (
			self.llm.startswith("o1")
			or self.llm.startswith("o3")
			or self.llm.startswith("o4")
			or self.llm.startswith("gpt-5")
		):
			tasks = [
				self.get_result_reasoning(prompt, **openai_chat_params)
				for prompt in prompts
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
		if kwargs.get("logprobs") is not None:
			kwargs.pop("logprobs")
			logger.warning(
				"parameter logprob does not effective. It always set to False."
			)
		if kwargs.get("n") is not None:
			kwargs.pop("n")
			logger.warning("parameter n does not effective. It always set to 1.")

		prompts = list(
			map(
				lambda prompt: truncate_by_token(
					prompt, self.tokenizer, self.max_token_size
				),
				prompts,
			)
		)

		openai_chat_params = pop_params(self.client.responses.parse, kwargs)
		loop = get_event_loop()
		tasks = [
			self.get_structured_result(prompt, output_cls, **openai_chat_params)
			for prompt in prompts
		]
		result = loop.run_until_complete(process_batch(tasks, self.batch))
		return result

	async def astream(self, prompt: Union[str, List[Dict]], **kwargs):
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
			messages=parse_prompt(prompt),
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

	def stream(self, prompt: Union[str, List[Dict]], **kwargs):
		raise NotImplementedError("stream method is not implemented yet.")

	async def get_structured_result(
		self, prompt: Union[str, List[Dict]], output_cls, **kwargs
	):
		if self.llm.startswith("gpt-3.5") or self.llm in [
			"gpt-4",
			"gpt-4-0613",
			"gpt-4-32k",
			"gpt-4-32k-0613",
			"gpt-4-turbo",
		]:
			raise ValueError("structured output is supported after the gpt-4o model.")

		response = await self.client.responses.parse(
			model=self.llm,
			input=parse_prompt(prompt),
			text_format=output_cls,
			**kwargs,
		)
		return response.output_parsed

	async def get_result(self, prompt: Union[str, List[dict]], **kwargs):
		logprobs = True
		messages = parse_prompt(prompt)

		response = await self.client.chat.completions.create(
			model=self.llm,
			messages=messages,
			logprobs=logprobs,
			n=1,
			**kwargs,
		)
		choice = response.choices[0]
		answer = choice.message.content
		logprobs = [x.logprob for x in choice.logprobs.content]
		tokens = [
			self.tokenizer.encode(x.token, allowed_special="all")[0]
			for x in choice.logprobs.content
		]
		if len(tokens) != len(logprobs):
			raise ValueError("tokens and logprobs size is different.")
		return answer, tokens, logprobs

	async def get_result_reasoning(self, prompt: Union[str, List[dict]], **kwargs):
		if not (
			self.llm.startswith("o1")
			or self.llm.startswith("o3")
			or self.llm.startswith("o4")
			or self.llm.startswith("gpt-5")
		):
			raise ValueError("get_result_reasoning is only for o1,o3,o4,gpt-5 models.")
		# The default temperature for the o1 model is 1. 1 is only supported.
		# See https://platform.openai.com/docs/guides/reasoning about beta limitation of o1 models.
		unsupported_params = [
			"temperature",
			"top_p",
			"presence_penalty",
			"frequency_penalty",
			"logprobs",
			"top_logprobs",
			"logit_bias",
		]
		kwargs["max_completion_tokens"] = kwargs.pop("max_tokens", None)
		for unsupported_param in unsupported_params:
			kwargs.pop(unsupported_param, None)
		messages = parse_prompt(prompt)

		response = await self.client.chat.completions.create(
			model=self.llm,
			messages=messages,
			n=1,
			**kwargs,
		)
		answer = response.choices[0].message.content
		tokens = self.tokenizer.encode(answer, allowed_special="all")
		pseudo_log_probs = [0.5] * len(tokens)
		return answer, tokens, pseudo_log_probs


def truncate_by_token(
	prompt: Union[str, List[Dict]], tokenizer: Encoding, max_token_size: int
):
	if isinstance(prompt, list):
		prompt = tiktoken_messages_to_string(prompt)
	tokens = tokenizer.encode(prompt, allowed_special="all")
	return tokenizer.decode(tokens[:max_token_size])


def tiktoken_messages_to_string(messages: List[Dict[str, str]]) -> str:
	"""Convert chat messages to string format for accurate token counting"""
	formatted_parts = [
		f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
		for message in messages
	]
	formatted_parts.append("<|im_start|>assistant")
	full_string = "\n".join(formatted_parts)
	return full_string


def parse_prompt(prompt: Union[str, List[Dict]]) -> List[Dict]:
	if isinstance(prompt, str):
		return [{"role": "user", "content": prompt}]
	elif isinstance(prompt, list):
		return prompt
	else:
		raise ValueError("prompt must be a string or a list of dicts.")
