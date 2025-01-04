import logging
from typing import List, Tuple
import time

import pandas as pd
import requests
from asyncio import to_thread

from autorag.nodes.generator.base import BaseGenerator
from autorag.utils.util import get_event_loop, process_batch, result_to_dataframe

logger = logging.getLogger("AutoRAG")

DEFAULT_MAX_TOKENS = 4096  # Default token limit


class VllmAPI(BaseGenerator):
	def __init__(
		self,
		project_dir,
		llm: str,
		uri: str,
		max_tokens: int = None,
		batch: int = 16,
		*args,
		**kwargs,
	):
		"""
		VLLM API Wrapper for OpenAI-compatible chat/completions format.

		:param project_dir: Project directory.
		:param llm: Model name (e.g., LLaMA model).
		:param uri: VLLM API server URI.
		:param max_tokens: Maximum token limit.
		    Default is 4096.
		:param batch: Request batch size.
		    Default is 16.
		"""
		super().__init__(project_dir, llm, *args, **kwargs)
		assert batch > 0, "Batch size must be greater than 0."
		self.uri = uri.rstrip("/")  # Set API URI
		self.batch = batch
		# Use the provided max_tokens if available, otherwise use the default
		self.max_token_size = max_tokens if max_tokens else DEFAULT_MAX_TOKENS
		self.max_model_len = self.get_max_model_length()
		logger.info(f"{llm} max model length: {self.max_model_len}")

	@result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		prompts = self.cast_to_run(previous_result)
		return self._pure(prompts, **kwargs)

	def _pure(
		self, prompts: List[str], truncate: bool = True, **kwargs
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		"""
		Method to call the VLLM API to generate text.

		:param prompts: List of input prompts.
		:param truncate: Whether to truncate input prompts to fit within the token limit.
		:param kwargs: Additional options (e.g., temperature, top_p).
		:return: Generated text, token lists, and log probability lists.
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
			prompts = list(map(lambda p: self.truncate_by_token(p), prompts))
		loop = get_event_loop()
		tasks = [to_thread(self.get_result, prompt, **kwargs) for prompt in prompts]
		results = loop.run_until_complete(process_batch(tasks, self.batch))

		answer_result = list(map(lambda x: x[0], results))
		token_result = list(map(lambda x: x[1], results))
		logprob_result = list(map(lambda x: x[2], results))
		return answer_result, token_result, logprob_result

	def truncate_by_token(self, prompt: str) -> str:
		"""
		Function to truncate prompts to fit within the maximum token limit.
		"""
		tokens = self.encoding_for_model(prompt)["tokens"]  # Simple tokenization
		return self.decoding_for_model(tokens[: self.max_model_len])["prompt"]

	def call_vllm_api(self, prompt: str, **kwargs) -> dict:
		"""
		Calls the VLLM API to get chat/completions responses.

		:param prompt: Input prompt.
		:param kwargs: Additional API options (e.g., temperature, max_tokens).
		:return: API response.
		"""
		payload = {
			"model": self.llm,
			"messages": [{"role": "user", "content": prompt}],
			"temperature": kwargs.get("temperature", 0.4),
			"max_tokens": min(
				kwargs.get("max_tokens", self.max_token_size), self.max_token_size
			),
			"logprobs": True,
			"n": 1,
		}
		start_time = time.time()  # Record request start time
		response = requests.post(f"{self.uri}/v1/chat/completions", json=payload)
		end_time = time.time()  # Record request end time

		response.raise_for_status()
		elapsed_time = end_time - start_time  # Calculate elapsed time
		logger.info(
			f"Request chat completions to vllm server completed in {elapsed_time:.2f} seconds"
		)
		return response.json()

	# Additional method: abstract method implementation
	async def astream(self, prompt: str, **kwargs):
		"""
		Asynchronous streaming method not implemented.
		"""
		raise NotImplementedError("astream method is not implemented for VLLM API yet.")

	def stream(self, prompt: str, **kwargs):
		"""
		Synchronous streaming method not implemented.
		"""
		raise NotImplementedError("stream method is not implemented for VLLM API yet.")

	def get_result(self, prompt: str, **kwargs):
		response = self.call_vllm_api(prompt, **kwargs)
		choice = response["choices"][0]
		answer = choice["message"]["content"]

		# Handle cases where logprobs is None
		if choice.get("logprobs") and "content" in choice["logprobs"]:
			logprobs = list(map(lambda x: x["logprob"], choice["logprobs"]["content"]))
			tokens = list(
				map(
					lambda x: self.encoding_for_model(x["token"])["tokens"],
					choice["logprobs"]["content"],
				)
			)
		else:
			logprobs = []
			tokens = []

		return answer, tokens, logprobs

	def encoding_for_model(self, answer_piece: str):
		payload = {
			"model": self.llm,
			"prompt": answer_piece,
			"add_special_tokens": True,
		}
		response = requests.post(f"{self.uri}/tokenize", json=payload)
		response.raise_for_status()
		return response.json()

	def decoding_for_model(self, tokens: list[int]):
		payload = {
			"model": self.llm,
			"tokens": tokens,
		}
		response = requests.post(f"{self.uri}/detokenize", json=payload)
		response.raise_for_status()
		return response.json()

	def get_max_model_length(self):
		response = requests.get(f"{self.uri}/v1/models")
		response.raise_for_status()
		json_data = response.json()
		return json_data["data"][0]["max_model_len"]
