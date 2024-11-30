import gc
from copy import deepcopy
from typing import List, Tuple

import pandas as pd

from autorag.nodes.generator.base import BaseGenerator
from autorag.utils import result_to_dataframe
from autorag.utils.util import pop_params, to_list


class Vllm(BaseGenerator):
	def __init__(self, project_dir: str, llm: str, **kwargs):
		super().__init__(project_dir, llm, **kwargs)
		try:
			from vllm import SamplingParams, LLM
		except ImportError:
			raise ImportError(
				"Please install vllm library. You can install it by running `pip install vllm`."
			)

		model_from_kwargs = kwargs.pop("model", None)
		model = llm if model_from_kwargs is None else model_from_kwargs

		input_kwargs = deepcopy(kwargs)
		sampling_params_init_params = pop_params(
			SamplingParams.from_optional, input_kwargs
		)
		self.vllm_model = LLM(model, **input_kwargs)

		# delete not sampling param keys in the kwargs
		kwargs_keys = list(kwargs.keys())
		for key in kwargs_keys:
			if key not in sampling_params_init_params:
				kwargs.pop(key)

	def __del__(self):
		try:
			import torch
			import contextlib

			if torch.cuda.is_available():
				from vllm.distributed.parallel_state import (
					destroy_model_parallel,
					destroy_distributed_environment,
				)

				destroy_model_parallel()
				destroy_distributed_environment()
				del self.vllm_model.llm_engine.model_executor
				del self.vllm_model
				with contextlib.suppress(AssertionError):
					torch.distributed.destroy_process_group()
				gc.collect()
				torch.cuda.empty_cache()
				torch.cuda.synchronize()
		except ImportError:
			del self.vllm_model

		super().__del__()

	@result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		prompts = self.cast_to_run(previous_result)
		return self._pure(prompts, **kwargs)

	def _pure(
		self, prompts: List[str], **kwargs
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		"""
		Vllm module.
		It gets the VLLM instance and returns generated texts by the input prompt.
		You can set logprobs to get the log probs of the generated text.
		Default logprobs is 1.

		:param prompts: A list of prompts.
		:param kwargs: The extra parameters for generating the text.
		:return: A tuple of three elements.
		    The first element is a list of generated text.
		    The second element is a list of generated text's token ids.
		    The third element is a list of generated text's log probs.
		"""
		try:
			from vllm.outputs import RequestOutput
			from vllm.sequence import SampleLogprobs
			from vllm import SamplingParams
		except ImportError:
			raise ImportError(
				"Please install vllm library. You can install it by running `pip install vllm`."
			)

		if "logprobs" not in kwargs:
			kwargs["logprobs"] = 1

		sampling_params = pop_params(SamplingParams.from_optional, kwargs)
		generate_params = SamplingParams(**sampling_params)
		results: List[RequestOutput] = self.vllm_model.generate(
			prompts, generate_params
		)
		generated_texts = list(map(lambda x: x.outputs[0].text, results))
		generated_token_ids = list(map(lambda x: x.outputs[0].token_ids, results))
		log_probs: List[SampleLogprobs] = list(
			map(lambda x: x.outputs[0].logprobs, results)
		)
		generated_log_probs = list(
			map(
				lambda x: list(map(lambda y: y[0][y[1]].logprob, zip(x[0], x[1]))),
				zip(log_probs, generated_token_ids),
			)
		)
		return (
			to_list(generated_texts),
			to_list(generated_token_ids),
			to_list(generated_log_probs),
		)

	async def astream(self, prompt: str, **kwargs):
		raise NotImplementedError

	def stream(self, prompt: str, **kwargs):
		raise NotImplementedError
