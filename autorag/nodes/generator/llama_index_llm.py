from typing import List, Tuple

import pandas as pd
from llama_index.core.base.llms.base import BaseLLM
from transformers import AutoTokenizer

from autorag import generator_models
from autorag.nodes.generator.base import BaseGenerator
from autorag.utils.util import (
	get_event_loop,
	process_batch,
	result_to_dataframe,
	pop_params,
)


class LlamaIndexLLM(BaseGenerator):
	def __init__(self, project_dir: str, llm: str, batch: int = 16, *args, **kwargs):
		"""
		Initialize the Llama Index LLM module.

		:param project_dir: The project directory.
		:param llm: A llama index LLM instance.
		:param batch: The batch size for llm.
			Set low if you face some errors.
			Default is 16.
		:param kwargs: The extra parameters for initializing the llm instance.
		"""
		super().__init__(project_dir=project_dir, llm=llm)
		if self.llm not in generator_models.keys():
			raise ValueError(
				f"{self.llm} is not a valid llm name. Please check the llm name."
				"You can check valid llm names from autorag.generator_models."
			)
		self.batch = batch
		llm_class = generator_models[self.llm]

		if llm_class.class_name() in [
			"HuggingFace_LLM",
			"HuggingFaceInferenceAPI",
			"TextGenerationInference",
		]:
			model_name = kwargs.pop("model", None)
			if model_name is not None:
				kwargs["model_name"] = model_name
			else:
				if "model_name" not in kwargs.keys():
					raise ValueError(
						"`model` or `model_name` parameter must be provided for using huggingfacellm."
					)
			kwargs["tokenizer_name"] = kwargs["model_name"]
		self.llm_instance: BaseLLM = llm_class(**pop_params(llm_class.__init__, kwargs))

	def __del__(self):
		super().__del__()
		del self.llm_instance

	@result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		prompts = self.cast_to_run(previous_result=previous_result)
		return self._pure(prompts)

	def _pure(
		self,
		prompts: List[str],
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		"""
		Llama Index LLM module.
		It gets the LLM instance from llama index, and returns generated text by the input prompt.
		It does not generate the right log probs, but it returns the pseudo log probs,
		which are not meant to be used for other modules.

		:param prompts: A list of prompts.
		:return: A tuple of three elements.
			The first element is a list of a generated text.
			The second element is a list of generated text's token ids, used tokenizer is GPT2Tokenizer.
			The third element is a list of generated text's pseudo log probs.
		"""
		tasks = [self.llm_instance.acomplete(prompt) for prompt in prompts]
		loop = get_event_loop()
		results = loop.run_until_complete(process_batch(tasks, batch_size=self.batch))

		generated_texts = list(map(lambda x: x.text, results))
		tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
		tokenized_ids = tokenizer(generated_texts).data["input_ids"]
		pseudo_log_probs = list(map(lambda x: [0.5] * len(x), tokenized_ids))
		return generated_texts, tokenized_ids, pseudo_log_probs

	async def astream(self, prompt: str, **kwargs):
		async for completion_response in await self.llm_instance.astream_complete(
			prompt
		):
			yield completion_response.text

	def stream(self, prompt: str, **kwargs):
		for completion_response in self.llm_instance.stream_complete(prompt):
			yield completion_response.text
