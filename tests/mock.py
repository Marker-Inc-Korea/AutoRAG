import time
from random import random
from typing import Optional, Callable, Sequence, Any, List, Union

import tiktoken
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.base.llms.types import (
	ChatMessage,
	LLMMetadata,
	CompletionResponseGen,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CompletionResponse, CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.types import PydanticProgramMode
from openai.types.chat import (
	ChatCompletion,
	ChatCompletionMessage,
	ChatCompletionTokenLogprob,
)
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class MockLLM(CustomLLM):
	max_tokens: Optional[int]

	def __init__(
		self,
		max_tokens: Optional[int] = None,
		callback_manager: Optional[CallbackManager] = None,
		system_prompt: Optional[str] = None,
		messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
		completion_to_prompt: Optional[Callable[[str], str]] = None,
		pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
		**kwargs: Any,
	) -> None:
		super().__init__(
			max_tokens=max_tokens,
			callback_manager=callback_manager,
			system_prompt=system_prompt,
			messages_to_prompt=messages_to_prompt,
			completion_to_prompt=completion_to_prompt,
			pydantic_program_mode=pydantic_program_mode,
		)

	@classmethod
	def class_name(cls) -> str:
		return "MockLLM"

	@property
	def metadata(self) -> LLMMetadata:
		return LLMMetadata(num_output=self.max_tokens or -1)

	def _generate_text(self, length: int) -> str:
		return " ".join(["text" for _ in range(length)])

	@llm_completion_callback()
	def complete(
		self, prompt: str, formatted: bool = False, **kwargs: Any
	) -> CompletionResponse:
		response_text = (
			self._generate_text(self.max_tokens) if self.max_tokens else prompt
		)

		return CompletionResponse(
			text=response_text,
		)

	@llm_completion_callback()
	def stream_complete(
		self, prompt: str, formatted: bool = False, **kwargs: Any
	) -> CompletionResponseGen:
		def gen_prompt() -> CompletionResponseGen:
			for ch in prompt:
				yield CompletionResponse(
					text=prompt,
					delta=ch,
				)

		def gen_response(max_tokens: int) -> CompletionResponseGen:
			for i in range(max_tokens):
				response_text = self._generate_text(i)
				yield CompletionResponse(
					text=response_text,
					delta="text ",
				)

		return gen_response(self.max_tokens) if self.max_tokens else gen_prompt()


async def mock_openai_chat_create(self, messages, model, **kwargs):
	tokenizer = tiktoken.encoding_for_model(model)
	tokens = tokenizer.encode(str(messages[0]["content"]))
	if len(tokens) > 16_385:
		raise ValueError("The maximum number of tokens is 16_385")

	return ChatCompletion(
		id="test_id",
		choices=[
			Choice(
				finish_reason="stop",
				index=0,
				logprobs=ChoiceLogprobs(
					content=[
						ChatCompletionTokenLogprob(
							token="Why",
							logprob=-0.445,
							top_logprobs=[],
						),
						ChatCompletionTokenLogprob(
							token=" not",
							logprob=-0.223,
							top_logprobs=[],
						),
						ChatCompletionTokenLogprob(
							token="<|end|>",
							logprob=-0.443,
							top_logprobs=[],
						),
					]
				),
				message=ChatCompletionMessage(
					content="Why not",
					role="assistant",
				),
			)
		],
		created=int(time.time()),
		model=model,
		object="chat.completion",
	)


def mock_get_text_embedding_batch(
	self,
	texts: List[str],
	show_progress: bool = False,
	**kwargs: Any,
) -> List[Embedding]:
	return [[random() for _ in range(1536)] for _ in range(len(texts))]


class MockDeepevalMetric(BaseMetric):
	def __init__(
		self,
		threshold: float = 0.5,
		n: int = 5,
		model: Optional[Union[str, DeepEvalBaseLLM]] = None,
		assessment_questions: Optional[List[str]] = None,
		include_reason: bool = True,
		async_mode=True,
		strict_mode: bool = False,
		verbose_mode: bool = False,
	):
		self.threshold = 1 if strict_mode else threshold

		if assessment_questions is not None and len(assessment_questions) == 0:
			self.assessment_questions = None
		else:
			self.assessment_questions = assessment_questions

		self.include_reason = include_reason
		self.n = n
		self.async_mode = async_mode
		self.strict_mode = strict_mode
		self.verbose_mode = verbose_mode

	def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
		return 0.1

	async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
		return 0.2

	def is_successful(self) -> bool:
		return True

	@property
	def __name__(self):
		return "Mock Deepeval Base Metric"
