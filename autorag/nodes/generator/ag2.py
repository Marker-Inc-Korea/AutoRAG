"""AG2 multi-agent generator module for AutoRAG."""

import logging
import os
from typing import Dict, List, Tuple, Union

import tiktoken

from autogen import AssistantAgent, LLMConfig, UserProxyAgent

from autorag.nodes.generator.base import BaseGenerator
from autorag.utils.util import (
	get_event_loop,
	process_batch,
	result_to_dataframe,
)

logger = logging.getLogger("AutoRAG")

DEFAULT_SYSTEM_MESSAGE = (
	"You are a precise and helpful assistant. Given the context and "
	"question below, provide a comprehensive and accurate answer based "
	"solely on the provided context. If the context doesn't contain "
	"enough information, say so clearly. Be concise but thorough."
)


class AG2Generator(BaseGenerator):
	"""
	AG2 multi-agent generator for AutoRAG.

	Uses AG2's multi-agent conversation framework where a user proxy agent
	and an assistant agent collaborate to generate answers based on
	retrieved context.

	It returns pseudo token ids and log probs since AG2 does not support logprobs.

	:param project_dir: The project directory.
	:param llm: The LLM model name (e.g., "gpt-4o-mini", "gpt-4o").
		Default is "gpt-4o-mini".
	:param batch: Batch size for parallel generation. Default is 16.
	:param api_key: The API key. If not provided, reads from OPENAI_API_KEY env var.
	:param api_type: The API type for AG2 LLMConfig (e.g., "openai").
		Default is "openai".
	:param max_turns: Maximum number of conversation turns for the agent chat.
		Default is 1.
	:param system_message: Custom system message for the assistant agent.
		If not provided, uses a default RAG-optimized prompt.
	"""

	def __init__(
		self,
		project_dir,
		llm: str = "gpt-4o-mini",
		batch: int = 16,
		*args,
		**kwargs,
	):
		super().__init__(project_dir, llm, *args, **kwargs)
		assert batch > 0, "batch size must be greater than 0."
		self.batch = batch

		self.api_key = kwargs.pop("api_key", None) or os.environ.get(
			"OPENAI_API_KEY", ""
		)
		self.api_type = kwargs.pop("api_type", "openai")
		self.max_turns = kwargs.pop("max_turns", 1)
		self.system_message = (
			kwargs.pop("system_message", None) or DEFAULT_SYSTEM_MESSAGE
		)

		try:
			self.tokenizer = tiktoken.encoding_for_model(self.llm)
		except KeyError:
			self.tokenizer = tiktoken.get_encoding("o200k_base")

	@result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
	def pure(self, previous_result, *args, **kwargs):
		prompts = self.cast_to_run(previous_result)
		return self._pure(prompts, **kwargs)

	def _pure(
		self,
		prompts: Union[List[str], List[List[dict]]],
		**kwargs,
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		"""
		AG2 multi-agent generator module.
		Uses AG2's multi-agent conversation framework for generating answers.
		It returns pseudo token ids and log probs.

		:param prompts: A list of prompts.
		:param kwargs: Optional parameters passed to the AG2 LLMConfig.
		:return: A tuple of three elements.
			The first element is a list of generated text.
			The second element is a list of generated text's pseudo token ids.
			The third element is a list of generated text's pseudo log probs.
		"""
		loop = get_event_loop()
		tasks = [self._agenerate_single(prompt) for prompt in prompts]
		result = loop.run_until_complete(process_batch(tasks, self.batch))
		answer_result = list(map(lambda x: x[0], result))
		token_result = list(map(lambda x: x[1], result))
		logprob_result = list(map(lambda x: x[2], result))
		return answer_result, token_result, logprob_result

	async def _agenerate_single(
		self, prompt: Union[str, List[Dict]]
	) -> Tuple[str, List[int], List[float]]:
		"""Generate a single answer using AG2 agents."""
		llm_config = LLMConfig(
			{
				"model": self.llm,
				"api_key": self.api_key,
				"api_type": self.api_type,
			}
		)

		assistant = AssistantAgent(
			name="rag_assistant",
			system_message=self.system_message,
			llm_config=llm_config,
		)

		user_proxy = UserProxyAgent(
			name="user_proxy",
			human_input_mode="NEVER",
			code_execution_config=False,
			is_termination_msg=lambda x: (
				x.get("content", "") and "TERMINATE" in x.get("content", "")
			),
		)

		message = _format_prompt(prompt)

		try:
			chat_result = await user_proxy.a_run(
				assistant, message=message, max_turns=self.max_turns
			)
			await chat_result.process()

			answer = _extract_last_assistant_message(assistant, user_proxy)
			tokens = self.tokenizer.encode(answer, allowed_special="all")
			pseudo_log_probs = [0.5] * len(tokens)
			return answer, tokens, pseudo_log_probs

		except Exception as e:
			logger.warning(f"AG2 generation failed: {e}")
			empty_tokens = self.tokenizer.encode("", allowed_special="all")
			return "", empty_tokens, [0.5] * len(empty_tokens)

	async def astream(self, prompt: Union[str, List[Dict]], **kwargs):
		"""
		AG2 multi-agent conversations do not support native streaming.
		This method runs the full generation and yields the complete result.
		"""
		llm_config = LLMConfig(
			{
				"model": self.llm,
				"api_key": self.api_key,
				"api_type": self.api_type,
			}
		)

		assistant = AssistantAgent(
			name="rag_assistant",
			system_message=self.system_message,
			llm_config=llm_config,
		)

		user_proxy = UserProxyAgent(
			name="user_proxy",
			human_input_mode="NEVER",
			code_execution_config=False,
			is_termination_msg=lambda x: (
				x.get("content", "") and "TERMINATE" in x.get("content", "")
			),
		)

		message = _format_prompt(prompt)

		chat_result = await user_proxy.a_run(
			assistant, message=message, max_turns=self.max_turns
		)
		await chat_result.process()

		answer = _extract_last_assistant_message(assistant, user_proxy)
		yield answer

	def stream(self, prompt: Union[str, List[Dict]], **kwargs):
		raise NotImplementedError("stream method is not implemented yet.")


def _format_prompt(prompt: Union[str, List[Dict]]) -> str:
	"""Convert prompt to a single string message for the AG2 user proxy."""
	if isinstance(prompt, str):
		return prompt
	elif isinstance(prompt, list):
		parts = []
		for msg in prompt:
			role = msg.get("role", "")
			content = msg.get("content", "")
			if role == "system":
				parts.append(f"System: {content}")
			elif role == "user":
				parts.append(content)
			else:
				parts.append(f"{role}: {content}")
		return "\n\n".join(parts)
	else:
		raise ValueError("prompt must be a string or a list of dicts.")


def _extract_last_assistant_message(assistant, user_proxy) -> str:
	"""Extract the last assistant message from the AG2 chat history."""
	messages = assistant.chat_messages.get(user_proxy, [])
	for msg in reversed(messages):
		content = msg.get("content", "")
		if content:
			if "TERMINATE" in content:
				content = content.replace("TERMINATE", "").strip()
			if content:
				return content
	return ""
