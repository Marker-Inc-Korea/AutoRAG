import os
from unittest.mock import patch, MagicMock

import openai.resources.chat
import pandas as pd
import pytest
import tiktoken

from autorag.nodes.generator import MiniMaxLLM
from autorag.nodes.generator.minimax_llm import (
	truncate_by_token,
	strip_think_tags,
	parse_prompt,
	messages_to_string,
	MAX_TOKEN_DICT,
	MINIMAX_API_BASE,
)
from autorag.schema.module import Module
from autorag.support import get_support_modules
from tests.autorag.nodes.generator.test_generator_base import (
	prompts,
	check_generated_texts,
	check_generated_tokens,
	check_generated_log_probs,
	chat_prompts,
)


async def mock_minimax_chat_create(self, messages, model, **kwargs):
	"""Mock async function that returns an OpenAI-like response for MiniMax."""
	mock_message = MagicMock()
	mock_message.content = "Why not"

	mock_choice = MagicMock()
	mock_choice.message = mock_message

	mock_response = MagicMock()
	mock_response.choices = [mock_choice]
	mock_response.model = model

	return mock_response


async def mock_minimax_chat_create_with_think_tags(self, messages, model, **kwargs):
	"""Mock that returns response with think tags."""
	mock_message = MagicMock()
	mock_message.content = "<think>Let me think about this...</think>The answer is 42."

	mock_choice = MagicMock()
	mock_choice.message = mock_message

	mock_response = MagicMock()
	mock_response.choices = [mock_choice]
	mock_response.model = model

	return mock_response


async def mock_minimax_chat_create_simple(self, messages, model, **kwargs):
	"""Mock that returns a simple response without think tags."""
	mock_message = MagicMock()
	mock_message.content = "Simple answer."

	mock_choice = MagicMock()
	mock_choice.message = mock_message

	mock_response = MagicMock()
	mock_response.choices = [mock_choice]
	mock_response.model = model

	return mock_response


class MockStreamChunk:
	def __init__(self, content):
		mock_delta = MagicMock()
		mock_delta.content = content

		mock_choice = MagicMock()
		mock_choice.delta = mock_delta

		self.choices = [mock_choice]


class MockAsyncStream:
	def __init__(self, chunks):
		self.chunks = [MockStreamChunk(chunk) for chunk in chunks]
		self._index = 0

	def __aiter__(self):
		self._index = 0
		return self

	async def __anext__(self):
		if self._index >= len(self.chunks):
			raise StopAsyncIteration
		chunk = self.chunks[self._index]
		self._index += 1
		return chunk


async def mock_minimax_chat_create_stream(
	self, messages, model, stream=False, **kwargs
):
	if stream:
		return MockAsyncStream(
			["<thi", "nk>hidden reasoning", "</think>Visible", " answer"]
		)
	return await mock_minimax_chat_create(self, messages, model, **kwargs)


@pytest.fixture
def minimax_m27_instance():
	return MiniMaxLLM(
		project_dir=".",
		llm="MiniMax-M2.7",
		api_key="mock_minimax_api_key",
	)


@pytest.fixture
def minimax_m25_highspeed_instance():
	return MiniMaxLLM(
		project_dir=".",
		llm="MiniMax-M2.5-highspeed",
		api_key="mock_minimax_api_key",
	)


@pytest.fixture
def minimax_unknown_model_instance():
	return MiniMaxLLM(
		project_dir=".",
		llm="MiniMax-Unknown",
		api_key="mock_minimax_api_key",
	)


# --- Unit Tests ---


class TestMiniMaxLLMInit:
	"""Test MiniMaxLLM initialization."""

	def test_init_known_model(self, minimax_m27_instance):
		assert minimax_m27_instance.llm == "MiniMax-M2.7"
		assert minimax_m27_instance.batch == 16
		assert minimax_m27_instance.max_token_size == MAX_TOKEN_DICT["MiniMax-M2.7"] - 7

	def test_init_highspeed_model(self, minimax_m25_highspeed_instance):
		assert minimax_m25_highspeed_instance.llm == "MiniMax-M2.5-highspeed"
		assert (
			minimax_m25_highspeed_instance.max_token_size
			== MAX_TOKEN_DICT["MiniMax-M2.5-highspeed"] - 7
		)

	def test_init_unknown_model_uses_default(self, minimax_unknown_model_instance):
		assert minimax_unknown_model_instance.max_token_size == 204_800 - 7

	def test_init_custom_batch(self):
		instance = MiniMaxLLM(
			project_dir=".",
			llm="MiniMax-M2.7",
			batch=8,
			api_key="mock_key",
		)
		assert instance.batch == 8

	def test_init_custom_base_url(self):
		instance = MiniMaxLLM(
			project_dir=".",
			llm="MiniMax-M2.7",
			api_key="mock_key",
			base_url="https://custom.api.example.com/v1",
		)
		assert instance.client.base_url.host == "custom.api.example.com"

	def test_init_invalid_batch(self):
		with pytest.raises(AssertionError):
			MiniMaxLLM(
				project_dir=".",
				llm="MiniMax-M2.7",
				batch=0,
				api_key="mock_key",
			)


class TestMiniMaxModuleRegistration:
	def test_support_module_resolution(self):
		assert get_support_modules("minimax_llm") is MiniMaxLLM
		assert get_support_modules("MiniMaxLLM") is MiniMaxLLM

	def test_module_from_dict(self):
		module = Module.from_dict({"module_type": "minimax_llm", "llm": "MiniMax-M2.7"})
		assert module.module is MiniMaxLLM


class TestMiniMaxLLMPure:
	"""Test MiniMaxLLM._pure method."""

	@patch.object(
		openai.resources.chat.completions.AsyncCompletions,
		"create",
		mock_minimax_chat_create,
	)
	def test_pure_string_prompts(self, minimax_m27_instance):
		answers, tokens, log_probs = minimax_m27_instance._pure(
			prompts, temperature=0.5
		)
		check_generated_texts(answers)
		check_generated_tokens(tokens)
		check_generated_log_probs(log_probs)

	@patch.object(
		openai.resources.chat.completions.AsyncCompletions,
		"create",
		mock_minimax_chat_create,
	)
	def test_pure_chat_prompts(self, minimax_m27_instance):
		answers, tokens, log_probs = minimax_m27_instance._pure(chat_prompts)
		check_generated_texts(answers)
		check_generated_tokens(tokens)
		check_generated_log_probs(log_probs)

	def test_pure_chat_prompts_preserve_roles(self, minimax_m27_instance):
		captured_messages = []

		async def capture_chat_create(self, messages, model, **kwargs):
			captured_messages.append(messages)
			return await mock_minimax_chat_create(self, messages, model, **kwargs)

		with patch.object(
			openai.resources.chat.completions.AsyncCompletions,
			"create",
			capture_chat_create,
		):
			minimax_m27_instance._pure(chat_prompts[:1])

		assert captured_messages[0] == chat_prompts[0]
		assert captured_messages[0][0]["role"] == "system"
		assert captured_messages[0][1]["role"] == "user"

	@patch.object(
		openai.resources.chat.completions.AsyncCompletions,
		"create",
		mock_minimax_chat_create,
	)
	def test_pure_strips_logprobs_param(self, minimax_m27_instance):
		answers, tokens, log_probs = minimax_m27_instance._pure(
			prompts, logprobs=True, n=3
		)
		check_generated_texts(answers)

	@patch.object(
		openai.resources.chat.completions.AsyncCompletions,
		"create",
		mock_minimax_chat_create,
	)
	def test_pure_temperature_clamping(self, minimax_m27_instance):
		# Temperature > 1.0 should be clamped
		answers, tokens, log_probs = minimax_m27_instance._pure(
			prompts, temperature=1.5
		)
		check_generated_texts(answers)

	@patch.object(
		openai.resources.chat.completions.AsyncCompletions,
		"create",
		mock_minimax_chat_create,
	)
	def test_pure_temperature_zero(self, minimax_m27_instance):
		# Temperature 0 should work fine
		answers, tokens, log_probs = minimax_m27_instance._pure(prompts, temperature=0)
		check_generated_texts(answers)


class TestMiniMaxLLMThinkingStrip:
	"""Test thinking tag stripping."""

	@patch.object(
		openai.resources.chat.completions.AsyncCompletions,
		"create",
		mock_minimax_chat_create_with_think_tags,
	)
	def test_strips_think_tags(self, minimax_m27_instance):
		answers, tokens, log_probs = minimax_m27_instance._pure(prompts[:1])
		assert answers[0] == "The answer is 42."

	@patch.object(
		openai.resources.chat.completions.AsyncCompletions,
		"create",
		mock_minimax_chat_create_simple,
	)
	def test_no_think_tags(self, minimax_m27_instance):
		answers, tokens, log_probs = minimax_m27_instance._pure(prompts[:1])
		assert answers[0] == "Simple answer."


class TestMiniMaxLLMTruncation:
	"""Test prompt truncation."""

	@patch.object(
		openai.resources.chat.completions.AsyncCompletions,
		"create",
		mock_minimax_chat_create,
	)
	def test_truncate_long_prompt(self, minimax_m25_highspeed_instance):
		# Create a very long prompt
		long_prompt = " ".join([f"word {i} is here" for i in range(100_000)])
		answers, tokens, log_probs = minimax_m25_highspeed_instance._pure(
			[long_prompt] * 3
		)
		check_generated_texts(answers)

	@patch.object(
		openai.resources.chat.completions.AsyncCompletions,
		"create",
		mock_minimax_chat_create,
	)
	def test_no_truncation(self, minimax_m27_instance):
		answers, _, _ = minimax_m27_instance._pure(prompts, truncate=False)
		check_generated_texts(answers)


class TestMiniMaxLLMNode:
	"""Test MiniMaxLLM as a node (run_evaluator)."""

	@patch.object(
		openai.resources.chat.completions.AsyncCompletions,
		"create",
		mock_minimax_chat_create,
	)
	def test_run_evaluator(self):
		previous_result = pd.DataFrame(
			{"prompts": prompts, "qid": ["id-1", "id-2", "id-3"]}
		)
		result_df = MiniMaxLLM.run_evaluator(
			project_dir=".",
			previous_result=previous_result,
			llm="MiniMax-M2.7",
			api_key="mock_minimax_api_key",
			temperature=0.5,
		)
		check_generated_texts(result_df["generated_texts"].tolist())
		check_generated_tokens(result_df["generated_tokens"].tolist())
		check_generated_log_probs(result_df["generated_log_probs"].tolist())


class TestMiniMaxLLMStream:
	"""Test streaming methods."""

	def test_stream_not_implemented(self, minimax_m27_instance):
		with pytest.raises(NotImplementedError):
			minimax_m27_instance.stream("Hello")

	@pytest.mark.asyncio()
	@patch.object(
		openai.resources.chat.completions.AsyncCompletions,
		"create",
		mock_minimax_chat_create_stream,
	)
	async def test_astream_strips_think_tags(self, minimax_m27_instance):
		results = []
		async for partial in minimax_m27_instance.astream("Hello"):
			results.append(partial)

		assert results[-1] == "Visible answer"
		assert all("<think>" not in partial for partial in results)
		assert all("hidden reasoning" not in partial for partial in results)


class TestParsePrompt:
	"""Test the parse_prompt utility function."""

	def test_string_prompt(self):
		result = parse_prompt("Hello world")
		assert result == [{"role": "user", "content": "Hello world"}]

	def test_list_prompt(self):
		messages = [
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": "Hello"},
		]
		result = parse_prompt(messages)
		assert result == messages

	def test_invalid_prompt(self):
		with pytest.raises(ValueError):
			parse_prompt(123)


class TestMessagesToString:
	"""Test messages_to_string conversion."""

	def test_basic_conversion(self):
		messages = [
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": "Hi"},
		]
		result = messages_to_string(messages)
		assert "<|im_start|>user" in result
		assert "Hello" in result
		assert "<|im_start|>assistant" in result


class TestTruncateByToken:
	"""Test truncate_by_token function."""

	def test_truncate_string(self):
		tokenizer = tiktoken.get_encoding("o200k_base")
		# Short prompt should not be truncated
		result = truncate_by_token("Hello world", tokenizer, 100)
		assert "Hello world" in result

	def test_truncate_messages(self):
		tokenizer = tiktoken.get_encoding("o200k_base")
		messages = [{"role": "user", "content": "Hello world"}]
		result = truncate_by_token(messages, tokenizer, 100)
		assert isinstance(result, list)
		assert result == messages


class TestThinkTagHelpers:
	def test_strip_think_tags(self):
		assert (
			strip_think_tags("<think>Reasoning...</think>The answer is 42.")
			== "The answer is 42."
		)


class TestMaxTokenDict:
	"""Test MAX_TOKEN_DICT constants."""

	def test_known_models(self):
		assert "MiniMax-M2.7" in MAX_TOKEN_DICT
		assert "MiniMax-M2.7-highspeed" in MAX_TOKEN_DICT
		assert "MiniMax-M2.5" in MAX_TOKEN_DICT
		assert "MiniMax-M2.5-highspeed" in MAX_TOKEN_DICT

	def test_m27_context(self):
		assert MAX_TOKEN_DICT["MiniMax-M2.7"] == 1_048_576

	def test_m25_highspeed_context(self):
		assert MAX_TOKEN_DICT["MiniMax-M2.5-highspeed"] == 204_800


class TestMiniMaxAPIBase:
	"""Test API base URL constant."""

	def test_api_base_url(self):
		assert MINIMAX_API_BASE == "https://api.minimax.io/v1"


# --- Integration Tests ---


@pytest.mark.skipif(
	not os.getenv("MINIMAX_API_KEY"),
	reason="Integration test requires real MiniMax API key.",
)
class TestMiniMaxLLMIntegration:
	"""Integration tests that call the real MiniMax API."""

	def test_real_api_call(self):
		"""Test with real API - requires MINIMAX_API_KEY env var."""
		instance = MiniMaxLLM(
			project_dir=".",
			llm="MiniMax-M2.7",
		)
		answers, tokens, log_probs = instance._pure(
			["What is 2 + 2?"], temperature=0.1, max_tokens=50
		)
		assert len(answers) == 1
		assert isinstance(answers[0], str)
		assert len(answers[0]) > 0

	@pytest.mark.asyncio()
	async def test_real_astream(self):
		"""Test streaming with real API."""
		import asyncstdlib as a

		instance = MiniMaxLLM(
			project_dir=".",
			llm="MiniMax-M2.7",
		)
		result = []
		async for i, s in a.enumerate(
			instance.astream("What is 2 + 2?", temperature=0.1, max_tokens=50)
		):
			assert isinstance(s, str)
			result.append(s)
			if i >= 1:
				assert len(result[i]) >= len(result[i - 1])
		assert len(result) > 0

	def test_real_m25_highspeed(self):
		"""Test M2.5-highspeed model."""
		instance = MiniMaxLLM(
			project_dir=".",
			llm="MiniMax-M2.5-highspeed",
		)
		answers, tokens, log_probs = instance._pure(
			["Tell me a joke."], temperature=0.5, max_tokens=100
		)
		assert len(answers) == 1
		assert isinstance(answers[0], str)
