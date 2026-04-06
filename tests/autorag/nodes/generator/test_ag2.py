"""Tests for AG2 generator module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from autorag.nodes.generator import AG2Generator
from autorag.nodes.generator.ag2 import (
	DEFAULT_SYSTEM_MESSAGE,
	_extract_last_assistant_message,
	_format_prompt,
)
from autorag.schema.module import Module
from autorag.support import get_support_modules
from tests.autorag.nodes.generator.test_generator_base import (
	check_generated_log_probs,
	check_generated_texts,
	check_generated_tokens,
	chat_prompts,
	prompts,
)


def _make_mock_ag2(answer="Why not"):
	"""Create mock AG2 classes that return a fixed answer."""
	mock_assistant_cls = MagicMock()
	mock_user_proxy_cls = MagicMock()
	mock_llm_config_cls = MagicMock()

	mock_assistant = MagicMock()
	mock_assistant_cls.return_value = mock_assistant

	mock_user_proxy = MagicMock()
	mock_user_proxy_cls.return_value = mock_user_proxy

	mock_chat_result = MagicMock()
	mock_chat_result.process = AsyncMock()
	mock_user_proxy.a_run = AsyncMock(return_value=mock_chat_result)

	mock_assistant.chat_messages = {
		mock_user_proxy: [
			{"role": "assistant", "content": f"{answer} TERMINATE"},
		]
	}

	return mock_assistant_cls, mock_user_proxy_cls, mock_llm_config_cls


@pytest.fixture
def ag2_instance():
	return AG2Generator(
		project_dir=".",
		llm="gpt-4o-mini",
		api_key="mock_ag2_api_key",
	)


@pytest.fixture
def ag2_custom_instance():
	return AG2Generator(
		project_dir=".",
		llm="gpt-4o",
		batch=8,
		api_key="mock_ag2_api_key",
		api_type="openai",
		max_turns=3,
		system_message="Custom prompt.",
	)


# --- Unit Tests ---


class TestAG2GeneratorInit:
	"""Test AG2Generator initialization."""

	def test_default_init(self, ag2_instance):
		assert ag2_instance.llm == "gpt-4o-mini"
		assert ag2_instance.batch == 16
		assert ag2_instance.api_type == "openai"
		assert ag2_instance.max_turns == 1
		assert ag2_instance.system_message == DEFAULT_SYSTEM_MESSAGE

	def test_custom_init(self, ag2_custom_instance):
		assert ag2_custom_instance.llm == "gpt-4o"
		assert ag2_custom_instance.batch == 8
		assert ag2_custom_instance.max_turns == 3
		assert ag2_custom_instance.system_message == "Custom prompt."

	def test_init_invalid_batch(self):
		with pytest.raises(AssertionError):
			AG2Generator(
				project_dir=".",
				llm="gpt-4o-mini",
				batch=0,
				api_key="mock_key",
			)


class TestAG2ModuleRegistration:
	"""Test AG2Generator module registration in support.py."""

	def test_support_module_resolution(self):
		assert get_support_modules("ag2") is AG2Generator
		assert get_support_modules("AG2Generator") is AG2Generator

	def test_module_from_dict(self):
		module = Module.from_dict({"module_type": "ag2", "llm": "gpt-4o-mini"})
		assert module.module is AG2Generator


class TestAG2GeneratorPure:
	"""Test AG2Generator._pure method."""

	@patch("autorag.nodes.generator.ag2.UserProxyAgent")
	@patch("autorag.nodes.generator.ag2.AssistantAgent")
	@patch("autorag.nodes.generator.ag2.LLMConfig")
	def test_pure_string_prompts(
		self, mock_llm_config, mock_assistant_cls, mock_user_proxy_cls, ag2_instance
	):
		mock_assistant = MagicMock()
		mock_assistant_cls.return_value = mock_assistant
		mock_user_proxy = MagicMock()
		mock_user_proxy_cls.return_value = mock_user_proxy

		mock_chat_result = MagicMock()
		mock_chat_result.process = AsyncMock()
		mock_user_proxy.a_run = AsyncMock(return_value=mock_chat_result)

		mock_assistant.chat_messages = {
			mock_user_proxy: [
				{"role": "assistant", "content": "Why not TERMINATE"},
			]
		}

		answers, tokens, log_probs = ag2_instance._pure(prompts)
		check_generated_texts(answers)
		check_generated_tokens(tokens)
		check_generated_log_probs(log_probs)

	@patch("autorag.nodes.generator.ag2.UserProxyAgent")
	@patch("autorag.nodes.generator.ag2.AssistantAgent")
	@patch("autorag.nodes.generator.ag2.LLMConfig")
	def test_pure_chat_prompts(
		self, mock_llm_config, mock_assistant_cls, mock_user_proxy_cls, ag2_instance
	):
		mock_assistant = MagicMock()
		mock_assistant_cls.return_value = mock_assistant
		mock_user_proxy = MagicMock()
		mock_user_proxy_cls.return_value = mock_user_proxy

		mock_chat_result = MagicMock()
		mock_chat_result.process = AsyncMock()
		mock_user_proxy.a_run = AsyncMock(return_value=mock_chat_result)

		mock_assistant.chat_messages = {
			mock_user_proxy: [
				{"role": "assistant", "content": "Why not TERMINATE"},
			]
		}

		answers, tokens, log_probs = ag2_instance._pure(chat_prompts)
		check_generated_texts(answers)
		check_generated_tokens(tokens)
		check_generated_log_probs(log_probs)

	@patch("autorag.nodes.generator.ag2.UserProxyAgent")
	@patch("autorag.nodes.generator.ag2.AssistantAgent")
	@patch("autorag.nodes.generator.ag2.LLMConfig")
	def test_pure_uses_correct_llm_config(
		self, mock_llm_config, mock_assistant_cls, mock_user_proxy_cls
	):
		mock_assistant = MagicMock()
		mock_assistant_cls.return_value = mock_assistant
		mock_user_proxy = MagicMock()
		mock_user_proxy_cls.return_value = mock_user_proxy

		mock_chat_result = MagicMock()
		mock_chat_result.process = AsyncMock()
		mock_user_proxy.a_run = AsyncMock(return_value=mock_chat_result)

		mock_assistant.chat_messages = {
			mock_user_proxy: [
				{"role": "assistant", "content": "answer TERMINATE"},
			]
		}

		gen = AG2Generator(
			project_dir=".", llm="gpt-4o", api_key="my-key", api_type="openai"
		)
		gen._pure(["test"])

		mock_llm_config.assert_called_with(
			{
				"model": "gpt-4o",
				"api_key": "my-key",
				"api_type": "openai",
			}
		)

	@patch("autorag.nodes.generator.ag2.UserProxyAgent")
	@patch("autorag.nodes.generator.ag2.AssistantAgent")
	@patch("autorag.nodes.generator.ag2.LLMConfig")
	def test_pure_handles_exception(
		self, mock_llm_config, mock_assistant_cls, mock_user_proxy_cls, ag2_instance
	):
		mock_assistant = MagicMock()
		mock_assistant_cls.return_value = mock_assistant
		mock_user_proxy = MagicMock()
		mock_user_proxy_cls.return_value = mock_user_proxy

		mock_user_proxy.a_run = AsyncMock(side_effect=Exception("API error"))

		answers, tokens, log_probs = ag2_instance._pure(["test"])
		assert len(answers) == 1
		assert answers[0] == ""

	@patch("autorag.nodes.generator.ag2.UserProxyAgent")
	@patch("autorag.nodes.generator.ag2.AssistantAgent")
	@patch("autorag.nodes.generator.ag2.LLMConfig")
	def test_pure_empty_messages(
		self, mock_llm_config, mock_assistant_cls, mock_user_proxy_cls, ag2_instance
	):
		mock_assistant = MagicMock()
		mock_assistant_cls.return_value = mock_assistant
		mock_user_proxy = MagicMock()
		mock_user_proxy_cls.return_value = mock_user_proxy

		mock_chat_result = MagicMock()
		mock_chat_result.process = AsyncMock()
		mock_user_proxy.a_run = AsyncMock(return_value=mock_chat_result)

		mock_assistant.chat_messages = {mock_user_proxy: []}

		answers, _, _ = ag2_instance._pure(["test"])
		assert answers[0] == ""


class TestAG2GeneratorNode:
	"""Test AG2Generator as a node (run_evaluator)."""

	@patch("autorag.nodes.generator.ag2.UserProxyAgent")
	@patch("autorag.nodes.generator.ag2.AssistantAgent")
	@patch("autorag.nodes.generator.ag2.LLMConfig")
	def test_run_evaluator(
		self, mock_llm_config, mock_assistant_cls, mock_user_proxy_cls
	):
		mock_assistant = MagicMock()
		mock_assistant_cls.return_value = mock_assistant
		mock_user_proxy = MagicMock()
		mock_user_proxy_cls.return_value = mock_user_proxy

		mock_chat_result = MagicMock()
		mock_chat_result.process = AsyncMock()
		mock_user_proxy.a_run = AsyncMock(return_value=mock_chat_result)

		mock_assistant.chat_messages = {
			mock_user_proxy: [
				{"role": "assistant", "content": "Why not TERMINATE"},
			]
		}

		previous_result = pd.DataFrame(
			{"prompts": prompts, "qid": ["id-1", "id-2", "id-3"]}
		)
		result_df = AG2Generator.run_evaluator(
			project_dir=".",
			previous_result=previous_result,
			llm="gpt-4o-mini",
			api_key="mock_ag2_api_key",
		)
		check_generated_texts(result_df["generated_texts"].tolist())
		check_generated_tokens(result_df["generated_tokens"].tolist())
		check_generated_log_probs(result_df["generated_log_probs"].tolist())


class TestAG2GeneratorStream:
	"""Test streaming methods."""

	def test_stream_not_implemented(self, ag2_instance):
		with pytest.raises(NotImplementedError):
			ag2_instance.stream("Hello")

	@pytest.mark.asyncio()
	@patch("autorag.nodes.generator.ag2.UserProxyAgent")
	@patch("autorag.nodes.generator.ag2.AssistantAgent")
	@patch("autorag.nodes.generator.ag2.LLMConfig")
	async def test_astream(
		self, mock_llm_config, mock_assistant_cls, mock_user_proxy_cls, ag2_instance
	):
		mock_assistant = MagicMock()
		mock_assistant_cls.return_value = mock_assistant
		mock_user_proxy = MagicMock()
		mock_user_proxy_cls.return_value = mock_user_proxy

		mock_chat_result = MagicMock()
		mock_chat_result.process = AsyncMock()
		mock_user_proxy.a_run = AsyncMock(return_value=mock_chat_result)

		mock_assistant.chat_messages = {
			mock_user_proxy: [
				{"role": "assistant", "content": "The answer TERMINATE"},
			]
		}

		results = []
		async for partial in ag2_instance.astream("Hello"):
			results.append(partial)

		assert len(results) == 1
		assert results[0] == "The answer"


class TestFormatPrompt:
	"""Test the _format_prompt utility function."""

	def test_string_prompt(self):
		result = _format_prompt("Hello world")
		assert result == "Hello world"

	def test_list_prompt(self):
		messages = [
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": "Hello"},
		]
		result = _format_prompt(messages)
		assert "System: You are helpful." in result
		assert "Hello" in result

	def test_invalid_prompt(self):
		with pytest.raises(ValueError):
			_format_prompt(123)


class TestExtractLastAssistantMessage:
	"""Test the _extract_last_assistant_message utility function."""

	def test_extracts_last_message(self):
		assistant = MagicMock()
		user_proxy = MagicMock()
		assistant.chat_messages = {
			user_proxy: [
				{"role": "assistant", "content": "First answer"},
				{"role": "assistant", "content": "Second answer TERMINATE"},
			]
		}
		result = _extract_last_assistant_message(assistant, user_proxy)
		assert result == "Second answer"

	def test_empty_messages(self):
		assistant = MagicMock()
		user_proxy = MagicMock()
		assistant.chat_messages = {user_proxy: []}
		result = _extract_last_assistant_message(assistant, user_proxy)
		assert result == ""

	def test_only_terminate(self):
		assistant = MagicMock()
		user_proxy = MagicMock()
		assistant.chat_messages = {
			user_proxy: [
				{"role": "assistant", "content": "TERMINATE"},
			]
		}
		result = _extract_last_assistant_message(assistant, user_proxy)
		assert result == ""
