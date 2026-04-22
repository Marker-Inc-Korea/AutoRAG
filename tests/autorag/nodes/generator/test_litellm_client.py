from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from autorag.nodes.generator import LiteLLMGenerator
from autorag.nodes.generator.litellm_client import parse_prompt, truncate_by_token
from autorag.schema.module import Module
from autorag.support import get_support_modules
from tests.autorag.nodes.generator.test_generator_base import (
	prompts,
	check_generated_texts,
	check_generated_tokens,
	check_generated_log_probs,
	chat_prompts,
)


def _mock_litellm_response(content="Why not"):
	mock_message = MagicMock()
	mock_message.content = content

	mock_choice = MagicMock()
	mock_choice.message = mock_message

	mock_response = MagicMock()
	mock_response.choices = [mock_choice]
	mock_response.model = "gpt-4o-mini"
	mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=2)

	return mock_response


@pytest.fixture
def litellm_instance():
	return LiteLLMGenerator(
		project_dir=".",
		llm="gpt-4o-mini",
	)


@pytest.fixture
def litellm_instance_with_key():
	return LiteLLMGenerator(
		project_dir=".",
		llm="anthropic/claude-sonnet-4-20250514",
		api_key="sk-test-key",
		api_base="https://my-proxy.com",
	)


class TestLiteLLMGeneratorInit:
	def test_init_default(self, litellm_instance):
		assert litellm_instance.llm == "gpt-4o-mini"
		assert litellm_instance.batch == 16
		assert litellm_instance.api_key is None
		assert litellm_instance.api_base is None

	def test_init_with_credentials(self, litellm_instance_with_key):
		assert litellm_instance_with_key.llm == "anthropic/claude-sonnet-4-20250514"
		assert litellm_instance_with_key.api_key == "sk-test-key"
		assert litellm_instance_with_key.api_base == "https://my-proxy.com"

	def test_init_custom_batch(self):
		instance = LiteLLMGenerator(project_dir=".", llm="gpt-4o", batch=8)
		assert instance.batch == 8

	def test_init_invalid_batch(self):
		with pytest.raises(AssertionError):
			LiteLLMGenerator(project_dir=".", llm="gpt-4o", batch=0)


class TestLiteLLMModuleRegistration:
	def test_support_module_resolution(self):
		assert get_support_modules("litellm_client") is LiteLLMGenerator
		assert get_support_modules("LiteLLMGenerator") is LiteLLMGenerator

	def test_module_from_dict(self):
		module = Module.from_dict(
			{"module_type": "litellm_client", "llm": "gpt-4o-mini"}
		)
		assert module.module is LiteLLMGenerator


class TestLiteLLMGeneratorPure:
	@patch("autorag.nodes.generator.litellm_client.litellm")
	def test_pure_string_prompts(self, mock_litellm, litellm_instance):
		mock_litellm.acompletion = AsyncMock(return_value=_mock_litellm_response())

		answers, tokens, log_probs = litellm_instance._pure(prompts, temperature=0.5)
		check_generated_texts(answers)
		check_generated_tokens(tokens)
		check_generated_log_probs(log_probs)

	@patch("autorag.nodes.generator.litellm_client.litellm")
	def test_pure_chat_prompts(self, mock_litellm, litellm_instance):
		mock_litellm.acompletion = AsyncMock(return_value=_mock_litellm_response())

		answers, tokens, log_probs = litellm_instance._pure(chat_prompts)
		check_generated_texts(answers)
		check_generated_tokens(tokens)
		check_generated_log_probs(log_probs)

	@patch("autorag.nodes.generator.litellm_client.litellm")
	def test_pure_forwards_api_key(self, mock_litellm, litellm_instance_with_key):
		mock_litellm.acompletion = AsyncMock(return_value=_mock_litellm_response())

		litellm_instance_with_key._pure(prompts[:1])

		call_kwargs = mock_litellm.acompletion.call_args[1]
		assert call_kwargs["api_key"] == "sk-test-key"
		assert call_kwargs["api_base"] == "https://my-proxy.com"

	@patch("autorag.nodes.generator.litellm_client.litellm")
	def test_pure_sets_drop_params(self, mock_litellm, litellm_instance):
		mock_litellm.acompletion = AsyncMock(return_value=_mock_litellm_response())

		litellm_instance._pure(prompts[:1])

		call_kwargs = mock_litellm.acompletion.call_args[1]
		assert call_kwargs["drop_params"] is True

	@patch("autorag.nodes.generator.litellm_client.litellm")
	def test_pure_omits_api_key_when_none(self, mock_litellm, litellm_instance):
		mock_litellm.acompletion = AsyncMock(return_value=_mock_litellm_response())

		litellm_instance._pure(prompts[:1])

		call_kwargs = mock_litellm.acompletion.call_args[1]
		assert "api_key" not in call_kwargs


class TestLiteLLMGeneratorStream:
	def test_stream_not_implemented(self, litellm_instance):
		with pytest.raises(NotImplementedError):
			litellm_instance.stream("Hello")


class TestParsePrompt:
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


class TestLiteLLMImportError:
	def test_raises_import_error_without_litellm(self):
		with patch.dict("sys.modules", {"litellm": None}):
			with pytest.raises(ImportError, match="litellm is not installed"):
				LiteLLMGenerator(project_dir=".", llm="gpt-4o")
