from types import SimpleNamespace
from unittest.mock import patch

import openai.resources.chat
import openai.resources.responses
import pandas as pd
import pytest
from pydantic import BaseModel

from autorag.nodes.generator import OpenAILLM
from autorag.nodes.generator.openai_llm import (
	GPT_5_LONG_CONTEXT,
	get_max_token_size,
)
from tests.autorag.nodes.generator.test_generator_base import (
	prompts,
	check_generated_texts,
	check_generated_tokens,
	check_generated_log_probs,
	chat_prompts,
)
from tests.delete_tests import is_github_action
from tests.mock import mock_openai_chat_create


@pytest.fixture
def openai_llm_instance():
	return OpenAILLM(
		project_dir=".", llm="gpt-3.5-turbo", api_key="mock_openai_api_key"
	)


@pytest.fixture
def openai_gpt_4_1_instance():
	return OpenAILLM(
		project_dir=".",
		llm="gpt-4.1",
		api_key="mock_openai_api_key",
	)


@pytest.fixture
def openai_reasoning_instance():
	return OpenAILLM(
		project_dir=".",
		llm="o4-mini",
		api_key="mock_openai_api_key",
	)


@pytest.fixture
def openai_gpt_5_instance():
	return OpenAILLM(
		project_dir=".",
		llm="gpt-5.6-pro",
		api_key="mock_openai_api_key",
	)


async def mock_openai_responses_create(*args, **kwargs):
	return SimpleNamespace(output_text="Why not")


@patch.object(
	openai.resources.responses.AsyncResponses,
	"create",
	mock_openai_responses_create,
)
def test_openai_llm_gpt_5(openai_gpt_5_instance):
	answers, tokens, log_probs = openai_gpt_5_instance._pure(
		prompts,
		reasoning={"effort": "high"},
		text={"verbosity": "low"},
	)
	check_generated_texts(answers)
	check_generated_tokens(tokens)
	check_generated_log_probs(log_probs)


@pytest.mark.parametrize(
	"model_name",
	["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.4"],
)
@patch.object(
	openai.resources.responses.AsyncResponses,
	"create",
	mock_openai_responses_create,
)
def test_openai_llm_latest_models(model_name):
	instance = OpenAILLM(project_dir=".", llm=model_name, api_key="mock_openai_api_key")
	assert instance.max_token_size == GPT_5_LONG_CONTEXT - 7
	answers, tokens, log_probs = instance._pure(
		prompts,
		reasoning={"effort": "medium"},
		text={"verbosity": "low"},
	)
	check_generated_texts(answers)
	check_generated_tokens(tokens)
	check_generated_log_probs(log_probs)


def test_get_max_token_size():
	# exact matches
	assert get_max_token_size("gpt-5.6-sol") == GPT_5_LONG_CONTEXT
	assert get_max_token_size("gpt-4o-mini") == 128_000
	# dated snapshots fall back to the base model entry
	assert get_max_token_size("gpt-5.6-sol-2026-07-09") == GPT_5_LONG_CONTEXT
	assert get_max_token_size("gpt-4o-mini-2024-07-18") == 128_000
	# unknown gpt-5.4+ variants fall back to the family context window
	assert get_max_token_size("gpt-5.6-ultra") == GPT_5_LONG_CONTEXT
	# retired models (gpt-5, gpt-5.1, ...) and unknown models return None
	assert get_max_token_size("gpt-5") is None
	assert get_max_token_size("gpt-5.1") is None
	assert get_max_token_size("gpt-5-pro") is None
	assert get_max_token_size("gpt-5-whatever") is None
	assert get_max_token_size("not-a-model") is None


def test_retired_models_raise():
	for retired_model in ["gpt-5", "gpt-5.1", "gpt-5-mini", "o1-preview", "gpt-4-32k"]:
		with pytest.raises(ValueError, match="does not supported"):
			OpenAILLM(project_dir=".", llm=retired_model, api_key="mock_openai_api_key")


@patch.object(
	openai.resources.chat.completions.AsyncCompletions,
	"create",
	mock_openai_chat_create,
)
def test_openai_llm(openai_llm_instance):
	answers, tokens, log_probs = openai_llm_instance._pure(
		prompts, temperature=0.5, logprobs=False, n=3
	)
	check_generated_texts(answers)
	check_generated_tokens(tokens)
	check_generated_log_probs(log_probs)

	answers, tokens, log_probs = openai_llm_instance._pure(chat_prompts)
	check_generated_texts(answers)
	check_generated_tokens(tokens)
	check_generated_log_probs(log_probs)


@patch.object(
	openai.resources.chat.completions.AsyncCompletions,
	"create",
	mock_openai_chat_create,
)
def test_openai_llm_gpt_41(openai_gpt_4_1_instance):
	answers, tokens, log_probs = openai_gpt_4_1_instance._pure(
		prompts, temperature=0.5, logprobs=False, n=3
	)
	check_generated_texts(answers)
	check_generated_tokens(tokens)
	check_generated_log_probs(log_probs)


@patch.object(
	openai.resources.chat.completions.AsyncCompletions,
	"create",
	mock_openai_chat_create,
)
def test_openai_llm_reasoning(openai_reasoning_instance):
	answer, tokens, log_probs = openai_reasoning_instance._pure(
		prompts,
		temperature=0.5,
		top_p=0.9,
		max_tokens=256,
		logprobs=True,
		top_logprobs=3,
		logit_bias=0.9,
	)
	check_generated_texts(answer)
	check_generated_tokens(tokens)
	check_generated_log_probs(log_probs)


@patch.object(
	openai.resources.chat.completions.AsyncCompletions,
	"create",
	mock_openai_chat_create,
)
def test_openai_llm_node():
	previous_result = pd.DataFrame(
		{"prompts": prompts, "qid": ["id-1", "id-2", "id-3"]}
	)
	result_df = OpenAILLM.run_evaluator(
		project_dir=".",
		previous_result=previous_result,
		llm="gpt-4o-mini",
		api_key="mock_openai_api_key",
		temperature=0.5,
	)
	check_generated_texts(result_df["generated_texts"].tolist())
	check_generated_tokens(result_df["generated_tokens"].tolist())
	check_generated_log_probs(result_df["generated_log_probs"].tolist())


@patch.object(
	openai.resources.chat.completions.AsyncCompletions,
	"create",
	mock_openai_chat_create,
)
def test_openai_llm_truncate(openai_llm_instance):
	prompt = [
		f"havertz on the block and I am {i}th player on the Arsenal."
		for i in range(50_000)
	]
	prompt = " ".join(prompt)
	answers, tokens, log_probs = openai_llm_instance._pure([prompt] * 3)
	check_generated_texts(answers)
	check_generated_tokens(tokens)
	check_generated_log_probs(log_probs)


class TestResponse(BaseModel):
	name: str
	phone_number: str
	age: int
	is_dead: bool


async def mock_gen_gt_response(*args, **kwargs):
	return SimpleNamespace(
		output_parsed=TestResponse(
			name="John Doe",
			phone_number="1234567890",
			age=30,
			is_dead=False,
		)
	)


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it uses the real OpenAI API.",
)
@patch.object(
	openai.resources.responses.AsyncResponses,
	"parse",
	mock_gen_gt_response,
)
def test_openai_llm_structured():
	llm = OpenAILLM(project_dir=".", llm="gpt-4o-mini-2024-07-18")
	prompt = """You must transform the user introduction to json format. You have to extract four information: name, phone number, age, and is_dead.
Hello, my name is John Doe. My phone number is 1234567890. I am 30 years old. I am alive. I am good at soccer."""

	response = llm.structured_output([prompt], TestResponse)
	assert isinstance(response[0], TestResponse)
	assert response[0].name == "John Doe"
	assert response[0].phone_number == "1234567890"
	assert response[0].age == 30
	assert response[0].is_dead is False

	llm = OpenAILLM(project_dir=".", llm="gpt-3.5-turbo")
	with pytest.raises(ValueError):
		llm.structured_output([prompt], TestResponse)


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it uses the real OpenAI API.",
)
@pytest.mark.asyncio()
async def test_openai_llm_astream():
	import asyncstdlib as a

	llm_instance = OpenAILLM(project_dir=".", llm="gpt-4o-mini-2024-07-18")
	result = []
	async for i, s in a.enumerate(
		llm_instance.astream("Hello. Tell me about who is Kai Havertz")
	):
		assert isinstance(s, str)
		result.append(s)
		if i >= 1:
			assert len(result[i]) >= len(result[i - 1])
