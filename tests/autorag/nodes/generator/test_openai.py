import time
from unittest.mock import patch

import openai.resources.chat
import pandas as pd
import pytest
from pydantic import BaseModel

from autorag.nodes.generator import OpenAILLM
from tests.autorag.nodes.generator.test_generator_base import (
	prompts,
	check_generated_texts,
	check_generated_tokens,
	check_generated_log_probs,
)
from tests.delete_tests import is_github_action
from tests.mock import mock_openai_chat_create
import openai.resources.beta.chat
from openai.types.chat import (
	ParsedChatCompletion,
	ParsedChatCompletionMessage,
	ParsedChoice,
)


@pytest.fixture
def openai_llm_instance():
	return OpenAILLM(
		project_dir=".", llm="gpt-3.5-turbo", api_key="mock_openai_api_key"
	)


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


async def mock_gen_gt_response(*args, **kwargs) -> ParsedChatCompletion[TestResponse]:
	return ParsedChatCompletion(
		id="test_id",
		choices=[
			ParsedChoice(
				finish_reason="stop",
				index=0,
				message=ParsedChatCompletionMessage(
					parsed=TestResponse(
						name="John Doe",
						phone_number="1234567890",
						age=30,
						is_dead=False,
					),
					role="assistant",
				),
			)
		],
		created=int(time.time()),
		model="gpt-4o-mini-2024-07-18",
		object="chat.completion",
	)


@patch.object(
	openai.resources.beta.chat.completions.AsyncCompletions,
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
